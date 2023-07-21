use crate::*;
use core::f32::consts::PI;
use crossbeam_channel as channel;
use rustfft::num_complex::Complex;
use rustfft::FftPlanner;
use std::sync::{Arc, RwLock};
use std::thread;
use weresocool_ring_buffer::{RingBuffer, RingBufferError};

pub struct WscFFT {
    buffer: RingBuffer,
    window: Vec<f32>,
    fft: Arc<dyn rustfft::Fft<f32>>,
    complex_input: Vec<Complex<f32>>,
    magnitude_buffer: Vec<f32>,
    max_freq: f32,
}

fn amplitude_to_db(amplitude: f32) -> f32 {
    20.0 * (amplitude + 1e-6).log10()
}

fn equal_loudness_contour_correction(bin_index: f32, amplitude: f32) -> f32 {
    let freq = bin_index * 48_000.0 * 2.0 / 2048.0;
    // This is just a basic approximation using the ISO 226:2003 equal-loudness contour for phon level 80.
    let correction = 3.64 * (freq / 1000.0).powf(-0.8)
        - 6.5 * (-0.6 * (freq / 1000.0 - 3.3).exp2())
        + 0.001 * (freq / 1000.0).powf(4.0);
    amplitude * correction
}

fn sigmoid(x: f32) -> f32 {
    let steepness = 10.0; // Experiment with this value
    let bias = 0.0; // This value shifts the sigmoid function to the right
    1.0 / (1.0 + ((-steepness * x) + bias).exp())
}

impl WscFFT {
    pub fn spawn(
        buffer_size: usize,
        ring_buffer_size: usize,
        sample_rate: usize,
        receiver: channel::Receiver<Vec<f32>>,
    ) -> (
        Box<dyn Fn() -> Vec<f32> + Send + Sync>,
        thread::JoinHandle<()>,
    ) {
        let fft = WscFFT::new(buffer_size, ring_buffer_size, sample_rate);

        let fft_arc = Arc::new(RwLock::new(fft));

        let fft_arc_clone = fft_arc.clone();

        let read_fn = Box::new(move || {
            let fft = fft_arc.read().unwrap();
            // let read_buffer_arc = fft.buffer.read();
            // let read_buffer_guard = read_buffer_arc.read().unwrap();
            // read_buffer_guard.to_owned()
            fft.buffer.read()
        });

        let handle = thread::spawn(move || {
            for mut large_input_buffer in receiver.iter() {
                let mut fft = fft_arc_clone.write().unwrap();
                let chunk_size = 1024 * 2;
                let input_buffers: Vec<_> = large_input_buffer.chunks_mut(chunk_size).collect();
                let mut buffer_vec: Vec<f32>;
                for input_buffer in input_buffers {
                    buffer_vec = input_buffer.to_vec();
                    if let Err(e) = fft.process(&mut buffer_vec) {
                        eprintln!("fft.process failed: {:?}", e);
                    }
                }
            }
        });

        (read_fn, handle)
    }

    pub fn new(buffer_size: usize, ring_buffer_size: usize, sample_rate: usize) -> Self {
        let buffer = RingBuffer::new(buffer_size, ring_buffer_size, sample_rate);
        let window = hamming_window(buffer_size);

        let mut planner = FftPlanner::new();
        let fft = planner.plan_fft_forward(buffer_size);

        Self {
            buffer,
            window,
            fft,
            max_freq: 1.0,
            complex_input: vec![Complex { re: 0.0, im: 0.0 }; buffer_size],
            magnitude_buffer: vec![0.0; buffer_size],
        }
    }

    pub fn read(&mut self) -> Vec<f32> {
        self.buffer.read()
    }

    pub fn write(&mut self) -> Result<(), RingBufferError> {
        self.buffer.write(self.magnitude_buffer.clone())
    }

    pub fn apply_window(buffer: &mut [f32], window: &[f32]) {
        for (b, w) in buffer.iter_mut().zip(window.iter()) {
            *b *= w;
        }
    }

    fn _log_bin_fft(fft_res: &Vec<Complex<f32>>) -> Vec<Complex<f32>> {
        let min_bin = 1.0;
        let max_bin = (fft_res.len() / 2) as f32;
        let bins_per_octave = 12.0; // number of bins per octave
        let min_freq = 27.5; // frequency of low A (A1)
        let sample_rate = 44100.0; // sample rate of the audio
        let factor = bins_per_octave / (2.0f32).log2();

        let mut log_binned = vec![Complex::new(0.0, 0.0); fft_res.len()];
        let mut bin_sums = vec![0.0; fft_res.len()];
        let mut bin_counts = vec![0; fft_res.len()];

        for (i, &value) in fft_res.iter().enumerate() {
            if i < 2 {
                continue;
            } // skip DC and Nyquist
            let freq = i as f32 * sample_rate / fft_res.len() as f32;
            let bin = ((freq / min_freq).log2() * factor).round() as usize;
            if bin < min_bin as usize || bin >= max_bin as usize {
                continue;
            }
            log_binned[bin] += value;
            bin_sums[bin] += value.norm();
            bin_counts[bin] += 1;
        }

        for (bin, &count) in bin_counts.iter().enumerate() {
            if count > 0 {
                log_binned[bin] /= count as f32;
            }
        }

        log_binned
    }

    pub fn process(&mut self, buffer: &mut [f32]) -> Result<(), Box<dyn std::error::Error>> {
        WscFFT::apply_window(buffer, &self.window);

        for (complex, &f) in self.complex_input.iter_mut().zip(buffer.iter()) {
            complex.re = f;
            complex.im = 0.0;
        }

        self.fft.process(&mut self.complex_input);

        let mut norm_values: Vec<f32> = self
            .complex_input
            .iter()
            .map(|c| c.norm().powf(0.25))
            .collect();

        smooth_inplace(&mut norm_values, 5);

        if let Some(max_norm) = norm_values
            .iter()
            .fold(Some(std::f32::NEG_INFINITY), |a, &b| match a {
                None => Some(b),
                Some(a) => Some(a.max(b)),
            })
        {
            self.max_freq = self.max_freq.max(max_norm);
        }

        self.magnitude_buffer.clear();
        self.magnitude_buffer
            .extend(norm_values.iter().map(|&norm| norm / self.max_freq));

        self.write()?;

        Ok(())
    }
}

fn smooth_inplace(buffer: &mut [f32], window_size: usize) {
    let padding = vec![0.0; window_size / 2];
    let extended_buffer = [&padding[..], &buffer[..], &padding[..]].concat();

    for i in 0..buffer.len() {
        let window = &extended_buffer[i..i + window_size];
        let sum: f32 = window.iter().sum();
        buffer[i] = sum / window_size as f32;
    }
}

fn smooth(buffer: &Vec<f32>, window_size: usize) -> Vec<f32> {
    let mut result = Vec::new();
    let padding = vec![0.0; window_size / 2];
    let extended_buffer = [&padding[..], &buffer[..], &padding[..]].concat();

    for i in 0..buffer.len() {
        let window = &extended_buffer[i..i + window_size];
        let sum: f32 = window.iter().sum();
        result.push(sum / window_size as f32);
    }
    result
}

fn _create_mel_filter_bank(
    num_filters: usize,
    min_freq: f32,
    max_freq: f32,
    fft_size: usize,
    sample_rate: usize,
) -> Vec<Vec<f32>> {
    let min_mel = 2595.0 * (1.0 + min_freq / 700.0).log10();
    let max_mel = 2595.0 * (1.0 + max_freq / 700.0).log10();

    let mels: Vec<f32> = (0..=num_filters + 1)
        .map(|i| min_mel + (max_mel - min_mel) * i as f32 / (num_filters + 1) as f32)
        .collect();

    let freqs: Vec<f32> = mels
        .iter()
        .map(|&m| 700.0 * (10.0f32.powf(m / 2595.0) - 1.0))
        .collect();

    let bins: Vec<usize> = freqs
        .iter()
        .map(|&f| (fft_size as f32 * f / sample_rate as f32).round() as usize)
        .collect();

    let mut filters = vec![vec![0.0; fft_size / 2]; num_filters];

    for i in 0..num_filters {
        for f in bins[i]..bins[i + 1] {
            filters[i][f] = (f - bins[i]) as f32 / (bins[i + 1] - bins[i]) as f32;
        }
        for f in bins[i + 1]..bins[i + 2] {
            filters[i][f] = 1.0 - (f - bins[i + 1]) as f32 / (bins[i + 2] - bins[i + 1]) as f32;
        }
    }

    filters
}

pub fn validate_frequency(data: Vec<f32>, freq: f32, sample_rate: f32, buffer_size: usize) {
    let max_index = data
        .iter()
        .enumerate()
        .take(data.len() / 2 - 1)
        .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .unwrap()
        .0;

    let expected_max_index = (freq * buffer_size as f32 / sample_rate).round() as usize;
    assert_eq!(max_index, expected_max_index);

    let recovered_freq = max_index as f32 * sample_rate / buffer_size as f32;
    let freq_error = (recovered_freq - freq).abs();
    dbg!(recovered_freq, freq_error);
    assert!(
        freq_error < 0.7,
        "Frequency error is too high: {}",
        freq_error
    );
}

pub fn generate_sine_wave(freq: f32, sample_rate: f32, buffer_size: usize) -> Vec<f32> {
    (0..buffer_size)
        .map(|i| ((2.0 * PI * freq * i as f32 / sample_rate).sin()))
        .collect()
}

fn _low_pass_filter(buffer: &mut Vec<f32>, window_size: usize) {
    let mut sum = 0.0;
    let scale = 1.0 / window_size as f32;

    for i in 0..buffer.len() {
        sum += buffer[i];

        if i >= window_size {
            sum -= buffer[i - window_size]
        }

        buffer[i] = sum * scale;
    }
}

// #[cfg(test)]
// mod wsc_fft_test {
// use super::*;

// #[test]
// fn test_wsc_fft() {
// let sample_rate = 44100.0;
// let freq = 430.0;
// let buffer_size = 2048;

// let sine_wave = generate_sine_wave(freq, sample_rate, buffer_size);
// let (s, r) = channel::unbounded();
// let (read_fn, _) = WscFFT::spawn(buffer_size, r.clone());

// s.send(sine_wave).unwrap();
// thread::sleep(std::time::Duration::from_nanos(1));

// let result = read_fn();

// validate_frequency(result, freq, sample_rate, buffer_size);
// }
// }
