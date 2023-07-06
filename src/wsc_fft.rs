use crate::*;
use core::f32::consts::PI;
use crossbeam_channel as channel;
use std::sync::{Arc, RwLock};
use std::thread;
use weresocool_double_buffer::{DoubleBuffer, DoubleBufferError};

pub struct WscFFT {
    buffer: DoubleBuffer<f32>,
    window: Vec<f32>,
}

impl WscFFT {
    pub fn spawn(
        buffer_size: usize,
        r: channel::Receiver<Vec<f32>>,
    ) -> (
        Box<dyn Fn() -> Vec<f32> + Send + Sync>,
        thread::JoinHandle<()>,
    ) {
        let fft = WscFFT::new(buffer_size);

        let fft_arc = std::sync::Arc::new(std::sync::Mutex::new(fft));

        let fft_arc_clone = fft_arc.clone();

        let read_fn = Box::new(move || {
            let fft = fft_arc.lock().unwrap();
            let read_buffer_arc = fft.buffer.read();
            let read_buffer_guard = read_buffer_arc.read().unwrap();
            read_buffer_guard.to_owned()
        });

        let handle = thread::spawn(move || {
            for mut input_buffer in r.iter() {
                let mut fft = fft_arc_clone.lock().unwrap();
                if let Err(e) = fft.process(&mut input_buffer) {
                    eprintln!("fft.process failed: {:?}", e);
                }
            }
        });

        (read_fn, handle)
    }

    pub fn new(buffer_size: usize) -> Self {
        let buffer = DoubleBuffer::new(buffer_size);
        let window = hann_window(buffer_size);

        Self { buffer, window }
    }

    pub fn get_read_fn(&mut self) -> Box<dyn Fn() -> Vec<f32> + Send> {
        let read_buffer_arc = self.read();
        Box::new(move || {
            let read_buffer_guard = read_buffer_arc.read().unwrap();
            read_buffer_guard.clone()
        })
    }

    pub fn read(&mut self) -> Arc<RwLock<Vec<f32>>> {
        self.buffer.read()
    }

    pub fn write(&self, data: Vec<f32>) -> Result<(), DoubleBufferError> {
        self.buffer.write(data)
    }

    pub fn apply_window(buffer: &mut [f32], window: &[f32]) {
        for (b, w) in buffer.iter_mut().zip(window.iter()) {
            *b *= w;
        }
    }

    pub fn process(&mut self, buffer: &mut Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
        let mut magnitude_buffer = process_buffer(buffer)?;
        WscFFT::apply_window(&mut magnitude_buffer, &self.window);
        self.write(magnitude_buffer)?;

        Ok(())
    }
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

#[cfg(test)]
mod wsc_fft_test {
    use super::*;

    #[test]
    fn test_wsc_fft() {
        let sample_rate = 44100.0;
        let freq = 430.0;
        let buffer_size = 2048;

        let sine_wave = generate_sine_wave(freq, sample_rate, buffer_size);
        let (s, r) = channel::unbounded();
        let (read_fn, _) = WscFFT::spawn(buffer_size, r.clone());

        s.send(sine_wave).unwrap();
        thread::sleep(std::time::Duration::from_nanos(1));

        let result = read_fn();

        validate_frequency(result, freq, sample_rate, buffer_size);
    }
}
