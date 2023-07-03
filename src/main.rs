mod test_fft;
mod test_mel;
use hound::WavReader;
use num_complex::Complex;
use plotters::prelude::*;
use std::error::Error;
use std::f32::consts::PI;
use std::fs::File;
use std::io::BufReader;

fn hz_to_mel(hz: f32) -> f32 {
    2595.0 * (hz / 700.0 + 1.0).log10()
}

fn mel_to_hz(mel: f32) -> f32 {
    700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
}

fn fft_in_place(x: &mut [Complex<f32>]) {
    let n = x.len();
    if n <= 1 {
        return;
    }

    let mut even: Vec<Complex<f32>> = x.iter().step_by(2).cloned().collect();
    let mut odd: Vec<Complex<f32>> = x.iter().skip(1).step_by(2).cloned().collect();

    fft_in_place(&mut even);
    fft_in_place(&mut odd);

    for i in 0..n / 2 {
        let t = Complex::from_polar(1.0, -2.0 * PI * (i as f32) / (n as f32)) * odd[i];
        x[i] = even[i] + t;
        x[i + n / 2] = even[i] - t;
    }
}

/// Calculate the center frequencies for the Mel scale filters
fn calculate_center_freqs(num_filters: usize, sample_rate: f32) -> Vec<f32> {
    let max_mel = hz_to_mel(sample_rate / 2.0);
    let mel_increment = max_mel / (num_filters + 1) as f32;
    (0..=num_filters + 1)
        .map(|i| mel_to_hz(mel_increment * i as f32))
        .collect::<Vec<_>>()
}

/// Calculate the bin frequencies
fn calculate_bin_freqs(fft_size: usize, sample_rate: f32) -> Vec<f32> {
    (0..=fft_size / 2)
        .map(|i| sample_rate * (i as f32) / (fft_size as f32))
        .collect::<Vec<_>>()
}

/// Create a Mel filter bank
fn create_mel_filter_bank(num_filters: usize, fft_size: usize, sample_rate: f32) -> Vec<Vec<f32>> {
    let mut filter_bank = vec![vec![0.0; fft_size / 2 + 1]; num_filters];

    // Calculate center frequencies for each filter
    let center_freqs = calculate_center_freqs(num_filters, sample_rate);

    // Calculate bin frequencies
    let bin_freqs = calculate_bin_freqs(fft_size, sample_rate);

    for (i, filter) in filter_bank.iter_mut().enumerate() {
        let (start, center, end) = (center_freqs[i], center_freqs[i + 1], center_freqs[i + 2]);

        for (j, value) in filter.iter_mut().enumerate() {
            if bin_freqs[j] >= start && bin_freqs[j] <= center {
                *value = (bin_freqs[j] - start) / (center - start);
            } else if bin_freqs[j] > center && bin_freqs[j] <= end {
                *value = (end - bin_freqs[j]) / (end - center);
            }
        }
    }

    filter_bank
}

fn f32_to_complex(signal: &Vec<f32>) -> Vec<Complex<f32>> {
    signal
        .into_iter()
        .map(|val| Complex::new(*val, 0.0))
        .collect()
}

fn complex_to_f32(signal: Vec<Complex<f32>>) -> Vec<f32> {
    signal.into_iter().map(|val| val.re).collect()
}

fn process_and_draw_buffer(buffer: &mut Vec<f32>) -> Result<(), Box<dyn std::error::Error>> {
    // Convert buffer to complex values
    let mut complex_buffer = f32_to_complex(&buffer);

    // Perform FFT in-place
    fft_in_place(&mut complex_buffer);

    // Compute magnitudes
    let magnitude_buffer: Vec<_> = complex_buffer.iter().map(|c| c.norm()).collect();

    // Draw the buffer
    draw_buffer(&magnitude_buffer, 0.0, 0.2)?;

    // Clear the buffer for the next round of samples
    buffer.clear();

    Ok(())
}

fn process_buffer(buffer: &mut Vec<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    // Convert buffer to complex values
    let mut complex_buffer = f32_to_complex(&buffer);

    // Perform FFT in-place
    fft_in_place(&mut complex_buffer);

    // Compute magnitudes and return
    let magnitude_buffer: Vec<_> = complex_buffer.iter().map(|c| c.norm()).collect();

    // Clear the buffer for the next round of samples
    buffer.clear();

    Ok(magnitude_buffer)
}

fn draw_buffer(
    magnitude_buffer: &[f32],
    y_min: f32,
    y_max: f32,
) -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("fft_plot.png", (800, 600)).into_drawing_area();
    root.fill(&WHITE)?;

    let x: Vec<_> = (0..magnitude_buffer.len()).collect();
    let y: Vec<_> = magnitude_buffer.iter().cloned().collect();

    let mut chart = ChartBuilder::on(&root)
        .margin(10)
        .x_label_area_size(40)
        .y_label_area_size(40)
        .build_ranged(0..magnitude_buffer.len(), y_min..y_max)?;

    chart.configure_mesh().draw()?;
    chart.draw_series(LineSeries::new(x.into_iter().zip(y.into_iter()), &BLUE))?;

    Ok(())
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let reader = File::open("./src/simple.wav").expect("Failed to open file");
    let reader = BufReader::new(reader);
    let mut wav_reader = WavReader::new(reader).expect("Failed to read wav");
    println!("{:?}", &wav_reader.spec());

    let buffer_size = 1024 * 8; // Change as needed
    let mut buffer = Vec::with_capacity(buffer_size);

    for sample in wav_reader.samples::<f32>() {
        let sample = sample.expect("Failed to read sample");
        buffer.push(sample as f32);

        if buffer.len() >= buffer_size {
            let magnitude_buffer = process_buffer(&mut buffer)?;
            draw_buffer(&magnitude_buffer, 0.0, 0.2)?;
        }
    }

    // Don't forget to process the remaining samples in the buffer!
    if !buffer.is_empty() {
        let magnitude_buffer = process_buffer(&mut buffer)?;
        draw_buffer(&magnitude_buffer, 0.0, 0.2)?;
    }

    Ok(())
}
