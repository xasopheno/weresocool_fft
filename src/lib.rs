pub mod helpers;
mod test_fft;
mod test_mel;
pub mod wsc_fft;
pub use crate::wsc_fft::WscFFT;
use num_complex::Complex;
use plotters::prelude::*;
use std::f32::consts::PI;

pub fn fft_in_place(x: &mut [Complex<f32>]) {
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

pub fn hann_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.5 * (1.0 - (2.0 * PI * i as f32 / (size as f32 - 1.0)).cos()))
        .collect()
}

pub fn hamming_window(size: usize) -> Vec<f32> {
    (0..size)
        .map(|i| 0.54 - 0.46 * (2.0 * PI * i as f32 / (size as f32 - 1.0)).cos())
        .collect()
}

pub fn apply_window(buffer: &mut [f32], window: &[f32]) {
    for (b, w) in buffer.iter_mut().zip(window.iter()) {
        *b *= w;
    }
}

/// Calculate the bin frequencies
pub fn calculate_bin_freqs(fft_size: usize, sample_rate: f32) -> Vec<f32> {
    (0..=fft_size / 2)
        .map(|i| sample_rate * (i as f32) / (fft_size as f32))
        .collect::<Vec<_>>()
}

pub fn f32_to_complex(signal: &Vec<f32>) -> Vec<Complex<f32>> {
    signal
        .into_iter()
        .map(|val| Complex::new(*val, 0.0))
        .collect()
}

pub fn complex_to_f32(signal: Vec<Complex<f32>>) -> Vec<f32> {
    signal.into_iter().map(|val| val.re).collect()
}

pub fn process_buffer(buffer: &mut Vec<f32>) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let mut complex_buffer = f32_to_complex(&buffer);

    fft_in_place(&mut complex_buffer);

    Ok(complex_buffer.iter().map(|c| c.norm()).collect())
}

pub fn draw_buffer(
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
        .build_cartesian_2d(0..magnitude_buffer.len(), y_min..y_max)?;

    chart.configure_mesh().draw()?;
    chart.draw_series(LineSeries::new(x.into_iter().zip(y.into_iter()), &BLUE))?;

    Ok(())
}
