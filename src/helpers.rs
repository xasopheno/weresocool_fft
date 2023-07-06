use std::f32::consts::PI;

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
        freq_error < 0.1,
        "Frequency error is too high: {}",
        freq_error
    );
}

pub fn generate_sine_wave(freq: f32, sample_rate: f32, buffer_size: usize) -> Vec<f32> {
    (0..buffer_size)
        .map(|i| ((2.0 * PI * freq * i as f32 / sample_rate).sin()))
        .collect()
}
