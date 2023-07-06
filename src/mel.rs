// /// Create a Mel filter bank
// fn create_mel_filter_bank(num_filters: usize, fft_size: usize, sample_rate: f32) -> Vec<Vec<f32>> {
// let mut filter_bank = vec![vec![0.0; fft_size / 2 + 1]; num_filters];

// // Calculate center frequencies for each filter
// let center_freqs = calculate_center_freqs(num_filters, sample_rate);

// // Calculate bin frequencies
// let bin_freqs = calculate_bin_freqs(fft_size, sample_rate);

// for (i, filter) in filter_bank.iter_mut().enumerate() {
// let (start, center, end) = (center_freqs[i], center_freqs[i + 1], center_freqs[i + 2]);

// for (j, value) in filter.iter_mut().enumerate() {
// if bin_freqs[j] >= start && bin_freqs[j] <= center {
// *value = (bin_freqs[j] - start) / (center - start);
// } else if bin_freqs[j] > center && bin_freqs[j] <= end {
// *value = (end - bin_freqs[j]) / (end - center);
// }
// }
// }

// filter_bank
// }
//
// /// Calculate the center frequencies for the Mel scale filters
// fn calculate_center_freqs(num_filters: usize, sample_rate: f32) -> Vec<f32> {
// let max_mel = hz_to_mel(sample_rate / 2.0);
// let mel_increment = max_mel / (num_filters + 1) as f32;
// (0..=num_filters + 1)
// .map(|i| mel_to_hz(mel_increment * i as f32))
// .collect::<Vec<_>>()
// }
//

// fn hz_to_mel(hz: f32) -> f32 {
// 2595.0 * (hz / 700.0 + 1.0).log10()
// }

// fn mel_to_hz(mel: f32) -> f32 {
// 700.0 * (10.0f32.powf(mel / 2595.0) - 1.0)
// }
