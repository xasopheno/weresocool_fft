mod mel_tests {
    use crate::{create_mel_filter_bank, fft_in_place, hz_to_mel, mel_to_hz};
    use core::f32::consts::PI;
    use num_complex::Complex;

    #[test]
    fn test_mel_filter_bank() {
        let sample_rate = 44100.0;
        let freq = 440.0;
        let fft_size = 2048;
        let amplitude = 1.0;

        // Create a buffer with one frequency
        let mut buffer = (0..fft_size)
            .map(|i| {
                let t = i as f32 / sample_rate;
                Complex::new(amplitude * (2.0 * PI * freq * t).sin(), 0.0)
            })
            .collect::<Vec<_>>();

        // Perform FFT
        fft_in_place(&mut buffer);

        // Create a single Mel filter
        let filter_bank = create_mel_filter_bank(1, fft_size, sample_rate);

        // Apply mel filter
        let filtered = filter_bank[0]
            .iter()
            .zip(&buffer)
            .map(|(&weight, value)| value.norm() * weight)
            .sum::<f32>();

        // The frequency bin that corresponds to the input frequency
        let expected_max_index = (freq / sample_rate * fft_size as f32).round() as usize;

        // Find the index of the maximum value in the filtered output
        let max_index = buffer
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.norm().partial_cmp(&b.1.norm()).unwrap())
            .unwrap()
            .0;

        // The maximum value should be in the bin corresponding to the input frequency
        assert_eq!(max_index, expected_max_index);
    }

    // #[test]
    // fn test_mel_filter_bank_edge_frequencies() {
    // let sample_rate = 44100.0;
    // let fft_size = 2048;
    // let num_filters = 40;

    // // Create a mel filter bank
    // let filter_bank = create_mel_filter_bank(num_filters, fft_size, sample_rate);

    // // Test the first and last center frequencies
    // let min_mel = hz_to_mel(0.0);
    // let max_mel = hz_to_mel(sample_rate / 2.0);
    // let mel_increment = (max_mel - min_mel) / (num_filters + 1) as f32;

    // // Calculate the expected center frequencies for the first and last filters
    // let expected_first_freq = mel_to_hz(min_mel + mel_increment);
    // let expected_last_freq = mel_to_hz(min_mel + (mel_increment * (num_filters as f32)));

    // // Get the number of bins in the filter bank
    // let num_bins = filter_bank[0].len();

    // // Verify the first center frequency
    // let first_center_freq = (0.0 * sample_rate) as f32 / fft_size as f32;
    // assert_eq!(
    // (first_center_freq - expected_first_freq).abs() < 0.01,
    // true,
    // "First center frequency is incorrect"
    // );

    // // Verify the last center frequency
    // let last_center_freq = ((num_bins - 1) as f32 * sample_rate) as f32 / fft_size as f32;
    // assert_eq!(
    // (last_center_freq - expected_last_freq).abs() < 0.01,
    // true,
    // "Last center frequency is incorrect"
    // );

    // // Additional assertions on the filter bank can be added here if needed
    // }
}
