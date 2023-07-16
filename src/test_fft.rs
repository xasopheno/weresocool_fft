#[cfg(test)]
mod test_fft {
    use crate::*;
    use num_complex::Complex;
    use num_traits::Zero;
    use rustfft::algorithm::Radix4;
    use rustfft::{Fft, FftDirection};

    #[test]
    fn test_fft_of_zeros() {
        let mut input: Vec<Complex<f32>> = vec![Complex::zero(); 4];
        fft_in_place(&mut input);
        assert!(input.iter().all(|&x| x == Complex::zero()));
    }

    #[test]
    fn test_fft_of_constant() {
        let mut input: Vec<Complex<f32>> = vec![Complex::new(1.0, 0.0); 4];
        fft_in_place(&mut input);
        assert_eq!(input[0], Complex::new(4.0, 0.0));
        assert!(input[1..].iter().all(|&x| x == Complex::zero()));
    }

    #[test]
    fn test_fft_and_ifft_inverse() {
        let original: Vec<Complex<f32>> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(3.0, 3.0),
            Complex::new(4.0, 4.0),
        ];
        let mut copy = original.clone();
        fft_in_place(&mut copy);

        // Create a FFT instance
        let fft = Radix4::new(copy.len(), FftDirection::Inverse);
        fft.process(&mut copy);

        for v in &mut copy {
            *v /= original.len() as f32;
        }

        for (x, y) in original.iter().zip(copy.iter()) {
            assert!((x.re - y.re).abs() < 1e-6);
            assert!((x.im - y.im).abs() < 1e-6);
        }
    }

    #[test]
    fn test_parsevals_theorem() {
        let mut input: Vec<Complex<f32>> = vec![
            Complex::new(0.0, 0.0),
            Complex::new(1.0, 1.0),
            Complex::new(3.0, 3.0),
            Complex::new(4.0, 4.0),
        ];

        let time_domain_sum: f32 = input.iter().map(|x| x.norm_sqr()).sum();

        fft_in_place(&mut input);

        let freq_domain_sum: f32 =
            input.iter().map(|x| x.norm_sqr()).sum::<f32>() / input.len() as f32;

        assert!((time_domain_sum - freq_domain_sum).abs() < 1e-6);
    }

    #[test]
    fn test_frequency_resolution() {
        let sample_rate = 8000.0;
        let duration = 1.0;
        let freq = 440.0;
        let amplitude = 1.0;

        let num_samples = (sample_rate * duration) as usize;
        let t = (0..num_samples)
            .map(|i| i as f32 / sample_rate)
            .collect::<Vec<_>>();

        let signal = t
            .iter()
            .map(|t| Complex::new(amplitude * (2.0 * PI * freq * t).sin(), 0.0))
            .collect::<Vec<_>>();

        let mut fft_output = signal.clone();

        fft_in_place(&mut fft_output);

        let magnitudes = fft_output.iter().map(|c| c.norm()).collect::<Vec<_>>();
        let max_index = magnitudes
            .iter()
            .take(num_samples / 2)
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;

        let expected_max_index = (freq * num_samples as f32 / sample_rate).round() as usize;
        assert_eq!(max_index, expected_max_index);

        let recovered_freq = max_index as f32 * sample_rate / num_samples as f32;
        let freq_error = (recovered_freq - freq).abs();
        assert!(
            freq_error < 0.01,
            "Frequency error is too high: {}",
            freq_error
        );
    }
}
