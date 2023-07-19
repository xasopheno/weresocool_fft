use crossbeam_channel as channel;
use std::thread;

use weresocool_fft::{helpers::generate_sine_wave, wsc_fft::WscFFT};

fn main() {
    let sample_rate = 44100.0;
    let freq = 430.0;
    let buffer_size = 2048;
    let ring_buffer_size = 8;

    let sine_wave = generate_sine_wave(freq, sample_rate, buffer_size);
    let (s, r) = channel::unbounded();
    let (read_fn, _) = WscFFT::spawn(buffer_size, ring_buffer_size, sample_rate as usize, r);

    s.send(sine_wave).unwrap();
    thread::sleep(std::time::Duration::from_nanos(1));

    let result = read_fn();
    dbg!(result);
}
