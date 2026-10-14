//! PCM の形式の変換。Python の audioop (wyoming の AudioChunkConverter が使う) と同じ計算をする

use crate::domain::audio::Audio;

fn read_raw(data: &[u8], width: usize, index: usize) -> i32 {
    let bytes = &data[index..index + width];
    match width {
        1 => i32::from(bytes[0] as i8),
        2 => i32::from(i16::from_le_bytes([bytes[0], bytes[1]])),
        3 => (i32::from_le_bytes([0, bytes[0], bytes[1], bytes[2]])) >> 8,
        4 => i32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]),
        _ => unreachable!("幅は 1〜4 バイト"),
    }
}

fn write_raw(out: &mut Vec<u8>, width: usize, value: i32) {
    out.extend_from_slice(&value.to_le_bytes()[..width]);
}

/// 32bit に揃えた値として読む
fn read_sample32(data: &[u8], width: usize, index: usize) -> i32 {
    let value = read_raw(data, width, index);
    match width {
        1 => value << 24,
        2 => value << 16,
        3 => value << 8,
        _ => value,
    }
}

fn write_sample32(out: &mut Vec<u8>, width: usize, value: i32) {
    let value = match width {
        1 => value >> 24,
        2 => value >> 16,
        3 => value >> 8,
        _ => value,
    };
    write_raw(out, width, value);
}

fn bounds(width: usize) -> (f64, f64) {
    let max = ((1i64 << (width * 8 - 1)) - 1) as f64;
    (-max - 1.0, max)
}

fn fbound(value: f64, min: f64, max: f64) -> i32 {
    let value = if value > max {
        max
    } else if value < min + 1.0 {
        min
    } else {
        value
    };
    value.floor() as i32
}

/// 8bit PCM の符号なし・符号付きの切り替え
fn bias_8bit(data: &[u8], bias: i16) -> Vec<u8> {
    data.iter().map(|&b| (i16::from(b) + bias).rem_euclid(256) as u8).collect()
}

fn lin2lin(data: &[u8], width: usize, new_width: usize) -> Vec<u8> {
    if width == new_width {
        return data.to_vec();
    }
    let mut out = Vec::with_capacity(data.len() / width * new_width);
    for index in (0..data.len()).step_by(width) {
        write_sample32(&mut out, new_width, read_sample32(data, width, index));
    }
    out
}

fn tomono(data: &[u8], width: usize) -> Vec<u8> {
    let (min, max) = bounds(width);
    let mut out = Vec::with_capacity(data.len() / 2);
    for index in (0..data.len()).step_by(width * 2) {
        let left = f64::from(read_raw(data, width, index));
        let right = f64::from(read_raw(data, width, index + width));
        write_raw(&mut out, width, fbound(left + right, min, max));
    }
    out
}

fn tostereo(data: &[u8], width: usize) -> Vec<u8> {
    let (min, max) = bounds(width);
    let mut out = Vec::with_capacity(data.len() * 2);
    for index in (0..data.len()).step_by(width) {
        let sample = fbound(f64::from(read_raw(data, width, index)), min, max);
        write_raw(&mut out, width, sample);
        write_raw(&mut out, width, sample);
    }
    out
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// audioop.ratecv の状態。チャンクをまたいでリサンプリングを続けるために使う
#[derive(Debug, Clone)]
struct RateState {
    d: i64,
    samples: Vec<(i32, i32)>,
}

fn ratecv(
    data: &[u8],
    width: usize,
    channels: usize,
    in_rate: u32,
    out_rate: u32,
    state: Option<RateState>,
) -> (Vec<u8>, RateState) {
    let divisor = gcd(in_rate.into(), out_rate.into());
    let in_rate = i64::from(in_rate) / divisor;
    let out_rate = i64::from(out_rate) / divisor;

    let (mut d, samples) = match state {
        Some(state) => (state.d, state.samples),
        None => (-out_rate, vec![(0, 0); channels]),
    };
    let (mut prev, mut cur): (Vec<i32>, Vec<i32>) = samples.into_iter().unzip();

    let frame_size = width * channels;
    let mut frames = data.len() / frame_size;
    let mut out = Vec::with_capacity(((frames as i64 * out_rate / in_rate + 1) as usize) * frame_size);
    let mut position = 0;
    loop {
        while d < 0 {
            if frames == 0 {
                return (out, RateState { d, samples: prev.into_iter().zip(cur).collect() });
            }
            for channel in 0..channels {
                prev[channel] = cur[channel];
                cur[channel] = read_sample32(data, width, position);
                position += width;
            }
            frames -= 1;
            d += out_rate;
        }
        while d >= 0 {
            for channel in 0..channels {
                let value = (f64::from(prev[channel]) * d as f64 + f64::from(cur[channel]) * (out_rate - d) as f64)
                    / out_rate as f64;
                write_sample32(&mut out, width, value as i32);
            }
            d -= in_rate;
        }
    }
}

/// 音声を指定の形式に変換する。リサンプリングの状態を持つため、続いた音声には同じ変換器を使い回す
#[derive(Debug, Clone)]
pub struct AudioConverter {
    rate: u32,
    width: u16,
    channels: u16,
    rate_state: Option<RateState>,
}

impl AudioConverter {
    pub fn new(rate: u32, width: u16, channels: u16) -> Self {
        Self { rate, width, channels, rate_state: None }
    }

    /// 形式は 1〜4 バイト、モノラルかステレオで、データはフレーム境界に揃っている必要がある
    pub fn convert(&mut self, audio: &Audio) -> Audio {
        if audio.rate == self.rate && audio.width == self.width && audio.channels == self.channels {
            return audio.clone();
        }

        let mut data = audio.data.clone();
        let mut width = usize::from(audio.width);
        if audio.width != self.width {
            if width == 1 {
                data = bias_8bit(&data, -128);
            }
            data = lin2lin(&data, width, usize::from(self.width));
            width = usize::from(self.width);
            if width == 1 {
                data = bias_8bit(&data, 128);
            }
        }

        if audio.channels != self.channels {
            data = if self.channels == 1 { tomono(&data, width) } else { tostereo(&data, width) };
        }

        if audio.rate != self.rate {
            let (converted, state) =
                ratecv(&data, width, usize::from(self.channels), audio.rate, self.rate, self.rate_state.take());
            data = converted;
            self.rate_state = Some(state);
        }

        Audio::new(self.rate, self.width, self.channels, data)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_width_and_channels() {
        let mut converter = AudioConverter::new(8000, 1, 1);
        // Python 版と同じく、8bit 符号なしに変えてからモノラルにまとめる
        let audio = converter.convert(&Audio::new(8000, 2, 2, vec![0x00, 0x01, 0x00, 0x01]));
        assert_eq!(audio.data, vec![0x80]);
        let mut converter = AudioConverter::new(8000, 2, 2);
        let audio = converter.convert(&Audio::new(8000, 1, 1, vec![0x90]));
        assert_eq!(audio.data, vec![0x00, 0x10, 0x00, 0x10]);
    }

    #[test]
    fn resamples_with_state_across_chunks() {
        let mut converter = AudioConverter::new(16000, 2, 1);
        let chunk = Audio::new(8000, 2, 1, [1000i16.to_le_bytes(); 100].concat());
        let first = converter.convert(&chunk);
        let second = converter.convert(&chunk);
        assert_eq!(first.data.len() / 2 + second.data.len() / 2, 400 - 1);
        assert_eq!(&first.data[..4], &[0xe8, 0x03, 0xe8, 0x03]);

        let mut converter = AudioConverter::new(22050, 2, 1);
        let chunk = Audio::new(11025, 2, 2, [1u8, 0, 1, 0].repeat(1102));
        let sizes: Vec<usize> = (0..4).map(|_| converter.convert(&chunk).data.len()).collect();
        assert_eq!(sizes, [4406, 4408, 4408, 4408]);
    }
}
