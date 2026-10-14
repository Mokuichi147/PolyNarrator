use thiserror::Error;

/// 音声形式として受け付ける上限。異常な形式の音声で無音などに大量のメモリを使わないようにする
pub const MAX_RATE: i64 = 384_000;
pub const MAX_WIDTH: i64 = 4;
// 形式の変換がモノラルとステレオにしか対応していないため
pub const MAX_CHANNELS: i64 = 2;
/// 1回の合成で受け取る音声の上限。音声を送り続ける異常なサーバーでメモリを使い切らないようにする
pub const MAX_AUDIO_BYTES: usize = 1 << 30;

/// PCM音声データ
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Audio {
    pub rate: u32,
    pub width: u16,
    pub channels: u16,
    pub data: Vec<u8>,
}

impl Audio {
    pub fn new(rate: u32, width: u16, channels: u16, data: Vec<u8>) -> Self {
        Self { rate, width, channels, data }
    }

    pub fn frame_size(&self) -> usize {
        usize::from(self.width) * usize::from(self.channels)
    }
}

/// 文の内容ではなくサーバーの不具合とみなす音声
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct InvalidAudio(pub String);

/// 扱えない形式やフレーム境界の合わない音声は、文の内容ではなくサーバーの不具合とみなす
pub fn validate_format(rate: i64, width: i64, channels: i64, size: usize) -> Result<(), InvalidAudio> {
    let valid_format =
        0 < rate && rate <= MAX_RATE && 0 < width && width <= MAX_WIDTH && 0 < channels && channels <= MAX_CHANNELS;
    if !valid_format || size % (width * channels) as usize != 0 {
        return Err(InvalidAudio(format!(
            "TTSサーバーから不正な音声が返されました(rate={rate}, width={width}, channels={channels}, {size}バイト)"
        )));
    }
    Ok(())
}

/// 空の音声や不正な形式の音声は、文の内容ではなくサーバーの不具合とみなす
pub fn validate_audio(audio: &Audio) -> Result<(), InvalidAudio> {
    if audio.data.is_empty() {
        return Err(InvalidAudio("TTSサーバーから空の音声が返されました".to_string()));
    }
    validate_format(audio.rate.into(), audio.width.into(), audio.channels.into(), audio.data.len())
}
