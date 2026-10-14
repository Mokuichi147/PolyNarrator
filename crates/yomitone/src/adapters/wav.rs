use std::fs;
use std::io::{self, BufWriter, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

use crate::adapters::audio_convert::AudioConverter;
use crate::application::ports::AudioWriter;
use crate::domain::audio::{validate_audio, Audio};

/// ストリーミングでデータ長が分からないWAVのヘッダーに入る値
pub const WAV_UNKNOWN_SIZE: u32 = 0xFFFF_FFFF;

const WAVE_FORMAT_PCM: u16 = 0x0001;
const WAVE_FORMAT_EXTENSIBLE: u16 = 0xFFFE;
const KSDATAFORMAT_SUBTYPE_PCM: [u8; 16] =
    [0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x10, 0x00, 0x80, 0x00, 0x00, 0xAA, 0x00, 0x38, 0x9B, 0x71];

fn u16_at(content: &[u8], position: usize) -> Option<u16> {
    content.get(position..position + 2).map(|b| u16::from_le_bytes([b[0], b[1]]))
}

fn u32_at(content: &[u8], position: usize) -> Option<u32> {
    content.get(position..position + 4).map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
}

/// WAVの data チャンクに宣言されたバイト数。見つからない場合は None
pub fn wav_data_size(content: &[u8]) -> Option<u32> {
    let mut position = 12usize;
    while position + 8 <= content.len() {
        let size = u32_at(content, position + 4)?;
        if &content[position..position + 4] == b"data" {
            return Some(size);
        }
        position = position.checked_add(8 + size as usize + (size & 1) as usize)?;
    }
    None
}

/// 読み込めない WAV
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct WavError(pub String);

/// PCM の WAV を読み込む。data チャンクが宣言より短い場合は、あるだけを返す
pub fn read_wav(content: &[u8]) -> Result<Audio, WavError> {
    let error = |message: &str| WavError(message.to_string());
    if content.get(0..4) != Some(b"RIFF") {
        return Err(error("file does not start with RIFF id"));
    }
    if content.get(8..12) != Some(b"WAVE") {
        return Err(error("not a WAVE file"));
    }

    let mut format: Option<(u32, u16, u16)> = None;
    let mut position = 12usize;
    while position + 8 <= content.len() {
        let id = &content[position..position + 4];
        let size = u32_at(content, position + 4).unwrap_or(0) as usize;
        let body = position + 8;
        if id == b"fmt " {
            let field = |offset: usize| u16_at(content, body + offset).filter(|_| offset + 2 <= size);
            let (Some(tag), Some(channels), Some(rate), Some(bits)) =
                (field(0), field(2), u32_at(content, body + 4).filter(|_| size >= 8), field(14))
            else {
                return Err(error("fmt chunk is too short"));
            };
            if tag == WAVE_FORMAT_EXTENSIBLE {
                if content.get(body + 24..body + 40) != Some(&KSDATAFORMAT_SUBTYPE_PCM[..]) || size < 40 {
                    return Err(error("unknown extended format"));
                }
            } else if tag != WAVE_FORMAT_PCM {
                return Err(WavError(format!("unknown format: {tag}")));
            }
            if channels == 0 {
                return Err(error("bad # of channels"));
            }
            let width = bits.div_ceil(8);
            if width == 0 {
                return Err(error("bad sample width"));
            }
            format = Some((rate, width, channels));
        } else if id == b"data" {
            let Some((rate, width, channels)) = format else {
                return Err(error("data chunk before fmt chunk"));
            };
            let frame_size = usize::from(width) * usize::from(channels);
            let wanted = size / frame_size * frame_size;
            let available = content.len() - body;
            let data = content[body..body + wanted.min(available)].to_vec();
            return Ok(Audio::new(rate, width, channels, data));
        }
        position = body + size + (size & 1);
    }
    Err(error("fmt chunk and/or data chunk missing"))
}

/// 無音を一度に確保しないよう、最大1秒ずつ書き出す
fn write_silence(out: &mut impl Write, format: &Audio, frames: usize) -> io::Result<()> {
    // 8bit PCM は符号なしのため 0x80 が無音
    let sample = if format.width == 1 { 0x80 } else { 0x00 };
    let frame_size = format.frame_size();
    let block = vec![sample; frame_size * frames.min(format.rate as usize)];
    let mut remaining = frames;
    while remaining > 0 {
        let count = remaining.min(format.rate as usize);
        out.write_all(&block[..count * frame_size])?;
        remaining -= count;
    }
    Ok(())
}

fn write_header(out: &mut impl Write, format: &Audio, data_size: u32) -> io::Result<()> {
    let block_align = format.width * format.channels;
    out.write_all(b"RIFF")?;
    out.write_all(&(36 + data_size + (data_size & 1)).to_le_bytes())?;
    out.write_all(b"WAVEfmt ")?;
    out.write_all(&16u32.to_le_bytes())?;
    out.write_all(&WAVE_FORMAT_PCM.to_le_bytes())?;
    out.write_all(&format.channels.to_le_bytes())?;
    out.write_all(&format.rate.to_le_bytes())?;
    out.write_all(&(format.rate * u32::from(block_align)).to_le_bytes())?;
    out.write_all(&block_align.to_le_bytes())?;
    out.write_all(&(format.width * 8).to_le_bytes())?;
    out.write_all(b"data")?;
    out.write_all(&data_size.to_le_bytes())
}

#[derive(Debug, Error)]
pub enum WriteWavError {
    #[error("pause には0以上を指定してください")]
    InvalidPause,
    #[error("{0}")]
    InvalidAudio(#[from] crate::domain::audio::InvalidAudio),
    #[error("WAVファイルが大きすぎます")]
    TooLarge,
    #[error("{0}")]
    Io(#[from] io::Error),
}

/// 音声を順に連結して1つのWAVファイルに書き出す。形式が異なる音声は先頭に合わせて変換する
pub fn write_wav(path: &Path, segments: &[Audio], pause: f64) -> Result<(), WriteWavError> {
    if pause.is_nan() || pause < 0.0 {
        return Err(WriteWavError::InvalidPause);
    }
    let Some(first) = segments.first() else {
        return Ok(());
    };
    for segment in segments {
        validate_audio(segment)?;
    }
    let silence_frames = (f64::from(first.rate) * pause) as usize;

    // 途中で失敗しても書きかけのファイルが残らないよう、同じフォルダに一意な一時ファイルを作ってから置き換える。
    // 長いファイル名でも名前の長さの上限を超えないよう、一時ファイルの名前は出力名によらない短いものにする
    let directory = match path.parent() {
        Some(parent) if !parent.as_os_str().is_empty() => parent,
        _ => Path::new("."),
    };
    fs::create_dir_all(directory)?;
    let temp = tempfile::Builder::new().prefix(".yomitone-").suffix(".wav.tmp").tempfile_in(directory)?;
    {
        let mut out = BufWriter::new(temp.as_file());
        write_header(&mut out, first, 0)?;
        let mut data_size: u64 = 0;
        for (index, segment) in segments.iter().enumerate() {
            if index > 0 {
                write_silence(&mut out, first, silence_frames)?;
                data_size += (silence_frames * first.frame_size()) as u64;
            }
            // 変換器はリサンプリングの状態を持つため、音声ごとに作り直す
            let converted = AudioConverter::new(first.rate, first.width, first.channels).convert(segment);
            out.write_all(&converted.data)?;
            data_size += converted.data.len() as u64;
        }
        let data_size =
            u32::try_from(data_size).ok().filter(|&size| size < u32::MAX - 36).ok_or(WriteWavError::TooLarge)?;
        if data_size & 1 == 1 {
            out.write_all(&[0])?;
        }
        out.seek(SeekFrom::Start(0))?;
        write_header(&mut out, first, data_size)?;
        out.flush()?;
    }
    temp.persist(path).map_err(|e| e.error)?;
    Ok(())
}

/// 出力先に書き出せない
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[error("{0}")]
pub struct OutputDirectoryError(pub String);

fn display_paths(paths: &[PathBuf]) -> String {
    paths.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join(", ")
}

/// 入力ファイルごとに <入力ファイル名>.wav をフォルダに書き出す
pub struct WavDirectory {
    directory: PathBuf,
    /// 音声と音声の間に入れる無音の秒数
    pause: f64,
}

impl WavDirectory {
    pub fn new(directory: impl Into<PathBuf>, pause: f64) -> Self {
        Self { directory: directory.into(), pause }
    }

    pub fn path_for(&self, source: &str) -> PathBuf {
        let stem = Path::new(source).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_default();
        self.directory.join(format!("{stem}.wav"))
    }

    /// 書き出しを始める前に、出力先を作成して書き込めるか、出力ファイルが衝突しないかを確認する
    pub fn prepare<S: AsRef<str>>(&self, sources: &[S]) -> Result<(), OutputDirectoryError> {
        fs::create_dir_all(&self.directory).and_then(|_| tempfile::tempfile_in(&self.directory)).map_err(|e| {
            OutputDirectoryError(format!("音声の出力先に書き込めません({}): {e}", self.directory.display()))
        })?;

        let outputs: Vec<PathBuf> = sources.iter().map(|s| self.path_for(s.as_ref())).collect();
        // 大文字小文字や Unicode 正規化を区別しないファイルシステムでも上書きし合わないよう、正規化した名前で比べる
        let keys: Vec<String> = outputs
            .iter()
            .map(|p| p.file_name().unwrap_or_default().to_string_lossy().nfc().collect::<String>().to_lowercase())
            .collect();
        let mut duplicates: Vec<PathBuf> = outputs
            .iter()
            .zip(&keys)
            .filter(|(_, key)| keys.iter().filter(|k| k == key).count() > 1)
            .map(|(p, _)| p.clone())
            .collect();
        duplicates.sort();
        duplicates.dedup();
        if !duplicates.is_empty() {
            return Err(OutputDirectoryError(format!(
                "複数の入力ファイルの出力先が同じになります: {}",
                display_paths(&duplicates)
            )));
        }

        // シンボリックリンクは置き換えるとリンク自体が通常のファイルに変わるため、通常のファイル以外は受け付けない
        let conflicts: Vec<PathBuf> = outputs
            .into_iter()
            .filter(|p| match fs::symlink_metadata(p) {
                Ok(metadata) => !metadata.file_type().is_file(),
                Err(_) => false,
            })
            .collect();
        if !conflicts.is_empty() {
            return Err(OutputDirectoryError(format!(
                "音声の出力先に通常のファイル以外のものがあります: {}",
                display_paths(&conflicts)
            )));
        }
        Ok(())
    }
}

impl AudioWriter for WavDirectory {
    fn write(&mut self, source: &str, segments: &[Audio]) -> Result<String, String> {
        let path = self.path_for(source);
        write_wav(&path, segments, self.pause).map_err(|e| e.to_string())?;
        Ok(path.display().to_string())
    }

    fn remove(&mut self, source: &str) -> Result<(), String> {
        match fs::remove_file(self.path_for(source)) {
            Err(e) if e.kind() != io::ErrorKind::NotFound => Err(e.to_string()),
            _ => Ok(()),
        }
    }
}
