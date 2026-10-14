//! Wyoming プロトコル(JSON の見出し行と、続くデータ・音声のバイト列からなるイベントの送受信)で音声合成する

use std::io::{self, BufRead, BufReader, Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

use indexmap::IndexMap;
use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::adapters::audio_convert::AudioConverter;
use crate::application::ports::{SpeechSynthesizer, TtsError};
use crate::domain::audio::{validate_audio, validate_format, Audio, MAX_AUDIO_BYTES};

/// 送るイベントに付けるプロトコルのバージョン
const PROTOCOL_VERSION: &str = "1.10.2";
/// 見出し行の長さの上限。壊れた応答で読み続けないようにする
const MAX_HEADER_LINE: u64 = 1 << 20;

const DISCONNECTED: &str = "Wyomingサーバーとの接続が切断されました";

struct Event {
    kind: String,
    data: Map<String, Value>,
    payload: Vec<u8>,
}

trait Stream: Read + Write {}
impl<T: Read + Write> Stream for T {}

struct Connection {
    reader: BufReader<Box<dyn Stream>>,
}

fn io_error(error: io::Error) -> String {
    format!("Wyomingサーバーとの通信に失敗しました: {error}")
}

impl Connection {
    fn open(uri: &str, timeout: Duration) -> Result<Self, String> {
        let stream: Box<dyn Stream> = if let Some(address) = uri.strip_prefix("tcp://") {
            let address = address.trim_end_matches('/');
            let addresses: Vec<_> =
                address.to_socket_addrs().map_err(|e| format!("{address} を解決できません: {e}"))?.collect();
            let mut last_error = None;
            let mut connected = None;
            for address in addresses {
                match TcpStream::connect_timeout(&address, timeout) {
                    Ok(stream) => {
                        connected = Some(stream);
                        break;
                    }
                    Err(e) => last_error = Some(e),
                }
            }
            let stream = connected.ok_or_else(|| match last_error {
                Some(e) => format!("{address} に接続できません: {e}"),
                None => format!("{address} を解決できません"),
            })?;
            stream.set_read_timeout(Some(timeout)).map_err(io_error)?;
            stream.set_write_timeout(Some(timeout)).map_err(io_error)?;
            Box::new(stream)
        } else if let Some(path) = uri.strip_prefix("unix://") {
            Self::open_unix(path, timeout)?
        } else {
            return Err(format!("対応していないURIです(tcp:// または unix:// を指定してください): {uri}"));
        };
        Ok(Self { reader: BufReader::new(stream) })
    }

    #[cfg(unix)]
    fn open_unix(path: &str, timeout: Duration) -> Result<Box<dyn Stream>, String> {
        let stream =
            std::os::unix::net::UnixStream::connect(path).map_err(|e| format!("{path} に接続できません: {e}"))?;
        stream.set_read_timeout(Some(timeout)).map_err(io_error)?;
        stream.set_write_timeout(Some(timeout)).map_err(io_error)?;
        Ok(Box::new(stream))
    }

    #[cfg(not(unix))]
    fn open_unix(_path: &str, _timeout: Duration) -> Result<Box<dyn Stream>, String> {
        Err("この環境では unix:// に対応していません".to_string())
    }

    fn write_event(&mut self, kind: &str, data: Value) -> Result<(), String> {
        let mut header = json!({"type": kind, "version": PROTOCOL_VERSION});
        let data_bytes = match data {
            Value::Object(map) if !map.is_empty() => Some(serde_json::to_vec(&map).expect("JSON に変換できる")),
            _ => None,
        };
        if let Some(bytes) = &data_bytes {
            header["data_length"] = json!(bytes.len());
        }
        let stream = self.reader.get_mut();
        let mut line = serde_json::to_vec(&header).expect("JSON に変換できる");
        line.push(b'\n');
        if let Some(bytes) = data_bytes {
            line.extend(bytes);
        }
        stream.write_all(&line).and_then(|_| stream.flush()).map_err(io_error)
    }

    /// 接続が切れた場合や読めない見出しの場合は None
    fn read_event(&mut self) -> Result<Option<Event>, String> {
        let mut line = Vec::new();
        (&mut self.reader).take(MAX_HEADER_LINE).read_until(b'\n', &mut line).map_err(io_error)?;
        if line.is_empty() || line.last() != Some(&b'\n') {
            return Ok(None);
        }
        let Ok(Value::Object(header)) = serde_json::from_slice::<Value>(&line) else {
            return Ok(None);
        };
        let Some(kind) = header.get("type").and_then(Value::as_str).map(str::to_string) else {
            return Ok(None);
        };
        let mut data = match header.get("data") {
            Some(Value::Object(map)) => map.clone(),
            _ => Map::new(),
        };
        let length = |key: &str| header.get(key).and_then(Value::as_u64).unwrap_or(0) as usize;

        let data_length = length("data_length");
        if data_length > 0 {
            let bytes = self.read_exact(data_length)?;
            match serde_json::from_slice::<Value>(&bytes) {
                Ok(Value::Object(extra)) => data.extend(extra),
                _ => return Ok(None),
            }
        }
        let payload_length = length("payload_length");
        // 大量のメモリを確保しないよう、音声の上限を超える長さは読まない
        if payload_length > MAX_AUDIO_BYTES {
            return Err(format!("Wyomingサーバーから受け取った音声が上限({MAX_AUDIO_BYTES}バイト)を超えました"));
        }
        let payload = if payload_length > 0 { self.read_exact(payload_length)? } else { Vec::new() };
        Ok(Some(Event { kind, data, payload }))
    }

    fn read_exact(&mut self, length: usize) -> Result<Vec<u8>, String> {
        let mut bytes = vec![0; length];
        self.reader.read_exact(&mut bytes).map_err(|e| {
            if e.kind() == io::ErrorKind::UnexpectedEof {
                DISCONNECTED.to_string()
            } else {
                io_error(e)
            }
        })?;
        Ok(bytes)
    }
}

fn error_text(event: &Event) -> String {
    event.data.get("text").and_then(Value::as_str).unwrap_or_default().to_string()
}

#[derive(Deserialize)]
struct Info {
    #[serde(default)]
    tts: Vec<TtsProgram>,
}

#[derive(Deserialize)]
struct TtsProgram {
    installed: bool,
    #[serde(default)]
    voices: Vec<TtsVoice>,
}

#[derive(Deserialize)]
struct TtsVoice {
    name: String,
    installed: bool,
    #[serde(default)]
    languages: Vec<String>,
    #[serde(default)]
    speakers: Option<Vec<TtsVoiceSpeaker>>,
}

#[derive(Deserialize)]
struct TtsVoiceSpeaker {
    name: String,
}

/// 合成に使う音声と話者
#[derive(Debug, Clone, PartialEq, Eq)]
struct SynthesizeVoice {
    name: String,
    speaker: Option<String>,
}

/// Wyomingプロトコルで音声合成する
pub struct WyomingClient {
    uri: String,
    timeout: Duration,
    language: Option<String>,
    /// 指定できる全ての音声。複数話者モデルは「音声名:話者名」と、既定の話者を使う「音声名」の両方で指定できる
    voices: IndexMap<String, SynthesizeVoice>,
    /// 言語が一致し、音声一覧として使う音声
    listed: Vec<String>,
    /// 受け取る音声の上限
    pub max_audio_bytes: usize,
}

impl WyomingClient {
    /// サーバーから音声一覧を取得する。取得できない場合はエラー
    pub fn new(uri: &str, timeout: f64, language: Option<&str>) -> Result<Self, String> {
        let mut client = Self {
            uri: uri.to_string(),
            timeout: Duration::from_secs_f64(timeout),
            language: language.map(str::to_string),
            voices: IndexMap::new(),
            listed: Vec::new(),
            max_audio_bytes: MAX_AUDIO_BYTES,
        };
        client.describe()?;
        Ok(client)
    }

    fn connect(&self) -> Result<Connection, String> {
        Connection::open(&self.uri, self.timeout)
    }

    fn match_language(&self, languages: &[String]) -> bool {
        let Some(language) = self.language.as_deref().filter(|l| !l.is_empty()) else {
            return true;
        };
        if languages.is_empty() {
            return true;
        }
        let normalize = |l: &str| l.to_lowercase().replace('_', "-");
        let requested = normalize(language);
        // 「ja」と「ja-JP」のように、一方が他方の地域指定付きであれば一致とみなす
        languages.iter().map(|l| normalize(l)).any(|lang| {
            lang == requested
                || lang.starts_with(&format!("{requested}-"))
                || requested.starts_with(&format!("{lang}-"))
        })
    }

    fn describe(&mut self) -> Result<(), String> {
        let mut connection = self.connect()?;
        connection.write_event("describe", json!({}))?;
        let info = loop {
            let event = connection.read_event()?.ok_or_else(|| DISCONNECTED.to_string())?;
            match event.kind.as_str() {
                "info" => {
                    break serde_json::from_value::<Info>(Value::Object(event.data))
                        .map_err(|e| format!("Wyomingサーバーの音声一覧を読めませんでした: {e}"))?
                }
                "error" => return Err(format!("Wyomingサーバーがエラーを返しました: {}", error_text(&event))),
                _ => {}
            }
        };

        // Synthesize ではプログラムを選べず、サーバー既定のプログラムで合成されるため、最初のプログラムの音声だけを使う
        let Some(program) = info.tts.into_iter().find(|p| p.installed) else {
            return Ok(());
        };
        for voice in program.voices.into_iter().filter(|v| v.installed) {
            let matched = self.match_language(&voice.languages);
            self.voices.insert(voice.name.clone(), SynthesizeVoice { name: voice.name.clone(), speaker: None });
            match voice.speakers.filter(|s| !s.is_empty()) {
                // 複数話者モデルは話者ごとに別の音声として扱う
                Some(speakers) => {
                    for speaker in speakers {
                        let key = format!("{}:{}", voice.name, speaker.name);
                        self.voices.insert(
                            key.clone(),
                            SynthesizeVoice { name: voice.name.clone(), speaker: Some(speaker.name) },
                        );
                        if matched {
                            self.listed.push(key);
                        }
                    }
                }
                None if matched => self.listed.push(voice.name),
                None => {}
            }
        }
        Ok(())
    }

    fn too_large(&self) -> TtsError {
        TtsError::Fatal(format!("Wyomingサーバーから受け取った音声が上限({}バイト)を超えました", self.max_audio_bytes))
    }
}

fn integer(data: &Map<String, Value>, key: &str) -> Result<i64, TtsError> {
    data.get(key)
        .and_then(Value::as_i64)
        .ok_or_else(|| TtsError::Fatal(format!("Wyomingサーバーの音声に {key} がありません")))
}

impl SpeechSynthesizer for WyomingClient {
    fn list_voices(&self) -> Vec<String> {
        self.listed.clone()
    }

    fn unknown_voices(&self, voices: &[String]) -> Vec<String> {
        voices.iter().filter(|v| !self.voices.contains_key(*v)).cloned().collect()
    }

    fn synthesize(&mut self, text: &str, voice: &str) -> Result<Audio, TtsError> {
        let synthesize_voice = self
            .voices
            .get(voice)
            .cloned()
            .unwrap_or_else(|| SynthesizeVoice { name: voice.to_string(), speaker: None });
        let mut voice_data = json!({"name": synthesize_voice.name});
        if let Some(speaker) = synthesize_voice.speaker {
            voice_data["speaker"] = json!(speaker);
        }

        let mut connection = self.connect().map_err(TtsError::Fatal)?;
        connection.write_event("synthesize", json!({"text": text, "voice": voice_data})).map_err(TtsError::Fatal)?;

        let mut audio: Option<Audio> = None;
        // 変換器はリサンプリングの状態を持つため、連続した同じ形式のチャンクでは使い回す
        let mut converter: Option<(AudioConverter, (i64, i64, i64))> = None;
        loop {
            let event = connection
                .read_event()
                .map_err(TtsError::Fatal)?
                .ok_or_else(|| TtsError::Fatal(DISCONNECTED.into()))?;
            match event.kind.as_str() {
                "audio-chunk" => {
                    let format = (
                        integer(&event.data, "rate")?,
                        integer(&event.data, "width")?,
                        integer(&event.data, "channels")?,
                    );
                    let (rate, width, channels) = format;
                    // 後続のチャンクの形式が変わる場合もあるため、変換する前にチャンクごとに確認する
                    validate_format(rate, width, channels, event.payload.len()).map_err(|e| TtsError::Fatal(e.0))?;
                    let chunk = Audio::new(rate as u32, width as u16, channels as u16, event.payload);
                    let output =
                        audio.get_or_insert_with(|| Audio::new(chunk.rate, chunk.width, chunk.channels, Vec::new()));
                    if converter.as_ref().map(|(_, f)| *f) != Some(format) {
                        converter = Some((AudioConverter::new(output.rate, output.width, output.channels), format));
                    }
                    // 変換で大きくなる場合も含め、変換して保持する前に上限を超えないかを見積もり(切り上げ)、
                    // リサンプリングの端数で見積もりを超える場合に備えて変換後の実際の大きさでも確認する
                    let output_rate = u128::from(output.rate) * u128::from(output.width) * u128::from(output.channels);
                    let input_rate = (rate * width * channels) as u128;
                    let converted_size = (chunk.data.len() as u128 * output_rate).div_ceil(input_rate);
                    if output.data.len() as u128 + converted_size > self.max_audio_bytes as u128 {
                        return Err(self.too_large());
                    }
                    let converted = converter.as_mut().expect("直前に作成している").0.convert(&chunk);
                    output.data.extend(converted.data);
                    if output.data.len() > self.max_audio_bytes {
                        return Err(self.too_large());
                    }
                }
                "audio-stop" => break,
                // Wyomingのエラーは原因を区別できないため文の失敗として扱う。
                // 設定やサーバーの問題であれば、未成功の音声での失敗や連続した失敗として中断される
                "error" => return Err(TtsError::Input(error_text(&event))),
                _ => {}
            }
        }

        // エラーも音声も返さないのは文の内容ではなくサーバーの不具合とみなす
        let audio = audio.ok_or_else(|| TtsError::Fatal("Wyomingサーバーから音声が返されませんでした".into()))?;
        validate_audio(&audio).map_err(|e| TtsError::Fatal(e.0))?;
        Ok(audio)
    }
}
