//! テスト用のサーバー。移植前の Python 版のテストと同じ応答を返す

#![allow(dead_code)]

use std::io::{BufRead, BufReader, Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::thread;

use serde_json::{json, Map, Value};

/// 応答を切り替えるキー。「名前:セリフ」の形式ではセリフ部分を使う
pub fn response_key(text: &str) -> String {
    let last = text.rsplit(':').next().unwrap_or(text);
    last.strip_suffix('」').unwrap_or(last).to_string()
}

pub fn wav_bytes(rate: u32, frames: usize) -> Vec<u8> {
    let data_size = (frames * 2) as u32;
    let mut out = Vec::new();
    out.extend(b"RIFF");
    out.extend((36 + data_size).to_le_bytes());
    out.extend(b"WAVEfmt ");
    out.extend(16u32.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(1u16.to_le_bytes());
    out.extend(rate.to_le_bytes());
    out.extend((rate * 2).to_le_bytes());
    out.extend(2u16.to_le_bytes());
    out.extend(16u16.to_le_bytes());
    out.extend(b"data");
    out.extend(data_size.to_le_bytes());
    out.extend([1u8, 0].repeat(frames));
    out
}

/// WAV の (rate, width, channels, data)
pub fn read_wav_file(path: &std::path::Path) -> (u32, u16, u16, Vec<u8>) {
    let content = std::fs::read(path).unwrap();
    let audio = yomitone::adapters::wav::read_wav(&content).unwrap();
    (audio.rate, audio.width, audio.channels, audio.data)
}

// --- HTTP ---

pub struct HttpRequest {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: Vec<u8>,
}

impl HttpRequest {
    pub fn json(&self) -> Value {
        serde_json::from_slice(&self.body).unwrap()
    }

    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers.iter().find(|(k, _)| k.eq_ignore_ascii_case(name)).map(|(_, v)| v.as_str())
    }
}

pub struct HttpResponse {
    pub status: u16,
    pub content_type: &'static str,
    pub body: Vec<u8>,
}

fn read_request(stream: &TcpStream) -> Option<HttpRequest> {
    let mut reader = BufReader::new(stream);
    let mut line = String::new();
    reader.read_line(&mut line).ok()?;
    let mut parts = line.split_whitespace();
    let method = parts.next()?.to_string();
    let path = parts.next()?.to_string();
    let mut headers = Vec::new();
    loop {
        let mut line = String::new();
        reader.read_line(&mut line).ok()?;
        let line = line.trim_end();
        if line.is_empty() {
            break;
        }
        let (key, value) = line.split_once(':')?;
        headers.push((key.trim().to_string(), value.trim().to_string()));
    }
    let length = headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case("content-length"))
        .and_then(|(_, v)| v.parse().ok())
        .unwrap_or(0);
    let mut body = vec![0; length];
    reader.read_exact(&mut body).ok()?;
    Some(HttpRequest { method, path, headers, body })
}

/// 1リクエストごとに handler の応答を返す HTTP サーバー。base URL と受け取ったリクエストを返す
pub fn http_server(
    handler: impl Fn(&HttpRequest) -> HttpResponse + Send + Sync + 'static,
) -> (String, Arc<Mutex<Vec<HttpRequest>>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let handler = Arc::new(handler);
    {
        let requests = requests.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(mut stream) = stream else { continue };
                let requests = requests.clone();
                let handler = handler.clone();
                thread::spawn(move || {
                    let Some(request) = read_request(&stream) else { return };
                    let response = handler(&request);
                    requests.lock().unwrap().push(request);
                    let reason = if response.status == 200 { "OK" } else { "Error" };
                    let head = format!(
                        "HTTP/1.1 {} {reason}\r\nContent-Type: {}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                        response.status,
                        response.content_type,
                        response.body.len()
                    );
                    let _ = stream.write_all(head.as_bytes());
                    let _ = stream.write_all(&response.body);
                });
            }
        });
    }
    (format!("http://{address}"), requests)
}

/// OpenAI 互換の /v1/audio/speech。入力の文字列で応答を切り替える
pub fn speech_server() -> (String, Arc<Mutex<Vec<HttpRequest>>>) {
    let (url, requests) = http_server(|request| {
        let body = request.json();
        let input = body["input"].as_str().unwrap_or_default().to_string();
        let mut status = match response_key(&input).as_str() {
            "bad" => 400,
            "unprocessable" => 422,
            "unauthorized" => 401,
            _ => 200,
        };
        if body["model"] == "unknown-model" {
            status = 400;
        }
        if status != 200 {
            let payload = json!({"error": {"message": input, "type": "invalid_request_error"}});
            return HttpResponse { status, content_type: "application/json", body: payload.to_string().into_bytes() };
        }
        let mut payload = wav_bytes(24000, if input == "zero" { 0 } else { 240 });
        match input.as_str() {
            "notwav" => payload = b"not a wav file".to_vec(),
            "truncated" => payload.truncate(payload.len() - 100),
            // 最大値の直前の(ストリーミング用ではない)データ長
            "near-limit" => payload[40..44].copy_from_slice(&[0xfe, 0xff, 0xff, 0xff]),
            // ストリーミング用にRIFFとdataのサイズを最大値にしたヘッダー
            "streaming" => {
                payload[4..8].copy_from_slice(&[0xff; 4]);
                payload[40..44].copy_from_slice(&[0xff; 4]);
            }
            _ => {}
        }
        HttpResponse { status: 200, content_type: "audio/wav", body: payload }
    });
    (format!("{url}/v1"), requests)
}

// --- Wyoming ---

pub struct WyomingState {
    /// "info" / "error" / "none"
    pub describe: String,
    pub info: Value,
    /// 受け取った Synthesize の data
    pub requests: Vec<Value>,
}

fn tts_voice(name: &str, languages: &[&str], speakers: Option<&[&str]>, installed: bool) -> Value {
    json!({
        "name": name,
        "attribution": {"name": "test", "url": ""},
        "installed": installed,
        "description": null,
        "version": null,
        "languages": languages,
        "speakers": speakers.map(|s| s.iter().map(|n| json!({"name": n})).collect::<Vec<_>>()),
    })
}

fn tts_program(name: &str, installed: bool, voices: Vec<Value>) -> Value {
    json!({
        "name": name,
        "attribution": {"name": "test", "url": ""},
        "installed": installed,
        "description": null,
        "version": null,
        "voices": voices,
        "supports_synthesize_streaming": false,
    })
}

pub fn wyoming_info() -> Value {
    json!({"asr": [], "tts": [
        tts_program("tts", true, vec![
            tts_voice("ja-a", &["ja_JP"], None, true),
            tts_voice("ja-multi", &["ja"], Some(&["s1", "s2"]), true),
            tts_voice("en-a", &["en_US"], None, true),
            tts_voice("ja-missing", &["ja"], None, false),
        ]),
        tts_program("not-installed", false, vec![tts_voice("ja-other", &["ja"], None, true)]),
        tts_program("second", true, vec![tts_voice("ja-second", &["ja"], None, true)]),
    ], "handle": [], "intent": [], "wake": [], "mic": [], "snd": []})
}

/// Python の wyoming ライブラリと同じく、data を見出しとは別のバイト列で送る
fn write_event(stream: &mut TcpStream, kind: &str, data: Value, payload: &[u8]) {
    let mut header = json!({"type": kind, "version": "1.10.2"});
    let data_bytes = match &data {
        Value::Object(map) if !map.is_empty() => Some(serde_json::to_vec(map).unwrap()),
        _ => None,
    };
    if let Some(bytes) = &data_bytes {
        header["data_length"] = json!(bytes.len());
    }
    if !payload.is_empty() {
        header["payload_length"] = json!(payload.len());
    }
    let mut out = serde_json::to_vec(&header).unwrap();
    out.push(b'\n');
    out.extend(data_bytes.unwrap_or_default());
    out.extend(payload);
    let _ = stream.write_all(&out);
}

fn read_event(stream: &TcpStream) -> Option<(String, Map<String, Value>)> {
    let mut reader = BufReader::new(stream);
    let mut line = Vec::new();
    reader.read_until(b'\n', &mut line).ok()?;
    let header: Value = serde_json::from_slice(&line).ok()?;
    let mut data = header.get("data").and_then(Value::as_object).cloned().unwrap_or_default();
    if let Some(length) = header.get("data_length").and_then(Value::as_u64) {
        let mut bytes = vec![0; length as usize];
        reader.read_exact(&mut bytes).ok()?;
        data.extend(serde_json::from_slice::<Map<String, Value>>(&bytes).ok()?);
    }
    Some((header["type"].as_str()?.to_string(), data))
}

fn chunk(stream: &mut TcpStream, rate: i64, width: i64, channels: i64, audio: &[u8]) {
    write_event(
        stream,
        "audio-chunk",
        json!({"rate": rate, "width": width, "channels": channels, "timestamp": null}),
        audio,
    );
}

fn handle_wyoming(mut stream: TcpStream, state: &Mutex<WyomingState>) {
    let Some((kind, data)) = read_event(&stream) else { return };
    let stop = json!({"timestamp": null});
    if kind == "describe" {
        let (mode, info) = {
            let state = state.lock().unwrap();
            (state.describe.clone(), state.info.clone())
        };
        match mode.as_str() {
            "error" => write_event(&mut stream, "error", json!({"text": "describe failed"}), &[]),
            "info" => write_event(&mut stream, "info", info, &[]),
            _ => {}
        }
    } else if kind == "synthesize" {
        let text = data["text"].as_str().unwrap_or_default().to_string();
        state.lock().unwrap().requests.push(Value::Object(data));
        let s = &mut stream;
        match response_key(&text).as_str() {
            "error" => write_event(s, "error", json!({"text": "synthesis failed"}), &[]),
            "empty" => write_event(s, "audio-stop", stop, &[]),
            "partial-error" => {
                chunk(s, 22050, 2, 1, &[1, 0].repeat(100));
                write_event(s, "error", json!({"text": "failed midway"}), &[]);
            }
            "mixed" => {
                chunk(s, 22050, 2, 1, &[1, 0].repeat(2205));
                for _ in 0..4 {
                    chunk(s, 11025, 2, 2, &[1, 0, 1, 0].repeat(1102));
                }
                write_event(s, "audio-stop", stop, &[]);
            }
            "bad-second" => {
                chunk(s, 22050, 2, 1, &[1, 0].repeat(100));
                chunk(s, 384001, 2, 1, &[1, 0].repeat(100));
                write_event(s, "audio-stop", stop, &[]);
            }
            "odd" => {
                chunk(s, 22050, 2, 1, &[1]);
                write_event(s, "audio-stop", stop, &[]);
            }
            "empty-chunk" => {
                chunk(s, 22050, 2, 1, &[]);
                write_event(s, "audio-stop", stop, &[]);
            }
            "disconnect" => {}
            _ => {
                write_event(
                    s,
                    "audio-start",
                    json!({"rate": 22050, "width": 2, "channels": 1, "timestamp": null}),
                    &[],
                );
                for _ in 0..3 {
                    chunk(s, 22050, 2, 1, &[1, 0].repeat(100));
                }
                write_event(s, "audio-stop", stop, &[]);
            }
        }
    }
}

/// Wyoming サーバー。tcp:// の URI と、応答を切り替える状態を返す
pub fn wyoming_server() -> (String, Arc<Mutex<WyomingState>>) {
    let listener = TcpListener::bind("127.0.0.1:0").unwrap();
    let address = listener.local_addr().unwrap();
    let state =
        Arc::new(Mutex::new(WyomingState { describe: "info".into(), info: wyoming_info(), requests: Vec::new() }));
    {
        let state = state.clone();
        thread::spawn(move || {
            for stream in listener.incoming() {
                let Ok(stream) = stream else { continue };
                let state = state.clone();
                thread::spawn(move || handle_wyoming(stream, &state));
            }
        });
    }
    (format!("tcp://{address}"), state)
}

/// 何も待ち受けていないポート
pub fn unused_tcp_port() -> u16 {
    TcpListener::bind("127.0.0.1:0").unwrap().local_addr().unwrap().port()
}
