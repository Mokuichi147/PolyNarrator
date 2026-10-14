use std::io::Read;
use std::time::Duration;

use serde_json::json;

use crate::adapters::http::{openai_error_message, send_with_retry, HttpFailure, OPENAI_RETRY};
use crate::adapters::wav::{read_wav, wav_data_size, WAV_UNKNOWN_SIZE};
use crate::application::ports::{SpeechSynthesizer, TtsError};
use crate::domain::audio::{validate_audio, Audio, MAX_AUDIO_BYTES};

/// WAVのヘッダーなど音声以外の部分として許容する大きさ
pub const WAV_HEADER_ALLOWANCE: usize = 1 << 16;

/// OpenAI互換の /v1/audio/speech で音声合成する
pub struct OpenAiSpeechClient {
    agent: ureq::Agent,
    base_url: String,
    model: String,
    api_key: String,
    /// 受け取る音声の上限
    pub max_audio_bytes: usize,
    pub header_allowance: usize,
}

impl OpenAiSpeechClient {
    pub fn new(base_url: &str, model: &str, api_key: &str, timeout: f64) -> Self {
        let timeout = Duration::from_secs_f64(timeout);
        let agent =
            ureq::AgentBuilder::new().timeout_connect(timeout).timeout_read(timeout).timeout_write(timeout).build();
        Self {
            agent,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            api_key: api_key.to_string(),
            max_audio_bytes: MAX_AUDIO_BYTES,
            header_allowance: WAV_HEADER_ALLOWANCE,
        }
    }

    fn too_large(&self) -> TtsError {
        TtsError::Fatal(format!("TTSサーバーから受け取った音声が上限({}バイト)を超えました", self.max_audio_bytes))
    }

    fn decode(&self, content: &[u8]) -> Result<Audio, TtsError> {
        // PCM以外のWAVは読み込めないため、文の内容ではなくサーバーの問題とみなす
        let audio = read_wav(content).map_err(|e| TtsError::Fatal(e.to_string()))?;
        if audio.data.len() > self.max_audio_bytes {
            return Err(self.too_large());
        }
        // ストリーミング用にデータ長を最大値にしたヘッダーを除き、ヘッダーより短いデータは途中で切れた応答とみなす。
        // 1フレーム未満の端数は聴感上の影響がないため、フレーム単位で比べる
        let frame_size = audio.frame_size();
        let declared = wav_data_size(content).filter(|&d| d != WAV_UNKNOWN_SIZE);
        if let Some(frames) = declared.and_then(|d| (d as usize).checked_div(frame_size)) {
            let expected = frames * frame_size;
            if audio.data.len() < expected {
                return Err(TtsError::Fatal(format!(
                    "TTSサーバーから途中で切れた音声が返されました({}/{expected}バイト)",
                    audio.data.len()
                )));
            }
        }
        validate_audio(&audio).map_err(|e| TtsError::Fatal(e.to_string()))?;
        Ok(audio)
    }
}

impl SpeechSynthesizer for OpenAiSpeechClient {
    fn list_voices(&self) -> Vec<String> {
        // 音声一覧を取得する標準APIがないため、利用者が指定する
        Vec::new()
    }

    fn synthesize(&mut self, text: &str, voice: &str) -> Result<Audio, TtsError> {
        let url = format!("{}/audio/speech", self.base_url);
        let body = json!({"input": text, "model": self.model, "voice": voice, "response_format": "wav"});
        let authorization = format!("Bearer {}", self.api_key);
        let response =
            send_with_retry(&OPENAI_RETRY, |_| self.agent.post(&url).set("Authorization", &authorization), &body)
                .map_err(|failure| {
                    let message = openai_error_message(&failure);
                    match failure {
                        HttpFailure::Status { status: 400 | 422, .. } => TtsError::Input(message),
                        // 認証・接続・サーバー内部のエラーは続けても失敗するだけなので中断させる
                        _ => TtsError::Fatal(message),
                    }
                })?;

        // 巨大な応答でメモリを使い切らないよう、上限を超えた時点で読むのをやめる
        let limit = self.max_audio_bytes.saturating_add(self.header_allowance);
        let mut content = Vec::new();
        response
            .into_reader()
            .take(limit as u64 + 1)
            .read_to_end(&mut content)
            .map_err(|e| TtsError::Fatal(format!("TTSサーバーの応答を読めませんでした: {e}")))?;
        if content.len() > limit {
            return Err(self.too_large());
        }
        self.decode(&content)
    }
}
