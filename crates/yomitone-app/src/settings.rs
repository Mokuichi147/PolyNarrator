//! 画面で入力する設定。API キー以外は次に起動したときのために保存する

use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};
use yomitone::setup::{self, JevOptions, LlmOptions, TtsBackend, TtsOptions};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SpeakerBackend {
    #[default]
    Llm,
    Jev,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsBackendSetting {
    #[default]
    Openai,
    Wyoming,
}

impl TtsBackendSetting {
    pub fn backend(self) -> TtsBackend {
        match self {
            TtsBackendSetting::Openai => TtsBackend::OpenAi,
            TtsBackendSetting::Wyoming => TtsBackend::Wyoming,
        }
    }
}

/// 画面の設定。文字列の項目は空のとき「指定しない」を表す
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct AppSettings {
    pub input_folder: String,
    pub export_audio: bool,
    pub output_folder: String,

    pub llm_host: String,
    pub llm_port: String,
    pub llm_model: String,

    pub speaker_backend: SpeakerBackend,
    pub jev_model: String,
    pub jev_base_url: String,
    /// 平文でファイルに残さないよう保存しない
    #[serde(skip)]
    pub jev_api_key: String,

    pub tts_backend: TtsBackendSetting,
    pub tts_url: String,
    pub tts_model: String,
    /// 平文でファイルに残さないよう保存しない
    #[serde(skip)]
    pub tts_api_key: String,
    pub tts_language: String,
    pub tts_voices: String,
    pub tts_narrator_voice: String,
    pub tts_male_voices: String,
    pub tts_female_voices: String,
    pub tts_pause: f64,
    pub tts_max_chars: usize,
    pub tts_timeout: f64,
}

impl Default for AppSettings {
    fn default() -> Self {
        let llm = LlmOptions::default();
        let tts = TtsOptions::default();
        Self {
            input_folder: String::new(),
            export_audio: false,
            output_folder: String::new(),
            llm_host: llm.host,
            llm_port: llm.port,
            llm_model: llm.model,
            speaker_backend: SpeakerBackend::default(),
            jev_model: String::new(),
            jev_base_url: String::new(),
            jev_api_key: String::new(),
            tts_backend: TtsBackendSetting::default(),
            tts_url: String::new(),
            tts_model: tts.model,
            tts_api_key: String::new(),
            tts_language: tts.language,
            tts_voices: String::new(),
            tts_narrator_voice: String::new(),
            tts_male_voices: String::new(),
            tts_female_voices: String::new(),
            tts_pause: setup::DEFAULT_TTS_PAUSE,
            tts_max_chars: setup::DEFAULT_TTS_MAX_CHARS,
            tts_timeout: tts.timeout,
        }
    }
}

fn optional(value: &str) -> Option<String> {
    let value = value.trim();
    (!value.is_empty()).then(|| value.to_string())
}

impl AppSettings {
    pub fn llm_options(&self) -> LlmOptions {
        let default = LlmOptions::default();
        LlmOptions {
            host: optional(&self.llm_host).unwrap_or(default.host),
            port: optional(&self.llm_port).unwrap_or(default.port),
            model: optional(&self.llm_model).unwrap_or(default.model),
        }
    }

    pub fn jev_options(&self) -> JevOptions {
        JevOptions {
            api_key: optional(&self.jev_api_key),
            model: optional(&self.jev_model),
            base_url: optional(&self.jev_base_url),
        }
    }

    pub fn tts_options(&self) -> TtsOptions {
        TtsOptions {
            backend: self.tts_backend.backend(),
            url: optional(&self.tts_url),
            model: optional(&self.tts_model).unwrap_or_else(|| setup::DEFAULT_OPENAI_TTS_MODEL.into()),
            api_key: optional(&self.tts_api_key),
            language: self.tts_language.trim().to_string(),
            voices: setup::split_voices(&self.tts_voices),
            narrator_voice: optional(&self.tts_narrator_voice),
            male_voices: setup::split_voices(&self.tts_male_voices),
            female_voices: setup::split_voices(&self.tts_female_voices),
            timeout: self.tts_timeout,
        }
    }

    /// 処理を始められない設定であれば、直すべき項目を伝えるメッセージを返す
    pub fn validate(&self) -> Result<(), String> {
        if self.input_folder.trim().is_empty() {
            return Err("入力フォルダを選んでください".into());
        }
        if self.export_audio {
            if self.output_folder.trim().is_empty() {
                return Err("音声の出力先フォルダを選んでください".into());
            }
            if !(0.0..=setup::MAX_TTS_PAUSE).contains(&self.tts_pause) {
                return Err(format!("「音声合成」の行間の無音は0〜{}秒で指定してください", setup::MAX_TTS_PAUSE));
            }
            if !(self.tts_timeout.is_finite() && self.tts_timeout > 0.0) {
                return Err("「音声合成」のタイムアウトは0より大きい秒数で指定してください".into());
            }
        }
        Ok(())
    }
}

/// 設定を保存する場所。OS ごとの設定フォルダの下に置く
pub fn default_path() -> Option<PathBuf> {
    dirs::config_dir().map(|dir| dir.join("Yomitone").join("settings.json"))
}

/// 保存した設定を読む。無い・読めない場合は既定の設定
pub fn load(path: &Path) -> AppSettings {
    fs::read_to_string(path).ok().and_then(|text| serde_json::from_str(&text).ok()).unwrap_or_default()
}

pub fn save(path: &Path, settings: &AppSettings) -> io::Result<()> {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let text = serde_json::to_string_pretty(settings).map_err(io::Error::other)?;
    fs::write(path, text + "\n")
}
