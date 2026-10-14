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

impl SpeakerBackend {
    /// 設定する画面の名前
    pub const fn page_title(self) -> &'static str {
        match self {
            SpeakerBackend::Llm => "LLM",
            SpeakerBackend::Jev => "Jev",
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TtsBackendSetting {
    #[default]
    Openai,
    Wyoming,
}

impl TtsBackendSetting {
    /// 設定する画面の名前
    pub const fn page_title(self) -> &'static str {
        match self {
            TtsBackendSetting::Openai => "TTS（OpenAI互換API）",
            TtsBackendSetting::Wyoming => "TTS（Wyoming）",
        }
    }

    pub fn backend(self) -> TtsBackend {
        match self {
            TtsBackendSetting::Openai => TtsBackend::OpenAi,
            TtsBackendSetting::Wyoming => TtsBackend::Wyoming,
        }
    }
}

/// TTS サーバーごとの設定。音声名はサーバーごとに違うため、音声の割り当てもサーバーごとに持つ
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TtsServerSettings {
    /// 空のときはプロトコルごとの既定の URL
    pub url: String,
    /// OpenAI 互換 API のモデル名
    pub model: String,
    /// OpenAI 互換 API のキー。平文でファイルに残さないよう保存しない
    #[serde(skip)]
    pub api_key: String,
    /// Wyoming で音声一覧を絞り込む言語
    pub language: String,
    pub voices: String,
    pub narrator_voice: String,
    pub male_voices: String,
    pub female_voices: String,
    pub max_chars: usize,
    pub pause: f64,
    pub timeout: f64,
}

impl Default for TtsServerSettings {
    fn default() -> Self {
        let tts = TtsOptions::default();
        Self {
            url: String::new(),
            model: tts.model,
            api_key: String::new(),
            language: tts.language,
            voices: String::new(),
            narrator_voice: String::new(),
            male_voices: String::new(),
            female_voices: String::new(),
            max_chars: setup::DEFAULT_TTS_MAX_CHARS,
            pause: setup::DEFAULT_TTS_PAUSE,
            timeout: tts.timeout,
        }
    }
}

impl TtsServerSettings {
    pub fn options(&self, backend: TtsBackend) -> TtsOptions {
        TtsOptions {
            backend,
            url: optional(&self.url),
            model: optional(&self.model).unwrap_or_else(|| setup::DEFAULT_OPENAI_TTS_MODEL.into()),
            api_key: optional(&self.api_key),
            language: self.language.trim().to_string(),
            voices: setup::split_voices(&self.voices),
            narrator_voice: optional(&self.narrator_voice),
            male_voices: setup::split_voices(&self.male_voices),
            female_voices: setup::split_voices(&self.female_voices),
            timeout: self.timeout,
        }
    }

    /// 合成を始められない設定であれば、直すべき項目を伝えるメッセージを返す。page は設定する画面の名前
    pub fn validate(&self, page: &str) -> Result<(), String> {
        if !(0.0..=setup::MAX_TTS_PAUSE).contains(&self.pause) {
            return Err(format!("「{page}」の行間の無音は0〜{}秒で指定してください", setup::MAX_TTS_PAUSE));
        }
        if !(self.timeout.is_finite() && self.timeout > 0.0) {
            return Err(format!("「{page}」のタイムアウトは0より大きい秒数で指定してください"));
        }
        Ok(())
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

    /// 話者推測に使うもの
    pub speaker_backend: SpeakerBackend,
    pub jev_model: String,
    pub jev_base_url: String,
    /// 平文でファイルに残さないよう保存しない
    #[serde(skip)]
    pub jev_api_key: String,

    /// 音声合成に使うもの
    pub tts_backend: TtsBackendSetting,
    pub openai_tts: TtsServerSettings,
    pub wyoming_tts: TtsServerSettings,
}

impl Default for AppSettings {
    fn default() -> Self {
        let llm = LlmOptions::default();
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
            openai_tts: TtsServerSettings::default(),
            wyoming_tts: TtsServerSettings::default(),
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

    pub fn tts_server(&self, backend: TtsBackendSetting) -> &TtsServerSettings {
        match backend {
            TtsBackendSetting::Openai => &self.openai_tts,
            TtsBackendSetting::Wyoming => &self.wyoming_tts,
        }
    }

    pub fn tts_server_mut(&mut self, backend: TtsBackendSetting) -> &mut TtsServerSettings {
        match backend {
            TtsBackendSetting::Openai => &mut self.openai_tts,
            TtsBackendSetting::Wyoming => &mut self.wyoming_tts,
        }
    }

    /// 音声合成に使う TTS サーバーの設定
    pub fn selected_tts(&self) -> &TtsServerSettings {
        self.tts_server(self.tts_backend)
    }

    pub fn tts_options(&self) -> TtsOptions {
        self.selected_tts().options(self.tts_backend.backend())
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
            self.selected_tts().validate(self.tts_backend.page_title())?;
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
