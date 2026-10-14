//! 設定から各層の実装を組み立てる。コマンドラインと GUI で共通に使う

use std::path::Path;

use thiserror::Error;

use crate::adapters::jev::JevSpeakerEstimator;
use crate::adapters::llm::OpenAiChatModel;
use crate::adapters::openai_speech::OpenAiSpeechClient;
use crate::adapters::text_files::{list_text_files, read_novel};
use crate::adapters::wav::WavDirectory;
use crate::adapters::wyoming::WyomingClient;
use crate::application::narrate_novels::Chapter;
use crate::application::ports::{CharacterExtractor, SpeakerEstimator, SpeechSynthesizer};
use crate::domain::voice_assigner::{VoiceAssigner, VoiceAssignerError};

pub const DEFAULT_LLM_HOST: &str = "localhost";
pub const DEFAULT_LLM_PORT: &str = "1234";
pub const DEFAULT_LLM_MODEL: &str = "qwen3.8-flash-next-iq3_s";
pub const DEFAULT_OPENAI_TTS_URL: &str = "http://localhost:8880/v1";
pub const DEFAULT_OPENAI_TTS_MODEL: &str = "tts-1";
pub const DEFAULT_WYOMING_URL: &str = "tcp://localhost:10200";
pub const DEFAULT_TTS_LANGUAGE: &str = "ja";
pub const DEFAULT_TTS_PAUSE: f64 = 0.3;
/// 無音はメモリ上に作るため、極端な値で大量のメモリを確保しないよう上限を設ける
pub const MAX_TTS_PAUSE: f64 = 60.0;
pub const DEFAULT_TTS_MAX_CHARS: usize = 200;
pub const DEFAULT_TTS_TIMEOUT: f64 = 60.0;

/// 登場人物の抽出と、LLM での話者の推測に使うサーバー
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LlmOptions {
    pub host: String,
    pub port: String,
    pub model: String,
}

impl Default for LlmOptions {
    fn default() -> Self {
        Self { host: DEFAULT_LLM_HOST.into(), port: DEFAULT_LLM_PORT.into(), model: DEFAULT_LLM_MODEL.into() }
    }
}

/// 話者推測に使う Jev 互換 API。指定がない項目は環境変数、既定値の順に使う
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct JevOptions {
    pub api_key: Option<String>,
    pub model: Option<String>,
    pub base_url: Option<String>,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum TtsBackend {
    #[default]
    OpenAi,
    Wyoming,
}

impl TtsBackend {
    pub fn name(self) -> &'static str {
        match self {
            TtsBackend::OpenAi => "openai",
            TtsBackend::Wyoming => "wyoming",
        }
    }

    pub fn default_url(self) -> &'static str {
        match self {
            TtsBackend::OpenAi => DEFAULT_OPENAI_TTS_URL,
            TtsBackend::Wyoming => DEFAULT_WYOMING_URL,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct TtsOptions {
    pub backend: TtsBackend,
    /// 未指定時はプロトコルごとの既定の URL
    pub url: Option<String>,
    /// OpenAI 互換 API で使うモデル名
    pub model: String,
    /// OpenAI 互換 API のキー。未指定時は環境変数 OPENAI_API_KEY
    pub api_key: Option<String>,
    /// Wyoming で音声一覧を絞り込む言語。空文字で絞り込まない
    pub language: String,
    /// 使用する音声。Wyoming で空の場合はサーバーの一覧を使う
    pub voices: Vec<String>,
    /// ナレーターの音声。未指定時は音声一覧の先頭
    pub narrator_voice: Option<String>,
    pub male_voices: Vec<String>,
    pub female_voices: Vec<String>,
    /// TTSサーバーへのリクエストタイムアウト秒数
    pub timeout: f64,
}

impl Default for TtsOptions {
    fn default() -> Self {
        Self {
            backend: TtsBackend::default(),
            url: None,
            model: DEFAULT_OPENAI_TTS_MODEL.into(),
            api_key: None,
            language: DEFAULT_TTS_LANGUAGE.into(),
            voices: Vec::new(),
            narrator_voice: None,
            male_voices: Vec::new(),
            female_voices: Vec::new(),
            timeout: DEFAULT_TTS_TIMEOUT,
        }
    }
}

impl TtsOptions {
    pub fn url(&self) -> &str {
        self.url.as_deref().filter(|u| !u.is_empty()).unwrap_or(self.backend.default_url())
    }
}

/// 処理を始める前に見つかった、設定や環境の問題
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SetupError {
    #[error("入力フォルダを読み込めませんでした({folder}): {error}")]
    InputFolder { folder: String, error: String },
    #[error("入力ファイルを読み込めませんでした({path}): {error}")]
    InputFile { path: String, error: String },
    #[error("{0}")]
    Output(String),
    #[error("Wyomingサーバー({uri})から音声一覧を取得できませんでした: {error}")]
    WyomingUnreachable { uri: String, error: String },
    #[error("TTSサーバーにない音声が指定されています: {}", .0.join(", "))]
    UnknownVoices(Vec<String>),
    /// Wyoming サーバーの一覧に、言語の一致する音声がない
    #[error("TTSサーバーから利用できる音声を取得できませんでした")]
    NoServerVoices,
    /// OpenAI 互換 API は音声一覧を取得できないため、利用者が指定する必要がある
    #[error("利用する音声が指定されていません")]
    VoicesRequired,
    #[error("{0}")]
    Assigner(VoiceAssignerError),
    #[error("{0}")]
    Jev(String),
}

/// カンマ区切りの音声の一覧
pub fn split_voices(value: &str) -> Vec<String> {
    value.split(',').map(str::trim).filter(|v| !v.is_empty()).map(str::to_string).collect()
}

/// LLMやTTSサーバーでの処理を始める前に、全ての入力ファイルを読み込んでおく
pub fn load_chapters(folder: &Path) -> Result<Vec<Chapter>, SetupError> {
    let files = list_text_files(folder)
        .map_err(|e| SetupError::InputFolder { folder: folder.display().to_string(), error: e.to_string() })?;
    files
        .into_iter()
        .map(|file| {
            let path = folder.join(file);
            let novel = read_novel(&path)
                .map_err(|e| SetupError::InputFile { path: path.display().to_string(), error: e.to_string() })?;
            Ok(Chapter::new(path.display().to_string(), novel))
        })
        .collect()
}

/// LLMやTTSサーバーでの処理を始める前に、出力先に書き出せるかを確認する
pub fn prepare_output(directory: &Path, pause: f64, chapters: &[Chapter]) -> Result<WavDirectory, SetupError> {
    let writer = WavDirectory::new(directory, pause);
    let sources: Vec<&str> = chapters.iter().map(|c| c.source.as_str()).collect();
    writer.prepare(&sources).map_err(|e| SetupError::Output(e.to_string()))?;
    Ok(writer)
}

/// 登場人物の抽出と、LLM での話者の推測
pub fn create_chat_models(options: &LlmOptions) -> (Box<dyn CharacterExtractor>, Box<dyn SpeakerEstimator>) {
    let model = || OpenAiChatModel::new(&options.host, &options.port, &options.model);
    (Box::new(model()), Box::new(model()))
}

pub fn create_jev(options: &JevOptions) -> Result<Box<dyn SpeakerEstimator>, SetupError> {
    let estimator = JevSpeakerEstimator::new(options.api_key.clone(), options.model.clone(), options.base_url.clone())
        .map_err(SetupError::Jev)?;
    Ok(Box::new(estimator))
}

/// TTS サーバーにつなぎ、使う音声を決める。音声の一覧は割り当てに使う順で返す
pub fn create_tts(options: &TtsOptions) -> Result<(Box<dyn SpeechSynthesizer>, VoiceAssigner), SetupError> {
    let wyoming = options.backend == TtsBackend::Wyoming;
    let url = options.url();
    let client: Box<dyn SpeechSynthesizer> = if wyoming {
        let client = WyomingClient::new(url, options.timeout, Some(&options.language))
            .map_err(|error| SetupError::WyomingUnreachable { uri: url.to_string(), error })?;
        Box::new(client)
    } else {
        let api_key = options
            .api_key
            .clone()
            .filter(|k| !k.is_empty())
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_else(|| "not-needed".into());
        Box::new(OpenAiSpeechClient::new(url, &options.model, &api_key, options.timeout))
    };

    let explicit = !options.voices.is_empty();
    let mut voices = if explicit { options.voices.clone() } else { client.list_voices() };
    let narrator_voice = options.narrator_voice.clone().filter(|v| !v.is_empty());
    let role_voices: Vec<String> = narrator_voice
        .iter()
        .cloned()
        .chain(options.male_voices.iter().cloned())
        .chain(options.female_voices.iter().cloned())
        .collect();
    let unknown = client.unknown_voices(&[voices.clone(), role_voices.clone()].concat());
    if !unknown.is_empty() {
        return Err(SetupError::UnknownVoices(unknown));
    }
    if !explicit && wyoming {
        // サーバーの一覧を使う場合、言語の絞り込みや複数話者モデルの既定話者で一覧に出ない音声も、明示されていれば使う
        for voice in role_voices {
            if !voices.contains(&voice) {
                voices.push(voice);
            }
        }
    }
    if voices.is_empty() {
        return Err(if wyoming { SetupError::NoServerVoices } else { SetupError::VoicesRequired });
    }

    let assigner =
        VoiceAssigner::new(voices, narrator_voice, options.male_voices.clone(), options.female_voices.clone())
            .map_err(SetupError::Assigner)?;
    Ok((client, assigner))
}
