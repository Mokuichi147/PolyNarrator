//! コマンドライン引数から各層の実装を組み立てて実行する

mod args;
mod console;

use std::path::Path;
use std::rc::Rc;

use clap::CommandFactory;
use yomitone::adapters::wav::WavDirectory;
use yomitone::application::narrate_novels::{Chapter, NarrateNovels, NarrationError};
use yomitone::application::ports::{CharacterExtractor, Presenter, SpeakerEstimator, SpeechSynthesizer};
use yomitone::application::speaker_estimation::SpeakerEstimation;
use yomitone::application::speech_synthesis::{AudioExporter, SentenceSynthesizer};
use yomitone::domain::voice_assigner::VoiceAssigner;
use yomitone::setup::{self, JevOptions, LlmOptions, SetupError, TtsOptions};

pub use crate::args::{Args, SpeakerBackend, TtsBackend, MAX_PAUSE};
pub use crate::console::ConsolePresenter;

/// 終了コード 1 で終了する理由
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError(pub String);

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

/// 設定の問題を、直すべきオプションが分かるメッセージにする
impl From<SetupError> for CliError {
    fn from(error: SetupError) -> Self {
        CliError(match error {
            SetupError::NoServerVoices => format!("{error}。--tts-voices や --tts-language を確認してください"),
            SetupError::VoicesRequired => "利用する音声を --tts-voices で指定してください".to_string(),
            SetupError::Assigner(_) => format!("{error}。--tts-voices に含まれる音声を指定してください"),
            SetupError::Jev(_) => format!("{error}。--jev-api-key で指定してください"),
            _ => error.to_string(),
        })
    }
}

fn fail<T>(message: impl Into<String>) -> Result<T, CliError> {
    Err(CliError(message.into()))
}

pub type ChatModels = (Box<dyn CharacterExtractor>, Box<dyn SpeakerEstimator>);
pub type Tts = (Box<dyn SpeechSynthesizer>, VoiceAssigner);
type Factory<T> = Box<dyn Fn(&Args) -> T>;

/// 外部のサーバーにつながる実装の作り方。テストでは置き換える
pub struct Factories {
    /// 登場人物の抽出と、LLM での話者の推測
    pub chat_model: Factory<ChatModels>,
    pub jev: Factory<Result<Box<dyn SpeakerEstimator>, CliError>>,
    pub tts: Factory<Result<Tts, CliError>>,
    pub presenter: Box<dyn Fn() -> Rc<dyn Presenter>>,
}

impl Default for Factories {
    fn default() -> Self {
        Self {
            chat_model: Box::new(|args| setup::create_chat_models(&llm_options(args))),
            jev: Box::new(|args| Ok(setup::create_jev(&jev_options(args))?)),
            tts: Box::new(create_tts),
            presenter: Box::new(|| Rc::new(ConsolePresenter::default())),
        }
    }
}

pub fn llm_options(args: &Args) -> LlmOptions {
    LlmOptions { host: args.host.clone(), port: args.port.clone(), model: args.model.clone() }
}

pub fn jev_options(args: &Args) -> JevOptions {
    JevOptions { api_key: args.jev_api_key.clone(), model: args.jev_model.clone(), base_url: args.jev_base_url.clone() }
}

pub fn tts_options(args: &Args) -> TtsOptions {
    let split = |value: &Option<String>| setup::split_voices(value.as_deref().unwrap_or_default());
    TtsOptions {
        backend: match args.tts_backend {
            TtsBackend::Openai => setup::TtsBackend::OpenAi,
            TtsBackend::Wyoming => setup::TtsBackend::Wyoming,
        },
        url: args.tts_url.clone(),
        model: args.tts_model.clone(),
        api_key: args.tts_api_key.clone(),
        language: args.tts_language.clone(),
        voices: split(&args.tts_voices),
        narrator_voice: args.tts_narrator_voice.clone(),
        male_voices: split(&args.tts_male_voices),
        female_voices: split(&args.tts_female_voices),
        timeout: args.tts_timeout,
    }
}

/// LLMやTTSサーバーでの処理を始める前に、全ての入力ファイルを読み込んでおく
pub fn load_chapters(folder: &str) -> Result<Vec<Chapter>, CliError> {
    Ok(setup::load_chapters(Path::new(folder))?)
}

/// LLMやTTSサーバーでの処理を始める前に、出力先に書き出せるかを確認する
pub fn prepare_output(directory: &str, pause: f64, chapters: &[Chapter]) -> Result<WavDirectory, CliError> {
    Ok(setup::prepare_output(Path::new(directory), pause, chapters)?)
}

pub fn create_tts(args: &Args) -> Result<Tts, CliError> {
    let options = tts_options(args);
    let (client, assigner) = setup::create_tts(&options)?;
    println!("TTS 初期化: backend={}, 音声一覧=[{}]", options.backend.name(), assigner.voices().join(", "));
    Ok((client, assigner))
}

pub fn run(args: &Args, factories: &Factories) -> Result<(), CliError> {
    // 出力先を作る前に入力フォルダを確認する(出力先の作成で入力フォルダが作られてしまわないように)
    let mut chapters = load_chapters(&args.folder)?;
    let writer = match args.tts_output.as_deref() {
        None => None,
        Some("") => Args::command()
            .error(clap::error::ErrorKind::InvalidValue, "--tts-output に出力先のディレクトリを指定してください")
            .exit(),
        Some(directory) => Some(prepare_output(directory, args.tts_pause, &chapters)?),
    };

    let presenter = (factories.presenter)();
    let (extractor, llm_estimator) = (factories.chat_model)(args);
    let estimator = match args.speaker_backend {
        SpeakerBackend::Llm => llm_estimator,
        SpeakerBackend::Jev => (factories.jev)(args)?,
    };
    let exporter = match writer {
        None => None,
        Some(writer) => {
            let (client, assigner) = (factories.tts)(args)?;
            let synthesizer = SentenceSynthesizer::new(client, assigner, presenter.clone(), args.tts_max_chars);
            Some(AudioExporter::new(synthesizer, Box::new(writer), presenter.clone()))
        }
    };

    let speaker_estimation = SpeakerEstimation::new(estimator, presenter.clone());
    let mut use_case = NarrateNovels::new(extractor, speaker_estimation, presenter, exporter);
    let report = use_case.run(&mut chapters).map_err(|e| match e {
        NarrationError::Fatal(message) => CliError(format!("処理を中断しました: {message}")),
        aborted => CliError(aborted.to_string()),
    })?;

    if !report.failed_sources.is_empty() {
        return fail(format!("\n音声合成に失敗した文があります: {}", report.failed_sources.join(", ")));
    }
    Ok(())
}
