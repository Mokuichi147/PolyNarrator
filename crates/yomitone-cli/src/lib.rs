//! コマンドライン引数から各層の実装を組み立てて実行する

mod args;
mod console;

use std::path::Path;
use std::rc::Rc;

use clap::CommandFactory;
use yomitone::adapters::jev::JevSpeakerEstimator;
use yomitone::adapters::llm::OpenAiChatModel;
use yomitone::adapters::openai_speech::OpenAiSpeechClient;
use yomitone::adapters::text_files::{list_text_files, read_novel};
use yomitone::adapters::wav::WavDirectory;
use yomitone::adapters::wyoming::WyomingClient;
use yomitone::application::narrate_novels::{Chapter, NarrateNovels, NarrationError};
use yomitone::application::ports::{CharacterExtractor, Presenter, SpeakerEstimator, SpeechSynthesizer};
use yomitone::application::speaker_estimation::SpeakerEstimation;
use yomitone::application::speech_synthesis::{AudioExporter, SentenceSynthesizer};
use yomitone::domain::voice_assigner::VoiceAssigner;

pub use crate::args::{Args, SpeakerBackend, TtsBackend, MAX_PAUSE};
pub use crate::console::ConsolePresenter;

pub const DEFAULT_OPENAI_TTS_URL: &str = "http://localhost:8880/v1";
pub const DEFAULT_WYOMING_URL: &str = "tcp://localhost:10200";

/// 終了コード 1 で終了する理由
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CliError(pub String);

impl std::fmt::Display for CliError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for CliError {}

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
            chat_model: Box::new(|args| {
                let model = || OpenAiChatModel::new(&args.host, &args.port, &args.model);
                (Box::new(model()), Box::new(model()))
            }),
            jev: Box::new(|args| {
                JevSpeakerEstimator::new(args.jev_api_key.clone(), args.jev_model.clone(), args.jev_base_url.clone())
                    .map(|e| Box::new(e) as Box<dyn SpeakerEstimator>)
                    .map_err(CliError)
            }),
            tts: Box::new(create_tts),
            presenter: Box::new(|| Rc::new(ConsolePresenter::default())),
        }
    }
}

pub fn split_voices(value: Option<&str>) -> Vec<String> {
    value.unwrap_or_default().split(',').map(str::trim).filter(|v| !v.is_empty()).map(str::to_string).collect()
}

/// LLMやTTSサーバーでの処理を始める前に、全ての入力ファイルを読み込んでおく
pub fn load_chapters(folder: &str) -> Result<Vec<Chapter>, CliError> {
    let files = list_text_files(Path::new(folder))
        .map_err(|e| CliError(format!("入力フォルダを読み込めませんでした({folder}): {e}")))?;
    files
        .into_iter()
        .map(|file| {
            let path = Path::new(folder).join(file);
            let novel = read_novel(&path)
                .map_err(|e| CliError(format!("入力ファイルを読み込めませんでした({}): {e}", path.display())))?;
            Ok(Chapter::new(path.display().to_string(), novel))
        })
        .collect()
}

/// LLMやTTSサーバーでの処理を始める前に、出力先に書き出せるかを確認する
pub fn prepare_output(directory: &str, pause: f64, chapters: &[Chapter]) -> Result<WavDirectory, CliError> {
    let writer = WavDirectory::new(directory, pause);
    let sources: Vec<&str> = chapters.iter().map(|c| c.source.as_str()).collect();
    writer.prepare(&sources).map_err(|e| CliError(e.to_string()))?;
    Ok(writer)
}

pub fn create_tts(args: &Args) -> Result<Tts, CliError> {
    let wyoming = args.tts_backend == TtsBackend::Wyoming;
    let client: Box<dyn SpeechSynthesizer> = if wyoming {
        let uri = args.tts_url.as_deref().unwrap_or(DEFAULT_WYOMING_URL);
        let client = WyomingClient::new(uri, args.tts_timeout, Some(&args.tts_language))
            .map_err(|e| CliError(format!("Wyomingサーバー({uri})から音声一覧を取得できませんでした: {e}")))?;
        Box::new(client)
    } else {
        let api_key = args
            .tts_api_key
            .clone()
            .or_else(|| std::env::var("OPENAI_API_KEY").ok())
            .unwrap_or_else(|| "not-needed".into());
        let url = args.tts_url.as_deref().unwrap_or(DEFAULT_OPENAI_TTS_URL);
        Box::new(OpenAiSpeechClient::new(url, &args.tts_model, &api_key, args.tts_timeout))
    };

    let explicit_voices = split_voices(args.tts_voices.as_deref());
    let mut voices = if explicit_voices.is_empty() { client.list_voices() } else { explicit_voices };
    let male_voices = split_voices(args.tts_male_voices.as_deref());
    let female_voices = split_voices(args.tts_female_voices.as_deref());
    let role_voices: Vec<String> = args
        .tts_narrator_voice
        .iter()
        .filter(|v| !v.is_empty())
        .cloned()
        .chain(male_voices.clone())
        .chain(female_voices.clone())
        .collect();
    let unknown = client.unknown_voices(&[voices.clone(), role_voices.clone()].concat());
    if !unknown.is_empty() {
        return fail(format!("TTSサーバーにない音声が指定されています: {}", unknown.join(", ")));
    }
    if args.tts_voices.as_deref().unwrap_or_default().is_empty() && wyoming {
        // サーバーの一覧を使う場合、言語の絞り込みや複数話者モデルの既定話者で一覧に出ない音声も、明示されていれば使う
        for voice in role_voices {
            if !voices.contains(&voice) {
                voices.push(voice);
            }
        }
    }
    if voices.is_empty() {
        if wyoming {
            return fail("TTSサーバーから利用できる音声を取得できませんでした。--tts-voices や --tts-language を確認してください");
        }
        return fail("利用する音声を --tts-voices で指定してください");
    }

    let narrator_voice = args.tts_narrator_voice.clone().filter(|v| !v.is_empty());
    let assigner = VoiceAssigner::new(voices.clone(), narrator_voice, male_voices, female_voices)
        .map_err(|e| CliError(format!("{e}。--tts-voices に含まれる音声を指定してください")))?;
    let backend = if wyoming { "wyoming" } else { "openai" };
    println!("TTS 初期化: backend={backend}, 音声一覧=[{}]", voices.join(", "));
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
