//! 読み上げの処理。画面を止めないよう別のスレッドで動かし、経過は Event で画面へ送る

use std::path::Path;
use std::rc::Rc;

use indexmap::IndexMap;
use yomitone::application::cancel::CancelFlag;
use yomitone::application::narrate_novels::{NarrateNovels, NarrationError};
use yomitone::application::ports::{Presenter, SynthesisResult};
use yomitone::application::speaker_estimation::SpeakerEstimation;
use yomitone::application::speech_synthesis::{AudioExporter, SentenceSynthesizer};
use yomitone::domain::character::Character;
use yomitone::domain::novel::Sentence;
use yomitone::setup::{self, SetupError};

use crate::settings::{AppSettings, SpeakerBackend};

/// 処理の経過。送った順に画面へ届く
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// 経過の表示に足す行
    Log(String),
    /// 入力を読み込んだ。章ごとの行数と、音声を書き出すか
    Loaded {
        counts: Vec<usize>,
        synthesize: bool,
    },
    ChapterStarted {
        index: usize,
        characters: Vec<Character>,
    },
    /// 1行の話者を決めた(推測できなかった行を含む)
    SentenceEstimated,
    SynthesisProgressed {
        done: usize,
        total: usize,
    },
    /// 全ての章を終えた。voices は名前ごとの音声で、音声を書き出さない場合は None
    Finished {
        characters: Vec<Character>,
        voices: Option<Vec<(String, String)>>,
    },
    Done(Outcome),
}

/// 処理の終わり方
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Outcome {
    Completed,
    /// 合成に失敗した文がある章
    PartiallyFailed(Vec<String>),
    Cancelled,
    Failed(String),
}

/// 設定の問題を、画面のどこを直せばよいか分かるメッセージにする
pub fn describe_setup_error(error: &SetupError) -> String {
    match error {
        SetupError::NoServerVoices => {
            format!("{error}。「声の割り当て」の使用する音声か、「音声合成」の言語を確認してください")
        }
        SetupError::VoicesRequired => "「声の割り当て」で使用する音声を指定してください".into(),
        SetupError::Assigner(_) => format!("{error}。「声の割り当て」の使用する音声に含まれる音声を指定してください"),
        SetupError::Jev(_) => format!("{error}。「話者推測」で API キーを入力してください"),
        _ => error.to_string(),
    }
}

fn speaker_name(sentence: &Sentence) -> &str {
    sentence.speaker.as_ref().map(|s| s.name.as_str()).unwrap_or_default()
}

/// 経過を Event にして送る
struct EventPresenter<F: Fn(Event)> {
    send: F,
}

impl<F: Fn(Event)> EventPresenter<F> {
    fn log(&self, line: impl Into<String>) {
        (self.send)(Event::Log(line.into()));
    }
}

impl<F: Fn(Event)> Presenter for EventPresenter<F> {
    fn chapter_started(&self, index: usize, source: &str, characters: &[Character]) {
        let name = Path::new(source).file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or(source.into());
        self.log(format!("── {name}"));
        (self.send)(Event::ChapterStarted { index, characters: characters.to_vec() });
    }

    fn character_extraction_failed(&self, error: &str) {
        self.log(format!("登場人物を抽出できなかったため、以前の一覧を使います: {error}"));
    }

    fn narration_assigned(&self, sentence: &Sentence) {
        if !sentence.text.is_empty() {
            self.log(format!("[{}] {}", speaker_name(sentence), sentence.text));
        }
        (self.send)(Event::SentenceEstimated);
    }

    fn speaker_estimated(&self, sentence: &Sentence, confidence: Option<f64>) {
        match confidence {
            Some(confidence) => self.log(format!("[{}] ({confidence:.2}) {}", speaker_name(sentence), sentence.text)),
            None => self.log(format!("[{}] {}", speaker_name(sentence), sentence.text)),
        }
        (self.send)(Event::SentenceEstimated);
    }

    fn speaker_estimation_failed(&self, sentence: &Sentence, error: Option<&str>) {
        let reason = error.map(|e| format!(": {e}")).unwrap_or_default();
        self.log(format!("[{}] {} (推測できませんでした{reason})", speaker_name(sentence), sentence.text));
        (self.send)(Event::SentenceEstimated);
    }

    fn context_shrunk(&self, previous_count: usize) {
        self.log(format!("入力長超過のため、文脈を直前{previous_count}文に縮小して再試行します"));
    }

    fn sentence_synthesis_failed(&self, voice: &str, text: &str, error: &str) {
        self.log(format!("音声合成に失敗しました ({voice}): {error} … {text}"));
    }

    fn synthesis_progressed(&self, done: usize, total: usize) {
        (self.send)(Event::SynthesisProgressed { done, total });
    }

    fn audio_exported(&self, location: Option<&str>, result: &SynthesisResult) {
        let failures = result.failures.len();
        self.log(match location {
            None if failures > 0 => "全ての文の合成に失敗したため、音声を書き出しませんでした".to_string(),
            None => "読み上げられる文がないため、音声を書き出しませんでした".to_string(),
            Some(location) if failures > 0 => {
                format!("音声を書き出しました(合成に失敗した{failures}文を除く): {location}")
            }
            Some(location) => format!("音声を書き出しました: {location}"),
        });
    }

    fn finished(&self, characters: &[Character], voices: Option<&IndexMap<String, String>>) {
        let voices = voices.map(|v| v.iter().map(|(name, voice)| (name.clone(), voice.clone())).collect());
        (self.send)(Event::Finished { characters: characters.to_vec(), voices });
    }
}

fn narrate(settings: &AppSettings, cancel: CancelFlag, send: impl Fn(Event) + 'static) -> Result<Outcome, String> {
    settings.validate()?;
    let describe = |e: SetupError| describe_setup_error(&e);
    let mut chapters = setup::load_chapters(Path::new(settings.input_folder.trim())).map_err(describe)?;
    if chapters.is_empty() {
        return Err(format!("入力フォルダに .txt ファイルがありません({})", settings.input_folder.trim()));
    }
    let writer = if settings.export_audio {
        Some(
            setup::prepare_output(Path::new(settings.output_folder.trim()), settings.tts_pause, &chapters)
                .map_err(describe)?,
        )
    } else {
        None
    };
    let counts = chapters.iter().map(|c| c.novel.sentences.len()).collect();
    send(Event::Loaded { counts, synthesize: writer.is_some() });

    let presenter: Rc<dyn Presenter> = Rc::new(EventPresenter { send });
    let (extractor, llm_estimator) = setup::create_chat_models(&settings.llm_options());
    let estimator = match settings.speaker_backend {
        SpeakerBackend::Llm => llm_estimator,
        SpeakerBackend::Jev => setup::create_jev(&settings.jev_options()).map_err(describe)?,
    };
    let exporter = match writer {
        None => None,
        Some(writer) => {
            let options = settings.tts_options();
            let (client, assigner) = setup::create_tts(&options).map_err(describe)?;
            let synthesizer = SentenceSynthesizer::new(client, assigner, presenter.clone(), settings.tts_max_chars);
            Some(AudioExporter::new(synthesizer, Box::new(writer), presenter.clone()))
        }
    };

    let speaker_estimation = SpeakerEstimation::new(estimator, presenter.clone());
    let mut use_case = NarrateNovels::new(extractor, speaker_estimation, presenter, exporter).with_cancel(cancel);
    match use_case.run(&mut chapters) {
        Ok(report) if report.failed_sources.is_empty() => Ok(Outcome::Completed),
        Ok(report) => Ok(Outcome::PartiallyFailed(report.failed_sources)),
        Err(NarrationError::Cancelled) => Ok(Outcome::Cancelled),
        Err(error) => Err(error.to_string()),
    }
}

/// 設定に従って読み上げ、経過を send で送る。最後に必ず Event::Done を送る
pub fn run(settings: AppSettings, cancel: CancelFlag, send: impl Fn(Event) + Clone + 'static) {
    let outcome = narrate(&settings, cancel, send.clone()).unwrap_or_else(Outcome::Failed);
    send(Event::Done(outcome));
}
