use std::cell::RefCell;
use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use indexmap::IndexMap;
use polynarrator::application::narrate_novels::{Chapter, NarrateNovels, NarrationError};
use polynarrator::application::ports::*;
use polynarrator::application::speaker_estimation::SpeakerEstimation;
use polynarrator::application::speech_synthesis::{AudioExporter, SentenceSynthesizer, MAX_CONSECUTIVE_FAILURES};
use polynarrator::domain::audio::Audio;
use polynarrator::domain::character::Character;
use polynarrator::domain::novel::{Novel, Sentence};
use polynarrator::domain::voice_assigner::VoiceAssigner;

fn character(name: &str) -> Character {
    Character::new(name, "")
}

fn pcm() -> Audio {
    Audio::new(16000, 2, 1, [1u8, 0].repeat(160))
}

fn sentences(texts: &[&str]) -> Vec<Sentence> {
    texts.iter().map(|t| Sentence::new(*t)).collect()
}

fn aborted(result: Result<impl std::fmt::Debug, NarrationError>) -> String {
    match result {
        Err(NarrationError::SynthesisAborted(message)) => message,
        other => panic!("中断されていません: {other:?}"),
    }
}

type Fail = Box<dyn Fn(&str) -> Option<TtsError>>;

#[derive(Clone)]
struct FakeSynthesizer {
    fail: Rc<Fail>,
    requests: Rc<RefCell<Vec<(String, String)>>>,
}

impl FakeSynthesizer {
    fn new(fail: impl Fn(&str) -> Option<TtsError> + 'static) -> Self {
        Self { fail: Rc::new(Box::new(fail)), requests: Rc::default() }
    }

    fn ok() -> Self {
        Self::new(|_| None)
    }

    fn texts(&self) -> Vec<String> {
        self.requests.borrow().iter().map(|(t, _)| t.clone()).collect()
    }
}

impl SpeechSynthesizer for FakeSynthesizer {
    fn list_voices(&self) -> Vec<String> {
        Vec::new()
    }

    fn synthesize(&mut self, text: &str, voice: &str) -> Result<Audio, TtsError> {
        self.requests.borrow_mut().push((text.to_string(), voice.to_string()));
        match (self.fail)(text) {
            Some(error) => Err(error),
            None => Ok(pcm()),
        }
    }
}

fn input_error(text: &str) -> Option<TtsError> {
    let _ = text;
    Some(TtsError::Input("bad".into()))
}

fn sentence_synthesizer(client: &FakeSynthesizer, voices: &[&str], max_chars: usize) -> SentenceSynthesizer {
    let assigner = VoiceAssigner::with_voices(voices.iter().copied()).unwrap();
    SentenceSynthesizer::new(Box::new(client.clone()), assigner, Rc::new(SilentPresenter), max_chars)
}

fn failures(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
}

// --- 文ごとの合成 ---

#[test]
fn synthesize_skips_unreadable_without_request() {
    let client = FakeSynthesizer::ok();
    let result = sentence_synthesizer(&client, &["n"], 0)
        .synthesize(&sentences(&["", "あ", "＊\u{3000}＊\u{3000}＊", "「………………」", "い"]))
        .unwrap();
    assert_eq!(client.texts(), ["あ", "い"]);
    assert_eq!(result.segments.len(), 2);
    assert_eq!(result.skipped, 2);
    assert!(result.failures.is_empty());
}

#[test]
fn synthesize_skipped_counts_symbol_lines_only() {
    let result = sentence_synthesizer(&FakeSynthesizer::ok(), &["n"], 0)
        .synthesize(&sentences(&["", "＊\u{3000}＊\u{3000}＊", ""]))
        .unwrap();
    assert_eq!(result.skipped, 1);
}

#[test]
fn synthesize_records_input_errors_and_continues() {
    let client = FakeSynthesizer::new(|text| if text == "い" { input_error(text) } else { None });
    let result = sentence_synthesizer(&client, &["n"], 0).synthesize(&sentences(&["あ", "い", "う"])).unwrap();
    assert_eq!(result.segments.len(), 2);
    assert_eq!(result.failures, failures(&[("い", "bad")]));
}

#[test]
fn synthesize_aborts_after_consecutive_input_errors() {
    let client = FakeSynthesizer::new(|text| if text != "ok" { input_error(text) } else { None });
    let mut texts = vec!["ok"];
    texts.extend(["あ"; 10]);
    let message = aborted(sentence_synthesizer(&client, &["n"], 0).synthesize(&sentences(&texts)));
    assert!(message.contains("連続"), "{message}");
    assert_eq!(client.texts().len(), 1 + MAX_CONSECUTIVE_FAILURES);
}

#[test]
fn synthesize_drops_whole_line_when_a_chunk_fails() {
    for failing in ["一文目。", "二文目。", "三文目。"] {
        let client = FakeSynthesizer::new(move |text| if text == failing { input_error(text) } else { None });
        let line = "一文目。二文目。三文目。";
        let result = sentence_synthesizer(&client, &["n"], 4).synthesize(&sentences(&["ok", line, "ok"])).unwrap();
        assert_eq!(result.segments.len(), 2);
        assert_eq!(result.failures, failures(&[(line, "bad")]));
    }
}

#[test]
fn synthesize_partial_line_success_counts_as_voice_success() {
    // 先頭の塊が成功していれば、その音声は使えるとみなして行の失敗として続行する
    let client = FakeSynthesizer::new(|text| if text == "二文目。" { input_error(text) } else { None });
    let result =
        sentence_synthesizer(&client, &["n"], 4).synthesize(&sentences(&["一文目。二文目。", "三文目。"])).unwrap();
    assert_eq!(result.failures, failures(&[("一文目。二文目。", "bad")]));
    assert_eq!(result.segments.len(), 1);
}

#[test]
fn synthesize_counts_split_line_as_one_failure() {
    let client = FakeSynthesizer::new(|text| if text.starts_with('い') { input_error(text) } else { None });
    let line = "い。".repeat(4);
    let mut texts = vec!["ok"];
    texts.extend(vec![line.as_str(); MAX_CONSECUTIVE_FAILURES - 1]);
    texts.push("ok");
    let result = sentence_synthesizer(&client, &["n"], 2).synthesize(&sentences(&texts)).unwrap();
    assert_eq!(result.failures.len(), MAX_CONSECUTIVE_FAILURES - 1);
}

#[test]
fn synthesize_counts_consecutive_failures_across_files() {
    let client = FakeSynthesizer::new(|text| if text != "ok" { input_error(text) } else { None });
    let mut synthesizer = sentence_synthesizer(&client, &["n"], 0);
    let mut texts = vec!["ok"];
    texts.extend(vec!["あ"; MAX_CONSECUTIVE_FAILURES - 1]);
    let result = synthesizer.synthesize(&sentences(&texts)).unwrap();
    assert_eq!(result.failures.len(), MAX_CONSECUTIVE_FAILURES - 1);
    assert!(aborted(synthesizer.synthesize(&sentences(&["い"]))).contains("連続"));
}

#[test]
fn synthesize_aborts_when_voice_never_succeeded() {
    // モデル名や音声名の誤りで 400 が返る場合を想定し、成功したことのない音声での失敗は中断する
    let client =
        FakeSynthesizer::new(|text| if text == "B" { Some(TtsError::Input("unknown voice".into())) } else { None });
    let speakers = [Sentence::spoken_by("A", character("A")), Sentence::spoken_by("B", character("B"))];
    let message = aborted(sentence_synthesizer(&client, &["n", "a", "b"], 0).synthesize(&speakers));
    assert!(message.contains("最初の合成"), "{message}");
}

#[test]
fn synthesize_remembers_succeeded_voices_across_calls() {
    let client = FakeSynthesizer::new(|text| if text == "い" { input_error(text) } else { None });
    let mut synthesizer = sentence_synthesizer(&client, &["n"], 0);
    synthesizer.synthesize(&sentences(&["あ"])).unwrap();
    let result = synthesizer.synthesize(&sentences(&["い", "う"])).unwrap();
    assert_eq!(result.failures, failures(&[("い", "bad")]));
}

#[test]
fn synthesize_aborts_on_other_errors() {
    let client = FakeSynthesizer::new(|_| Some(TtsError::Fatal("down".into())));
    assert_eq!(aborted(sentence_synthesizer(&client, &["n"], 0).synthesize(&sentences(&["あ", "い"]))), "down");
    assert_eq!(client.texts().len(), 1);
}

#[test]
fn synthesize_splits_long_text_with_same_voice() {
    let client = FakeSynthesizer::ok();
    let result = sentence_synthesizer(&client, &["n", "a"], 4)
        .synthesize(&[Sentence::spoken_by("一文目。二文目。", character("A"))])
        .unwrap();
    let expected = [("一文目。".to_string(), "a".to_string()), ("二文目。".to_string(), "a".to_string())];
    assert_eq!(*client.requests.borrow(), expected);
    assert_eq!(result.segments.len(), 2);
}

#[test]
fn synthesize_uses_configured_failure_limit() {
    let client = FakeSynthesizer::new(|text| if text != "ok" { input_error(text) } else { None });
    let mut synthesizer = sentence_synthesizer(&client, &["n"], 0);
    synthesizer.max_consecutive_failures = 2;
    assert!(aborted(synthesizer.synthesize(&sentences(&["ok", "あ", "い", "う"]))).contains("2回連続"));
}

// --- 章ごとの書き出し ---

#[derive(Clone, Default)]
struct FakeWriter {
    fail: bool,
    written: Rc<RefCell<HashMap<String, usize>>>,
    removed: Rc<RefCell<Vec<String>>>,
}

impl AudioWriter for FakeWriter {
    fn write(&mut self, source: &str, segments: &[Audio]) -> Result<String, String> {
        if self.fail {
            return Err("disk full".into());
        }
        self.written.borrow_mut().insert(source.to_string(), segments.len());
        Ok(format!("out/{source}.wav"))
    }

    fn remove(&mut self, source: &str) -> Result<(), String> {
        if self.fail {
            return Err("is a directory".into());
        }
        self.removed.borrow_mut().push(source.to_string());
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq)]
enum Event {
    Chapter(String, Vec<String>),
    ExtractionFailed(String),
    Estimated(String, String, Option<f64>),
    EstimationFailed(String, Option<String>),
    Shrunk(usize),
    Exported(Option<String>, usize),
    Finished(Vec<String>, Option<IndexMap<String, String>>),
}

#[derive(Default)]
struct RecordingPresenter {
    events: RefCell<Vec<Event>>,
}

fn names(characters: &[Character]) -> Vec<String> {
    characters.iter().map(|c| c.name.clone()).collect()
}

impl Presenter for RecordingPresenter {
    fn chapter_started(&self, _index: usize, source: &str, characters: &[Character]) {
        self.events.borrow_mut().push(Event::Chapter(source.into(), names(characters)));
    }

    fn character_extraction_failed(&self, error: &str) {
        self.events.borrow_mut().push(Event::ExtractionFailed(error.into()));
    }

    fn speaker_estimated(&self, sentence: &Sentence, confidence: Option<f64>) {
        let speaker = sentence.speaker.as_ref().unwrap().name.clone();
        self.events.borrow_mut().push(Event::Estimated(sentence.text.clone(), speaker, confidence));
    }

    fn speaker_estimation_failed(&self, sentence: &Sentence, error: Option<&str>) {
        self.events.borrow_mut().push(Event::EstimationFailed(sentence.text.clone(), error.map(String::from)));
    }

    fn context_shrunk(&self, previous_count: usize) {
        self.events.borrow_mut().push(Event::Shrunk(previous_count));
    }

    fn audio_exported(&self, location: Option<&str>, result: &SynthesisResult) {
        self.events.borrow_mut().push(Event::Exported(location.map(String::from), result.failures.len()));
    }

    fn finished(&self, characters: &[Character], voices: Option<&IndexMap<String, String>>) {
        self.events.borrow_mut().push(Event::Finished(names(characters), voices.cloned()));
    }
}

fn exporter(client: &FakeSynthesizer, writer: &FakeWriter, presenter: Rc<dyn Presenter>) -> AudioExporter {
    let assigner = VoiceAssigner::with_voices(["n"]).unwrap();
    let synthesizer = SentenceSynthesizer::new(Box::new(client.clone()), assigner, presenter.clone(), 0);
    AudioExporter::new(synthesizer, Box::new(writer.clone()), presenter)
}

#[test]
fn export_writes_audio() {
    let (writer, presenter) = (FakeWriter::default(), Rc::new(RecordingPresenter::default()));
    let failed = exporter(&FakeSynthesizer::ok(), &writer, presenter.clone())
        .export("1.txt", &Novel::new(sentences(&["あ", "い"])))
        .unwrap();
    assert!(!failed);
    assert_eq!(*writer.written.borrow(), HashMap::from([("1.txt".to_string(), 2)]));
    assert_eq!(*presenter.events.borrow(), [Event::Exported(Some("out/1.txt.wav".into()), 0)]);
}

#[test]
fn export_reports_partial_failure() {
    let writer = FakeWriter::default();
    let client = FakeSynthesizer::new(|text| if text == "い" { input_error(text) } else { None });
    let failed = exporter(&client, &writer, Rc::new(SilentPresenter))
        .export("1.txt", &Novel::new(sentences(&["あ", "い"])))
        .unwrap();
    assert!(failed);
    assert_eq!(*writer.written.borrow(), HashMap::from([("1.txt".to_string(), 1)]));
}

#[test]
fn export_removes_stale_output_when_nothing_to_read() {
    let (writer, presenter) = (FakeWriter::default(), Rc::new(RecordingPresenter::default()));
    let failed = exporter(&FakeSynthesizer::ok(), &writer, presenter.clone())
        .export("1.txt", &Novel::new(sentences(&["＊\u{3000}＊\u{3000}＊", ""])))
        .unwrap();
    assert!(!failed);
    assert_eq!(*writer.removed.borrow(), ["1.txt"]);
    assert_eq!(*presenter.events.borrow(), [Event::Exported(None, 0)]);
}

#[test]
fn export_aborts_on_writer_error() {
    for (text, message) in [("あ", "disk full"), ("＊\u{3000}＊\u{3000}＊", "is a directory")] {
        let writer = FakeWriter { fail: true, ..FakeWriter::default() };
        let result = exporter(&FakeSynthesizer::ok(), &writer, Rc::new(SilentPresenter))
            .export("1.txt", &Novel::new(sentences(&[text])));
        assert_eq!(aborted(result), message);
    }
}

#[test]
fn export_aborts_on_fatal_synthesis_error_without_writing() {
    let writer = FakeWriter::default();
    let client = FakeSynthesizer::new(|text| if text == "い" { Some(TtsError::Fatal("down".into())) } else { None });
    let result =
        exporter(&client, &writer, Rc::new(SilentPresenter)).export("1.txt", &Novel::new(sentences(&["あ", "い"])));
    assert_eq!(aborted(result), "down");
    assert!(writer.written.borrow().is_empty() && writer.removed.borrow().is_empty());
}

// --- 話者の推測 ---

type Answer = Result<Option<SpeakerGuess>, EstimationError>;
/// 推測器に渡された (前の文, 推測する文, 後の文)
type Call = (Vec<String>, String, Vec<String>);

/// 推測ごとに answers の結果を順に返す
#[derive(Clone, Default)]
struct FakeEstimator {
    answers: Rc<RefCell<VecDeque<Answer>>>,
    calls: Rc<RefCell<Vec<Call>>>,
}

impl FakeEstimator {
    fn new(answers: impl IntoIterator<Item = Answer>) -> Self {
        Self { answers: Rc::new(RefCell::new(answers.into_iter().collect())), calls: Rc::default() }
    }
}

fn texts(sentences: &[Sentence]) -> Vec<String> {
    sentences.iter().map(|s| s.text.clone()).collect()
}

impl SpeakerEstimator for FakeEstimator {
    fn estimate_speaker(
        &mut self,
        _candidates: &[Character],
        previous: &[Sentence],
        target: &Sentence,
        following: &[Sentence],
    ) -> Answer {
        self.calls.borrow_mut().push((texts(previous), target.text.clone(), texts(following)));
        self.answers.borrow_mut().pop_front().expect("想定より多く呼ばれた")
    }
}

fn guess(index: i64) -> Answer {
    Ok(Some(SpeakerGuess::new(index)))
}

fn dialogue_novel() -> Novel {
    Novel {
        sentences: sentences(&["地の文", "「あ」", "「い」", "「う」"]),
        characters: vec![character("花子"), character("太郎")],
    }
}

fn estimation(estimator: &FakeEstimator, presenter: Rc<dyn Presenter>) -> SpeakerEstimation {
    SpeakerEstimation::new(Box::new(estimator.clone()), presenter)
}

fn speaker_names(novel: &Novel) -> Vec<String> {
    novel.sentences.iter().map(|s| s.speaker.as_ref().unwrap().name.clone()).collect()
}

#[test]
fn speaker_estimation_assigns_candidates() {
    let mut novel = dialogue_novel();
    let presenter = Rc::new(RecordingPresenter::default());
    let estimator = FakeEstimator::new([Ok(Some(SpeakerGuess::with_confidence(1, 0.9))), guess(2), guess(0)]);
    estimation(&estimator, presenter.clone()).run(&mut novel).unwrap();
    assert_eq!(speaker_names(&novel), ["ナレーター", "花子", "太郎", "ナレーター"]);
    assert_eq!(presenter.events.borrow()[0], Event::Estimated("「あ」".into(), "花子".into(), Some(0.9)));
    // セリフではない文は推測しない
    let targets: Vec<String> = estimator.calls.borrow().iter().map(|(_, t, _)| t.clone()).collect();
    assert_eq!(targets, ["「あ」", "「い」", "「う」"]);
}

#[test]
fn speaker_estimation_falls_back_to_narrator() {
    let answers = [
        (Ok(None), None),
        (guess(3), None),
        (guess(-1), None),
        (Err(EstimationError::Failed("bad response".into())), Some("bad response")),
    ];
    for (answer, error) in answers {
        let mut novel =
            Novel { sentences: sentences(&["「あ」"]), characters: vec![character("花子"), character("太郎")] };
        let presenter = Rc::new(RecordingPresenter::default());
        estimation(&FakeEstimator::new([answer]), presenter.clone()).run(&mut novel).unwrap();
        assert!(novel.sentences[0].speaker.as_ref().unwrap().is_narrator());
        assert_eq!(*presenter.events.borrow(), [Event::EstimationFailed("「あ」".into(), error.map(String::from))]);
    }
}

#[test]
fn speaker_estimation_passes_context() {
    let mut novel = dialogue_novel();
    let estimator = FakeEstimator::new([guess(1), guess(2), guess(1)]);
    let mut estimation = estimation(&estimator, Rc::new(SilentPresenter));
    estimation.previous_count = 2;
    estimation.following_count = 1;
    estimation.run(&mut novel).unwrap();
    let calls = estimator.calls.borrow();
    assert_eq!(calls[2], (vec!["「あ」".to_string(), "「い」".to_string()], "「う」".to_string(), vec![]));
    assert_eq!(calls[0], (vec!["地の文".to_string()], "「あ」".to_string(), vec!["「い」".to_string()]));
}

#[test]
fn speaker_estimation_estimates_all_sentences_unless_dialogue_only() {
    let mut novel = dialogue_novel();
    let estimator = FakeEstimator::new([guess(0), guess(0), guess(0), guess(0)]);
    let mut estimation = estimation(&estimator, Rc::new(SilentPresenter));
    estimation.dialogue_only = false;
    estimation.run(&mut novel).unwrap();
    assert_eq!(estimator.calls.borrow().len(), 4);
}

fn too_long(message: &str) -> Answer {
    Err(EstimationError::ContextTooLong(message.into()))
}

#[test]
fn speaker_estimation_shrinks_context_when_too_long() {
    let mut texts: Vec<String> = (0..8).map(|i| format!("文{i}")).collect();
    texts.push("「あ」".into());
    let mut novel =
        Novel { sentences: texts.iter().map(Sentence::new).collect(), characters: vec![character("花子")] };
    let presenter = Rc::new(RecordingPresenter::default());
    let estimator = FakeEstimator::new([too_long("long"), too_long("long"), guess(1)]);
    let mut estimation = estimation(&estimator, presenter.clone());
    estimation.previous_count = 8;
    estimation.run(&mut novel).unwrap();
    let lengths: Vec<usize> = estimator.calls.borrow().iter().map(|(p, _, _)| p.len()).collect();
    assert_eq!(lengths, [8, 4, 2]);
    assert_eq!(novel.sentences.last().unwrap().speaker.as_ref().unwrap().name, "花子");
    assert_eq!(presenter.events.borrow()[..2], [Event::Shrunk(4), Event::Shrunk(2)]);
}

#[test]
fn speaker_estimation_gives_up_when_too_long_without_context() {
    let mut novel = Novel { sentences: sentences(&["前", "「あ」"]), characters: vec![character("花子")] };
    let presenter = Rc::new(RecordingPresenter::default());
    let estimator = FakeEstimator::new([too_long("long"), too_long("long"), too_long("still long")]);
    let mut estimation = estimation(&estimator, presenter.clone());
    estimation.previous_count = 2;
    estimation.run(&mut novel).unwrap();
    let lengths: Vec<usize> = estimator.calls.borrow().iter().map(|(p, _, _)| p.len()).collect();
    assert_eq!(lengths, [1, 1, 0]);
    assert!(novel.sentences.last().unwrap().speaker.as_ref().unwrap().is_narrator());
    assert_eq!(
        presenter.events.borrow().last().unwrap(),
        &Event::EstimationFailed("「あ」".into(), Some("still long".into()))
    );
}

#[test]
fn speaker_estimation_stops_on_fatal_errors() {
    let estimator = FakeEstimator::new([Err(EstimationError::Fatal("down".into()))]);
    let result = estimation(&estimator, Rc::new(SilentPresenter)).run(&mut Novel::new(sentences(&["「あ」"])));
    assert_eq!(result, Err(NarrationError::Fatal("down".into())));
}

// --- 全体の流れ ---

type Extracted = Result<Vec<Character>, ExtractionError>;

#[derive(Clone, Default)]
struct FakeExtractor {
    results: Rc<RefCell<VecDeque<Extracted>>>,
    known: Rc<RefCell<Vec<Vec<String>>>>,
}

impl FakeExtractor {
    fn new(results: impl IntoIterator<Item = Extracted>) -> Self {
        Self { results: Rc::new(RefCell::new(results.into_iter().collect())), known: Rc::default() }
    }
}

impl CharacterExtractor for FakeExtractor {
    fn extract_characters(&mut self, _novel: &Novel, known: &[Character]) -> Extracted {
        self.known.borrow_mut().push(names(known));
        self.results.borrow_mut().pop_front().expect("想定より多く呼ばれた")
    }
}

#[test]
fn narrate_novels_keeps_previous_characters_when_extraction_fails() {
    let extractor =
        FakeExtractor::new([Ok(vec![character("花子")]), Err(ExtractionError::Failed("broken".into())), Ok(vec![])]);
    let presenter = Rc::new(RecordingPresenter::default());
    let mut chapters: Vec<Chapter> =
        (0..3).map(|i| Chapter::new(format!("{i}.txt"), Novel::new(sentences(&["地の文"])))).collect();
    let speaker_estimation = estimation(&FakeEstimator::default(), Rc::new(SilentPresenter));
    let report = NarrateNovels::new(Box::new(extractor.clone()), speaker_estimation, presenter.clone(), None)
        .run(&mut chapters)
        .unwrap();
    assert_eq!(*extractor.known.borrow(), [vec![], vec!["花子".to_string()], vec!["花子".to_string()]]);
    assert_eq!(names(&report.characters), ["花子"]);
    assert!(chapters.iter().all(|c| names(&c.novel.characters) == ["花子"]));
    assert!(presenter.events.borrow().contains(&Event::ExtractionFailed("broken".into())));
    assert_eq!(presenter.events.borrow().last().unwrap(), &Event::Finished(vec!["花子".into()], None));
}

#[test]
fn narrate_novels_stops_on_fatal_extraction_error() {
    let extractor = FakeExtractor::new([Err(ExtractionError::Fatal("down".into()))]);
    let mut chapters = vec![Chapter::new("1.txt", Novel::new(sentences(&["地の文"])))];
    let speaker_estimation = estimation(&FakeEstimator::default(), Rc::new(SilentPresenter));
    let result =
        NarrateNovels::new(Box::new(extractor), speaker_estimation, Rc::new(SilentPresenter), None).run(&mut chapters);
    assert_eq!(result, Err(NarrationError::Fatal("down".into())));
}

#[test]
fn narrate_novels_exports_each_chapter() {
    let extractor = FakeExtractor::new([Ok(vec![character("花子")]), Ok(vec![character("花子")])]);
    let writer = FakeWriter::default();
    let presenter = Rc::new(RecordingPresenter::default());
    let client = FakeSynthesizer::new(|text| if text == "失敗" { input_error(text) } else { None });
    let mut chapters = vec![
        Chapter::new("1.txt", Novel::new(sentences(&["あ"]))),
        Chapter::new("2.txt", Novel::new(sentences(&["い", "失敗"]))),
    ];
    let speaker_estimation = estimation(&FakeEstimator::default(), Rc::new(SilentPresenter));
    let exporter = exporter(&client, &writer, presenter.clone());
    let report = NarrateNovels::new(Box::new(extractor), speaker_estimation, presenter.clone(), Some(exporter))
        .run(&mut chapters)
        .unwrap();
    assert_eq!(*writer.written.borrow(), HashMap::from([("1.txt".to_string(), 1), ("2.txt".to_string(), 1)]));
    assert_eq!(report.failed_sources, ["2.txt"]);
    let voices = IndexMap::from([("ナレーター".to_string(), "n".to_string())]);
    assert_eq!(presenter.events.borrow().last().unwrap(), &Event::Finished(vec!["花子".into()], Some(voices)));
}
