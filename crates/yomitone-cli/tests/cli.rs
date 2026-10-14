#[path = "../../yomitone/tests/support/mod.rs"]
mod support;

use std::cell::{Cell, RefCell};
use std::fs;
use std::io::Write;
use std::path::Path;
use std::process::Command;
use std::rc::Rc;

use clap::Parser;
use serde_json::{json, Value};
use support::*;
use yomitone::adapters::wav::WavDirectory;
use yomitone::application::narrate_novels::NarrationError;
use yomitone::application::ports::*;
use yomitone::application::speech_synthesis::{AudioExporter, SentenceSynthesizer};
use yomitone::domain::audio::Audio;
use yomitone::domain::character::Character;
use yomitone::domain::novel::{Novel, Sentence};
use yomitone::domain::voice_assigner::VoiceAssigner;
use yomitone_cli::*;

fn parse(args: &[&str]) -> Result<Args, clap::Error> {
    Args::try_parse_from(["yomitone"].iter().chain(args).chain(&["data"]))
}

// --- CLI引数 ---

#[test]
fn parser_defaults() {
    let args = parse(&[]).unwrap();
    assert_eq!((args.tts_pause, args.tts_max_chars, args.tts_timeout), (0.3, 200, 60.0));
    assert_eq!(
        (args.host.as_str(), args.port.as_str(), args.model.as_str()),
        ("localhost", "1234", "qwen3.8-flash-next-iq3_s")
    );
}

#[test]
fn parser_rejects_invalid_values() {
    let cases = [
        ["--tts-pause", "-1"],
        ["--tts-pause", "nan"],
        ["--tts-pause", "inf"],
        ["--tts-pause", "60.1"],
        ["--tts-pause", "1e10"],
        ["--tts-max-chars", "-1"],
        ["--tts-max-chars", "1.5"],
        ["--tts-timeout", "0"],
        ["--tts-timeout", "-1"],
        ["--tts-timeout", "nan"],
        ["--tts-timeout", "inf"],
    ];
    for args in cases {
        assert!(parse(&args).is_err(), "{args:?}");
    }
}

#[test]
fn parser_accepts_boundaries() {
    let args = parse(&["--tts-pause", "0", "--tts-max-chars", "0"]).unwrap();
    assert_eq!((args.tts_pause, args.tts_max_chars), (0.0, 0));
    assert_eq!(parse(&["--tts-pause", "60"]).unwrap().tts_pause, 60.0);
}

// --- 入力ファイル ---

#[test]
fn load_chapters_missing_folder() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("file.txt"), "").unwrap();
    for folder in [dir.path().join("missing"), dir.path().join("file.txt")] {
        let error = load_chapters(folder.to_str().unwrap()).unwrap_err();
        assert!(error.0.contains("入力フォルダ"), "{error}");
    }
}

#[test]
fn load_chapters_in_order() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("2.txt"), "い\n").unwrap();
    fs::write(dir.path().join("1.txt"), "あ\n「う」\n").unwrap();
    let chapters = load_chapters(dir.path().to_str().unwrap()).unwrap();
    let sources: Vec<&str> = chapters.iter().map(|c| c.source.as_str()).collect();
    assert_eq!(
        sources,
        [dir.path().join("1.txt").display().to_string(), dir.path().join("2.txt").display().to_string()]
    );
    let texts: Vec<&str> = chapters[0].novel.sentences.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["あ", "「う」"]);
}

// --- 音声合成の初期化 ---

fn tts_error(args: &[&str]) -> String {
    match create_tts(&parse(args).unwrap()) {
        Err(error) => error.0,
        Ok(_) => panic!("エラーになりません: {args:?}"),
    }
}

fn tts_voices(args: &[&str]) -> (Vec<String>, String) {
    let (_, assigner) = create_tts(&parse(args).unwrap()).unwrap();
    (assigner.voices().to_vec(), assigner.narrator_voice().to_string())
}

#[test]
fn create_tts_openai_requires_voices() {
    for extra in [&[][..], &["--tts-narrator-voice", "alloy"], &["--tts-male-voices", "alloy"]] {
        let args: Vec<&str> = ["--tts-output", "out"].iter().chain(extra).copied().collect();
        assert!(tts_error(&args).contains("--tts-voices"));
    }
}

#[test]
fn create_tts_rejects_voice_outside_list() {
    assert!(tts_error(&["--tts-output", "out", "--tts-voices", "a,b", "--tts-narrator-voice", "z"]).contains('z'));
}

#[test]
fn create_tts_openai() {
    let (voices, narrator) = tts_voices(&["--tts-output", "out", "--tts-voices", "a, b,", "--tts-male-voices", "b"]);
    assert_eq!(voices, ["a", "b"]);
    assert_eq!(narrator, "a");
}

#[test]
fn create_tts_wyoming_uses_server_voices() {
    let (uri, _) = wyoming_server();
    let (voices, _) = tts_voices(&["--tts-output", "out", "--tts-backend", "wyoming", "--tts-url", &uri]);
    assert_eq!(voices, ["ja-a", "ja-multi:s1", "ja-multi:s2"]);
}

#[test]
fn create_tts_wyoming_accepts_known_voice_outside_list() {
    let (uri, _) = wyoming_server();
    // 複数話者モデルの既定話者や、言語の絞り込みで一覧に出ない音声も明示すれば使える
    let (voices, narrator) = tts_voices(&[
        "--tts-output",
        "out",
        "--tts-backend",
        "wyoming",
        "--tts-url",
        &uri,
        "--tts-narrator-voice",
        "ja-multi",
        "--tts-male-voices",
        "en-a,ja-a",
    ]);
    assert_eq!(voices, ["ja-a", "ja-multi:s1", "ja-multi:s2", "ja-multi", "en-a"]);
    assert_eq!(narrator, "ja-multi");
}

#[test]
fn create_tts_wyoming_explicit_voice_when_language_matches_nothing() {
    let (uri, _) = wyoming_server();
    let (voices, _) = tts_voices(&[
        "--tts-output",
        "out",
        "--tts-backend",
        "wyoming",
        "--tts-url",
        &uri,
        "--tts-language",
        "fr",
        "--tts-narrator-voice",
        "en-a",
    ]);
    assert_eq!(voices, ["en-a"]);
}

#[test]
fn create_tts_wyoming_errors() {
    let (uri, _) = wyoming_server();
    let base = ["--tts-output", "out", "--tts-backend", "wyoming", "--tts-url", uri.as_str()];
    assert!(tts_error(&[&base[..], &["--tts-voices", "ja-a,nope"]].concat()).contains("nope"));
    assert!(tts_error(&[&base[..], &["--tts-language", "fr"]].concat()).contains("--tts-language"));
    let unreachable = format!("tcp://127.0.0.1:{}", unused_tcp_port());
    let error = tts_error(&[
        "--tts-output",
        "out",
        "--tts-backend",
        "wyoming",
        "--tts-url",
        &unreachable,
        "--tts-timeout",
        "2",
    ]);
    assert!(error.contains("音声一覧を取得できませんでした"), "{error}");
}

// --- ファイル単位の書き出しと表示 ---

#[derive(Clone, Default)]
struct Buffer(Rc<RefCell<Vec<u8>>>);

impl Write for Buffer {
    fn write(&mut self, data: &[u8]) -> std::io::Result<usize> {
        self.0.borrow_mut().extend_from_slice(data);
        Ok(data.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Buffer {
    fn text(&self) -> String {
        String::from_utf8(self.0.borrow().clone()).unwrap()
    }
}

#[derive(Clone, Default)]
struct FakeClient {
    failing: Vec<String>,
    fatal: Vec<String>,
    broken: Vec<String>,
    requests: Rc<RefCell<Vec<String>>>,
}

impl FakeClient {
    fn failing(texts: &[&str]) -> Self {
        Self { failing: texts.iter().map(|s| s.to_string()).collect(), ..Self::default() }
    }

    fn fatal(texts: &[&str]) -> Self {
        Self { fatal: texts.iter().map(|s| s.to_string()).collect(), ..Self::default() }
    }
}

impl SpeechSynthesizer for FakeClient {
    fn list_voices(&self) -> Vec<String> {
        Vec::new()
    }

    fn synthesize(&mut self, text: &str, _voice: &str) -> Result<Audio, TtsError> {
        self.requests.borrow_mut().push(text.to_string());
        let text = text.to_string();
        if self.fatal.contains(&text) {
            return Err(TtsError::Fatal("down".into()));
        }
        if self.failing.contains(&text) {
            return Err(TtsError::Input("bad".into()));
        }
        if self.broken.contains(&text) {
            return Ok(Audio::new(8000, 2, 1, vec![1, 0, 2]));
        }
        Ok(Audio::new(16000, 2, 1, [1u8, 0].repeat(160)))
    }
}

fn novel(texts: &[&str]) -> Novel {
    Novel::new(texts.iter().map(|t| Sentence::new(*t)).collect())
}

/// outfile と同じ名前の入力を、経過を表示しながら書き出す
fn console_exporter(outfile: &Path, client: &FakeClient, out: &Buffer) -> AudioExporter {
    let presenter: Rc<dyn Presenter> = Rc::new(ConsolePresenter::new(Box::new(out.clone())));
    let synthesizer = SentenceSynthesizer::new(
        Box::new(client.clone()),
        VoiceAssigner::with_voices(["n"]).unwrap(),
        presenter.clone(),
        200,
    );
    AudioExporter::new(synthesizer, Box::new(WavDirectory::new(outfile.parent().unwrap(), 0.1)), presenter)
}

#[test]
fn export_writes_wav() {
    let dir = tempfile::tempdir().unwrap();
    let outfile = dir.path().join("out").join("1.wav");
    let out = Buffer::default();
    let failed = console_exporter(&outfile, &FakeClient::default(), &out)
        .export("1.txt", &novel(&["あ", "＊\u{3000}＊\u{3000}＊", "い"]))
        .unwrap();
    assert!(!failed);
    assert!(outfile.exists());
    assert!(out.text().contains(&format!("音声を書き出しました: {}", outfile.display())), "{}", out.text());
    assert!(out.text().contains("1行は読み飛ばしました"));
}

#[test]
fn export_reports_partial_failure() {
    let dir = tempfile::tempdir().unwrap();
    let outfile = dir.path().join("1.wav");
    let out = Buffer::default();
    let failed = console_exporter(&outfile, &FakeClient::failing(&["い"]), &out)
        .export("1.txt", &novel(&["あ", "い", "う"]))
        .unwrap();
    assert!(failed);
    assert!(outfile.exists());
    assert!(out.text().contains("音声合成に失敗しました (n): bad\tい"));
    assert!(out.text().contains("合成に失敗した1文を除く"));
}

#[test]
fn export_removes_stale_wav_when_nothing_to_read() {
    let dir = tempfile::tempdir().unwrap();
    let outfile = dir.path().join("1.wav");
    fs::write(&outfile, b"old").unwrap();
    let out = Buffer::default();
    let failed = console_exporter(&outfile, &FakeClient::default(), &out)
        .export("1.txt", &novel(&["＊\u{3000}＊\u{3000}＊", ""]))
        .unwrap();
    assert!(!failed);
    assert!(!outfile.exists());
    assert!(out.text().contains("読み上げられる文がない"));
}

#[test]
fn export_all_failed() {
    let dir = tempfile::tempdir().unwrap();
    let outfile = dir.path().join("1.wav");
    fs::write(&outfile, b"old").unwrap();
    let out = Buffer::default();
    let mut exporter = console_exporter(&outfile, &FakeClient::failing(&["い", "う"]), &out);
    exporter.export("0.txt", &novel(&["あ"])).unwrap();
    assert!(exporter.export("1.txt", &novel(&["い", "う"])).unwrap());
    assert!(!outfile.exists());
    assert!(out.text().contains("全ての文の合成に失敗"));
}

fn aborted<T: std::fmt::Debug>(result: Result<T, NarrationError>) -> String {
    match result {
        Err(NarrationError::SynthesisAborted(message)) => message,
        other => panic!("中断されていません: {other:?}"),
    }
}

#[test]
fn export_aborts_on_fatal_error() {
    let dir = tempfile::tempdir().unwrap();
    let outfile = dir.path().join("1.wav");
    fs::write(&outfile, b"old").unwrap();
    let result = console_exporter(&outfile, &FakeClient::fatal(&["い"]), &Buffer::default())
        .export("1.txt", &novel(&["あ", "い"]));
    assert_eq!(aborted(result), "down");
    // 中断した場合は書きかけの音声で既存のファイルを上書きしない
    assert_eq!(fs::read(&outfile).unwrap(), b"old");
}

#[test]
fn export_aborts_when_output_path_is_directory() {
    let dir = tempfile::tempdir().unwrap();
    let outfile = dir.path().join("1.wav");
    fs::create_dir(&outfile).unwrap();
    for texts in [&["＊\u{3000}＊\u{3000}＊"][..], &["あ"]] {
        aborted(console_exporter(&outfile, &FakeClient::default(), &Buffer::default()).export("1.txt", &novel(texts)));
    }
}

#[test]
fn export_aborts_on_unsupported_audio() {
    let dir = tempfile::tempdir().unwrap();
    let outfile = dir.path().join("1.wav");
    let client = FakeClient { broken: vec!["い".into()], ..FakeClient::default() };
    let message =
        aborted(console_exporter(&outfile, &client, &Buffer::default()).export("1.txt", &novel(&["あ", "い"])));
    assert!(message.contains("不正な音声"), "{message}");
    assert!(!outfile.exists());
}

// --- CLI全体 ---

/// 1ファイル目と2ファイル目で登場人物の正式名が揺れる LLM。「名前:セリフ」の行をその登場人物のセリフとみなす
#[derive(Default)]
struct FakeAi {
    calls: Cell<usize>,
}

impl CharacterExtractor for FakeAi {
    fn extract_characters(&mut self, _novel: &Novel, _known: &[Character]) -> Result<Vec<Character>, ExtractionError> {
        self.calls.set(self.calls.get() + 1);
        if self.calls.get() == 1 {
            return Ok(vec![Character::new("田中花子", "").with_aliases(["花子"])]);
        }
        Ok(vec![Character::new("花子", ""), Character::new("次郎", "")])
    }
}

impl SpeakerEstimator for FakeAi {
    fn estimate_speaker(
        &mut self,
        candidates: &[Character],
        _p: &[Sentence],
        target: &Sentence,
        _f: &[Sentence],
    ) -> Result<Option<SpeakerGuess>, EstimationError> {
        let inner = target.text.trim_matches(|c| c == '「' || c == '」');
        let (name, text) = inner.split_once(':').unwrap_or((inner, ""));
        let index = candidates.iter().position(|c| c.name == name);
        Ok(index.filter(|_| !text.is_empty()).map(|i| SpeakerGuess::new(i as i64)))
    }
}

fn factories(client: &FakeClient, voices: &[&str], out: &Buffer) -> Factories {
    let client = client.clone();
    let assigner = VoiceAssigner::with_voices(voices.iter().copied()).unwrap();
    let out = out.clone();
    Factories {
        chat_model: Box::new(|_| (Box::new(FakeAi::default()), Box::new(FakeAi::default()))),
        jev: Box::new(|_| Ok(Box::new(FakeAi::default()))),
        tts: Box::new(move |_| Ok((Box::new(client.clone()), assigner.clone()))),
        presenter: Box::new(move || Rc::new(ConsolePresenter::new(Box::new(out.clone())))),
    }
}

fn panic_factories() -> Factories {
    Factories {
        chat_model: Box::new(|_| panic!("入力・出力先の確認前に話者推測を始めた")), ..Factories::default()
    }
}

struct Workspace {
    dir: tempfile::TempDir,
}

impl Workspace {
    fn new() -> Self {
        let workspace = Self { dir: tempfile::tempdir().unwrap() };
        fs::create_dir(workspace.data()).unwrap();
        workspace
    }

    fn with_novels() -> Self {
        let workspace = Self::new();
        workspace.write("1.txt", "あ\n＊\u{3000}＊\u{3000}＊\n「田中花子:い」\n");
        workspace.write("2.txt", "「花子:う」\n「次郎:え」\n");
        fs::write(workspace.data().join(".DS_Store"), [0, 1]).unwrap();
        workspace
    }

    fn data(&self) -> std::path::PathBuf {
        self.dir.path().join("data")
    }

    fn out(&self) -> std::path::PathBuf {
        self.dir.path().join("out")
    }

    fn write(&self, name: &str, text: &str) {
        fs::write(self.data().join(name), text).unwrap();
    }

    fn args(&self, extra: &[&str]) -> Args {
        let out = self.out().display().to_string();
        let data = self.data().display().to_string();
        let args: Vec<&str> = ["yomitone", "--tts-output", &out, "--tts-pause", "0"]
            .iter()
            .chain(extra)
            .copied()
            .chain([data.as_str()])
            .collect();
        Args::try_parse_from(args).unwrap()
    }

    fn outputs(&self) -> Vec<String> {
        let mut names: Vec<String> =
            fs::read_dir(self.out()).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
        names.sort();
        names
    }
}

#[test]
fn main_writes_wav_per_file() {
    let workspace = Workspace::with_novels();
    run(&workspace.args(&[]), &factories(&FakeClient::default(), &["n"], &Buffer::default())).unwrap();
    assert_eq!(workspace.outputs(), ["1.wav", "2.wav"]);
    let (_, _, _, data) = read_wav_file(&workspace.out().join("1.wav"));
    assert_eq!(data.len() / 2, 160 * 2);
}

#[test]
fn main_shows_speakers_and_summary() {
    let workspace = Workspace::with_novels();
    let out = Buffer::default();
    run(&workspace.args(&[]), &factories(&FakeClient::default(), &["n", "a", "b"], &out)).unwrap();
    let output = out.text();
    assert!(
        output.contains(&format!("1 {}\n  - 田中花子 (不明) [花子]\n", workspace.data().join("1.txt").display())),
        "{output}"
    );
    assert!(output.contains("ナレーター\tあ\n"));
    assert!(output.contains("田中花子\t「田中花子:い」\n"));
    assert!(output.contains("\n登場人物一覧\n- 花子 (不明) []\n- 次郎 (不明) []\n"));
    // 章によって正式名が揺れても同じ音声を使う
    assert!(output.contains("\n音声の割り当て\n- ナレーター: n\n- 田中花子: a\n- 花子: a\n- 次郎: b\n"), "{output}");
}

#[test]
fn main_exits_with_error_when_some_sentences_failed() {
    let workspace = Workspace::with_novels();
    let error =
        run(&workspace.args(&[]), &factories(&FakeClient::failing(&["「次郎:え」"]), &["n"], &Buffer::default()))
            .unwrap_err();
    assert!(error.0.contains("2.txt"), "{error}");
    assert_eq!(workspace.outputs(), ["1.wav", "2.wav"]);
}

#[test]
fn main_rejects_undecodable_input() {
    let workspace = Workspace::with_novels();
    fs::write(workspace.data().join("0.txt"), [0x82, 0xa0]).unwrap();
    let error = run(&workspace.args(&[]), &panic_factories()).unwrap_err();
    assert!(error.0.contains("0.txt"), "{error}");
}

#[test]
fn main_passes_max_chars_to_synthesis() {
    let workspace = Workspace::with_novels();
    let client = FakeClient::default();
    run(&workspace.args(&["--tts-max-chars", "3"]), &factories(&client, &["n"], &Buffer::default())).unwrap();
    let requests = client.requests.borrow();
    assert!(requests.len() > 4);
    assert!(requests.iter().all(|text| text.chars().count() <= 3), "{requests:?}");
}

#[test]
fn main_aborts_without_writing_wav_after_fatal_error() {
    let workspace = Workspace::with_novels();
    let error = run(&workspace.args(&[]), &factories(&FakeClient::fatal(&["「次郎:え」"]), &["n"], &Buffer::default()))
        .unwrap_err();
    assert!(error.0.contains("中断"), "{error}");
    assert_eq!(workspace.outputs(), ["1.wav"]);
}

#[test]
fn main_aborts_after_consecutive_failures() {
    let workspace = Workspace::new();
    workspace.write("1.txt", "あ\n");
    workspace.write("2.txt", &"「失敗」\n".repeat(5));
    let error = run(&workspace.args(&[]), &factories(&FakeClient::failing(&["「失敗」"]), &["n"], &Buffer::default()))
        .unwrap_err();
    assert!(error.0.contains("連続"), "{error}");
    assert_eq!(workspace.outputs(), ["1.wav"]);
}

#[test]
fn main_rejects_file_as_output() {
    let workspace = Workspace::with_novels();
    let file = workspace.dir.path().join("file");
    fs::write(&file, "").unwrap();
    let mut args = workspace.args(&[]);
    args.tts_output = Some(file.display().to_string());
    assert!(run(&args, &panic_factories()).unwrap_err().0.contains("書き込めません"));
}

#[test]
fn main_checks_input_folder_before_creating_output() {
    let dir = tempfile::tempdir().unwrap();
    let missing = dir.path().join("missing");
    let args = Args::try_parse_from([
        "yomitone",
        "--tts-output",
        &missing.join("out").display().to_string(),
        &missing.display().to_string(),
    ])
    .unwrap();
    assert!(run(&args, &panic_factories()).unwrap_err().0.contains("入力フォルダ"));
    assert!(!missing.exists());
}

#[cfg(unix)]
#[test]
fn main_rejects_unwritable_output_before_processing() {
    use std::os::unix::fs::PermissionsExt;
    let workspace = Workspace::with_novels();
    fs::create_dir(workspace.out()).unwrap();
    fs::set_permissions(workspace.out(), fs::Permissions::from_mode(0o555)).unwrap();
    // root は書き込み権限を無視するため、書き込めてしまう環境では確かめない
    if tempfile::tempfile_in(workspace.out()).is_err() {
        assert!(run(&workspace.args(&[]), &panic_factories()).unwrap_err().0.contains("書き込めません"));
    }
    fs::set_permissions(workspace.out(), fs::Permissions::from_mode(0o755)).unwrap();
}

#[test]
fn main_rejects_conflicting_output_before_processing() {
    let workspace = Workspace::with_novels();
    fs::create_dir_all(workspace.out().join("2.wav")).unwrap();
    assert!(run(&workspace.args(&[]), &panic_factories()).unwrap_err().0.contains("2.wav"));

    let workspace = Workspace::new();
    workspace.write(".txt", "あ");
    workspace.write(".txt.txt", "い");
    assert!(run(&workspace.args(&[]), &panic_factories()).unwrap_err().0.contains("出力先が同じ"));
}

#[test]
fn main_with_openai_client() {
    let (url, requests) = speech_server();
    let workspace = Workspace::new();
    workspace.write("1.txt", "あ\n＊\u{3000}＊\u{3000}＊\n「田中花子:い」\n");
    workspace.write("2.txt", "「花子:bad」\n「次郎:え」\n");
    let factories =
        Factories { tts: Box::new(create_tts), ..factories(&FakeClient::default(), &["n"], &Buffer::default()) };
    let args = workspace.args(&["--tts-url", &url, "--tts-model", "m", "--tts-voices", "n,a,b"]);
    let error = run(&args, &factories).unwrap_err();
    assert!(error.0.contains("2.txt"), "{error}");
    let sent: Vec<(Value, Value)> =
        requests.lock().unwrap().iter().map(|r| (r.json()["input"].clone(), r.json()["voice"].clone())).collect();
    let expected = [("あ", "n"), ("「田中花子:い」", "a"), ("「花子:bad」", "a"), ("「次郎:え」", "b")]
        .map(|(t, v)| (json!(t), json!(v)));
    assert_eq!(sent, expected);
    let (_, _, _, data) = read_wav_file(&workspace.out().join("2.wav"));
    assert_eq!(data.len() / 2, 240);
}

#[test]
fn main_with_wyoming_client() {
    let (uri, state) = wyoming_server();
    let workspace = Workspace::new();
    workspace.write("1.txt", "あ\n「田中花子:い」\n");
    workspace.write("2.txt", "「次郎:う」\n「花子:error」\n");
    let factories =
        Factories { tts: Box::new(create_tts), ..factories(&FakeClient::default(), &["n"], &Buffer::default()) };
    let args = workspace.args(&[
        "--tts-backend",
        "wyoming",
        "--tts-url",
        &uri,
        "--tts-language",
        "ja-JP",
        "--tts-narrator-voice",
        "ja-multi",
    ]);
    let error = run(&args, &factories).unwrap_err();
    assert!(error.0.contains("2.txt"), "{error}");
    let voices: Vec<Value> = state.lock().unwrap().requests.iter().map(|r| json!([r["text"], r["voice"]])).collect();
    assert_eq!(
        voices,
        [
            json!(["あ", {"name": "ja-multi"}]),
            json!(["「田中花子:い」", {"name": "ja-a"}]),
            json!(["「次郎:う」", {"name": "ja-multi", "speaker": "s1"}]),
            json!(["「花子:error」", {"name": "ja-a"}]),
        ]
    );
    assert_eq!(workspace.outputs(), ["1.wav", "2.wav"]);
}

#[test]
fn main_with_jev_speaker_backend() {
    let workspace = Workspace::with_novels();
    let client = FakeClient::default();
    let out = Buffer::default();
    let used = Rc::new(Cell::new(false));
    let mut factories = factories(&client, &["n", "a", "b"], &out);
    factories.chat_model = Box::new(|_| (Box::new(FakeAi::default()), Box::new(FailingEstimator)));
    factories.jev = Box::new({
        let used = used.clone();
        move |_| {
            used.set(true);
            Ok(Box::new(FakeAi::default()))
        }
    });
    run(&workspace.args(&["--speaker-backend", "jev"]), &factories).unwrap();
    assert!(used.get());
    assert!(out.text().contains("\n音声の割り当て\n- ナレーター: n\n- 田中花子: a\n- 花子: a\n- 次郎: b\n"));
    assert_eq!(*client.requests.borrow(), ["あ", "「田中花子:い」", "「花子:う」", "「次郎:え」"]);
}

struct FailingEstimator;

impl SpeakerEstimator for FailingEstimator {
    fn estimate_speaker(
        &mut self,
        _c: &[Character],
        _p: &[Sentence],
        _t: &Sentence,
        _f: &[Sentence],
    ) -> Result<Option<SpeakerGuess>, EstimationError> {
        panic!("LLM で話者を推測した")
    }
}

// --- 終了コード ---

fn exit_code(args: &[&str]) -> Option<i32> {
    Command::new(env!("CARGO_BIN_EXE_yomitone")).args(args).output().unwrap().status.code()
}

#[test]
fn exit_codes() {
    let workspace = Workspace::new();
    let data = workspace.data().display().to_string();
    // 引数の誤りは 2
    assert_eq!(exit_code(&["--tts-pause", "-1", &data]), Some(2));
    assert_eq!(exit_code(&["--tts-output", "", &data]), Some(2));
    // 処理を続けられない場合は 1
    assert_eq!(exit_code(&[&workspace.dir.path().join("missing").display().to_string()]), Some(1));
    // 入力ファイルがなければ何もせずに終わる
    assert_eq!(exit_code(&[&data]), Some(0));
}
