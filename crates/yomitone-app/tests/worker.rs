#[path = "../../yomitone/tests/support/mod.rs"]
mod support;

use std::fs;
use std::sync::{Arc, Mutex};

use serde_json::{json, Value};
use support::*;
use yomitone::application::cancel::CancelFlag;
use yomitone_app::settings::{self, AppSettings, SpeakerBackend};
use yomitone_app::worker::{run, Event, Outcome};

/// 「名前:セリフ」の行をその登場人物のセリフとみなす LLM サーバー
fn chat_server() -> String {
    let (url, _) = http_server(|request| {
        let body = request.json();
        let content = if body["response_format"]["json_schema"]["name"] == "Narrators" {
            json!({"narrators": [{"name": "花子", "portrait": "", "gender": "女性"}, {"name": "太郎", "portrait": ""}]})
                .to_string()
        } else {
            let prompt = body["messages"][2]["content"].as_str().unwrap_or_default();
            let target = prompt.split("推測したいセリフの内容:").nth(1).unwrap_or_default();
            let index = if target.contains("花子:") {
                1
            } else if target.contains("太郎:") {
                2
            } else {
                0
            };
            json!({"narrator_index": index}).to_string()
        };
        let payload = json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]});
        HttpResponse { status: 200, content_type: "application/json", body: payload.to_string().into_bytes() }
    });
    url
}

fn collect(settings: AppSettings, cancel: CancelFlag) -> Vec<Event> {
    let events = Arc::new(Mutex::new(Vec::new()));
    run(settings, cancel, {
        let events = events.clone();
        move |event| events.lock().unwrap().push(event)
    });
    let events = events.lock().unwrap().clone();
    events
}

fn outcome(events: &[Event]) -> &Outcome {
    match events.last() {
        Some(Event::Done(outcome)) => outcome,
        other => panic!("最後が Done ではありません: {other:?}"),
    }
}

struct Workspace {
    dir: tempfile::TempDir,
}

impl Workspace {
    fn new() -> Self {
        let dir = tempfile::tempdir().unwrap();
        fs::create_dir(dir.path().join("data")).unwrap();
        fs::write(dir.path().join("data").join("1.txt"), "地の文\n「花子:おはよう」\n").unwrap();
        fs::write(dir.path().join("data").join("2.txt"), "「太郎:やあ」\n＊　＊　＊\n").unwrap();
        Self { dir }
    }

    fn settings(&self, llm_url: &str) -> AppSettings {
        let port = llm_url.rsplit(':').next().unwrap().to_string();
        AppSettings {
            input_folder: self.dir.path().join("data").display().to_string(),
            output_folder: self.dir.path().join("out").display().to_string(),
            llm_host: "127.0.0.1".into(),
            llm_port: port,
            tts_pause: 0.0,
            ..AppSettings::default()
        }
    }
}

#[test]
fn narrates_and_exports_audio() {
    let llm = chat_server();
    let (speech, requests) = speech_server();
    let workspace = Workspace::new();
    let settings =
        AppSettings { export_audio: true, tts_url: speech, tts_voices: "n, a, b".into(), ..workspace.settings(&llm) };
    let events = collect(settings, CancelFlag::new());
    assert_eq!(outcome(&events), &Outcome::Completed, "{events:?}");
    assert!(events.contains(&Event::Loaded { counts: vec![2, 2], synthesize: true }));
    let estimated = events.iter().filter(|e| matches!(e, Event::SentenceEstimated)).count();
    assert_eq!(estimated, 4);
    assert!(events.contains(&Event::SynthesisProgressed { done: 2, total: 2 }));
    let voices = events.iter().find_map(|e| match e {
        Event::Finished { voices, .. } => voices.clone(),
        _ => None,
    });
    let expected = [("ナレーター", "n"), ("花子", "a"), ("太郎", "b")].map(|(a, b)| (a.to_string(), b.to_string()));
    assert_eq!(voices.unwrap(), expected);
    let sent: Vec<(Value, Value)> =
        requests.lock().unwrap().iter().map(|r| (r.json()["input"].clone(), r.json()["voice"].clone())).collect();
    assert_eq!(
        sent,
        [("地の文", "n"), ("「花子:おはよう」", "a"), ("「太郎:やあ」", "b")].map(|(t, v)| (json!(t), json!(v)))
    );
    assert!(workspace.dir.path().join("out").join("1.wav").exists());
    assert!(workspace.dir.path().join("out").join("2.wav").exists());
}

#[test]
fn estimates_speakers_without_audio() {
    let llm = chat_server();
    let workspace = Workspace::new();
    let events = collect(workspace.settings(&llm), CancelFlag::new());
    assert_eq!(outcome(&events), &Outcome::Completed, "{events:?}");
    assert!(events.contains(&Event::Loaded { counts: vec![2, 2], synthesize: false }));
    assert!(events.contains(&Event::Log("[花子] 「花子:おはよう」".into())), "{events:?}");
    assert!(events.iter().any(|e| matches!(e, Event::Finished { voices: None, .. })));
    assert!(!workspace.dir.path().join("out").exists());
}

#[test]
fn reports_partial_failure() {
    let llm = chat_server();
    let (speech, _) = speech_server();
    let workspace = Workspace::new();
    fs::write(workspace.dir.path().join("data").join("2.txt"), "「太郎:やあ」\n「太郎:bad」\n").unwrap();
    let settings =
        AppSettings { export_audio: true, tts_url: speech, tts_voices: "n,a,b".into(), ..workspace.settings(&llm) };
    match outcome(&collect(settings, CancelFlag::new())) {
        Outcome::PartiallyFailed(sources) => assert!(sources[0].ends_with("2.txt"), "{sources:?}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn stops_when_cancelled() {
    let llm = chat_server();
    let workspace = Workspace::new();
    let cancel = CancelFlag::new();
    cancel.cancel();
    let events = collect(workspace.settings(&llm), cancel);
    assert_eq!(outcome(&events), &Outcome::Cancelled);
    assert!(!events.iter().any(|e| matches!(e, Event::ChapterStarted { .. })));
}

#[test]
fn reports_setup_errors_with_where_to_fix() {
    let workspace = Workspace::new();
    let cases = [
        (
            AppSettings { input_folder: String::new(), ..workspace.settings("http://127.0.0.1:1") },
            "入力フォルダを選んで",
        ),
        (
            AppSettings { export_audio: true, output_folder: " ".into(), ..workspace.settings("http://127.0.0.1:1") },
            "出力先フォルダ",
        ),
        (AppSettings { export_audio: true, ..workspace.settings("http://127.0.0.1:1") }, "「声の割り当て」"),
        (
            AppSettings {
                speaker_backend: SpeakerBackend::Jev,
                jev_api_key: String::new(),
                ..workspace.settings("http://127.0.0.1:1")
            },
            "「話者推測」",
        ),
    ];
    for (settings, message) in cases {
        if settings.speaker_backend == SpeakerBackend::Jev && std::env::var("TYPESAFE_API_KEY").is_ok() {
            continue;
        }
        match outcome(&collect(settings, CancelFlag::new())) {
            Outcome::Failed(error) => assert!(error.contains(message), "{error}"),
            other => panic!("{other:?}"),
        }
    }
    let empty = tempfile::tempdir().unwrap();
    let settings = AppSettings { input_folder: empty.path().display().to_string(), ..AppSettings::default() };
    match outcome(&collect(settings, CancelFlag::new())) {
        Outcome::Failed(error) => assert!(error.contains(".txt ファイルがありません"), "{error}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn settings_are_saved_without_api_keys() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("Yomitone").join("settings.json");
    assert_eq!(settings::load(&path), AppSettings::default());
    let saved = AppSettings {
        input_folder: "/novel".into(),
        speaker_backend: SpeakerBackend::Jev,
        jev_api_key: "secret-jev".into(),
        tts_api_key: "secret-tts".into(),
        tts_voices: "a,b".into(),
        ..AppSettings::default()
    };
    settings::save(&path, &saved).unwrap();
    let text = fs::read_to_string(&path).unwrap();
    assert!(!text.contains("secret"), "{text}");
    let loaded = settings::load(&path);
    assert_eq!(loaded, AppSettings { jev_api_key: String::new(), tts_api_key: String::new(), ..saved });
    // 項目が足りない古い設定や壊れた設定でも起動できる
    fs::write(&path, r#"{"input_folder": "/old"}"#).unwrap();
    assert_eq!(settings::load(&path), AppSettings { input_folder: "/old".into(), ..AppSettings::default() });
    fs::write(&path, "{").unwrap();
    assert_eq!(settings::load(&path), AppSettings::default());
}
