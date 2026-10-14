mod support;

use serde_json::json;
use support::*;
use yomitone::diagnostics::{check_jev, check_llm, check_tts, CheckOutcome, CHECK_TEXT};
use yomitone::setup::{JevOptions, LlmOptions, SetupError, TtsBackend, TtsOptions};

fn json_response(body: serde_json::Value) -> HttpResponse {
    HttpResponse { status: 200, content_type: "application/json", body: body.to_string().into_bytes() }
}

fn llm_options(url: &str, model: &str) -> LlmOptions {
    let port = url.rsplit(':').next().unwrap().to_string();
    LlmOptions { host: "127.0.0.1".into(), port, model: model.into() }
}

fn describe(error: &SetupError) -> String {
    error.to_string()
}

fn failed(outcome: CheckOutcome) -> String {
    match outcome {
        CheckOutcome::Failed(message) => message,
        other => panic!("失敗になりません: {other:?}"),
    }
}

#[test]
fn llm_model_is_listed() {
    let (url, requests) =
        http_server(|_| json_response(json!({"object": "list", "data": [{"id": "model-a"}, {"id": "model-b"}]})));
    assert!(matches!(check_llm(&llm_options(&url, "model-b")), CheckOutcome::Ok(_)));
    match check_llm(&llm_options(&url, "missing")) {
        CheckOutcome::Warning(message) => assert!(message.contains("model-a, model-b"), "{message}"),
        other => panic!("{other:?}"),
    }
    let requests = requests.lock().unwrap();
    assert_eq!((requests[0].method.as_str(), requests[0].path.as_str()), ("GET", "/v1/models"));
}

#[test]
fn llm_unreachable_or_broken() {
    let message = failed(check_llm(&llm_options(&format!("http://127.0.0.1:{}", unused_tcp_port()), "m")));
    assert!(message.contains("接続できませんでした"), "{message}");
    let (url, _) =
        http_server(|_| HttpResponse { status: 404, content_type: "text/plain", body: b"not found".to_vec() });
    assert!(failed(check_llm(&llm_options(&url, "m"))).contains("404"));
}

#[test]
fn jev_authenticates_and_lists_models() {
    let (url, requests) = http_server(|request| {
        if request.header("authorization") != Some("Bearer key") {
            return HttpResponse {
                status: 401,
                content_type: "application/json",
                body: br#"{"error": "invalid api key"}"#.to_vec(),
            };
        }
        json_response(json!({"models": [{"name": "jev-1", "description": "", "release_date": "2026-01-01"}]}))
    });
    let options = |key: &str, model: &str| JevOptions {
        api_key: Some(key.into()),
        model: Some(model.into()),
        base_url: Some(url.clone()),
    };
    assert!(matches!(check_jev(&options("key", "jev-1")), CheckOutcome::Ok(_)));
    match check_jev(&options("key", "jev-latest")) {
        CheckOutcome::Warning(message) => assert!(message.contains("別名"), "{message}"),
        other => panic!("{other:?}"),
    }
    let message = failed(check_jev(&options("wrong", "jev-1")));
    assert!(message.contains("401") && message.contains("invalid api key"), "{message}");
    assert_eq!(requests.lock().unwrap()[0].path, "/v1/models");
}

#[test]
fn tts_openai_synthesizes_with_every_voice() {
    let (url, requests) = speech_server();
    let options = TtsOptions { url: Some(url.clone()), voices: vec!["n".into(), "a".into()], ..TtsOptions::default() };
    match check_tts(&options, describe) {
        CheckOutcome::Ok(message) => assert!(message.contains("n, a"), "{message}"),
        other => panic!("{other:?}"),
    }
    let sent: Vec<(String, String)> = requests
        .lock()
        .unwrap()
        .iter()
        .map(|r| (r.json()["input"].as_str().unwrap().into(), r.json()["voice"].as_str().unwrap().into()))
        .collect();
    assert_eq!(sent, [(CHECK_TEXT.to_string(), "n".to_string()), (CHECK_TEXT.to_string(), "a".to_string())]);

    let wrong_model = TtsOptions { model: "unknown-model".into(), ..options.clone() };
    assert!(failed(check_tts(&wrong_model, describe)).contains("音声 n で合成できませんでした。音声名やモデル名"));
    let unreachable = TtsOptions { url: Some(format!("http://127.0.0.1:{}/v1", unused_tcp_port())), ..options.clone() };
    assert!(failed(check_tts(&unreachable, describe)).contains("TTSサーバー(http://127.0.0.1:"));
    assert!(failed(check_tts(&TtsOptions { voices: vec![], ..options }, describe)).contains("音声が指定されていません"));
}

#[test]
fn tts_wyoming_checks_voices_and_synthesizes() {
    let (uri, state) = wyoming_server();
    let options = TtsOptions { backend: TtsBackend::Wyoming, url: Some(uri.clone()), ..TtsOptions::default() };
    match check_tts(&options, describe) {
        CheckOutcome::Ok(message) => assert!(message.contains("ja-a, ja-multi:s1, ja-multi:s2"), "{message}"),
        other => panic!("{other:?}"),
    }
    assert_eq!(state.lock().unwrap().requests.len(), 1);

    let unknown = TtsOptions { voices: vec!["nope".into()], ..options.clone() };
    assert!(failed(check_tts(&unknown, describe)).contains("nope"));
    state.lock().unwrap().describe = "error".into();
    assert!(failed(check_tts(&options, describe)).contains("describe failed"));
}
