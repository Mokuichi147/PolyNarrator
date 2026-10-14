//! LLM・Jev 互換 API へ送る要求本文が Python 版と同じであることを確かめる。
//! 期待値は fixtures/generate_python_requests.py で Python 版から作ったもの

mod support;

use polynarrator::adapters::jev::JevSpeakerEstimator;
use polynarrator::adapters::llm::OpenAiChatModel;
use polynarrator::application::ports::{CharacterExtractor, SpeakerEstimator, SpeakerGuess};
use polynarrator::domain::character::{Character, Gender};
use polynarrator::domain::novel::{Novel, Sentence};
use serde_json::{json, Value};
use support::{http_server, HttpResponse};

fn reply(request: &support::HttpRequest) -> HttpResponse {
    let body = if request.path.ends_with("/chat/completions") {
        let content = if request.json()["response_format"]["json_schema"]["name"] == "Narrators" {
            r#"{"narrators": []}"#
        } else {
            r#"{"narrator_index": 1}"#
        };
        json!({"choices": [{"index": 0, "message": {"role": "assistant", "content": content}}]})
    } else {
        json!({"model": "jev", "usage": {}, "answers": {"speaker": {"type": "choice", "choice": "花子", "confidence": 0.5, "probabilities": {}}}})
    };
    HttpResponse { status: 200, content_type: "application/json", body: body.to_string().into_bytes() }
}

#[test]
fn requests_match_python_version() {
    let (url, requests) = http_server(reply);
    let port = url.rsplit(':').next().unwrap().to_string();

    let hanako = Character::new("花子", "明るい'委員長\"")
        .with_aliases(["はな", "委員長'", "\\"])
        .with_gender(Some(Gender::Female));
    let taro = Character::new("太郎", "");
    let novel = Novel::from_lines(["地の文です。\n", "「こんにちは」\n", "\n", "「やあ」\n"]);
    let known = [Character::new("田中花子", "").with_aliases(["花子"]).with_gender(Some(Gender::Female)), taro.clone()];
    let candidates =
        [Character::narrator(), hanako.clone(), taro, Character::new("花子", "別人").with_gender(Some(Gender::Other))];
    let previous = [
        Sentence::spoken_by("地の文", Character::narrator()),
        Sentence::spoken_by("「あ」", hanako),
        Sentence::new("「い」"),
    ];
    let target = Sentence::new("「う」");
    let following = [Sentence::new("後"), Sentence::new("「え」")];

    let mut llm = OpenAiChatModel::new("127.0.0.1", &port, "test-model");
    assert_eq!(llm.extract_characters(&novel, &[]).unwrap(), []);
    llm.extract_characters(&novel, &known).unwrap();
    assert_eq!(llm.estimate_speaker(&candidates, &previous, &target, &following).unwrap(), Some(SpeakerGuess::new(1)));
    llm.estimate_speaker(&candidates, &[], &target, &[]).unwrap();
    let mut jev = JevSpeakerEstimator::new(Some("key".into()), Some("jev-test".into()), Some(url.clone())).unwrap();
    assert_eq!(
        jev.estimate_speaker(&candidates, &previous, &target, &following).unwrap(),
        Some(SpeakerGuess::with_confidence(1, 0.5))
    );

    let expected: Vec<Value> = serde_json::from_str(include_str!("fixtures/python_requests.json")).unwrap();
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), expected.len());
    for (request, expected) in requests.iter().zip(&expected) {
        assert_eq!(request.path, expected["path"]);
        assert_eq!(request.json(), expected["body"], "{}", request.path);
    }
    assert_eq!(requests[0].header("authorization"), Some("Bearer lm-studio"));
    assert_eq!(requests[4].header("authorization"), Some("Bearer key"));
}
