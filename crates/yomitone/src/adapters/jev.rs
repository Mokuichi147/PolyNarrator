use std::time::Duration;

use serde::Deserialize;
use serde_json::{json, Map, Value};

use crate::adapters::http::{send_with_retry, HttpFailure, RetryPolicy};
use crate::application::ports::{EstimationError, SpeakerEstimator, SpeakerGuess};
use crate::domain::character::Character;
use crate::domain::novel::Sentence;

pub const API_KEY_ENV: &str = "TYPESAFE_API_KEY";
pub const BASE_URL_ENV: &str = "TYPESAFE_BASE_URL";
pub const DEFAULT_MODEL_ENV: &str = "TYPESAFE_DEFAULT_MODEL";
pub const DEFAULT_BASE_URL: &str = "https://api.typesafe.ai";
pub const DEFAULT_MODEL: &str = "jev-latest";
const SYSTEM_ONE_PATH: &str = "/v1/systemone";
/// 1回の送受信のタイムアウト
const TIMEOUT: Duration = Duration::from_secs(10);
/// エラー応答の本文をメッセージに含める長さの上限
const MAX_ERROR_BODY_LENGTH: usize = 200;

/// 400/422のときに入力長超過とみなすエラーメッセージのキーワード
const INPUT_TOO_LONG_KEYWORDS: [&str; 8] = [
    "context length",
    "context window",
    "maximum token",
    "max token",
    "token limit",
    "too many tokens",
    "input too long",
    "too long",
];

/// TypeSafe の公式 SDK と同じ再試行の方針
const RETRY: RetryPolicy = RetryPolicy {
    max_retries: 2,
    retry_status: |status| status == 408 || status == 429 || (500..600).contains(&status),
    backoff_initial: Duration::from_millis(500),
    backoff_max: Duration::from_secs(5),
    retry_after_max: None,
    budget: Some(Duration::from_secs(30)),
};

/// 明示された値、環境変数、既定値の順に使う。空白だけの環境変数は無視する
fn resolve(value: Option<String>, env: &str) -> Option<String> {
    value.or_else(|| std::env::var(env).ok().map(|v| v.trim().to_string()).filter(|v| !v.is_empty()))
}

/// エラー応答の本文から、利用者に見せるメッセージを取り出す
fn extract_message(body: &Value) -> Option<String> {
    let text = |value: Option<&Value>| value.and_then(Value::as_str).map(str::to_string);
    match body {
        Value::String(s) if !s.is_empty() => Some(s.clone()),
        Value::Object(map) => {
            let (error, message, detail) = (map.get("error"), map.get("message"), map.get("detail"));
            text(error)
                .or_else(|| text(error.and_then(|e| e.get("message"))))
                .or_else(|| text(message))
                .or_else(|| text(detail))
                .or_else(|| text(detail.and_then(|d| d.get("message"))))
                .or_else(|| {
                    let entries = detail?.as_array()?;
                    let parts: Vec<String> = entries
                        .iter()
                        .filter_map(|entry| {
                            let msg = entry.get("msg")?.as_str()?;
                            let path = entry
                                .get("loc")
                                .and_then(Value::as_array)
                                .map(|loc| {
                                    loc.iter()
                                        .filter(|item| item.as_str() != Some("body"))
                                        .map(|item| {
                                            item.as_str().map(str::to_string).unwrap_or_else(|| item.to_string())
                                        })
                                        .collect::<Vec<_>>()
                                        .join(".")
                                })
                                .unwrap_or_default();
                            Some(if path.is_empty() { msg.to_string() } else { format!("{path}: {msg}") })
                        })
                        .collect();
                    (!parts.is_empty()).then(|| parts.join("; "))
                })
        }
        _ => None,
    }
}

#[derive(Deserialize)]
struct SystemOneResponse {
    #[allow(dead_code)]
    model: String,
    #[allow(dead_code)]
    usage: Map<String, Value>,
    #[serde(default)]
    answers: Map<String, Value>,
}

#[derive(Deserialize)]
struct ChoiceAnswer {
    choice: String,
    confidence: f64,
    #[allow(dead_code)]
    probabilities: Map<String, Value>,
}

/// Jev(TypeSafe System One API)互換モデルで話者を判定する
pub struct JevSpeakerEstimator {
    agent: ureq::Agent,
    api_key: String,
    base_url: String,
    model: String,
}

impl JevSpeakerEstimator {
    /// 指定がない項目は環境変数、既定値の順に使う。API キーがない場合はエラー
    pub fn new(api_key: Option<String>, model: Option<String>, base_url: Option<String>) -> Result<Self, String> {
        let api_key = resolve(api_key, API_KEY_ENV).ok_or_else(|| {
            format!("APIキーが指定されていません。--jev-api-key か環境変数 {API_KEY_ENV} を指定してください")
        })?;
        let base_url = resolve(base_url, BASE_URL_ENV).unwrap_or_else(|| DEFAULT_BASE_URL.to_string());
        let model = resolve(model, DEFAULT_MODEL_ENV).unwrap_or_else(|| DEFAULT_MODEL.to_string());
        let agent =
            ureq::AgentBuilder::new().timeout_connect(TIMEOUT).timeout_read(TIMEOUT).timeout_write(TIMEOUT).build();
        Ok(Self { agent, api_key, base_url: base_url.trim_end_matches('/').to_string(), model })
    }

    /// 選択肢と、その説明。キーの順に候補の位置と対応する
    fn build_criteria(candidates: &[Character]) -> Map<String, Value> {
        let mut criteria = Map::new();
        for character in candidates {
            // 選択肢のキーは一意である必要があるため、名前が重複した場合は未使用になるまで番号を付ける
            let mut key = character.name.clone();
            let mut suffix = 2;
            while criteria.contains_key(&key) {
                key = format!("{} ({suffix})", character.name);
                suffix += 1;
            }
            criteria.insert(
                key,
                json!({
                    "性別": character.gender.map(|g| g.label()).unwrap_or("不明"),
                    "別名": character.aliases,
                    "説明": character.portrait,
                }),
            );
        }
        criteria
    }

    fn build_state(previous: &[Sentence], target: &Sentence, following: &[Sentence]) -> Value {
        let history: Vec<Value> = previous
            .iter()
            .map(|s| match &s.speaker {
                Some(speaker) if !speaker.is_narrator() => json!({"話者": speaker.name, "セリフ": s.text}),
                _ => json!({"セリフ": s.text}),
            })
            .collect();
        json!({
            "今までの内容": history,
            "推測したいセリフの内容": target.text,
            "後の内容": following.iter().map(|s| s.text.as_str()).collect::<Vec<_>>(),
        })
    }

    fn endpoint(&self) -> String {
        format!("POST {}{SYSTEM_ONE_PATH}", self.base_url)
    }

    /// SDK と同じ形のエラーメッセージにし、入力長超過かどうかを判定する
    fn map_failure(&self, failure: HttpFailure) -> EstimationError {
        match failure {
            HttpFailure::Status { status, body, request_id } => {
                let parsed: Option<Value> = serde_json::from_str(&body).ok();
                let message = parsed
                    .as_ref()
                    .and_then(extract_message)
                    .or_else(|| (!body.is_empty() && parsed.is_none()).then(|| body.clone()))
                    .unwrap_or_else(|| {
                        if body.is_empty() {
                            "status code (no body)".to_string()
                        } else if body.chars().count() > MAX_ERROR_BODY_LENGTH {
                            format!("{}…", body.chars().take(MAX_ERROR_BODY_LENGTH).collect::<String>())
                        } else {
                            body.clone()
                        }
                    });
                let mut error = format!("{}: {status} {message}", self.endpoint());
                if let Some(id) = request_id {
                    error.push_str(&format!(" (request_id={id})"));
                }
                if is_input_too_long(status, &format!("{error} {body}")) {
                    EstimationError::ContextTooLong(error)
                } else {
                    EstimationError::Failed(error)
                }
            }
            // 接続エラーやタイムアウトはこの文の推測を諦めてナレーター扱いにする
            HttpFailure::Transport { timeout: true, .. } => {
                EstimationError::Failed(format!("Request timed out (timeout={}).", TIMEOUT.as_secs_f64()))
            }
            HttpFailure::Transport { message, .. } => EstimationError::Failed(format!("Connection error: {message}")),
        }
    }

    fn invalid_response(&self, detail: impl std::fmt::Display) -> EstimationError {
        EstimationError::Failed(format!("{}: 200 Invalid response data: {detail}", self.endpoint()))
    }
}

fn is_input_too_long(status: u16, message: &str) -> bool {
    // 413はリクエストサイズ超過なので無条件に入力長超過とみなす
    if status == 413 {
        return true;
    }
    if status != 400 && status != 422 {
        return false;
    }
    let message = message.to_lowercase();
    INPUT_TOO_LONG_KEYWORDS.iter().any(|keyword| message.contains(keyword))
}

impl SpeakerEstimator for JevSpeakerEstimator {
    fn estimate_speaker(
        &mut self,
        candidates: &[Character],
        previous: &[Sentence],
        target: &Sentence,
        following: &[Sentence],
    ) -> Result<Option<SpeakerGuess>, EstimationError> {
        let criteria = Self::build_criteria(candidates);
        let body = json!({
            "state": Self::build_state(previous, target, following),
            "model": self.model,
            "questions": {
                "speaker": {
                    "type": "choice",
                    "criteria": criteria,
                    "instructions": "会話の内容から `推測したいセリフの内容` がどの登場人物による発言かを推測してください。\
                                     `今までの内容` と `後の内容` は前後の文脈です。\
                                     誰のセリフとも考えられない場合はナレーターを選択してください。",
                },
            },
        });

        let url = format!("{}{SYSTEM_ONE_PATH}", self.base_url);
        let user_agent = format!("yomitone/{}", env!("CARGO_PKG_VERSION"));
        let authorization = format!("Bearer {}", self.api_key);
        let request = |attempt: u32| {
            let request = self
                .agent
                .post(&url)
                .set("Authorization", &authorization)
                .set("Accept", "application/json")
                .set("User-Agent", &user_agent);
            if attempt > 0 {
                request.set("X-TypeSafe-Retry-Count", &attempt.to_string())
            } else {
                request
            }
        };
        let response = send_with_retry(&RETRY, request, &body).map_err(|failure| self.map_failure(failure))?;

        let response: SystemOneResponse = response.into_json().map_err(|e| self.invalid_response(e))?;
        // 未対応の回答種別は読み飛ばすため、回答が欠けている場合がある
        let Some(answer) = response.answers.get("speaker") else {
            return Ok(None);
        };
        match answer.get("type").and_then(Value::as_str) {
            None => Err(self.invalid_response("answers.speaker.type")),
            Some("choice") => {
                let answer: ChoiceAnswer =
                    serde_json::from_value(answer.clone()).map_err(|e| self.invalid_response(e))?;
                Ok(criteria
                    .keys()
                    .position(|key| *key == answer.choice)
                    .map(|index| SpeakerGuess::with_confidence(index as i64, answer.confidence)))
            }
            Some(_) => Ok(None),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::character::Gender;

    #[test]
    fn criteria_numbers_duplicated_names() {
        let candidates = [
            Character::narrator(),
            Character::new("花子", "p").with_gender(Some(Gender::Female)),
            Character::new("花子", "q"),
        ];
        let criteria = JevSpeakerEstimator::build_criteria(&candidates);
        assert_eq!(criteria.keys().collect::<Vec<_>>(), ["ナレーター", "花子", "花子 (2)"]);
        assert_eq!(criteria["花子"], json!({"性別": "女性", "別名": [], "説明": "p"}));
        assert_eq!(criteria["花子 (2)"]["性別"], "不明");
    }

    #[test]
    fn state_includes_speakers_except_narrator() {
        let previous = [
            Sentence::spoken_by("地の文", Character::narrator()),
            Sentence::spoken_by("「あ」", Character::new("花子", "p")),
        ];
        let state = JevSpeakerEstimator::build_state(&previous, &Sentence::new("「い」"), &[Sentence::new("後")]);
        assert_eq!(
            state,
            json!({
                "今までの内容": [{"セリフ": "地の文"}, {"話者": "花子", "セリフ": "「あ」"}],
                "推測したいセリフの内容": "「い」",
                "後の内容": ["後"],
            })
        );
    }

    #[test]
    fn input_too_long_detection() {
        assert!(is_input_too_long(413, "too large"));
        assert!(is_input_too_long(400, "bad {\"error\": \"context length exceeded\"}"));
        assert!(!is_input_too_long(400, "bad {\"error\": \"invalid\"}"));
        assert!(!is_input_too_long(500, "too long"));
    }

    #[test]
    fn extracts_error_message() {
        assert_eq!(extract_message(&json!({"error": {"message": "m"}})).as_deref(), Some("m"));
        assert_eq!(
            extract_message(&json!({"detail": [{"loc": ["body", "state"], "msg": "required"}]})).as_deref(),
            Some("state: required")
        );
        assert_eq!(extract_message(&json!({"other": 1})), None);
    }
}
