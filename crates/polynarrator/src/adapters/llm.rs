use std::sync::LazyLock;
use std::time::Duration;

use regex::Regex;
use serde::de::{self, DeserializeOwned};
use serde::{Deserialize, Deserializer};
use serde_json::{json, Value};

use crate::adapters::http::{openai_error_message, send_with_retry, HttpFailure, OPENAI_RETRY};
use crate::adapters::py_repr::{gender_repr, list_repr};
use crate::application::ports::{CharacterExtractor, EstimationError, ExtractionError, SpeakerEstimator, SpeakerGuess};
use crate::domain::character::{Character, Gender};
use crate::domain::novel::{Novel, Sentence};

// 応答のスキーマはプロンプトにも含めるため、Python 版の pydantic が作るものと同じ内容・表記にする。
// 変えるとモデルへの入力が変わる

/// Narrators のスキーマを Python の dict として文字列にしたもの
const NARRATORS_SCHEMA_REPR: &str = "{'$defs': {'Gender': {'enum': ['男性', '女性', 'その他'], 'title': 'Gender', 'type': 'string'}, 'Narrator': {'description': '登場人物の情報。これらの情報を元に会話内容の発言者を推測する', 'properties': {'name': {'description': 'キャラクター名。正式名称であることが好ましい', 'title': 'Name', 'type': 'string'}, 'portrait': {'description': '性格や来歴、外見といったキャラクターの特徴', 'title': 'Portrait', 'type': 'string'}, 'aliases': {'default': [], 'description': 'ニックネームや別称など', 'items': {'type': 'string'}, 'title': 'Aliases', 'type': 'array'}, 'gender': {'anyOf': [{'$ref': '#/$defs/Gender'}, {'type': 'null'}], 'default': None, 'description': '性別'}}, 'required': ['name', 'portrait'], 'title': 'Narrator', 'type': 'object'}}, 'description': '登場人物のリスト', 'properties': {'narrators': {'items': {'$ref': '#/$defs/Narrator'}, 'title': 'Narrators', 'type': 'array'}}, 'required': ['narrators'], 'title': 'Narrators', 'type': 'object'}";
const NARRATORS_SCHEMA_JSON: &str = r##"{"$defs": {"Gender": {"enum": ["男性", "女性", "その他"], "title": "Gender", "type": "string"}, "Narrator": {"description": "登場人物の情報。これらの情報を元に会話内容の発言者を推測する", "properties": {"name": {"description": "キャラクター名。正式名称であることが好ましい", "title": "Name", "type": "string"}, "portrait": {"description": "性格や来歴、外見といったキャラクターの特徴", "title": "Portrait", "type": "string"}, "aliases": {"default": [], "description": "ニックネームや別称など", "items": {"type": "string"}, "title": "Aliases", "type": "array"}, "gender": {"anyOf": [{"$ref": "#/$defs/Gender"}, {"type": "null"}], "default": null, "description": "性別"}}, "required": ["name", "portrait"], "title": "Narrator", "type": "object"}}, "description": "登場人物のリスト", "properties": {"narrators": {"items": {"$ref": "#/$defs/Narrator"}, "title": "Narrators", "type": "array"}}, "required": ["narrators"], "title": "Narrators", "type": "object"}"##;

/// NarratorResponse のスキーマを Python の dict として文字列にしたもの
const NARRATOR_RESPONSE_SCHEMA_REPR: &str = "{'description': '登場人物一覧のインデックス', 'properties': {'narrator_index': {'title': 'Narrator Index', 'type': 'integer'}}, 'required': ['narrator_index'], 'title': 'NarratorResponse', 'type': 'object'}";
const NARRATOR_RESPONSE_SCHEMA_JSON: &str = r#"{"description": "登場人物一覧のインデックス", "properties": {"narrator_index": {"title": "Narrator Index", "type": "integer"}}, "required": ["narrator_index"], "title": "NarratorResponse", "type": "object"}"#;

fn deserialize_gender<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Option<Gender>, D::Error> {
    match Option::<String>::deserialize(deserializer)? {
        None => Ok(None),
        Some(label) => Gender::from_label(&label)
            .map(Some)
            .ok_or_else(|| de::Error::custom(format!("性別の値が不正です: {label}"))),
    }
}

/// 登場人物の情報。これらの情報を元に会話内容の発言者を推測する
#[derive(Debug, Deserialize)]
pub struct Narrator {
    pub name: String,
    pub portrait: String,
    #[serde(default)]
    pub aliases: Vec<String>,
    #[serde(default, deserialize_with = "deserialize_gender")]
    pub gender: Option<Gender>,
}

impl Narrator {
    pub fn to_character(&self) -> Character {
        Character::new(&self.name, &self.portrait).with_aliases(self.aliases.clone()).with_gender(self.gender)
    }
}

/// 登場人物のリスト
#[derive(Debug, Deserialize)]
pub struct Narrators {
    pub narrators: Vec<Narrator>,
}

/// 整数のほか、小数部のない数値や数字の文字列も整数として受け付ける(pydantic と同じ)
fn deserialize_lax_int<'de, D: Deserializer<'de>>(deserializer: D) -> Result<i64, D::Error> {
    match Value::deserialize(deserializer)? {
        Value::Number(n) => n
            .as_i64()
            .or_else(|| n.as_f64().filter(|f| f.fract() == 0.0 && f.abs() < 9.0e15).map(|f| f as i64))
            .ok_or_else(|| de::Error::custom(format!("整数ではありません: {n}"))),
        Value::String(s) => s.trim().parse().map_err(|_| de::Error::custom(format!("整数ではありません: {s}"))),
        other => Err(de::Error::custom(format!("整数ではありません: {other}"))),
    }
}

/// 登場人物一覧のインデックス
#[derive(Debug, Deserialize)]
pub struct NarratorResponse {
    #[serde(deserialize_with = "deserialize_lax_int")]
    pub narrator_index: i64,
}

static JSON_SINGLE_BLOCK: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?s)([\{\[].*[\}\]])").unwrap());

/// JSON の前後に余計な文章が付いた応答も受け付ける
pub fn parse_json<T: DeserializeOwned>(data: &str) -> Result<T, serde_json::Error> {
    serde_json::from_str(data).or_else(|error| match JSON_SINGLE_BLOCK.find(data) {
        Some(block) => serde_json::from_str(block.as_str()),
        None => Err(error),
    })
}

enum ChatError {
    ContextTooLong(String),
    Fatal(String),
}

/// OpenAI API 互換の LLM サーバーで登場人物の抽出と話者の推測を行う
pub struct OpenAiChatModel {
    agent: ureq::Agent,
    base_url: String,
    model: String,
}

impl OpenAiChatModel {
    pub fn new(host: &str, port: &str, model: &str) -> Self {
        // OpenAI の公式 SDK の既定と同じタイムアウト
        let agent = ureq::AgentBuilder::new()
            .timeout_connect(Duration::from_secs(5))
            .timeout_read(Duration::from_secs(600))
            .timeout_write(Duration::from_secs(600))
            .build();
        Self { agent, base_url: format!("http://{host}:{port}/v1"), model: model.to_string() }
    }

    fn chat(&self, messages: Value, schema_json: &str) -> Result<Option<String>, ChatError> {
        let schema: Value = serde_json::from_str(schema_json).expect("スキーマは正しい JSON");
        let name = schema.get("title").and_then(Value::as_str).unwrap_or("response").to_string();
        let body = json!({
            "messages": messages,
            "model": self.model,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": name, "schema": schema},
            },
        });
        let url = format!("{}/chat/completions", self.base_url);
        let response =
            send_with_retry(&OPENAI_RETRY, |_| self.agent.post(&url).set("Authorization", "Bearer lm-studio"), &body)
                .map_err(|failure| {
                let message = openai_error_message(&failure);
                match failure {
                    HttpFailure::Status { status: 400, .. } if message.to_lowercase().contains("context") => {
                        ChatError::ContextTooLong(message)
                    }
                    _ => ChatError::Fatal(message),
                }
            })?;
        let response: Value =
            response.into_json().map_err(|e| ChatError::Fatal(format!("LLMサーバーの応答を読めませんでした: {e}")))?;
        let message = response
            .get("choices")
            .and_then(|c| c.get(0))
            .and_then(|c| c.get("message"))
            .ok_or_else(|| ChatError::Fatal(format!("LLMサーバーの応答に choices がありません: {response}")))?;
        Ok(message.get("content").and_then(Value::as_str).map(str::to_string))
    }
}

fn response_text(data: &Option<String>) -> &str {
    data.as_deref().unwrap_or("None")
}

impl CharacterExtractor for OpenAiChatModel {
    fn extract_characters(&mut self, novel: &Novel, known: &[Character]) -> Result<Vec<Character>, ExtractionError> {
        let mut messages = vec![json!({
            "role": "system",
            "content": format!(
                "ユーザーから与えられた小説の内容から登場人物を全て抽出し、指定されたJsonフォーマットで返答してください。\
                 同一の人物や一覧内で命名の揺れがないこと。\
                 今までの登場人物一覧が与え得られた場合は、内容を適宜更新すること。\n\
                 レスポンスフォーマット:\n{NARRATORS_SCHEMA_REPR}"
            ),
        })];

        if !known.is_empty() {
            let list: Vec<String> = known.iter().map(|n| format!("- {} ({})", n.name, gender_repr(n.gender))).collect();
            messages.push(json!({"role": "user", "content": format!("今までの登場人物一覧:\n{}", list.join("\n"))}));
        }

        let texts: Vec<&str> = novel.sentences.iter().map(|s| s.text.as_str()).collect();
        messages.push(json!({"role": "user", "content": format!("\n小説の内容:\n{}", texts.join("\n"))}));

        let data = self.chat(Value::Array(messages), NARRATORS_SCHEMA_JSON).map_err(|e| match e {
            // 入力が長すぎる章は抽出できないため、以前の登場人物一覧のまま続行する
            ChatError::ContextTooLong(e) => ExtractionError::Failed(e),
            ChatError::Fatal(e) => ExtractionError::Fatal(e),
        })?;
        let parsed = match &data {
            Some(text) => parse_json::<Narrators>(text).map_err(|e| e.to_string()),
            None => Err("応答が空です".to_string()),
        };
        parsed
            .map(|n| n.narrators.iter().map(Narrator::to_character).collect())
            .map_err(|e| ExtractionError::Failed(format!("{e}\n応答: {}", response_text(&data))))
    }
}

impl SpeakerEstimator for OpenAiChatModel {
    fn estimate_speaker(
        &mut self,
        candidates: &[Character],
        previous: &[Sentence],
        target: &Sentence,
        following: &[Sentence],
    ) -> Result<Option<SpeakerGuess>, EstimationError> {
        let history: Vec<String> = previous
            .iter()
            .map(|s| match &s.speaker {
                Some(speaker) if !speaker.is_narrator() => format!("{}\t{}", speaker.name, s.text),
                _ => format!("\t{}", s.text),
            })
            .collect();
        let upcoming: Vec<&str> = following.iter().map(|s| s.text.as_str()).collect();
        // モデルへの入力を変えないよう、行頭の字下げも含めて Python 版と同じ文字列にする
        let indent = " ".repeat(20);
        let content = format!(
            "\n{indent}今までの内容:\n{indent}{}\n\n{indent}推測したいセリフの内容:\n{indent}{}\n\n{indent}後の内容:\n{indent}{}\n{indent}",
            history.join("\n"),
            target.text,
            upcoming.join("\n"),
        );
        let list: Vec<String> = candidates
            .iter()
            .enumerate()
            .map(|(index, c)| {
                format!(
                    "{index}. {} (性別:{}, 別名:{}, 説明:{})",
                    c.name,
                    gender_repr(c.gender),
                    list_repr(&c.aliases),
                    c.portrait
                )
            })
            .collect();

        let messages = json!([
            {
                "role": "system",
                "content": format!(
                    "会話の内容から指定されたセリフがどの登場人物による発言かを推測し、指定されたJsonフォーマットで返答してください。\
                     会話は推定したい文とその前後の内容が与えられます。\
                     誰のセリフとも考えられない場合はナレーターを指定してください。\n\
                     レスポンスフォーマット:\n{NARRATOR_RESPONSE_SCHEMA_REPR}"
                ),
            },
            {"role": "user", "content": format!("登場人物一覧:\n{}", list.join("\n"))},
            {"role": "user", "content": content},
        ]);

        let data = self.chat(messages, NARRATOR_RESPONSE_SCHEMA_JSON).map_err(|e| match e {
            ChatError::ContextTooLong(e) => EstimationError::ContextTooLong(e),
            ChatError::Fatal(e) => EstimationError::Fatal(e),
        })?;
        let parsed = match &data {
            Some(text) => parse_json::<NarratorResponse>(text).map_err(|e| e.to_string()),
            None => Err("応答が空です".to_string()),
        };
        parsed
            .map(|r| Some(SpeakerGuess::new(r.narrator_index)))
            .map_err(|e| EstimationError::Failed(format!("{e}\n応答: {}", response_text(&data))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn schema_repr_and_json_describe_same_schema() {
        let to_json = |repr: &str| repr.replace('\'', "\"").replace("None", "null");
        let parse = |text: &str| serde_json::from_str::<Value>(text).unwrap();
        assert_eq!(parse(&to_json(NARRATORS_SCHEMA_REPR)), parse(NARRATORS_SCHEMA_JSON));
        assert_eq!(parse(&to_json(NARRATOR_RESPONSE_SCHEMA_REPR)), parse(NARRATOR_RESPONSE_SCHEMA_JSON));
    }

    #[test]
    fn parse_json_accepts_surrounding_text() {
        assert_eq!(parse_json::<NarratorResponse>(r#"{"narrator_index": 2}"#).unwrap().narrator_index, 2);
        assert_eq!(parse_json::<NarratorResponse>(r#"回答: {"narrator_index": 1} です"#).unwrap().narrator_index, 1);
        assert_eq!(parse_json::<NarratorResponse>(r#"{"narrator_index": "3"}"#).unwrap().narrator_index, 3);
        assert_eq!(parse_json::<NarratorResponse>(r#"{"narrator_index": 4.0}"#).unwrap().narrator_index, 4);
        assert!(parse_json::<NarratorResponse>("わかりません").is_err());
        assert!(parse_json::<NarratorResponse>(r#"{"narrator_index": 1.5}"#).is_err());
    }

    #[test]
    fn schema_converts_to_character() {
        let data = r#"{"narrators": [{"name": "花子", "portrait": "p", "aliases": ["はな"], "gender": "女性"}, {"name": "太郎", "portrait": "q", "extra": 1}]}"#;
        let characters: Vec<Character> =
            parse_json::<Narrators>(data).unwrap().narrators.iter().map(Narrator::to_character).collect();
        assert_eq!(
            characters,
            vec![
                Character::new("花子", "p").with_aliases(["はな"]).with_gender(Some(Gender::Female)),
                Character::new("太郎", "q"),
            ]
        );
        assert!(
            parse_json::<Narrators>(r#"{"narrators": [{"name": "花子", "portrait": "p", "gender": "不明"}]}"#).is_err()
        );
        assert!(parse_json::<Narrators>(r#"{"narrators": [{"name": "花子"}]}"#).is_err());
    }
}
