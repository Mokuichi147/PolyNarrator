//! 設定した接続先に正しくつながるかを確かめる。読み上げを始める前に設定を見直すために使う

use crate::adapters::jev::JevSpeakerEstimator;
use crate::adapters::llm::OpenAiChatModel;
use crate::application::ports::TtsError;
use crate::setup::{self, JevOptions, LlmOptions, SetupError, TtsBackend, TtsOptions};

/// 確認に合成する文
pub const CHECK_TEXT: &str = "これは接続の確認です。";

/// 確認の結果
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckOutcome {
    /// 設定どおりに使える
    Ok(String),
    /// つながったが、設定に気になる点がある(モデルが一覧にないなど)
    Warning(String),
    /// 使えない
    Failed(String),
}

impl CheckOutcome {
    pub fn message(&self) -> &str {
        match self {
            CheckOutcome::Ok(m) | CheckOutcome::Warning(m) | CheckOutcome::Failed(m) => m,
        }
    }
}

/// モデルの一覧に、使うモデルがあるかを伝える
fn check_model(model: &str, models: &[String]) -> CheckOutcome {
    if models.iter().any(|m| m == model) {
        CheckOutcome::Ok(format!("接続できました。モデル {model} を使えます"))
    } else if models.is_empty() {
        CheckOutcome::Warning(format!(
            "接続できましたが、サーバーのモデル一覧が空のため、モデル {model} があるか確かめられません"
        ))
    } else {
        CheckOutcome::Warning(format!(
            "接続できましたが、モデル {model} がサーバーのモデル一覧にありません。一覧にあるモデル: {}",
            models.join(", ")
        ))
    }
}

/// LLM サーバーにつながり、モデルが一覧にあるかを確かめる
pub fn check_llm(options: &LlmOptions) -> CheckOutcome {
    let model = OpenAiChatModel::new(&options.host, &options.port, &options.model);
    match model.list_models() {
        Ok(models) => check_model(&options.model, &models),
        Err(error) => CheckOutcome::Failed(format!("LLMサーバー({})に接続できませんでした: {error}", model.base_url())),
    }
}

/// Jev 互換 API にキーで認証でき、モデルが一覧にあるかを確かめる
pub fn check_jev(options: &JevOptions) -> CheckOutcome {
    let estimator =
        match JevSpeakerEstimator::new(options.api_key.clone(), options.model.clone(), options.base_url.clone()) {
            Ok(estimator) => estimator,
            Err(error) => return CheckOutcome::Failed(error),
        };
    match estimator.list_models() {
        Ok(models) => match check_model(estimator.model(), &models) {
            // 別名(jev-latest など)は一覧に無くても使える場合がある
            CheckOutcome::Warning(message) => {
                CheckOutcome::Warning(format!("{message}(別名の場合はそのまま使えることがあります)"))
            }
            outcome => outcome,
        },
        Err(error) => CheckOutcome::Failed(format!("Jev互換APIを利用できませんでした: {error}")),
    }
}

/// TTS サーバーにつながり、設定した音声で合成できるかを実際に短い文を合成して確かめる。
/// describe は設定の問題を利用者に伝えるメッセージにする
pub fn check_tts(options: &TtsOptions, describe: impl Fn(&SetupError) -> String) -> CheckOutcome {
    let (mut client, assigner) = match setup::create_tts(options) {
        Ok(tts) => tts,
        Err(error) => return CheckOutcome::Failed(describe(&error)),
    };
    // OpenAI 互換 API は音声名をサーバーで確かめられないため、指定された全ての音声で合成してみる。
    // Wyoming は音声名をサーバーの一覧と照合済みのため、ナレーターの音声だけで合成してみる
    let voices: Vec<String> = match options.backend {
        TtsBackend::OpenAi => assigner.voices().to_vec(),
        TtsBackend::Wyoming => vec![assigner.narrator_voice().to_string()],
    };
    for voice in &voices {
        match client.synthesize(CHECK_TEXT, voice) {
            Ok(_) => {}
            // 文の内容による失敗として返るのは、音声名やモデル名の誤りが多い
            Err(TtsError::Input(e)) => {
                return CheckOutcome::Failed(format!(
                    "音声 {voice} で合成できませんでした。音声名やモデル名を確認してください: {e}"
                ))
            }
            Err(TtsError::Fatal(e)) => {
                return CheckOutcome::Failed(format!("TTSサーバー({})で合成できませんでした: {e}", options.url()))
            }
        }
    }
    let mut message = format!("接続できました。音声 {} で合成できることを確かめました", voices.join(", "));
    if options.backend == TtsBackend::Wyoming {
        message.push_str(&format!("。使用する音声: {}", assigner.voices().join(", ")));
    }
    CheckOutcome::Ok(message)
}
