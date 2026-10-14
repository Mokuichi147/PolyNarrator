use clap::{Parser, ValueEnum};

/// 無音はメモリ上に作るため、極端な値で大量のメモリを確保しないよう上限を設ける
pub const MAX_PAUSE: f64 = 60.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum SpeakerBackend {
    Llm,
    Jev,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, ValueEnum)]
pub enum TtsBackend {
    Openai,
    Wyoming,
}

fn parse_float(value: &str) -> Result<f64, String> {
    value.trim().parse::<f64>().map_err(|_| format!("数値ではありません: {value}"))
}

fn non_negative_float(value: &str) -> Result<f64, String> {
    let number = parse_float(value)?;
    if !number.is_finite() || number < 0.0 {
        return Err("0以上の数値を指定してください".into());
    }
    Ok(number)
}

fn positive_float(value: &str) -> Result<f64, String> {
    let number = parse_float(value)?;
    if !number.is_finite() || number <= 0.0 {
        return Err("0より大きい数値を指定してください".into());
    }
    Ok(number)
}

fn pause_seconds(value: &str) -> Result<f64, String> {
    let number = non_negative_float(value)?;
    if number > MAX_PAUSE {
        return Err(format!("{MAX_PAUSE}秒以下を指定してください"));
    }
    Ok(number)
}

fn non_negative_int(value: &str) -> Result<usize, String> {
    let number: i64 = value.trim().parse().map_err(|_| format!("整数ではありません: {value}"))?;
    usize::try_from(number).map_err(|_| "0以上の整数を指定してください".into())
}

/// 小説の登場人物を推測し、登場人物ごとに声を変えて読み上げる
#[derive(Debug, Clone, Parser)]
#[command(version)]
pub struct Args {
    /// LLMサーバーのホスト名
    #[arg(long, default_value = "localhost")]
    pub host: String,
    /// LLMサーバーのポート番号
    #[arg(long, default_value = "1234")]
    pub port: String,
    /// LLMのモデル名
    #[arg(long, default_value = "qwen3.8-flash-next-iq3_s")]
    pub model: String,
    /// 話者推測に使うバックエンド
    #[arg(long, value_enum, default_value = "llm")]
    pub speaker_backend: SpeakerBackend,
    /// Jev互換モデル名(未指定時は TYPESAFE_DEFAULT_MODEL または jev-latest)
    #[arg(long)]
    pub jev_model: Option<String>,
    /// Jev互換APIのURL(未指定時は TYPESAFE_BASE_URL または https://api.typesafe.ai)
    #[arg(long)]
    pub jev_base_url: Option<String>,
    /// Jev互換APIのキー(未指定時は TYPESAFE_API_KEY)
    #[arg(long)]
    pub jev_api_key: Option<String>,
    /// 音声ファイルの出力先ディレクトリ(指定時のみ音声合成を行う)
    #[arg(long)]
    pub tts_output: Option<String>,
    /// 音声合成に使うプロトコル
    #[arg(long, value_enum, default_value = "openai")]
    pub tts_backend: TtsBackend,
    /// TTSサーバーのURL(未指定時は openai: http://localhost:8880/v1, wyoming: tcp://localhost:10200)
    #[arg(long)]
    pub tts_url: Option<String>,
    /// OpenAI互換APIで使うモデル名
    #[arg(long, default_value = "tts-1")]
    pub tts_model: String,
    /// OpenAI互換APIのキー(未指定時は OPENAI_API_KEY)
    #[arg(long)]
    pub tts_api_key: Option<String>,
    /// Wyomingで音声一覧を絞り込む言語(空文字で絞り込まない)
    #[arg(long, default_value = "ja")]
    pub tts_language: String,
    /// 使用する音声のカンマ区切り一覧(Wyomingでは未指定時にサーバーの一覧を使う)
    #[arg(long)]
    pub tts_voices: Option<String>,
    /// ナレーターの音声(未指定時は音声一覧の先頭)
    #[arg(long)]
    pub tts_narrator_voice: Option<String>,
    /// 男性の登場人物に優先して使う音声のカンマ区切り一覧
    #[arg(long)]
    pub tts_male_voices: Option<String>,
    /// 女性の登場人物に優先して使う音声のカンマ区切り一覧
    #[arg(long)]
    pub tts_female_voices: Option<String>,
    /// 行と行(長い行を分割した場合は分割した塊)の間に入れる無音の秒数
    #[arg(long, default_value = "0.3", value_parser = pause_seconds, allow_negative_numbers = true)]
    pub tts_pause: f64,
    /// 1回の合成に送る最大文字数。超える行は文末・読点・文字数の順で分割する(0で分割しない)
    #[arg(long, default_value = "200", value_parser = non_negative_int, allow_negative_numbers = true)]
    pub tts_max_chars: usize,
    /// TTSサーバーへのリクエストタイムアウト秒数
    #[arg(long, default_value = "60", value_parser = positive_float, allow_negative_numbers = true)]
    pub tts_timeout: f64,
    /// 章ごとに分けた UTF-8 の .txt ファイルを置いたフォルダ
    pub folder: String,
}
