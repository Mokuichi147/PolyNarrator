//! ユースケースが外部に求める機能。実装は adapters にあり、ユースケースはこのインターフェースだけに依存する

use indexmap::IndexMap;
use thiserror::Error;

use crate::domain::audio::Audio;
use crate::domain::character::Character;
use crate::domain::novel::{Novel, Sentence};

// --- 登場人物の抽出 ---

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum ExtractionError {
    /// 登場人物を抽出できなかった。以前の登場人物一覧のまま続行してよい
    #[error("{0}")]
    Failed(String),
    /// 接続できないなど、続けても失敗する
    #[error("{0}")]
    Fatal(String),
}

pub trait CharacterExtractor {
    /// 小説から登場人物を抽出する。known は以前のファイルまでの登場人物一覧
    fn extract_characters(&mut self, novel: &Novel, known: &[Character]) -> Result<Vec<Character>, ExtractionError>;
}

// --- 話者の推測 ---

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum EstimationError {
    /// 入力が長すぎて推測できなかった。文脈を減らせば推測できる可能性がある
    #[error("{0}")]
    ContextTooLong(String),
    /// この文の話者を推測できなかった。ナレーターとして続行してよい
    #[error("{0}")]
    Failed(String),
    /// 接続できないなど、続けても失敗する
    #[error("{0}")]
    Fatal(String),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpeakerGuess {
    /// 話者の候補一覧での位置。範囲外の値はモデルの誤りとして扱う
    pub index: i64,
    /// 推測の確信度。返さない推測器では None
    pub confidence: Option<f64>,
}

impl SpeakerGuess {
    pub fn new(index: i64) -> Self {
        Self { index, confidence: None }
    }

    pub fn with_confidence(index: i64, confidence: f64) -> Self {
        Self { index, confidence: Some(confidence) }
    }
}

pub trait SpeakerEstimator {
    /// target の話者を candidates から選ぶ。previous と following は前後の文脈。
    ///
    /// 回答が得られなかった場合は None を返し、入力が長すぎる場合は ContextTooLong を返す
    fn estimate_speaker(
        &mut self,
        candidates: &[Character],
        previous: &[Sentence],
        target: &Sentence,
        following: &[Sentence],
    ) -> Result<Option<SpeakerGuess>, EstimationError>;
}

// --- 音声合成 ---

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum TtsError {
    /// 文の内容が原因で合成できなかった。その文だけを読み飛ばして続行してよい
    #[error("{0}")]
    Input(String),
    /// 認証・接続・サーバー内部のエラーや、扱えない音声など。続けても失敗する
    #[error("{0}")]
    Fatal(String),
}

pub trait SpeechSynthesizer {
    /// サーバーから取得できる音声の一覧。取得できない場合は空
    fn list_voices(&self) -> Vec<String>;

    /// 指定された音声のうち、サーバーで使えないと分かっているもの
    fn unknown_voices(&self, _voices: &[String]) -> Vec<String> {
        Vec::new()
    }

    /// 音声合成する。文の内容が原因で失敗した場合は TtsError::Input を返す
    fn synthesize(&mut self, text: &str, voice: &str) -> Result<Audio, TtsError>;
}

pub trait AudioWriter {
    /// 入力 source に対応する出力先へ音声を順に連結して書き出し、出力先を返す
    fn write(&mut self, source: &str, segments: &[Audio]) -> Result<String, String>;

    /// 入力 source に対応する以前の出力を消す
    fn remove(&mut self, source: &str) -> Result<(), String>;
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct SynthesisResult {
    pub segments: Vec<Audio>,
    /// 合成に失敗した行と理由
    pub failures: Vec<(String, String)>,
    /// 記号だけで読み上げる文字がないため送信しなかった行の数(空行は含まない)
    pub skipped: usize,
}

// --- 進捗の表示 ---

/// 処理の経過を利用者に伝える。既定では何も表示しない
#[allow(unused_variables)]
pub trait Presenter {
    fn chapter_started(&self, index: usize, source: &str, characters: &[Character]) {}

    fn character_extraction_failed(&self, error: &str) {}

    /// セリフではないためナレーターを割り当てた
    fn narration_assigned(&self, sentence: &Sentence) {}

    fn speaker_estimated(&self, sentence: &Sentence, confidence: Option<f64>) {}

    /// 推測できなかったためナレーターを割り当てた
    fn speaker_estimation_failed(&self, sentence: &Sentence, error: Option<&str>) {}

    /// 入力長超過のため、文脈を直前 previous_count 文に縮小して再試行する
    fn context_shrunk(&self, previous_count: usize) {}

    fn sentence_synthesis_failed(&self, voice: &str, text: &str, error: &str) {}

    /// location は書き出した出力先。書き出さなかった場合は None
    fn audio_exported(&self, location: Option<&str>, result: &SynthesisResult) {}

    /// voices は名前ごとの音声。音声合成しない場合は None
    fn finished(&self, characters: &[Character], voices: Option<&IndexMap<String, String>>) {}
}

/// 何も表示しない Presenter
pub struct SilentPresenter;

impl Presenter for SilentPresenter {}
