use std::collections::HashSet;
use std::rc::Rc;

use crate::application::cancel::CancelFlag;
use crate::application::narrate_novels::NarrationError;
use crate::application::ports::{AudioWriter, Presenter, SpeechSynthesizer, SynthesisResult, TtsError};
use crate::domain::novel::{Novel, Sentence};
use crate::domain::text::{is_readable, split_for_tts};
use crate::domain::voice_assigner::VoiceAssigner;

/// 入力起因の失敗がこの回数だけ続いた場合は、文ではなくサーバー側の問題とみなして中断する
pub const MAX_CONSECUTIVE_FAILURES: usize = 5;

/// 文ごとに話者の音声で合成する。音声の割り当てと失敗の状況はファイルをまたいで引き継ぐ
pub struct SentenceSynthesizer {
    synthesizer: Box<dyn SpeechSynthesizer>,
    assigner: VoiceAssigner,
    presenter: Rc<dyn Presenter>,
    max_chars: usize,
    pub max_consecutive_failures: usize,
    /// 中止を求められたら次の文に進まない
    pub cancel: CancelFlag,
    // 一度でも合成に成功した音声と、入力起因の失敗が続いている回数
    succeeded_voices: HashSet<String>,
    consecutive_failures: usize,
}

impl SentenceSynthesizer {
    pub fn new(
        synthesizer: Box<dyn SpeechSynthesizer>,
        assigner: VoiceAssigner,
        presenter: Rc<dyn Presenter>,
        max_chars: usize,
    ) -> Self {
        Self {
            synthesizer,
            assigner,
            presenter,
            max_chars,
            max_consecutive_failures: MAX_CONSECUTIVE_FAILURES,
            cancel: CancelFlag::new(),
            succeeded_voices: HashSet::new(),
            consecutive_failures: 0,
        }
    }

    pub fn assigner(&self) -> &VoiceAssigner {
        &self.assigner
    }

    /// 文の内容による失敗は記録して続行し、それ以外の失敗は SynthesisAborted を返す
    pub fn synthesize(&mut self, sentences: &[Sentence]) -> Result<SynthesisResult, NarrationError> {
        let mut result = SynthesisResult::default();
        for (index, sentence) in sentences.iter().enumerate() {
            if self.cancel.is_cancelled() {
                return Err(NarrationError::Cancelled);
            }
            self.synthesize_sentence(sentence, &mut result)?;
            self.presenter.synthesis_progressed(index + 1, sentences.len());
        }
        Ok(result)
    }

    /// 1行を合成して result に加える。文の内容による失敗は result に記録する
    fn synthesize_sentence(&mut self, sentence: &Sentence, result: &mut SynthesisResult) -> Result<(), NarrationError> {
        if sentence.text.is_empty() {
            return Ok(());
        }
        if !is_readable(&sentence.text) {
            result.skipped += 1;
            return Ok(());
        }

        let voice = self.assigner.assign(sentence.speaker.as_ref());
        // 行の途中で発話が切れないよう、分割した塊が全て合成できた行だけを使う
        let mut segments = Vec::new();
        let mut failure = None;
        for chunk in split_for_tts(&sentence.text, self.max_chars) {
            match self.synthesizer.synthesize(&chunk, &voice) {
                Ok(audio) => {
                    segments.push(audio);
                    self.succeeded_voices.insert(voice.clone());
                }
                Err(TtsError::Input(e)) => {
                    failure = Some(e);
                    break;
                }
                Err(TtsError::Fatal(e)) => return Err(NarrationError::SynthesisAborted(e)),
            }
        }
        if let Some(error) = failure {
            self.record_failure(&voice, &sentence.text, &error)?;
            result.failures.push((sentence.text.clone(), error));
            return Ok(());
        }

        result.segments.extend(segments);
        self.consecutive_failures = 0;
        Ok(())
    }

    fn record_failure(&mut self, voice: &str, text: &str, error: &str) -> Result<(), NarrationError> {
        // 400 などはモデル名や音声名の誤りでも返るため、一度も成功していない音声での失敗は設定の問題とみなす
        if !self.succeeded_voices.contains(voice) {
            return Err(NarrationError::SynthesisAborted(format!(
                "音声 {voice} での最初の合成に失敗しました。モデル名や音声名を確認してください: {error}"
            )));
        }
        self.presenter.sentence_synthesis_failed(voice, text, error);
        self.consecutive_failures += 1;
        if self.consecutive_failures >= self.max_consecutive_failures {
            return Err(NarrationError::SynthesisAborted(format!(
                "音声合成が{}回連続で失敗しました: {error}",
                self.consecutive_failures
            )));
        }
        Ok(())
    }
}

/// 小説を音声合成して書き出す
pub struct AudioExporter {
    synthesizer: SentenceSynthesizer,
    writer: Box<dyn AudioWriter>,
    presenter: Rc<dyn Presenter>,
}

impl AudioExporter {
    pub fn new(synthesizer: SentenceSynthesizer, writer: Box<dyn AudioWriter>, presenter: Rc<dyn Presenter>) -> Self {
        Self { synthesizer, writer, presenter }
    }

    pub fn synthesizer(&self) -> &SentenceSynthesizer {
        &self.synthesizer
    }

    pub fn synthesizer_mut(&mut self) -> &mut SentenceSynthesizer {
        &mut self.synthesizer
    }

    /// 合成に失敗した文があれば true を返す。続けても失敗する場合は SynthesisAborted を返す
    pub fn export(&mut self, source: &str, novel: &Novel) -> Result<bool, NarrationError> {
        let result = self.synthesizer.synthesize(&novel.sentences)?;
        // 書き出せない出力先などは続けても失敗するため中断する
        let location = if result.segments.is_empty() {
            // 以前の実行結果を今回の結果と取り違えないよう、古い音声は消しておく
            self.writer.remove(source).map_err(NarrationError::SynthesisAborted)?;
            None
        } else {
            Some(self.writer.write(source, &result.segments).map_err(NarrationError::SynthesisAborted)?)
        };

        self.presenter.audio_exported(location.as_deref(), &result);
        Ok(!result.failures.is_empty())
    }
}
