use std::rc::Rc;

use thiserror::Error;

use crate::application::cancel::CancelFlag;
use crate::application::ports::{CharacterExtractor, ExtractionError, Presenter};
use crate::application::speaker_estimation::SpeakerEstimation;
use crate::application::speech_synthesis::AudioExporter;
use crate::domain::character::Character;
use crate::domain::novel::Novel;

/// 処理を続けられなくなった理由
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum NarrationError {
    /// LLM サーバーに接続できないなど、話者の推測を続けられない
    #[error("{0}")]
    Fatal(String),
    /// 続けても失敗するため音声合成を中断した
    #[error("音声合成を中断しました: {0}")]
    SynthesisAborted(String),
    /// 利用者が中止した
    #[error("中止しました")]
    Cancelled,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Chapter {
    /// 入力元を表す名前。表示と出力先の決定に使う
    pub source: String,
    pub novel: Novel,
}

impl Chapter {
    pub fn new(source: impl Into<String>, novel: Novel) -> Self {
        Self { source: source.into(), novel }
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct NarrationReport {
    pub characters: Vec<Character>,
    /// 合成に失敗した文がある章の入力元
    pub failed_sources: Vec<String>,
}

/// 章ごとに登場人物を抽出して話者を推測し、指定があれば音声を書き出す。登場人物の一覧は後の章に引き継ぐ
pub struct NarrateNovels {
    extractor: Box<dyn CharacterExtractor>,
    speaker_estimation: SpeakerEstimation,
    presenter: Rc<dyn Presenter>,
    exporter: Option<AudioExporter>,
    cancel: CancelFlag,
}

impl NarrateNovels {
    pub fn new(
        extractor: Box<dyn CharacterExtractor>,
        speaker_estimation: SpeakerEstimation,
        presenter: Rc<dyn Presenter>,
        exporter: Option<AudioExporter>,
    ) -> Self {
        Self { extractor, speaker_estimation, presenter, exporter, cancel: CancelFlag::new() }
    }

    /// cancel で中止を求められたら、文や章の区切りで止まって Cancelled を返す。
    /// 合成の途中で止まった章の音声は書き出さない
    pub fn with_cancel(mut self, cancel: CancelFlag) -> Self {
        self.speaker_estimation.cancel = cancel.clone();
        if let Some(exporter) = self.exporter.as_mut() {
            exporter.synthesizer_mut().cancel = cancel.clone();
        }
        self.cancel = cancel;
        self
    }

    pub fn exporter(&self) -> Option<&AudioExporter> {
        self.exporter.as_ref()
    }

    pub fn run(&mut self, chapters: &mut [Chapter]) -> Result<NarrationReport, NarrationError> {
        let mut report = NarrationReport::default();
        for (index, chapter) in chapters.iter_mut().enumerate() {
            if self.cancel.is_cancelled() {
                return Err(NarrationError::Cancelled);
            }
            let extracted = match self.extractor.extract_characters(&chapter.novel, &report.characters) {
                Ok(extracted) => extracted,
                Err(ExtractionError::Failed(e)) => {
                    self.presenter.character_extraction_failed(&e);
                    Vec::new()
                }
                Err(ExtractionError::Fatal(e)) => return Err(NarrationError::Fatal(e)),
            };
            if !extracted.is_empty() {
                report.characters = extracted;
            }
            self.presenter.chapter_started(index, &chapter.source, &report.characters);

            chapter.novel.characters = report.characters.clone();
            self.speaker_estimation.run(&mut chapter.novel)?;

            if let Some(exporter) = self.exporter.as_mut() {
                if exporter.export(&chapter.source, &chapter.novel)? {
                    report.failed_sources.push(chapter.source.clone());
                }
            }
        }

        let voices = self.exporter.as_ref().map(|e| e.synthesizer().assigner().assigned());
        self.presenter.finished(&report.characters, voices);
        Ok(report)
    }
}
