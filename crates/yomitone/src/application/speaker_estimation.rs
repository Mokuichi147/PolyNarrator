use std::rc::Rc;

use crate::application::cancel::CancelFlag;
use crate::application::narrate_novels::NarrationError;
use crate::application::ports::{EstimationError, Presenter, SpeakerEstimator, SpeakerGuess};
use crate::domain::character::Character;
use crate::domain::novel::{Novel, Sentence};

/// 小説の各文に話者を割り当てる。推測できなかった文はナレーターが読む
pub struct SpeakerEstimation {
    estimator: Box<dyn SpeakerEstimator>,
    presenter: Rc<dyn Presenter>,
    /// 文脈として渡す前後の文の数
    pub previous_count: usize,
    pub following_count: usize,
    /// 「」で囲まれたセリフだけを推測し、それ以外はナレーターにする
    pub dialogue_only: bool,
    /// 中止を求められたら次の文に進まない
    pub cancel: CancelFlag,
}

impl SpeakerEstimation {
    pub fn new(estimator: Box<dyn SpeakerEstimator>, presenter: Rc<dyn Presenter>) -> Self {
        Self {
            estimator,
            presenter,
            previous_count: 100,
            following_count: 0,
            dialogue_only: true,
            cancel: CancelFlag::new(),
        }
    }

    /// 推測できない文はナレーターにして続行し、続けても失敗するエラーだけを返す
    pub fn run(&mut self, novel: &mut Novel) -> Result<(), NarrationError> {
        let candidates: Vec<Character> =
            std::iter::once(Character::narrator()).chain(novel.characters.iter().cloned()).collect();
        for i in 0..novel.sentences.len() {
            if self.cancel.is_cancelled() {
                return Err(NarrationError::Cancelled);
            }
            if self.dialogue_only && !novel.sentences[i].is_dialogue() {
                novel.sentences[i].speaker = Some(candidates[0].clone());
                self.presenter.narration_assigned(&novel.sentences[i]);
                continue;
            }

            let end = (i + 1 + self.following_count).min(novel.sentences.len());
            let (preceding, rest) = novel.sentences.split_at(i);
            let (guess, error) = self.estimate(&candidates, preceding, &rest[0], &rest[1..end - i])?;
            let sentence = &mut novel.sentences[i];
            match guess.and_then(|g| {
                usize::try_from(g.index).ok().filter(|&index| index < candidates.len()).map(|index| (index, g))
            }) {
                Some((index, guess)) => {
                    sentence.speaker = Some(candidates[index].clone());
                    self.presenter.speaker_estimated(sentence, guess.confidence);
                }
                None => {
                    sentence.speaker = Some(candidates[0].clone());
                    self.presenter.speaker_estimation_failed(sentence, error.as_deref());
                }
            }
        }
        Ok(())
    }

    fn estimate(
        &mut self,
        candidates: &[Character],
        preceding: &[Sentence],
        sentence: &Sentence,
        following: &[Sentence],
    ) -> Result<(Option<SpeakerGuess>, Option<String>), NarrationError> {
        let mut count = self.previous_count;
        loop {
            let previous = &preceding[preceding.len().saturating_sub(count)..];
            match self.estimator.estimate_speaker(candidates, previous, sentence, following) {
                Ok(guess) => return Ok((guess, None)),
                // 入力長超過のときだけ文脈を縮小して再試行する
                Err(EstimationError::ContextTooLong(e)) => {
                    if count == 0 {
                        return Ok((None, Some(e)));
                    }
                    count /= 2;
                    self.presenter.context_shrunk(count);
                }
                Err(EstimationError::Failed(e)) => return Ok((None, Some(e))),
                Err(EstimationError::Fatal(e)) => return Err(NarrationError::Fatal(e)),
            }
        }
    }
}
