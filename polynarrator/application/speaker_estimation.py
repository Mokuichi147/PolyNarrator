from typing import List, Optional, Tuple

from polynarrator.application.ports import (
    ContextTooLongError,
    Presenter,
    SpeakerEstimationError,
    SpeakerEstimator,
    SpeakerGuess,
)
from polynarrator.domain.character import Character, narrator
from polynarrator.domain.novel import Novel, Sentence


class SpeakerEstimation:
    """小説の各文に話者を割り当てる。推測できなかった文はナレーターが読む"""

    def __init__(
        self,
        estimator: SpeakerEstimator,
        presenter: Presenter,
        previous_count: int = 100,
        following_count: int = 0,
        dialogue_only: bool = True,
    ):
        self.estimator = estimator
        self.presenter = presenter
        # 文脈として渡す前後の文の数
        self.previous_count = previous_count
        self.following_count = following_count
        # 「」で囲まれたセリフだけを推測し、それ以外はナレーターにする
        self.dialogue_only = dialogue_only

    def run(self, novel: Novel) -> None:
        candidates = [narrator(), *novel.characters]
        for i, sentence in enumerate(novel.sentences):
            if self.dialogue_only and not sentence.is_dialogue:
                sentence.speaker = candidates[0]
                self.presenter.narration_assigned(sentence)
                continue

            following = novel.sentences[i + 1:][:self.following_count]
            guess, error = self._estimate(candidates, novel.sentences[:i], sentence, following)
            if guess is not None and 0 <= guess.index < len(candidates):
                sentence.speaker = candidates[guess.index]
                self.presenter.speaker_estimated(sentence, guess.confidence)
            else:
                sentence.speaker = candidates[0]
                self.presenter.speaker_estimation_failed(sentence, error)

    def _estimate(
        self,
        candidates: List[Character],
        preceding: List[Sentence],
        sentence: Sentence,
        following: List[Sentence],
    ) -> Tuple[Optional[SpeakerGuess], Optional[Exception]]:
        count = self.previous_count
        while True:
            previous = preceding[-count:] if count > 0 else []
            try:
                return self.estimator.estimate_speaker(candidates, previous, sentence, following), None
            except ContextTooLongError as e:
                # 入力長超過のときだけ文脈を縮小して再試行する
                if count <= 0:
                    return None, e
                count //= 2
                self.presenter.context_shrunk(count)
            except SpeakerEstimationError as e:
                return None, e
