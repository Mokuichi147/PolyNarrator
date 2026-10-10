"""ユースケースが外部に求める機能。実装は adapters にあり、ユースケースはこのインターフェースだけに依存する"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

from polynarrator.domain.audio import Audio
from polynarrator.domain.character import Character
from polynarrator.domain.novel import Novel, Sentence


# --- 登場人物の抽出 ---

class CharacterExtractionError(Exception):
    """登場人物を抽出できなかった。以前の登場人物一覧のまま続行してよい"""


class CharacterExtractor(ABC):
    @abstractmethod
    def extract_characters(self, novel: Novel, known: List[Character]) -> List[Character]:
        """小説から登場人物を抽出する。known は以前のファイルまでの登場人物一覧"""


# --- 話者の推測 ---

class ContextTooLongError(Exception):
    """入力が長すぎて推測できなかった。文脈を減らせば推測できる可能性がある"""


class SpeakerEstimationError(Exception):
    """この文の話者を推測できなかった。ナレーターとして続行してよい"""


@dataclass
class SpeakerGuess:
    # 話者の候補一覧での位置
    index: int
    # 推測の確信度。返さない推測器では None
    confidence: Optional[float] = None


class SpeakerEstimator(ABC):
    @abstractmethod
    def estimate_speaker(
        self,
        candidates: List[Character],
        previous: List[Sentence],
        target: Sentence,
        following: List[Sentence],
    ) -> Optional[SpeakerGuess]:
        """target の話者を candidates から選ぶ。previous と following は前後の文脈。

        回答が得られなかった場合は None を返し、入力が長すぎる場合は ContextTooLongError を送出する
        """


# --- 音声合成 ---

class TtsInputError(Exception):
    """文の内容が原因で合成できなかった。その文だけを読み飛ばして続行してよい"""


class SpeechSynthesizer(ABC):
    @abstractmethod
    def list_voices(self) -> List[str]:
        """サーバーから取得できる音声の一覧。取得できない場合は空"""

    def unknown_voices(self, voices: Sequence[str]) -> List[str]:
        """指定された音声のうち、サーバーで使えないと分かっているもの"""
        return []

    @abstractmethod
    def synthesize(self, text: str, voice: str) -> Audio:
        """音声合成する。文の内容が原因で失敗した場合は TtsInputError を送出する"""


class AudioWriter(ABC):
    @abstractmethod
    def write(self, source: str, segments: List[Audio]) -> str:
        """入力 source に対応する出力先へ音声を順に連結して書き出し、出力先を返す"""

    @abstractmethod
    def remove(self, source: str) -> None:
        """入力 source に対応する以前の出力を消す"""


@dataclass
class SynthesisResult:
    segments: List[Audio] = field(default_factory = list)
    # 合成に失敗した行と理由
    failures: List[tuple[str, str]] = field(default_factory = list)
    # 記号だけで読み上げる文字がないため送信しなかった行の数(空行は含まない)
    skipped: int = 0


# --- 進捗の表示 ---

class Presenter:
    """処理の経過を利用者に伝える。既定では何も表示しない"""

    def chapter_started(self, index: int, source: str, characters: List[Character]) -> None:
        pass

    def character_extraction_failed(self, error: Exception) -> None:
        pass

    def narration_assigned(self, sentence: Sentence) -> None:
        """セリフではないためナレーターを割り当てた"""

    def speaker_estimated(self, sentence: Sentence, confidence: Optional[float]) -> None:
        pass

    def speaker_estimation_failed(self, sentence: Sentence, error: Optional[Exception]) -> None:
        """推測できなかったためナレーターを割り当てた"""

    def context_shrunk(self, previous_count: int) -> None:
        """入力長超過のため、文脈を直前 previous_count 文に縮小して再試行する"""

    def sentence_synthesis_failed(self, voice: str, text: str, error: Exception) -> None:
        pass

    def audio_exported(self, location: Optional[str], result: SynthesisResult) -> None:
        """location は書き出した出力先。書き出さなかった場合は None"""

    def finished(self, characters: List[Character], voices: Optional[Dict[str, str]]) -> None:
        """voices は名前ごとの音声。音声合成しない場合は None"""
