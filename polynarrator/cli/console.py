from typing import Dict, List, Optional

from polynarrator.application.ports import Presenter, SynthesisResult
from polynarrator.domain.character import Character
from polynarrator.domain.novel import Sentence


def _speaker_name(sentence: Sentence) -> str:
    return sentence.speaker.name if sentence.speaker is not None else ""


def _character_line(character: Character) -> str:
    return f"{character.name} ({character.gender}) {character.aliases}"


class ConsolePresenter(Presenter):
    """処理の経過を標準出力に表示する"""

    def chapter_started(self, index: int, source: str, characters: List[Character]) -> None:
        print(index + 1, source)
        print("\n".join(f"  - {_character_line(c)}" for c in characters))
        print()

    def character_extraction_failed(self, error: Exception) -> None:
        print(f"登場人物を抽出できませんでした: {error}")

    def narration_assigned(self, sentence: Sentence) -> None:
        print(f"{_speaker_name(sentence)}\t{sentence.text}")

    def speaker_estimated(self, sentence: Sentence, confidence: Optional[float]) -> None:
        if confidence is None:
            print(f"{_speaker_name(sentence)}\t{sentence.text}")
        else:
            print(f"{_speaker_name(sentence)}\t({confidence:.2f})\t{sentence.text}")

    def speaker_estimation_failed(self, sentence: Sentence, error: Optional[Exception]) -> None:
        if error is not None:
            print(error)
        print(f"失敗\t{_speaker_name(sentence)}\t{sentence.text}")

    def context_shrunk(self, previous_count: int) -> None:
        print(f"入力長超過のため、履歴を直前{previous_count}文に縮小して再試行します")

    def sentence_synthesis_failed(self, voice: str, text: str, error: Exception) -> None:
        print(f"音声合成に失敗しました ({voice}): {error}\t{text}")

    def audio_exported(self, location: Optional[str], result: SynthesisResult) -> None:
        if location is None:
            if len(result.failures) > 0:
                print("全ての文の合成に失敗したため、音声を書き出しませんでした")
            else:
                print("読み上げられる文がないため、音声を書き出しませんでした")
        elif len(result.failures) > 0:
            print(f"音声を書き出しました(合成に失敗した{len(result.failures)}文を除く): {location}")
        else:
            print(f"音声を書き出しました: {location}")
        if result.skipped > 0:
            print(f"記号だけで読み上げる文字がない{result.skipped}行は読み飛ばしました")

    def finished(self, characters: List[Character], voices: Optional[Dict[str, str]]) -> None:
        print("\n登場人物一覧")
        for character in characters:
            print(f"- {_character_line(character)}")

        if voices is not None:
            print("\n音声の割り当て")
            for name, voice in voices.items():
                print(f"- {name}: {voice}")
