from dataclasses import dataclass, field
from typing import Iterable, List, Optional

from polynarrator.domain.character import Character


@dataclass
class Sentence:
    text: str
    speaker: Optional[Character] = None

    @property
    def is_dialogue(self) -> bool:
        """行全体が「」で囲まれたセリフか"""
        return self.text.startswith("「") and self.text.endswith("」")


@dataclass
class Novel:
    sentences: List[Sentence] = field(default_factory = list)
    characters: List[Character] = field(default_factory = list)

    @classmethod
    def from_lines(cls, lines: Iterable[str]) -> "Novel":
        """1行を1つの文として扱う"""
        return cls(sentences = [Sentence(text = line.strip()) for line in lines])
