from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

NARRATOR_NAME = "ナレーター"


class Gender(Enum):
    MALE = "男性"
    FEMALE = "女性"
    OTHER = "その他"


@dataclass
class Character:
    """登場人物。これらの情報を元にセリフの話者を推測し、音声を割り当てる"""
    name: str
    portrait: str
    aliases: List[str] = field(default_factory = list)
    gender: Optional[Gender] = None

    @property
    def is_narrator(self) -> bool:
        return self.name == NARRATOR_NAME


def narrator() -> Character:
    """誰のセリフでもない文を読むナレーター"""
    return Character(name = NARRATOR_NAME, portrait = "世界観の説明などキャラクターの発言ではない内容のナレーションを行う")
