import unicodedata
from collections import Counter
from typing import Dict, List, Optional

from polynarrator.domain.character import NARRATOR_NAME, Character, Gender


class VoiceAssigner:
    """登場人物ごとに音声を割り当てる。同じ登場人物には常に同じ音声を使う"""

    def __init__(
        self,
        voices: List[str],
        narrator_voice: Optional[str] = None,
        male_voices: Optional[List[str]] = None,
        female_voices: Optional[List[str]] = None,
    ):
        if len(voices) == 0:
            raise ValueError("利用できる音声がありません")
        male_voices = male_voices or []
        female_voices = female_voices or []
        unknown = [v for v in [narrator_voice or voices[0], *male_voices, *female_voices] if v not in voices]
        if len(unknown) > 0:
            raise ValueError(f"音声一覧にない音声が指定されています: {unknown}")

        self.voices = voices
        self.narrator_voice = narrator_voice or voices[0]
        self.gender_voices: Dict[Gender, List[str]] = {
            Gender.MALE: male_voices,
            Gender.FEMALE: female_voices,
        }
        self.assigned: Dict[str, str] = {NARRATOR_NAME: self.narrator_voice}
        self.usage: Counter[str] = Counter([self.narrator_voice])
        # 人物ごとの音声と、名前・別名から人物への対応。複数の人物が持ち、どの人物か決められない別名は None にする
        self._person_voices: List[str] = []
        self._name_persons: Dict[str, int] = {}
        self._alias_persons: Dict[str, Optional[int]] = {}

    @staticmethod
    def _key(name: str) -> str:
        """前後の空白や Unicode の表記の違いで別人とみなさないよう、照合用に名前を正規化する"""
        return unicodedata.normalize("NFKC", name).strip()

    def _find_person(self, name: str, aliases: List[str]) -> Optional[int]:
        # 割り当て済みの名前は、後から別名で別の人物と結び付いても書き出し済みの音声と食い違わないよう元の人物のままにする
        if name in self._name_persons:
            return self._name_persons[name]
        # ファイルごとの抽出で正式名が揺れても、以前の別名や正式名と一致すれば同じ人物とみなす。
        # 汎用的な呼び名で別人と混同しないよう、別名同士の一致は使わない
        if self._alias_persons.get(name) is not None:
            return self._alias_persons[name]
        # 別名が複数の人物の正式名に一致する場合は、どの人物か決められないため新しい人物とする
        candidates = {self._name_persons[a] for a in aliases if a in self._name_persons}
        return candidates.pop() if len(candidates) == 1 else None

    def _new_person(self, gender: Optional[Gender]) -> int:
        pool = self.voices
        if gender is not None and len(self.gender_voices.get(gender, [])) > 0:
            pool = self.gender_voices[gender]

        # ナレーターと聞き分けられるよう、性別ごとの候補、全ての音声の順にナレーター以外の音声を探す
        candidates = (
            [v for v in pool if v != self.narrator_voice]
            or [v for v in self.voices if v != self.narrator_voice]
            or [self.narrator_voice]
        )
        voice = min(candidates, key = lambda v: self.usage[v])
        self.usage[voice] += 1
        self._person_voices.append(voice)
        return len(self._person_voices) - 1

    def assign(self, character: Optional[Character]) -> str:
        # 名前のない登場人物は誰か区別できないため、ナレーターとして読む
        if character is None:
            return self.narrator_voice
        name = self._key(character.name)
        if name == NARRATOR_NAME or not name:
            return self.narrator_voice
        aliases = [a for a in (self._key(a) for a in character.aliases) if a]

        person = self._find_person(name, aliases)
        if person is None:
            person = self._new_person(character.gender)

        voice = self._person_voices[person]
        self._name_persons[name] = person
        self.assigned[name] = voice
        for alias in aliases:
            if alias not in self._alias_persons:
                self._alias_persons[alias] = person
            elif self._alias_persons[alias] != person:
                self._alias_persons[alias] = None
        return voice
