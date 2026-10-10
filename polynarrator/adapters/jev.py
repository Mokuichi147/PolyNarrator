from typing import Dict, List, Optional

from typesafe_sdk import Choice, TypeSafeAPIError, TypeSafeClient, TypeSafeError

from polynarrator.application.ports import ContextTooLongError, SpeakerEstimationError, SpeakerEstimator, SpeakerGuess
from polynarrator.domain.character import Character
from polynarrator.domain.novel import Sentence


class JevSpeakerEstimator(SpeakerEstimator):
    """Jev(TypeSafe System One API)互換モデルで話者を判定する"""

    client: TypeSafeClient

    # 400/422のときに入力長超過とみなすエラーメッセージのキーワード
    INPUT_TOO_LONG_KEYWORDS = ("context length", "context window", "maximum token", "max token", "token limit", "too many tokens", "input too long", "too long")

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        self.client = TypeSafeClient(
            api_key = api_key,
            model = model,
            base_url = base_url,
        )

    def _build_criteria(self, candidates: List[Character]) -> Dict[str, dict]:
        criteria: Dict[str, dict] = {}
        for character in candidates:
            # 選択肢のキーは一意である必要があるため、名前が重複した場合は未使用になるまで番号を付ける
            key = character.name
            suffix = 2
            while key in criteria:
                key = f"{character.name} ({suffix})"
                suffix += 1
            criteria[key] = {
                "性別": character.gender.value if character.gender is not None else "不明",
                "別名": character.aliases,
                "説明": character.portrait,
            }
        return criteria

    def _is_input_too_long(self, error: TypeSafeAPIError) -> bool:
        # 413はリクエストサイズ超過なので無条件に入力長超過とみなす
        if error.status == 413:
            return True
        if error.status not in (400, 422):
            return False
        message = f"{error} {error.body}".lower()
        return any(keyword in message for keyword in self.INPUT_TOO_LONG_KEYWORDS)

    def estimate_speaker(
        self,
        candidates: List[Character],
        previous: List[Sentence],
        target: Sentence,
        following: List[Sentence],
    ) -> Optional[SpeakerGuess]:
        criteria = self._build_criteria(candidates)
        state = {
            "今までの内容": [
                {"話者": s.speaker.name, "セリフ": s.text} if s.speaker is not None and not s.speaker.is_narrator else {"セリフ": s.text}
                for s in previous
            ],
            "推測したいセリフの内容": target.text,
            "後の内容": [s.text for s in following],
        }
        question = Choice(
            instructions = \
                "会話の内容から `推測したいセリフの内容` がどの登場人物による発言かを推測してください。"\
                "`今までの内容` と `後の内容` は前後の文脈です。"\
                "誰のセリフとも考えられない場合はナレーターを選択してください。",
            criteria = criteria,
        )
        try:
            answer = self.client.system_one(state, {"speaker": question}).choices.get("speaker")
        except TypeSafeAPIError as e:
            if self._is_input_too_long(e):
                raise ContextTooLongError(str(e)) from e
            raise SpeakerEstimationError(str(e)) from e
        except TypeSafeError as e:
            # 接続エラーやタイムアウトはこの文の推測を諦めてナレーター扱いにする
            raise SpeakerEstimationError(str(e)) from e

        # 未対応の回答種別はSDKが読み飛ばすため、回答が欠けている場合がある
        keys = list(criteria)
        if answer is None or answer.choice not in keys:
            return None
        return SpeakerGuess(keys.index(answer.choice), answer.confidence)
