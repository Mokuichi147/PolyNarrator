from typing import Dict, List, Optional

from typesafe_sdk import Choice, TypeSafeAPIError, TypeSafeClient

from models.narrator import Narrator
from models.novel import Novel


class JevSpeakerEstimator:
    """Jev(TypeSafe System One API)互換モデルで話者を判定する"""

    client: TypeSafeClient

    def __init__(self, api_key: Optional[str] = None, model: Optional[str] = None, base_url: Optional[str] = None):
        self.client = TypeSafeClient(
            api_key = api_key,
            model = model,
            base_url = base_url,
        )

    def _build_criteria(self, narrators: List[Narrator]) -> Dict[str, dict]:
        criteria: Dict[str, dict] = {}
        for index, narrator in enumerate(narrators):
            # 選択肢のキーは一意である必要があるため、名前が重複した場合は番号を付ける
            key = narrator.name
            if key in criteria:
                key = f"{narrator.name} ({index})"
            criteria[key] = {
                "性別": narrator.gender.value if narrator.gender is not None else "不明",
                "別名": narrator.aliases,
                "説明": narrator.portrait,
            }
        return criteria

    def _ask(self, narrators: List[Narrator], pre_sentences: list, sentence_text: str, after_sentences: list):
        state = {
            "今までの内容": [
                {"話者": s.narrator.name, "セリフ": s.text} if s.narrator != None and s.narrator.name != "ナレーター" else {"セリフ": s.text}
                for s in pre_sentences
            ],
            "推測したいセリフの内容": sentence_text,
            "後の内容": [s.text for s in after_sentences],
        }
        question = Choice(
            instructions = \
                "会話の内容から `推測したいセリフの内容` がどの登場人物による発言かを推測してください。"\
                "`今までの内容` と `後の内容` は前後の文脈です。"\
                "誰のセリフとも考えられない場合はナレーターを選択してください。",
            criteria = self._build_criteria(narrators),
        )
        return self.client.system_one(state, {"speaker": question}).choices["speaker"]

    def set_estimation_narrator(self, novel: Novel, pre_max_count: int = 15, after_max_count: int = 1, corner_bracket_only: bool = False):
        narrators = [Narrator(name = "ナレーター", portrait = "世界観の説明などキャラクターの発言ではない内容のナレーションを行う")]
        narrators.extend(novel.narrators[:])
        criteria_keys = list(self._build_criteria(narrators).keys())

        for i in range(len(novel.sentences)):
            sentence = novel.sentences[i]
            after_sentences = novel.sentences[i+1:][:after_max_count]

            if corner_bracket_only and not (sentence.text.startswith("「") and sentence.text.endswith("」")):
                novel.sentences[i].narrator = narrators[0]
                print(f"{novel.sentences[i].narrator.name}\t{novel.sentences[i].text}")
                continue

            current_pre_max_count = pre_max_count
            answer = None
            while True:
                pre_sentences = novel.sentences[:i][-current_pre_max_count:] if current_pre_max_count > 0 else []
                try:
                    answer = self._ask(narrators, pre_sentences, sentence.text, after_sentences)
                    break
                except TypeSafeAPIError as e:
                    # 入力長超過などのリクエスト不正は履歴を縮小して再試行する
                    if e.status not in (400, 413, 422) or current_pre_max_count <= 0:
                        print(e)
                        break
                    current_pre_max_count = current_pre_max_count // 2
                    print(f"リクエストに失敗したため、履歴を直前{current_pre_max_count}文に縮小して再試行します")

            narrator_index: Optional[int] = None
            confidence: float = 0.0
            if answer is not None and answer.choice in criteria_keys:
                narrator_index = criteria_keys.index(answer.choice)
                confidence = answer.confidence

            if narrator_index is not None and 0 <= narrator_index < len(narrators):
                novel.sentences[i].narrator = narrators[narrator_index]
            else:
                novel.sentences[i].narrator = narrators[0]
                print(f"失敗\t", end="")

            print(f"{novel.sentences[i].narrator.name}\t({confidence:.2f})\t{novel.sentences[i].text}")
