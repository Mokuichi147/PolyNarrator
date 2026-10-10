import re
from typing import Annotated, List, Optional

from openai import BadRequestError, OpenAI
from openai.types.chat import ChatCompletionMessageParam
from pydantic import BaseModel, Field

from polynarrator.application.ports import (
    CharacterExtractionError,
    CharacterExtractor,
    ContextTooLongError,
    SpeakerEstimationError,
    SpeakerEstimator,
    SpeakerGuess,
)
from polynarrator.domain.character import Character, Gender
from polynarrator.domain.novel import Novel, Sentence

# 応答のスキーマはプロンプトにも含めるため、クラス名や説明を変えるとモデルへの入力が変わる

class Narrator(BaseModel):
    """登場人物の情報。これらの情報を元に会話内容の発言者を推測する"""

    name: Annotated[str, Field(description="キャラクター名。正式名称であることが好ましい")]
    portrait: Annotated[str, Field(description="性格や来歴、外見といったキャラクターの特徴")]
    aliases: Annotated[List[str], Field(description="ニックネームや別称など")] = []
    gender: Annotated[Optional[Gender], Field(description="性別")] = None

    def to_character(self) -> Character:
        return Character(name = self.name, portrait = self.portrait, aliases = list(self.aliases), gender = self.gender)


class Narrators(BaseModel):
    """登場人物のリスト"""
    narrators: List[Narrator]


class NarratorResponse(BaseModel):
    """登場人物一覧のインデックス"""
    narrator_index: int


JSON_SINGLE_BLOCK = re.compile(r'(?s)([\{\[].*[\}\]])')


def parse_json[T: BaseModel](data: str, model_cls: type[T]) -> T:
    """JSON の前後に余計な文章が付いた応答も受け付ける"""
    try:
        return model_cls.model_validate_json(data)
    except Exception:
        m = JSON_SINGLE_BLOCK.search(data)
        if m:
            return model_cls.model_validate_json(m.group(1))
        raise


class OpenAiChatModel(CharacterExtractor, SpeakerEstimator):
    """OpenAI API 互換の LLM サーバーで登場人物の抽出と話者の推測を行う"""

    def __init__(self, host: str, port: str, model: str):
        self.client = OpenAI(
            base_url = f"http://{host}:{port}/v1",
            api_key = "lm-studio",
        )
        self.model = model

    def _chat(self, messages: List[ChatCompletionMessageParam], schema: dict) -> Optional[str]:
        try:
            response = self.client.chat.completions.create(
                model = self.model,
                messages = messages,
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.get("title", "response"),
                        "schema": schema,
                    },
                },
            )
        except BadRequestError as e:
            message = str(e)
            if "context" in message.lower():
                raise ContextTooLongError(message) from e
            raise
        return response.choices[0].message.content

    def extract_characters(self, novel: Novel, known: List[Character]) -> List[Character]:
        schema = Narrators.model_json_schema()

        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": \
                    "ユーザーから与えられた小説の内容から登場人物を全て抽出し、指定されたJsonフォーマットで返答してください。"\
                    "同一の人物や一覧内で命名の揺れがないこと。"\
                    "今までの登場人物一覧が与え得られた場合は、内容を適宜更新すること。\n"\
                    f"レスポンスフォーマット:\n{schema}"
            }
        ]

        if len(known) > 0:
            messages.append(
                {
                    "role": "user",
                    "content": "今までの登場人物一覧:\n" + "\n".join([f"- {n.name} ({n.gender})" for n in known]),
                }
            )

        messages.append(
            {
                "role": "user",
                "content": "\n小説の内容:\n" + "\n".join([s.text for s in novel.sentences]),
            }
        )

        data = self._chat(messages, schema)
        try:
            return [n.to_character() for n in parse_json(data, Narrators).narrators]
        except Exception as e:
            raise CharacterExtractionError(f"{e}\n応答: {data}") from e

    def estimate_speaker(
        self,
        candidates: List[Character],
        previous: List[Sentence],
        target: Sentence,
        following: List[Sentence],
    ) -> Optional[SpeakerGuess]:
        schema = NarratorResponse.model_json_schema()

        history = "\n".join(
            f"{s.speaker.name}\t{s.text}" if s.speaker is not None and not s.speaker.is_narrator else f"\t{s.text}"
            for s in previous
        )
        upcoming = "\n".join(s.text for s in following)
        # モデルへの入力を変えないよう、行頭の字下げも含めて従来と同じ文字列にする
        indent = " " * 20
        content = (
            f"\n{indent}今までの内容:\n{indent}{history}\n"
            f"\n{indent}推測したいセリフの内容:\n{indent}{target.text}\n"
            f"\n{indent}後の内容:\n{indent}{upcoming}\n{indent}"
        )

        messages: List[ChatCompletionMessageParam] = [
            {
                "role": "system",
                "content": \
                    "会話の内容から指定されたセリフがどの登場人物による発言かを推測し、指定されたJsonフォーマットで返答してください。"\
                    "会話は推定したい文とその前後の内容が与えられます。"\
                    "誰のセリフとも考えられない場合はナレーターを指定してください。\n"\
                    f"レスポンスフォーマット:\n{schema}"
            },
            {
                "role": "user",
                "content": "登場人物一覧:\n" + "\n".join([f"{index}. {c.name} (性別:{c.gender}, 別名:{c.aliases}, 説明:{c.portrait})" for index, c in enumerate(candidates)]),
            },
            {
                "role": "user",
                "content": content,
            }
        ]

        data = self._chat(messages, schema)
        try:
            return SpeakerGuess(parse_json(data, NarratorResponse).narrator_index)
        except Exception as e:
            raise SpeakerEstimationError(f"{e}\n応答: {data}") from e
