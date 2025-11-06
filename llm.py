from typing import List, Optional
from ollama import Client, Message
from pydantic import RootModel

from models.narrator import Narrator
from models.novel import Novel

class Narrators(RootModel[List[Narrator]]):
    """登場人物のリスト"""
    pass

class NarratorIndex(RootModel[int]):
    """登場人物一覧のインデックス"""
    pass

class Ai:
    client: Client
    
    def __init__(self, host: str, port: str, model: str):
        self.client = Client(
            host = f"http://{host}:{port}"
        )
        self.model = model
    
    def get_narrators(self, novel: Novel, narrators: List[Narrator] = []) -> List[Narrator]:
        schema = Narrators.model_json_schema()
        
        messages: List[Message] = [
            {
                "role": "system",
                "content": \
                    "ユーザーから与えられた小説の内容から登場人物を全て抽出し、指定されたJsonフォーマットで返答してください。"\
                    "同一の人物や一覧内で命名の揺れがないこと。"\
                    "今までの登場人物一覧が与え得られた場合は、内容を適宜更新すること。"
            }
        ]
        
        if len(narrators) > 0:
            messages.append(
                {
                    "role": "user",
                    "content": "今までの登場人物一覧:\n" + "\n".join([f"- {n.name} ({n.gender})" for n in narrators]),
                }
            )
        
        messages.append(
            {
                "role": "user",
                "content": "\n小説の内容:\n" + "\n".join([s.text for s in novel.sentences]),
            }
        )
        
        response = self.client.chat(
            messages = messages,
            model = self.model,
            format = schema,
            #think = True,
            options = {
                "num_ctx": 16000
            }
        )
        
        data = response.message.content
        try:
            return Narrators.model_validate_json(data).root
        except:
            print("エラーが発生しました", response.message.content)
            return []
    
    def set_estimation_narrator(self, novel: Novel, pre_max_count: int = 15, after_max_count: int = 1, corner_bracket_only: bool = False):
        schema = NarratorIndex.model_json_schema()
        narrators = [Narrator(name = "ナレーター", portrait = "世界観の説明などキャラクターの発言ではない内容のナレーションを行う")]
        narrators.extend(novel.narrators[:])
        
        for i in range(len(novel.sentences)):
            pre_sentences = novel.sentences[:i][-pre_max_count:]
            sentence = novel.sentences[i]
            after_sentences = novel.sentences[i+1:][:after_max_count]
            
            if corner_bracket_only and not (sentence.text.startswith("「") and sentence.text.endswith("」")):
                novel.sentences[i].narrator = narrators[0]
                print(f"{novel.sentences[i].narrator.name}\t{novel.sentences[i].text}")
                continue
            
            pre_context = "\n".join(
                [
                    f"{s.narrator.name}\t{s.text}"
                    if s.narrator is not None and s.narrator.name != "ナレーター"
                    else f"\t{s.text}"
                    for s in pre_sentences
                ]
            )
            after_context = "\n".join([s.text for s in after_sentences])

            content: str = (
                "今までの内容:\n"
                f"{pre_context}\n\n"
                "推測したいセリフの内容:\n"
                f"{sentence.text}\n\n"
                "後の内容:\n"
                f"{after_context}"
            )
            
            messages: List[Message] = [
                {
                    "role": "system",
                    "content": \
                        "会話の内容から指定されたセリフがどの登場人物による発言かを推測し、指定されたJsonフォーマットで返答してください。"\
                        "会話は推定したい文とその前後の内容が与えられます。"\
                        "誰のセリフとも考えられない場合はナレーターを指定してください。"
                },
                {
                    "role": "user",
                    "content": "登場人物一覧:\n" + "\n".join([f"{index}. {narrator.name} (性別:{narrator.gender}, 別名:{narrator.aliases}, 説明:{narrator.portrait})" for index, narrator in enumerate(narrators)]),
                },
                {
                    "role": "user",
                    "content": content,
                }
            ]
            
            response = self.client.chat(
                messages = messages,
                model = self.model,
                format = schema,
                #think = True,
                options = {
                    "num_ctx": 16000
                }
            )
            
            data = response.message.content
            narrator_index: Optional[int] = None
            try:
                narrator_index = NarratorIndex.model_validate_json(data).root
            except:
                print("エラーが発生しました", response.message.content)

            if narrator_index is not None and 0 <= narrator_index < len(narrators):
                novel.sentences[i].narrator = narrators[narrator_index]
            else:
                novel.sentences[i].narrator = narrators[0]
                print(f"失敗\t", end="")
            
            print(f"{novel.sentences[i].narrator.name}\t{novel.sentences[i].text}")
