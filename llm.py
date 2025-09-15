from typing import List
from ollama import Client, Message
from pydantic import RootModel

from models.narrator import Narrator
from models.novel import Novel

class Narrators(RootModel[List[Narrator]]):
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
                    "今までの登場人物一覧が与え得られた場合は、内容を適宜更新すること。"\
                    f"\nフォーマット:\n{schema}",
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
                "content": "\n小説の内容:\n".join([s.text for s in novel.sentences]),
            }
        )
        
        response = self.client.chat(
            messages = messages,
            model = self.model,
            format = schema,
            think = True,
        )
        
        data = response.message.content
        try:
            return Narrators.model_validate_json(data).root
        except:
            print("エラーが発生しました", response.message.content)
            return []