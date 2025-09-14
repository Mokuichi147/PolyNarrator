import json
from typing import List
from ollama import Client
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
    
    def get_narrators(self, novel: Novel) -> List[Narrator]:
        schema = Narrators.model_json_schema()
        response = self.client.chat(
            messages=[
                {
                    "role": "system",
                    "content": f"小説の内容から登場人物の名前のリストを指定されたJsonフォーマットで返答してください。\nフォーマット:\n{schema}"
                },
                {
                    "role": "user",
                    "content": "\n".join([s.text for s in novel.sentences]),
                }
            ],
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