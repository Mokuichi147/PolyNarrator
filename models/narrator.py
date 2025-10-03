from typing import Annotated, List, Optional
from pydantic import BaseModel, Field

from models.gender import Gender

class Narrator(BaseModel):
    """登場人物の情報。これらの情報を元に会話内容の発言者を推測する"""
    
    name: Annotated[str, Field(description="キャラクター名。正式名称であることが好ましい")]
    portrait: Annotated[str, Field(description="性格や来歴、外見といったキャラクターの特徴")]
    aliases: Annotated[List[str], Field(description="ニックネームや別称など")] = []
    gender: Annotated[Optional[Gender], Field(description="性別")] = None