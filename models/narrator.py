from typing import List, Optional
from pydantic import BaseModel

from models.gender import Gender

class Narrator(BaseModel):
    name: str
    aliases: List[str] = []
    gender: Optional[Gender] = None
