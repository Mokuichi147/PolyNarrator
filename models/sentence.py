from typing import Optional
from pydantic import BaseModel
from models.narrator import Narrator

class Sentence(BaseModel):
    narrator: Optional[Narrator] = None
    text: str
