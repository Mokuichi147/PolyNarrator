from typing import Optional
from pydantic import BaseModel

from models.gender import Gender

class Narrator(BaseModel):
    name: str
    gender: Optional[Gender] = None
