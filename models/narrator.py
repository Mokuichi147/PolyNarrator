from typing import Optional
from pydantic import BaseModel

class Narrator(BaseModel):
    name: str
    gender: Optional[str] = None
