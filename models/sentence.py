from typing import Optional
from models.narrator import Narrator

class Sentence:
    narrator: Optional[Narrator] = None
    
    def __init__(self, text: str):
        self.text = text
