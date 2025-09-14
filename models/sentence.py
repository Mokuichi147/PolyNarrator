from typing import Optional
from models.narrator import Narrator

class Sentence:
    narrator: Optional[Narrator] = None
    
    def __init__(self, sentence: str):
        self.sentence = sentence
