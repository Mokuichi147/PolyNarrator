from typing import List

from models.narrator import Narrator
from models.sentence import Sentence


class Novel:
    def __init__(self) -> None:
        self.sentences: List[Sentence] = []
        self.narrators: List[Narrator] = []
    
    def load(self, filepath: str) -> None:
        self.sentences = []
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            self.sentences.append(Sentence(text=line.strip()))
