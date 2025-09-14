from typing import List
from models.narrator import Narrator
from models.sentence import Sentence

class Novel:
    sentences: List[Sentence] = []
    narrators: List[Narrator] = []
    
    def load(self, filepath: str):
        with open(filepath, "r", encoding="utf-8") as f:
            lines = f.readlines()
        
        for line in lines:
            self.sentences.append(Sentence(text = line.strip()))
