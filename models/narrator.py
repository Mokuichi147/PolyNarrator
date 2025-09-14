from typing import Optional


class Narrator:
    def __init__(self, name: str, gender: Optional[str] = None):
        self.name = name
        self.gender = gender