import re
import unicodedata
from typing import List


def is_readable(text: str) -> bool:
    """読み上げられる文字(文字・数字)を含むか。「＊　＊　＊」や「………」のような記号だけの文は False"""
    return any(unicodedata.category(c)[0] in ("L", "N") for c in text)


# 文末記号とその直後の閉じ括弧までを1文とする
_SENTENCE = re.compile(r"(?:[^。！？!?]+[。！？!?]*|[。！？!?]+)[」』）)]*")
# 読点とその直後の閉じ括弧までを1区切りとする
_CLAUSE = re.compile(r"(?:[^、，,]+[、，,]*|[、，,]+)[」』）)]*")


def _pack(pieces: List[str], max_chars: int) -> List[str]:
    """区切りを順に max_chars 以下の塊にまとめる。1つで max_chars を超える区切りはそのまま残す"""
    chunks: List[str] = []
    current = ""
    for piece in pieces:
        if current and len(current) + len(piece) > max_chars:
            chunks.append(current)
            current = ""
        current += piece
    if current:
        chunks.append(current)
    return chunks


def _clusters(text: str) -> List[str]:
    """濁点などの結合文字を直前の文字と同じ単位にまとめる"""
    clusters: List[str] = []
    for char in text:
        if clusters and unicodedata.category(char).startswith("M"):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _split_by_length(text: str, max_chars: int) -> List[str]:
    """文字数で区切る。句読点や括弧などの記号はできるだけ直前の文字と同じ塊に入れ、記号だけの塊ができないようにする"""
    units: List[List[str]] = []
    for cluster in _clusters(text):
        if units and (not is_readable(cluster) or not is_readable("".join(units[-1]))):
            units[-1].append(cluster)
        else:
            units.append([cluster])
    # 記号を含めた単位でも max_chars を超える場合は、上限を優先して結合文字の単位で区切る。
    # 結合文字を含む1文字だけで max_chars を超える場合は、文字を壊さないよう区切らない
    pieces = [piece for unit in units for piece in (_pack(unit, max_chars) if len("".join(unit)) > max_chars else ["".join(unit)])]
    return _pack(pieces, max_chars)


def split_for_tts(text: str, max_chars: int) -> List[str]:
    """max_chars を超える文を、文末・読点・文字数の順に区切りを優先して max_chars 以下の塊に分ける"""
    if max_chars <= 0 or len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    for sentence in _pack(_SENTENCE.findall(text), max_chars):
        for clause in _pack(_CLAUSE.findall(sentence), max_chars) if len(sentence) > max_chars else [sentence]:
            # 文末でも読点でも区切れない長さの場合は文字数で区切る
            chunks.extend(_split_by_length(clause, max_chars) if len(clause) > max_chars else [clause])
    return [c for c in chunks if is_readable(c)]
