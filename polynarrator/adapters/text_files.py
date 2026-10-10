import os
from typing import List

from natsort import natsorted

from polynarrator.domain.novel import Novel


def list_text_files(folder: str) -> List[str]:
    """フォルダ内のテキストファイル名を自然順で返す。.DS_Store や出力済みの音声などは対象外"""
    return natsorted(f for f in os.listdir(folder) if f.endswith(".txt") and os.path.isfile(os.path.join(folder, f)))


def read_novel(path: str) -> Novel:
    """UTF-8 のテキストファイルを1行1文の小説として読み込む"""
    with open(path, "r", encoding = "utf-8") as f:
        return Novel.from_lines(f.readlines())
