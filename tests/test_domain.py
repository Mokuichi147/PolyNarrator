from typing import List, Optional

import pytest

from polynarrator.domain.character import Character, Gender, narrator
from polynarrator.domain.novel import Novel, Sentence
from polynarrator.domain.text import is_readable, split_for_tts
from polynarrator.domain.voice_assigner import VoiceAssigner


def character(name: str, gender: Optional[Gender] = None, aliases: Optional[List[str]] = None) -> Character:
    return Character(name = name, portrait = "", gender = gender, aliases = aliases or [])


# --- 文と登場人物 ---

def test_novel_from_lines_strips_each_line():
    novel = Novel.from_lines(["あ\n", "　「い」 \n", "\n"])
    assert [s.text for s in novel.sentences] == ["あ", "「い」", ""]
    assert all(s.speaker is None for s in novel.sentences)


@pytest.mark.parametrize(("text", "expected"), [("「はい」", True), ("「はい」と言った", False), ("はい", False), ("「", False)])
def test_sentence_is_dialogue(text, expected):
    assert Sentence(text = text).is_dialogue == expected


def test_narrator_character():
    assert narrator().is_narrator
    assert not character("花子").is_narrator


# --- 読み上げ可否と分割 ---

@pytest.mark.parametrize("text", ["＊　＊　＊", "「………………」", "――――", "********", "　", "「！？」"])
def test_is_readable_symbols(text):
    assert not is_readable(text)


@pytest.mark.parametrize("text", ["こんにちは", "「はい」", "ＡＢＣ", "123", "「ああ……」", "ー"])
def test_is_readable_text(text):
    assert is_readable(text)


def test_split_for_tts_short_text_is_unchanged():
    assert split_for_tts("「はい。そうです」", 200) == ["「はい。そうです」"]
    assert split_for_tts("あ" * 300, 0) == ["あ" * 300]


def test_split_for_tts_splits_at_sentence_end():
    text = "一文目です。「二文目！」三文目です？四文目"
    chunks = split_for_tts(text, 8)
    assert "".join(chunks) == text
    assert chunks == ["一文目です。", "「二文目！」", "三文目です？", "四文目"]


def test_split_for_tts_packs_short_sentences():
    assert split_for_tts("あ。い。う。え。", 4) == ["あ。い。", "う。え。"]


def test_split_for_tts_splits_at_comma_then_by_length():
    assert split_for_tts("ああ、いいい、うう。", 6) == ["ああ、", "いいい、", "うう。"]
    assert split_for_tts("あ" * 7 + "、" + "い" * 3 + "。う", 5) == ["あああああ", "ああ、", "いいい。", "う"]


@pytest.mark.parametrize("text", ["あ" * 23, "あいうえおかきくけこ、さしすせそたちつてと。" * 3, "「" + "あ" * 30 + "」"])
def test_split_for_tts_never_exceeds_max_chars(text):
    chunks = split_for_tts(text, 7)
    assert all(len(c) <= 7 for c in chunks)
    assert "".join(chunks) == text


@pytest.mark.parametrize(("text", "max_chars"), [("あ。", 1), ("あ。」", 2), ("あ" + "…" * 10 + "い", 3), ("「" * 5 + "あ", 2)])
def test_split_for_tts_strictly_limits_chunks_with_symbols(text, max_chars):
    chunks = split_for_tts(text, max_chars)
    assert all(len(c) <= max_chars for c in chunks)
    assert all(is_readable(c) for c in chunks)


@pytest.mark.parametrize(("text", "max_chars", "expected"), [
    ("a\u0301\u0308b", 2, ["a\u0301\u0308", "b"]),  # 1文字で上限を超える結合文字列は区切らない
    ("\u30ab\u3099" * 3, 3, ["\u30ab\u3099", "\u30ab\u3099", "\u30ab\u3099"]),  # NFD の「ガ」
    ("\u30ab\u3099" * 3 + "。", 4, ["\u30ab\u3099" * 2, "\u30ab\u3099。"]),
])
def test_split_for_tts_keeps_combining_marks(text, max_chars, expected):
    assert split_for_tts(text, max_chars) == expected


@pytest.mark.parametrize("text", ["あああああ。い", "あああああ、い", "あああああ。」い", "「ああああ」"])
def test_split_for_tts_keeps_symbols_at_boundary(text):
    chunks = split_for_tts(text, 5)
    assert "".join(chunks) == text
    assert all(is_readable(c) for c in chunks)


def test_split_for_tts_drops_unreadable_chunks():
    assert split_for_tts("あいうえお。……。", 6) == ["あいうえお。"]


# --- 音声の割り当て ---

def test_voice_assigner_narrator_and_least_used():
    assigner = VoiceAssigner(["n", "a", "b"])
    assert assigner.assign(None) == "n"
    assert assigner.assign(character("ナレーター")) == "n"
    assert assigner.assign(character("A")) == "a"
    assert assigner.assign(character("B")) == "b"
    assert assigner.assign(character("C")) == "a"
    assert assigner.assign(character("A")) == "a"


def test_voice_assigner_single_voice_is_shared_with_narrator():
    assigner = VoiceAssigner(["n"])
    assert assigner.assign(character("A")) == "n"


def test_voice_assigner_gender_voices():
    assigner = VoiceAssigner(["n", "m", "f", "x"], narrator_voice = "x", male_voices = ["m"], female_voices = ["f"])
    assert assigner.assign(None) == "x"
    assert assigner.assign(character("太郎", Gender.MALE)) == "m"
    assert assigner.assign(character("花子", Gender.FEMALE)) == "f"
    assert assigner.assign(character("不明")) == "n"
    # 性別ごとの指定がない性別は全音声から選ぶ(使用数が同じなら一覧の先頭)
    assert assigner.assign(character("その他", Gender.OTHER)) == "n"


def test_voice_assigner_keeps_assigned_name_when_later_linked_by_alias():
    assigner = VoiceAssigner(["n", "a", "b"])
    assert assigner.assign(character("A")) == "a"
    assert assigner.assign(character("B")) == "b"
    # 後から B が A の別名と分かっても、書き出し済みの音声と食い違わないよう両者とも元の音声を維持する
    assert assigner.assign(character("B", aliases = ["A"])) == "b"
    assert assigner.assign(character("A")) == "a"


def test_voice_assigner_normalizes_names():
    assigner = VoiceAssigner(["n", "a", "b"])
    assert assigner.assign(character("A", aliases = ["\u30ab\u3099"])) == "a"
    assert assigner.assign(character(" A ")) == "a"
    assert assigner.assign(character("\u30ac")) == "a"  # NFD の別名と NFC の正式名
    assert assigner.assign(character("Ａ")) == "a"  # 全角
    assert assigner.assign(character(" ナレーター ")) == "n"
    assert list(assigner.assigned) == ["ナレーター", "A", "\u30ac"]


def test_voice_assigner_blank_name_and_alias():
    assigner = VoiceAssigner(["n", "a", "b"])
    assert assigner.assign(character("", Gender.MALE)) == "n"
    assert assigner.assign(character("　")) == "n"
    assert assigner.assign(character("A", aliases = ["", " "])) == "a"
    assert assigner.assign(character("B", aliases = [""])) == "b"


def test_voice_assigner_ignores_ambiguous_alias():
    assigner = VoiceAssigner(["n", "a", "b", "c"])
    assert assigner.assign(character("A", aliases = ["先生"])) == "a"
    assert assigner.assign(character("B", aliases = ["先生"])) == "b"
    # 複数の人物が持つ別名はどの人物か決められないため、新しい人物として扱う
    assert assigner.assign(character("先生")) == "c"


def test_voice_assigner_ignores_alias_shared_by_people_with_same_voice():
    assigner = VoiceAssigner(["n", "a", "b"], male_voices = ["a"], female_voices = ["a"])
    assert assigner.assign(character("A", Gender.MALE, aliases = ["X"])) == "a"
    assert assigner.assign(character("B", Gender.FEMALE, aliases = ["X"])) == "a"
    assert assigner.assign(character("X", Gender.OTHER)) == "b"


def test_voice_assigner_alias_of_same_person_with_varying_name():
    assigner = VoiceAssigner(["n", "a", "b"])
    assert assigner.assign(character("田中花子", aliases = ["花子", "委員長"])) == "a"
    # 正式名が揺れても同じ人物と分かっている名前から同じ別名が来た場合は、曖昧にしない
    assert assigner.assign(character("花子", aliases = ["委員長"])) == "a"
    assert assigner.assign(character("委員長")) == "a"


def test_voice_assigner_alias_matching_multiple_people_is_new_person():
    assigner = VoiceAssigner(["n", "a", "b", "c"])
    assert assigner.assign(character("A")) == "a"
    assert assigner.assign(character("B")) == "b"
    assert assigner.assign(character("C", aliases = ["A", "B"])) == "c"


def test_voice_assigner_gender_voices_only_narrator_falls_back():
    assigner = VoiceAssigner(["n", "x"], male_voices = ["n"])
    assert assigner.assign(character("太郎", Gender.MALE)) == "x"


def test_voice_assigner_matches_aliases_across_files():
    assigner = VoiceAssigner(["n", "a", "b", "c"])
    voice = assigner.assign(character("田中花子", aliases = ["花子", "委員長"]))
    # 後のファイルで正式名が別名の方に揺れた場合
    assert assigner.assign(character("花子")) == voice
    # 以前の正式名を別名に持つ場合
    assert assigner.assign(character("委員長花子", aliases = ["田中花子"])) == voice
    # 別名同士が一致するだけでは同一人物とみなさない
    assert assigner.assign(character("次郎", aliases = ["委員長"])) != voice


@pytest.mark.parametrize("kwargs", [
    {"narrator_voice": "z"},
    {"male_voices": ["z"]},
    {"female_voices": ["a", "z"]},
])
def test_voice_assigner_rejects_unknown_voices(kwargs):
    with pytest.raises(ValueError, match = "z"):
        VoiceAssigner(["a", "b"], **kwargs)


def test_voice_assigner_rejects_empty_voices():
    with pytest.raises(ValueError):
        VoiceAssigner([])
