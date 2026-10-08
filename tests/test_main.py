import os
import socket
import wave
from typing import List, Optional, Sequence

import pytest

import main
import tts
from main import build_parser, create_tts, list_text_files, synthesize_file
from models.narrator import Narrator
from models.novel import Novel
from models.sentence import Sentence
from tts import Audio, TtsClient, TtsInputError, VoiceAssigner


def parse(*args: str):
    return build_parser().parse_args([*args, "data"])


# --- CLI引数 ---

def test_parser_defaults():
    args = parse()
    assert (args.tts_pause, args.tts_max_chars, args.tts_timeout) == (0.3, 200, 60.0)


@pytest.mark.parametrize("args", [
    ["--tts-pause", "-1"],
    ["--tts-pause", "nan"],
    ["--tts-pause", "inf"],
    ["--tts-pause", "60.1"],
    ["--tts-pause", "1e10"],
    ["--tts-max-chars", "-1"],
    ["--tts-max-chars", "1.5"],
    ["--tts-timeout", "0"],
    ["--tts-timeout", "-1"],
    ["--tts-timeout", "nan"],
    ["--tts-timeout", "inf"],
])
def test_parser_rejects_invalid_values(args):
    with pytest.raises(SystemExit):
        parse(*args)


def test_parser_accepts_boundaries():
    args = parse("--tts-pause", "0", "--tts-max-chars", "0")
    assert (args.tts_pause, args.tts_max_chars) == (0.0, 0)
    assert parse("--tts-pause", "60").tts_pause == 60.0


# --- 入力ファイル ---

def test_list_text_files_missing_folder(tmp_path):
    (tmp_path / "file.txt").write_text("")
    with pytest.raises(SystemExit, match = "入力フォルダ"):
        list_text_files(str(tmp_path / "missing"))
    with pytest.raises(SystemExit, match = "入力フォルダ"):
        list_text_files(str(tmp_path / "file.txt"))


def test_list_text_files(tmp_path):
    for name in ["10.txt", "2.txt", "1.txt", ".DS_Store", "1.wav", "memo.md"]:
        (tmp_path / name).write_text("")
    (tmp_path / "dir.txt").mkdir()
    assert list_text_files(str(tmp_path)) == ["1.txt", "2.txt", "10.txt"]


# --- 音声合成の初期化 ---

@pytest.mark.parametrize("args", [[], ["--tts-narrator-voice", "alloy"], ["--tts-male-voices", "alloy"]])
def test_create_tts_openai_requires_voices(args):
    with pytest.raises(SystemExit, match = "--tts-voices"):
        create_tts(parse("--tts-output", "out", *args))


def test_create_tts_rejects_voice_outside_list():
    with pytest.raises(SystemExit, match = "z"):
        create_tts(parse("--tts-output", "out", "--tts-voices", "a,b", "--tts-narrator-voice", "z"))


def test_create_tts_openai(tmp_path):
    client, assigner = create_tts(parse("--tts-output", "out", "--tts-voices", "a,b", "--tts-male-voices", "b"))
    assert assigner.voices == ["a", "b"]
    assert assigner.narrator_voice == "a"


def test_create_tts_wyoming_uses_server_voices(wyoming_server):
    uri, _ = wyoming_server
    _, assigner = create_tts(parse("--tts-output", "out", "--tts-backend", "wyoming", "--tts-url", uri))
    assert assigner.voices == ["ja-a", "ja-multi:s1", "ja-multi:s2"]


def test_create_tts_wyoming_accepts_known_voice_outside_list(wyoming_server):
    uri, _ = wyoming_server
    # 複数話者モデルの既定話者や、言語の絞り込みで一覧に出ない音声も明示すれば使える
    _, assigner = create_tts(parse(
        "--tts-output", "out", "--tts-backend", "wyoming", "--tts-url", uri,
        "--tts-narrator-voice", "ja-multi", "--tts-male-voices", "en-a,ja-a",
    ))
    assert assigner.voices == ["ja-a", "ja-multi:s1", "ja-multi:s2", "ja-multi", "en-a"]
    assert assigner.narrator_voice == "ja-multi"


def test_create_tts_wyoming_explicit_voice_when_language_matches_nothing(wyoming_server):
    uri, _ = wyoming_server
    _, assigner = create_tts(parse(
        "--tts-output", "out", "--tts-backend", "wyoming", "--tts-url", uri, "--tts-language", "fr", "--tts-narrator-voice", "en-a",
    ))
    assert assigner.voices == ["en-a"]


def test_create_tts_wyoming_rejects_unknown_voice(wyoming_server):
    uri, _ = wyoming_server
    with pytest.raises(SystemExit, match = "nope"):
        create_tts(parse("--tts-output", "out", "--tts-backend", "wyoming", "--tts-url", uri, "--tts-voices", "ja-a,nope"))


def test_create_tts_wyoming_without_matching_voices(wyoming_server):
    uri, _ = wyoming_server
    with pytest.raises(SystemExit, match = "--tts-language"):
        create_tts(parse("--tts-output", "out", "--tts-backend", "wyoming", "--tts-url", uri, "--tts-language", "fr"))


def test_create_tts_wyoming_unreachable(unused_tcp_port):
    with pytest.raises(SystemExit, match = "音声一覧を取得できませんでした"):
        create_tts(parse("--tts-output", "out", "--tts-backend", "wyoming", "--tts-url", f"tcp://127.0.0.1:{unused_tcp_port}", "--tts-timeout", "2"))


@pytest.fixture
def unused_tcp_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


# --- ファイル単位の合成 ---

class FakeClient(TtsClient):
    def __init__(self, failing: Sequence[str] = (), fatal: Sequence[str] = ()):
        super().__init__()
        self.failing = failing
        self.fatal = fatal
        self.requests: List[str] = []

    def list_voices(self) -> List[str]:
        return []

    def synthesize(self, text: str, voice: str) -> Audio:
        self.requests.append(text)
        if text in self.fatal:
            raise ConnectionError("down")
        if text in self.failing:
            raise TtsInputError("bad")
        return Audio(16000, 2, 1, b"\x01\x00" * 160)


def novel(*texts: str) -> Novel:
    result = Novel()
    result.sentences = [Sentence(text = t) for t in texts]
    return result


def test_synthesize_file_writes_wav(tmp_path, capsys):
    outfile = tmp_path / "out" / "1.wav"
    failed = synthesize_file(novel("あ", "＊　＊　＊", "い"), outfile, FakeClient(), VoiceAssigner(["n"]), 200, 0.1)
    assert not failed
    assert outfile.exists()
    output = capsys.readouterr().out
    assert "音声を書き出しました" in output
    assert "1行は読み飛ばしました" in output


def test_synthesize_file_reports_partial_failure(tmp_path, capsys):
    outfile = tmp_path / "1.wav"
    failed = synthesize_file(novel("あ", "い", "う"), outfile, FakeClient(failing = ["い"]), VoiceAssigner(["n"]), 200, 0.1)
    assert failed
    assert outfile.exists()
    assert "合成に失敗した1文を除く" in capsys.readouterr().out


def test_synthesize_file_removes_stale_wav_when_nothing_to_read(tmp_path, capsys):
    outfile = tmp_path / "1.wav"
    outfile.write_bytes(b"old")
    failed = synthesize_file(novel("＊　＊　＊", ""), outfile, FakeClient(), VoiceAssigner(["n"]), 200, 0.1)
    assert not failed
    assert not outfile.exists()
    assert "読み上げられる文がない" in capsys.readouterr().out


def test_synthesize_file_all_failed(tmp_path, capsys):
    outfile = tmp_path / "1.wav"
    outfile.write_bytes(b"old")
    client = FakeClient(failing = ["い", "う"])
    assigner = VoiceAssigner(["n"])
    synthesize_file(novel("あ"), tmp_path / "0.wav", client, assigner, 200, 0.1)
    failed = synthesize_file(novel("い", "う"), outfile, client, assigner, 200, 0.1)
    assert failed
    assert not outfile.exists()
    assert "全ての文の合成に失敗" in capsys.readouterr().out


def test_synthesize_file_aborts_on_fatal_error(tmp_path):
    outfile = tmp_path / "1.wav"
    outfile.write_bytes(b"old")
    with pytest.raises(SystemExit, match = "中断"):
        synthesize_file(novel("あ", "い"), outfile, FakeClient(fatal = ["い"]), VoiceAssigner(["n"]), 200, 0.1)
    # 中断した場合は書きかけの音声で既存のファイルを上書きしない
    assert outfile.read_bytes() == b"old"


def test_synthesize_file_aborts_when_output_path_is_directory(tmp_path):
    outfile = tmp_path / "1.wav"
    outfile.mkdir()
    with pytest.raises(SystemExit, match = "中断"):
        synthesize_file(novel("＊　＊　＊"), outfile, FakeClient(), VoiceAssigner(["n"]), 200, 0.1)
    with pytest.raises(SystemExit, match = "中断"):
        synthesize_file(novel("あ"), outfile, FakeClient(), VoiceAssigner(["n"]), 200, 0.1)


def test_synthesize_file_aborts_on_unsupported_audio(tmp_path):
    class BrokenClient(FakeClient):
        def synthesize(self, text: str, voice: str) -> Audio:
            return Audio(8000, 2, 1, b"\x01\x00\x02") if text == "い" else super().synthesize(text, voice)

    outfile = tmp_path / "1.wav"
    with pytest.raises(SystemExit, match = "中断"):
        synthesize_file(novel("あ", "い"), outfile, BrokenClient(), VoiceAssigner(["n"]), 200, 0.1)
    assert not outfile.exists()


# --- CLI全体 ---

class FakeAi:
    """1ファイル目と2ファイル目で登場人物の正式名が揺れる話者推測"""

    def __init__(self, host: str, port: str, model: str):
        self.calls = 0

    def get_narrators(self, novel, narrators):
        self.calls += 1
        if self.calls == 1:
            return [Narrator(name = "田中花子", portrait = "", aliases = ["花子"])]
        return [Narrator(name = "花子", portrait = ""), Narrator(name = "次郎", portrait = "")]

    def set_estimation_narrator(self, novel, *args):
        # 「名前:セリフ」の行をその登場人物のセリフとみなす
        for sentence in novel.sentences:
            name, _, text = sentence.text.partition(":")
            sentence.narrator = next((n for n in novel.narrators if n.name == name), None) if text else None


def run_main(monkeypatch, tmp_path, client: TtsClient, assigner: Optional[VoiceAssigner] = None, args: Sequence[str] = ()) -> List[str]:
    data = tmp_path / "data"
    data.mkdir(exist_ok = True)
    (data / "1.txt").write_text("あ\n＊　＊　＊\n田中花子:い\n", encoding = "utf-8")
    (data / "2.txt").write_text("花子:う\n次郎:え\n", encoding = "utf-8")
    (data / ".DS_Store").write_bytes(b"\x00\x01")
    monkeypatch.setattr(main, "Ai", FakeAi)
    monkeypatch.setattr(main, "create_tts", lambda args: (client, assigner or VoiceAssigner(["n"])))
    monkeypatch.setattr("sys.argv", ["main.py", "--tts-output", str(tmp_path / "out"), "--tts-pause", "0", *args, str(data)])
    main.main()
    return sorted(p.name for p in (tmp_path / "out").iterdir())


def test_main_writes_wav_per_file(monkeypatch, tmp_path):
    assert run_main(monkeypatch, tmp_path, FakeClient()) == ["1.wav", "2.wav"]
    with wave.open(str(tmp_path / "out" / "1.wav"), "rb") as wav:
        assert wav.getnframes() == 160 * 2


def test_main_exits_with_error_when_some_sentences_failed(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match = "2.txt"):
        run_main(monkeypatch, tmp_path, FakeClient(failing = ["次郎:え"]))
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["1.wav", "2.wav"]


def test_main_keeps_voice_when_name_changes_across_files(monkeypatch, tmp_path):
    assigner = VoiceAssigner(["n", "a", "b"])
    run_main(monkeypatch, tmp_path, FakeClient(), assigner)
    assert assigner.assigned == {"ナレーター": "n", "田中花子": "a", "花子": "a", "次郎": "b"}


def test_main_rejects_undecodable_input(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "0.txt").write_bytes("あ".encode("shift_jis"))
    with pytest.raises(SystemExit, match = "0.txt"):
        run_main(monkeypatch, tmp_path, FakeClient())


def test_main_passes_max_chars_to_synthesis(monkeypatch, tmp_path):
    client = FakeClient()
    run_main(monkeypatch, tmp_path, client, args = ["--tts-max-chars", "3"])
    assert len(client.requests) > 4
    assert all(len(text) <= 3 for text in client.requests)


def test_main_aborts_without_writing_wav_after_consecutive_failures(monkeypatch, tmp_path):
    monkeypatch.setattr(tts, "MAX_CONSECUTIVE_FAILURES", 2)
    client = FakeClient(failing = ["田中花子:い", "花子:う"])
    with pytest.raises(SystemExit, match = "連続"):
        run_main(monkeypatch, tmp_path, client)
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["1.wav"]


@pytest.mark.parametrize("output", ["", "file"])
def test_main_rejects_invalid_output(monkeypatch, tmp_path, output):
    (tmp_path / "file").write_text("")
    (tmp_path / "data").mkdir()
    monkeypatch.setattr(main, "Ai", FakeAi)
    target = output and str(tmp_path / output)
    monkeypatch.setattr("sys.argv", ["main.py", "--tts-output", target, str(tmp_path / "data")])
    with pytest.raises(SystemExit) as e:
        main.main()
    assert e.value.code != 0


def test_main_with_openai_client(monkeypatch, tmp_path, speech_server):
    url, requests = speech_server
    data = tmp_path / "data"
    data.mkdir()
    (data / "1.txt").write_text("あ\n＊　＊　＊\n田中花子:い\n", encoding = "utf-8")
    (data / "2.txt").write_text("花子:bad\n次郎:え\n", encoding = "utf-8")
    monkeypatch.setattr(main, "Ai", FakeAi)
    monkeypatch.setattr("sys.argv", [
        "main.py", "--tts-output", str(tmp_path / "out"), "--tts-url", url, "--tts-model", "m",
        "--tts-voices", "n,a,b", "--tts-pause", "0", str(data),
    ])
    with pytest.raises(SystemExit, match = "2.txt"):
        main.main()
    assert [(r["input"], r["voice"]) for r in requests] == [
        ("あ", "n"), ("田中花子:い", "a"), ("花子:bad", "a"), ("次郎:え", "b"),
    ]
    with wave.open(str(tmp_path / "out" / "2.wav"), "rb") as wav:
        assert wav.getnframes() == 240


def test_main_checks_input_folder_before_creating_output(monkeypatch, tmp_path):
    missing = tmp_path / "missing"
    monkeypatch.setattr(main, "Ai", FakeAi)
    monkeypatch.setattr("sys.argv", ["main.py", "--tts-output", str(missing / "out"), str(missing)])
    with pytest.raises(SystemExit, match = "入力フォルダ"):
        main.main()
    assert not missing.exists()


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason = "root は書き込み権限を無視する")
def test_main_rejects_unwritable_output_before_processing(monkeypatch, tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "1.txt").write_text("あ", encoding = "utf-8")
    output = tmp_path / "out"
    output.mkdir()
    output.chmod(0o555)

    def fail(*args):
        raise AssertionError("出力先の確認前に話者推測を始めた")

    monkeypatch.setattr(main, "Ai", fail)
    monkeypatch.setattr("sys.argv", ["main.py", "--tts-output", str(output), str(tmp_path / "data")])
    try:
        with pytest.raises(SystemExit, match = "書き込めません"):
            main.main()
    finally:
        output.chmod(0o755)


def fail_if_called(*args):
    raise AssertionError("入力・出力先の確認前に話者推測を始めた")


def test_main_validates_all_inputs_before_processing(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "1.txt").write_text("あ", encoding = "utf-8")
    (data / "2.txt").write_bytes("い".encode("shift_jis"))
    monkeypatch.setattr(main, "Ai", fail_if_called)
    monkeypatch.setattr("sys.argv", ["main.py", "--tts-output", str(tmp_path / "out"), str(data)])
    with pytest.raises(SystemExit, match = "2.txt"):
        main.main()


def test_main_rejects_conflicting_output_path_before_processing(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "1.txt").write_text("あ", encoding = "utf-8")
    (data / "2.txt").write_text("い", encoding = "utf-8")
    (tmp_path / "out" / "2.wav").mkdir(parents = True)
    monkeypatch.setattr(main, "Ai", fail_if_called)
    monkeypatch.setattr("sys.argv", ["main.py", "--tts-output", str(tmp_path / "out"), str(data)])
    with pytest.raises(SystemExit, match = "2.wav"):
        main.main()


@pytest.mark.parametrize("names", [
    [".txt", ".txt.txt"],
    ["A.txt", "a.txt"],
    ["\u30ac.txt", "\u30ab\u3099.txt"],  # NFC の「ガ」と NFD の「カ + 濁点」
])
def test_check_output_dir_rejects_duplicate_output_names(tmp_path, names):
    with pytest.raises(SystemExit, match = "出力先が同じ"):
        main.check_output_dir(tmp_path / "out", names)


def test_main_rejects_duplicate_output_names_before_processing(monkeypatch, tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / ".txt").write_text("あ", encoding = "utf-8")
    (data / ".txt.txt").write_text("い", encoding = "utf-8")
    monkeypatch.setattr(main, "Ai", fail_if_called)
    monkeypatch.setattr("sys.argv", ["main.py", "--tts-output", str(tmp_path / "out"), str(data)])
    with pytest.raises(SystemExit, match = "出力先が同じ"):
        main.main()


def test_main_with_wyoming_client(monkeypatch, tmp_path, wyoming_server):
    uri, state = wyoming_server
    data = tmp_path / "data"
    data.mkdir()
    (data / "1.txt").write_text("あ\n田中花子:い\n", encoding = "utf-8")
    (data / "2.txt").write_text("次郎:う\n花子:error\n", encoding = "utf-8")
    monkeypatch.setattr(main, "Ai", FakeAi)
    monkeypatch.setattr("sys.argv", [
        "main.py", "--tts-output", str(tmp_path / "out"), "--tts-backend", "wyoming", "--tts-url", uri,
        "--tts-language", "ja-JP", "--tts-narrator-voice", "ja-multi", "--tts-pause", "0", str(data),
    ])
    with pytest.raises(SystemExit, match = "2.txt"):
        main.main()
    voices = [(r.text, r.voice.name, r.voice.speaker) for r in state["requests"]]
    assert voices == [
        ("あ", "ja-multi", None),
        ("田中花子:い", "ja-a", None),
        ("次郎:う", "ja-multi", "s1"),
        ("花子:error", "ja-a", None),
    ]
    assert sorted(p.name for p in (tmp_path / "out").iterdir()) == ["1.wav", "2.wav"]
