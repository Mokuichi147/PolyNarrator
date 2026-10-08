import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, List, Optional

import openai
import pytest

from models.gender import Gender
from models.narrator import Narrator
from models.sentence import Sentence
from tts import (
    MAX_CONSECUTIVE_FAILURES,
    Audio,
    OpenAiSpeechClient,
    TtsClient,
    TtsInputError,
    VoiceAssigner,
    WyomingClient,
    is_readable,
    split_for_tts,
    synthesize_sentences,
    write_wav,
)


def narrator(name: str, gender: Optional[Gender] = None, aliases: Optional[List[str]] = None) -> Narrator:
    return Narrator(name = name, portrait = "", gender = gender, aliases = aliases or [])


def pcm(rate: int = 16000, frames: int = 160, value: int = 1000) -> Audio:
    return Audio(rate, 2, 1, value.to_bytes(2, "little", signed = True) * frames)


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
    assert assigner.assign(narrator("ナレーター")) == "n"
    assert assigner.assign(narrator("A")) == "a"
    assert assigner.assign(narrator("B")) == "b"
    assert assigner.assign(narrator("C")) == "a"
    assert assigner.assign(narrator("A")) == "a"


def test_voice_assigner_single_voice_is_shared_with_narrator():
    assigner = VoiceAssigner(["n"])
    assert assigner.assign(narrator("A")) == "n"


def test_voice_assigner_gender_voices():
    assigner = VoiceAssigner(["n", "m", "f", "x"], narrator_voice = "x", male_voices = ["m"], female_voices = ["f"])
    assert assigner.assign(None) == "x"
    assert assigner.assign(narrator("太郎", Gender.MALE)) == "m"
    assert assigner.assign(narrator("花子", Gender.FEMALE)) == "f"
    assert assigner.assign(narrator("不明")) == "n"
    # 性別ごとの指定がない性別は全音声から選ぶ(使用数が同じなら一覧の先頭)
    assert assigner.assign(narrator("その他", Gender.OTHER)) == "n"


def test_voice_assigner_matches_aliases_across_files():
    assigner = VoiceAssigner(["n", "a", "b", "c"])
    voice = assigner.assign(narrator("モニカ・エヴァレット", aliases = ["モニカ", "沈黙の魔女"]))
    # 後のファイルで正式名が別名の方に揺れた場合
    assert assigner.assign(narrator("モニカ")) == voice
    # 以前の正式名を別名に持つ場合
    assert assigner.assign(narrator("沈黙の魔女モニカ", aliases = ["モニカ・エヴァレット"])) == voice
    # 別名同士が一致するだけでは同一人物とみなさない
    assert assigner.assign(narrator("ルイス", aliases = ["沈黙の魔女"])) != voice


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


# --- 文ごとの合成 ---

class FakeClient(TtsClient):
    def __init__(self, fail: Callable[[str], Optional[Exception]] = lambda text: None):
        super().__init__()
        self.fail = fail
        self.requests: List[tuple[str, str]] = []

    def list_voices(self) -> List[str]:
        return []

    def synthesize(self, text: str, voice: str) -> Audio:
        self.requests.append((text, voice))
        error = self.fail(text)
        if error is not None:
            raise error
        return pcm()


def sentences(*texts: str, speaker: Optional[Narrator] = None) -> List[Sentence]:
    return [Sentence(text = t, narrator = speaker) for t in texts]


def test_synthesize_sentences_skips_unreadable_without_request():
    client = FakeClient()
    result = synthesize_sentences(sentences("", "あ", "＊　＊　＊", "「………………」", "い"), client, VoiceAssigner(["n"]))
    assert [t for t, _ in client.requests] == ["あ", "い"]
    assert len(result.segments) == 2
    assert result.skipped == 2
    assert result.failures == []


def test_synthesize_sentences_records_input_errors_and_continues():
    client = FakeClient(lambda text: TtsInputError("bad") if text == "い" else None)
    result = synthesize_sentences(sentences("あ", "い", "う"), client, VoiceAssigner(["n"]))
    assert len(result.segments) == 2
    assert result.failures == [("い", "bad")]


def test_synthesize_sentences_aborts_after_consecutive_input_errors():
    client = FakeClient(lambda text: TtsInputError("bad") if text != "ok" else None)
    with pytest.raises(RuntimeError, match = "連続"):
        synthesize_sentences(sentences("ok", *["あ"] * 10), client, VoiceAssigner(["n"]))
    assert len(client.requests) == 1 + MAX_CONSECUTIVE_FAILURES


@pytest.mark.parametrize("failing", ["一文目。", "二文目。", "三文目。"])
def test_synthesize_sentences_drops_whole_line_when_a_chunk_fails(failing):
    client = FakeClient(lambda text: TtsInputError("bad") if text == failing else None)
    line = "一文目。二文目。三文目。"
    result = synthesize_sentences(sentences("ok", line, "ok"), client, VoiceAssigner(["n"]), max_chars = 4)
    assert len(result.segments) == 2
    assert result.failures == [(line, "bad")]


def test_synthesize_sentences_partial_line_success_counts_as_voice_success():
    # 先頭の塊が成功していれば、その音声は使えるとみなして行の失敗として続行する
    client = FakeClient(lambda text: TtsInputError("bad") if text == "二文目。" else None)
    result = synthesize_sentences(sentences("一文目。二文目。", "三文目。"), client, VoiceAssigner(["n"]), max_chars = 4)
    assert result.failures == [("一文目。二文目。", "bad")]
    assert len(result.segments) == 1


def test_synthesize_sentences_counts_split_line_as_one_failure():
    client = FakeClient(lambda text: TtsInputError("bad") if text.startswith("い") else None)
    lines = ["い。" * 4] * (MAX_CONSECUTIVE_FAILURES - 1)
    result = synthesize_sentences(sentences("ok", *lines, "ok"), client, VoiceAssigner(["n"]), max_chars = 2)
    assert len(result.failures) == MAX_CONSECUTIVE_FAILURES - 1


def test_synthesize_sentences_counts_consecutive_failures_across_files():
    client = FakeClient(lambda text: TtsInputError("bad") if text != "ok" else None)
    assigner = VoiceAssigner(["n"])
    result = synthesize_sentences(sentences("ok", *["あ"] * (MAX_CONSECUTIVE_FAILURES - 1)), client, assigner)
    assert len(result.failures) == MAX_CONSECUTIVE_FAILURES - 1
    with pytest.raises(RuntimeError, match = "連続"):
        synthesize_sentences(sentences("い"), client, assigner)


def test_synthesize_sentences_aborts_when_voice_never_succeeded():
    # モデル名や音声名の誤りで 400 が返る場合を想定し、成功したことのない音声での失敗は中断する
    client = FakeClient(lambda text: TtsInputError("unknown voice") if text == "B" else None)
    speakers = [Sentence(text = "A", narrator = narrator("A")), Sentence(text = "B", narrator = narrator("B"))]
    with pytest.raises(RuntimeError, match = "最初の合成"):
        synthesize_sentences(speakers, client, VoiceAssigner(["n", "a", "b"]))


def test_synthesize_sentences_remembers_succeeded_voices_across_calls():
    client = FakeClient(lambda text: TtsInputError("bad") if text == "い" else None)
    assigner = VoiceAssigner(["n"])
    synthesize_sentences(sentences("あ"), client, assigner)
    result = synthesize_sentences(sentences("い", "う"), client, assigner)
    assert result.failures == [("い", "bad")]


def test_synthesize_sentences_propagates_other_errors():
    client = FakeClient(lambda text: ConnectionError("down"))
    with pytest.raises(ConnectionError):
        synthesize_sentences(sentences("あ", "い"), client, VoiceAssigner(["n"]))
    assert len(client.requests) == 1


def test_synthesize_sentences_splits_long_text_with_same_voice():
    client = FakeClient()
    speaker = narrator("A")
    result = synthesize_sentences(sentences("一文目。二文目。", speaker = speaker), client, VoiceAssigner(["n", "a"]), max_chars = 4)
    assert client.requests == [("一文目。", "a"), ("二文目。", "a")]
    assert len(result.segments) == 2


# --- WAV出力 ---

def read_wav(path: Path) -> tuple[int, int, int, bytes]:
    with wave.open(str(path), "rb") as wav:
        return wav.getframerate(), wav.getsampwidth(), wav.getnchannels(), wav.readframes(wav.getnframes())


def test_write_wav_inserts_pause(tmp_path):
    path = tmp_path / "out" / "a.wav"
    write_wav(path, [pcm(frames = 100), pcm(frames = 50)], pause = 0.01)
    rate, width, channels, data = read_wav(path)
    assert (rate, width, channels) == (16000, 2, 1)
    assert len(data) == (100 + 160 + 50) * 2
    assert data[200:200 + 320] == bytes(320)
    assert not (tmp_path / "out" / "a.wav.tmp").exists()


def test_write_wav_converts_to_first_format(tmp_path):
    path = tmp_path / "a.wav"
    stereo = Audio(16000, 2, 2, b"\x01\x00\x02\x00" * 100)
    write_wav(path, [pcm(rate = 24000, frames = 240), pcm(rate = 16000, frames = 160), stereo], pause = 0)
    rate, width, channels, data = read_wav(path)
    assert (rate, width, channels) == (24000, 2, 1)
    # 16kHz の160フレーム(0.01秒)は24kHzでおよそ240フレームになる
    assert abs(len(data) // 2 - (240 + 240 + 150)) <= 4


def test_write_wav_8bit_silence_is_unsigned_center(tmp_path):
    path = tmp_path / "a.wav"
    segment = Audio(8000, 1, 1, b"\x90" * 80)
    write_wav(path, [segment, segment], pause = 0.01)
    _, width, _, data = read_wav(path)
    assert width == 1
    assert data == b"\x90" * 80 + b"\x80" * 80 + b"\x90" * 80


def test_write_wav_keeps_existing_file_on_conversion_error(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"old")
    broken = Audio(8000, 2, 1, b"\x01\x00\x02")
    with pytest.raises(Exception):
        write_wav(path, [pcm(), broken], pause = 0)
    assert path.read_bytes() == b"old"
    assert not (tmp_path / "a.wav.tmp").exists()


def test_write_wav_rejects_misaligned_pcm(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"old")
    with pytest.raises(RuntimeError, match = "不正な音声"):
        write_wav(path, [pcm(), Audio(16000, 2, 1, b"\x01\x00\x02")], pause = 0)
    assert path.read_bytes() == b"old"


def test_write_wav_without_segments_writes_nothing(tmp_path):
    path = tmp_path / "a.wav"
    write_wav(path, [], pause = 0.3)
    assert not path.exists()


def test_write_wav_rejects_negative_pause(tmp_path):
    with pytest.raises(ValueError):
        write_wav(tmp_path / "a.wav", [pcm()], pause = -1)
    assert list(tmp_path.iterdir()) == []


# --- OpenAI互換API ---

def wav_bytes(rate: int = 24000, frames: int = 240) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setframerate(rate)
        wav.setsampwidth(2)
        wav.setnchannels(1)
        wav.writeframes(b"\x01\x00" * frames)
    return buffer.getvalue()


@pytest.fixture
def speech_server():
    requests: List[dict] = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append({"path": self.path, **body})
            statuses = {"bad": 400, "unprocessable": 422, "unauthorized": 401}
            if body["model"] == "unknown-model":
                statuses = {body["input"]: 400}
            status = statuses.get(body["input"], 200)
            if status == 200:
                payload, content_type = wav_bytes(frames = 0 if body["input"] == "zero" else 240), "audio/wav"
            else:
                payload = json.dumps({"error": {"message": body["input"], "type": "invalid_request_error"}}).encode()
                content_type = "application/json"
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target = server.serve_forever, daemon = True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/v1", requests
    server.shutdown()
    server.server_close()


def test_openai_client_synthesize(speech_server):
    url, requests = speech_server
    audio = OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("こんにちは", "alloy")
    assert (audio.rate, audio.width, audio.channels, len(audio.data)) == (24000, 2, 1, 480)
    assert requests == [{"path": "/v1/audio/speech", "model": "tts-1", "voice": "alloy", "input": "こんにちは", "response_format": "wav"}]


@pytest.mark.parametrize("text", ["bad", "unprocessable"])
def test_openai_client_bad_request_is_input_error(speech_server, text):
    url, _ = speech_server
    with pytest.raises(TtsInputError):
        OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize(text, "alloy")


def test_openai_client_empty_wav_is_not_input_error(speech_server):
    url, _ = speech_server
    with pytest.raises(RuntimeError, match = "空の音声"):
        OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("zero", "alloy")


def test_openai_client_unknown_model_aborts(speech_server):
    url, _ = speech_server
    client = OpenAiSpeechClient(url, "unknown-model", "key", 10)
    with pytest.raises(RuntimeError, match = "最初の合成"):
        synthesize_sentences(sentences("こんにちは"), client, VoiceAssigner(["alloy"]))


def test_openai_client_auth_error_is_not_input_error(speech_server):
    url, _ = speech_server
    with pytest.raises(openai.AuthenticationError):
        OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("unauthorized", "alloy")


# --- Wyoming ---

def test_wyoming_lists_voices_by_language(wyoming_server):
    uri, _ = wyoming_server
    assert WyomingClient(uri, 5, "ja").list_voices() == ["ja-a", "ja-multi:s1", "ja-multi:s2"]
    assert WyomingClient(uri, 5, "").list_voices() == ["ja-a", "ja-multi:s1", "ja-multi:s2", "en-a"]


@pytest.mark.parametrize(("language", "expected"), [
    ("ja-JP", ["ja-a", "ja-multi:s1", "ja-multi:s2"]),
    ("ja_jp", ["ja-a", "ja-multi:s1", "ja-multi:s2"]),
    ("en", ["en-a"]),
    ("j", []),
])
def test_wyoming_language_matches_region(wyoming_server, language, expected):
    uri, _ = wyoming_server
    assert WyomingClient(uri, 5, language).list_voices() == expected


def test_wyoming_unknown_voices_ignores_language_filter(wyoming_server):
    uri, _ = wyoming_server
    client = WyomingClient(uri, 5, "ja")
    voices = ["ja-a", "en-a", "ja-multi", "ja-multi:s2", "ja-multi:s3", "ja-missing", "ja-other", "ja-second", "nope"]
    # 未インストールの音声と、合成時に選べない2番目以降のプログラムの音声は使えない
    assert client.unknown_voices(voices) == ["ja-multi:s3", "ja-missing", "ja-other", "ja-second", "nope"]


def test_wyoming_synthesize_with_speaker(wyoming_server):
    uri, state = wyoming_server
    audio = WyomingClient(uri, 5, "ja").synthesize("こんにちは", "ja-multi:s2")
    assert (audio.rate, audio.width, audio.channels, len(audio.data)) == (22050, 2, 1, 600)
    request = state["requests"][0]
    assert (request.text, request.voice.name, request.voice.speaker) == ("こんにちは", "ja-multi", "s2")


def test_wyoming_synthesis_error_is_input_error(wyoming_server):
    uri, _ = wyoming_server
    with pytest.raises(TtsInputError, match = "synthesis failed"):
        WyomingClient(uri, 5, "ja").synthesize("error", "ja-a")


def test_wyoming_error_after_success_is_recorded(wyoming_server):
    uri, _ = wyoming_server
    client = WyomingClient(uri, 5, "ja")
    result = synthesize_sentences(sentences("こんにちは", "error", "さようなら"), client, VoiceAssigner(["ja-a"]))
    assert len(result.segments) == 2
    assert result.failures == [("error", "synthesis failed")]


def test_wyoming_error_before_success_aborts(wyoming_server):
    uri, _ = wyoming_server
    with pytest.raises(RuntimeError, match = "最初の合成"):
        synthesize_sentences(sentences("error"), WyomingClient(uri, 5, "ja"), VoiceAssigner(["ja-a"]))


@pytest.mark.parametrize(("text", "message"), [("empty", "音声が返されませんでした"), ("empty-chunk", "空の音声"), ("odd", "不正な音声")])
def test_wyoming_broken_audio_is_not_input_error(wyoming_server, text, message):
    uri, _ = wyoming_server
    with pytest.raises(RuntimeError, match = message):
        WyomingClient(uri, 5, "ja").synthesize(text, "ja-a")


def test_wyoming_disconnect_is_not_input_error(wyoming_server):
    uri, _ = wyoming_server
    with pytest.raises(RuntimeError, match = "切断"):
        WyomingClient(uri, 5, "ja").synthesize("disconnect", "ja-a")


def test_wyoming_describe_error(wyoming_server):
    uri, state = wyoming_server
    state["describe"] = "error"
    with pytest.raises(RuntimeError, match = "describe failed"):
        WyomingClient(uri, 5, "ja")


def test_wyoming_describe_disconnect(wyoming_server):
    uri, state = wyoming_server
    state["describe"] = "none"
    with pytest.raises(RuntimeError, match = "切断"):
        WyomingClient(uri, 5, "ja")
