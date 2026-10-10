import dataclasses
import wave
from pathlib import Path
from typing import List

import openai
import pytest
from typesafe_sdk import ChoiceAnswer, TypeSafeAPIError, TypeSafeError

from conftest import wav_bytes
from polynarrator.adapters import openai_speech, wav, wyoming
from polynarrator.adapters.jev import JevSpeakerEstimator
from polynarrator.adapters.llm import NarratorResponse, Narrators, parse_json
from polynarrator.adapters.openai_speech import OpenAiSpeechClient
from polynarrator.adapters.text_files import list_text_files, read_novel
from polynarrator.adapters.wav import OutputDirectoryError, WavDirectory, wav_data_size, write_wav
from polynarrator.adapters.wyoming import WyomingClient
from polynarrator.application import speech_synthesis
from polynarrator.application.ports import (
    ContextTooLongError,
    Presenter,
    SpeakerEstimationError,
    SpeakerGuess,
    SpeechSynthesizer,
    TtsInputError,
)
from polynarrator.application.speech_synthesis import SentenceSynthesizer
from polynarrator.domain.audio import Audio
from polynarrator.domain.character import Character, Gender, narrator
from polynarrator.domain.novel import Sentence
from polynarrator.domain.voice_assigner import VoiceAssigner


def pcm(rate: int = 16000, frames: int = 160, value: int = 1000) -> Audio:
    return Audio(rate, 2, 1, value.to_bytes(2, "little", signed = True) * frames)


def sentences(*texts: str) -> List[Sentence]:
    return [Sentence(text = t) for t in texts]


def synthesizer(client: SpeechSynthesizer, voice: str) -> SentenceSynthesizer:
    return SentenceSynthesizer(client, VoiceAssigner([voice]), Presenter())


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
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["a.wav"]


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


def test_write_wav_keeps_existing_file_on_error_while_writing(tmp_path, monkeypatch):
    class FailingConverter(wav.AudioChunkConverter):
        calls = 0

        def convert(self, chunk):
            FailingConverter.calls += 1
            if FailingConverter.calls == 2:
                # 一時ファイルに1つ目の音声を書いた後で失敗させる
                assert len([p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]) == 1
                raise RuntimeError("convert failed")
            return super().convert(chunk)

    monkeypatch.setattr(wav, "AudioChunkConverter", FailingConverter)
    path = tmp_path / "a.wav"
    path.write_bytes(b"old")
    with pytest.raises(RuntimeError, match = "convert failed"):
        write_wav(path, [pcm(), pcm()], pause = 0)
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


def test_write_wav_keeps_existing_file_on_invalid_audio(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"old")
    broken = Audio(8000, 2, 1, b"\x01\x00\x02")
    with pytest.raises(Exception):
        write_wav(path, [pcm(), broken], pause = 0)
    assert path.read_bytes() == b"old"
    assert [p.name for p in tmp_path.iterdir()] == ["a.wav"]


def test_write_wav_with_long_file_name(tmp_path):
    path = tmp_path / ("あ" * 80 + ".wav")  # UTF-8 で 244 バイト
    write_wav(path, [pcm()], pause = 0)
    assert [p.name for p in tmp_path.iterdir()] == [path.name]


@pytest.mark.parametrize("audio", [
    Audio(0, 2, 1, b"\x01\x00"),
    Audio(1_000_000_000, 2, 1, b"\x01\x00"),
    Audio(16000, 8, 1, b"\x00" * 8),
    Audio(16000, 2, 100, b"\x00" * 200),
    Audio(16000, 2, 4, b"\x00" * 8),
])
def test_validate_audio_rejects_abnormal_format(tmp_path, audio):
    with pytest.raises(RuntimeError, match = "不正な音声"):
        write_wav(tmp_path / "a.wav", [audio, audio], pause = 60)
    assert list(tmp_path.iterdir()) == []


def test_write_wav_long_pause_is_written_in_blocks(tmp_path):
    path = tmp_path / "a.wav"
    write_wav(path, [pcm(frames = 10), pcm(frames = 10)], pause = 2.5)
    _, _, _, data = read_wav(path)
    assert len(data) == (10 + 40000 + 10) * 2
    assert data[20:-20] == bytes(80000)


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


@pytest.mark.parametrize("segments", [[], [pcm()]])
def test_write_wav_rejects_negative_pause(tmp_path, segments):
    with pytest.raises(ValueError):
        write_wav(tmp_path / "a.wav", segments, pause = -1)
    assert list(tmp_path.iterdir()) == []


# --- OpenAI互換API ---

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


def test_wav_data_size():
    content = wav_bytes(frames = 10)
    assert wav_data_size(content) == 20
    assert wav_data_size(content[:20]) is None


def test_openai_client_truncated_wav_is_not_input_error(speech_server):
    url, _ = speech_server
    with pytest.raises(RuntimeError, match = "途中で切れた"):
        OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("truncated", "alloy")


def test_openai_client_truncated_wav_near_size_limit(speech_server):
    url, _ = speech_server
    with pytest.raises(RuntimeError, match = "途中で切れた"):
        OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("near-limit", "alloy")


def test_openai_client_accepts_streaming_wav_header(speech_server):
    url, _ = speech_server
    audio = OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("streaming", "alloy")
    assert len(audio.data) == 480


def test_openai_client_invalid_wav_is_not_input_error(speech_server):
    url, _ = speech_server
    with pytest.raises(wave.Error):
        OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("notwav", "alloy")


def test_openai_client_rejects_too_large_audio(speech_server, monkeypatch):
    url, _ = speech_server
    monkeypatch.setattr(openai_speech, "MAX_AUDIO_BYTES", 100)
    monkeypatch.setattr(openai_speech, "WAV_HEADER_ALLOWANCE", 44)
    with pytest.raises(RuntimeError, match = "上限"):
        OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("こんにちは", "alloy")


def test_openai_client_rejects_pcm_over_limit_within_header_allowance(speech_server, monkeypatch):
    url, _ = speech_server
    # WAV全体(44 + 480バイト)はヘッダーの許容量の範囲に収まるが、PCM(480バイト)は上限を超える
    monkeypatch.setattr(openai_speech, "MAX_AUDIO_BYTES", 479)
    monkeypatch.setattr(openai_speech, "WAV_HEADER_ALLOWANCE", 100)
    with pytest.raises(RuntimeError, match = "上限"):
        OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("こんにちは", "alloy")


def test_openai_client_empty_wav_is_not_input_error(speech_server):
    url, _ = speech_server
    with pytest.raises(RuntimeError, match = "空の音声"):
        OpenAiSpeechClient(url, "tts-1", "key", 10).synthesize("zero", "alloy")


def test_openai_client_unknown_model_aborts(speech_server):
    url, _ = speech_server
    client = OpenAiSpeechClient(url, "unknown-model", "key", 10)
    with pytest.raises(RuntimeError, match = "最初の合成"):
        synthesizer(client, "alloy").synthesize(sentences("こんにちは"))


def test_openai_client_consecutive_bad_requests_across_files(speech_server, monkeypatch):
    url, requests = speech_server
    monkeypatch.setattr(speech_synthesis, "MAX_CONSECUTIVE_FAILURES", 2)
    client = OpenAiSpeechClient(url, "tts-1", "key", 10)
    sentence_synthesizer = synthesizer(client, "alloy")
    result = sentence_synthesizer.synthesize(sentences("こんにちは", "bad"))
    assert result.failures == [("bad", result.failures[0][1])]
    with pytest.raises(RuntimeError, match = "連続"):
        sentence_synthesizer.synthesize(sentences("bad", "さようなら"))
    assert [r["input"] for r in requests] == ["こんにちは", "bad", "bad"]


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


def test_wyoming_converts_chunks_to_first_format(wyoming_server):
    uri, _ = wyoming_server
    audio = WyomingClient(uri, 5, "ja").synthesize("mixed", "ja-a")
    assert (audio.rate, audio.width, audio.channels) == (22050, 2, 1)
    # 続く4つのチャンク(11025Hz ステレオ 各0.1秒)も 22050Hz モノラルに変換され、
    # 変換器の状態を引き継ぐためチャンク境界でフレーム数がずれない
    assert abs(len(audio.data) // 2 - (2205 + 2204 * 4)) <= 2


def test_wyoming_rejects_abnormal_later_chunk(wyoming_server):
    uri, _ = wyoming_server
    with pytest.raises(RuntimeError, match = "不正な音声"):
        WyomingClient(uri, 5, "ja").synthesize("bad-second", "ja-a")


def test_wyoming_error_after_partial_audio_is_input_error(wyoming_server):
    uri, _ = wyoming_server
    with pytest.raises(TtsInputError, match = "failed midway"):
        WyomingClient(uri, 5, "ja").synthesize("partial-error", "ja-a")


def test_wyoming_uses_first_installed_program(wyoming_server):
    uri, state = wyoming_server
    programs = state["info"].tts
    state["info"] = dataclasses.replace(state["info"], tts = [dataclasses.replace(programs[0], installed = False), *programs[1:]])
    assert WyomingClient(uri, 5, "ja").list_voices() == ["ja-second"]


def test_wyoming_synthesizes_with_first_installed_program_voice(wyoming_server):
    uri, state = wyoming_server
    programs = state["info"].tts
    state["info"] = dataclasses.replace(state["info"], tts = [dataclasses.replace(programs[0], installed = False), *programs[1:]])
    client = WyomingClient(uri, 5, "ja")
    client.synthesize("こんにちは", "ja-second")
    assert state["requests"][-1].voice.name == "ja-second"


def test_wyoming_rejects_too_large_audio(wyoming_server, monkeypatch):
    uri, _ = wyoming_server
    monkeypatch.setattr(wyoming, "MAX_AUDIO_BYTES", 500)
    converted: List[int] = []
    original = wyoming.AudioChunkConverter.convert

    def convert(self, chunk):
        converted.append(len(chunk.audio))
        return original(self, chunk)

    monkeypatch.setattr(wyoming.AudioChunkConverter, "convert", convert)
    with pytest.raises(RuntimeError, match = "上限"):
        WyomingClient(uri, 5, "ja").synthesize("こんにちは", "ja-a")
    # 200バイトのチャンクを2つ変換した後、3つ目は変換する前に拒否する
    assert converted == [200, 200]


def test_wyoming_checks_actual_size_after_conversion(wyoming_server, monkeypatch):
    uri, _ = wyoming_server

    class GrowingConverter(wyoming.AudioChunkConverter):
        def convert(self, chunk):
            # 見積もりより大きな結果を返す変換
            converted = super().convert(chunk)
            return dataclasses.replace(converted, audio = converted.audio * 2)

    monkeypatch.setattr(wyoming, "AudioChunkConverter", GrowingConverter)
    monkeypatch.setattr(wyoming, "MAX_AUDIO_BYTES", 500)
    with pytest.raises(RuntimeError, match = "上限"):
        WyomingClient(uri, 5, "ja").synthesize("こんにちは", "ja-a")


def test_wyoming_rejects_audio_too_large_after_conversion(wyoming_server, monkeypatch):
    uri, _ = wyoming_server
    # 「mixed」の後続チャンク(11025Hz ステレオ)は 22050Hz モノラルへの変換で同じ大きさになる
    monkeypatch.setattr(wyoming, "MAX_AUDIO_BYTES", 4410 + 4408 * 2)
    with pytest.raises(RuntimeError, match = "上限"):
        WyomingClient(uri, 5, "ja").synthesize("mixed", "ja-a")


def test_wyoming_without_installed_program(wyoming_server):
    uri, state = wyoming_server
    state["info"] = dataclasses.replace(state["info"], tts = [])
    assert WyomingClient(uri, 5, "ja").list_voices() == []


def test_wyoming_synthesis_error_is_input_error(wyoming_server):
    uri, _ = wyoming_server
    with pytest.raises(TtsInputError, match = "synthesis failed"):
        WyomingClient(uri, 5, "ja").synthesize("error", "ja-a")


def test_wyoming_error_after_success_is_recorded(wyoming_server):
    uri, _ = wyoming_server
    client = WyomingClient(uri, 5, "ja")
    result = synthesizer(client, "ja-a").synthesize(sentences("こんにちは", "error", "さようなら"))
    assert len(result.segments) == 2
    assert result.failures == [("error", "synthesis failed")]


def test_wyoming_error_before_success_aborts(wyoming_server):
    uri, _ = wyoming_server
    with pytest.raises(RuntimeError, match = "最初の合成"):
        synthesizer(WyomingClient(uri, 5, "ja"), "ja-a").synthesize(sentences("error"))


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


# --- 出力先フォルダ ---

def test_wav_directory_writes_and_removes_by_source_name(tmp_path):
    writer = WavDirectory(tmp_path / "out", pause = 0)
    location = writer.write("data/1.txt", [pcm()])
    assert location == str(tmp_path / "out" / "1.wav")
    writer.remove("other/1.txt")
    writer.remove("data/2.txt")
    assert list((tmp_path / "out").iterdir()) == []


def test_wav_directory_uses_pause(tmp_path):
    WavDirectory(tmp_path, pause = 0.01).write("a.txt", [pcm(frames = 10), pcm(frames = 10)])
    with wave.open(str(tmp_path / "a.wav"), "rb") as result:
        assert result.getnframes() == 10 + 160 + 10


@pytest.mark.parametrize("names", [
    [".txt", ".txt.txt"],
    ["A.txt", "a.txt"],
    ["ガ.txt", "ガ.txt"],  # NFC の「ガ」と NFD の「カ + 濁点」
])
def test_wav_directory_rejects_duplicate_output_names(tmp_path, names):
    with pytest.raises(OutputDirectoryError, match = "出力先が同じ"):
        WavDirectory(tmp_path / "out").prepare(names)


def test_wav_directory_rejects_symlink(tmp_path):
    target = tmp_path / "elsewhere.wav"
    target.write_bytes(b"")
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "1.wav").symlink_to(target)
    with pytest.raises(OutputDirectoryError, match = "通常のファイル以外"):
        WavDirectory(tmp_path / "out").prepare(["1.txt"])


def test_wav_directory_rejects_directory_at_output_path(tmp_path):
    (tmp_path / "out" / "2.wav").mkdir(parents = True)
    with pytest.raises(OutputDirectoryError, match = "2.wav"):
        WavDirectory(tmp_path / "out").prepare(["1.txt", "2.txt"])


def test_wav_directory_rejects_file_as_directory(tmp_path):
    (tmp_path / "file").write_text("")
    with pytest.raises(OutputDirectoryError, match = "書き込めません"):
        WavDirectory(tmp_path / "file").prepare(["1.txt"])


# --- 入力ファイル ---

def test_list_text_files(tmp_path):
    for name in ["10.txt", "2.txt", "1.txt", ".DS_Store", "1.wav", "memo.md"]:
        (tmp_path / name).write_text("")
    (tmp_path / "dir.txt").mkdir()
    assert list_text_files(str(tmp_path)) == ["1.txt", "2.txt", "10.txt"]


def test_list_text_files_missing_folder(tmp_path):
    (tmp_path / "file.txt").write_text("")
    with pytest.raises(OSError):
        list_text_files(str(tmp_path / "missing"))
    with pytest.raises(OSError):
        list_text_files(str(tmp_path / "file.txt"))


def test_read_novel(tmp_path):
    path = tmp_path / "1.txt"
    path.write_bytes("あ\r\n「い」\r\n\r\n".encode("utf-8"))
    assert [s.text for s in read_novel(str(path)).sentences] == ["あ", "「い」", ""]


def test_read_novel_rejects_undecodable_file(tmp_path):
    path = tmp_path / "1.txt"
    path.write_bytes("あ".encode("shift_jis"))
    with pytest.raises(UnicodeDecodeError):
        read_novel(str(path))


# --- LLM ---

def test_parse_json_accepts_surrounding_text():
    assert parse_json('{"narrator_index": 2}', NarratorResponse).narrator_index == 2
    assert parse_json('回答: {"narrator_index": 1} です', NarratorResponse).narrator_index == 1
    with pytest.raises(Exception):
        parse_json("わかりません", NarratorResponse)


def test_llm_schema_converts_to_character():
    data = '{"narrators": [{"name": "花子", "portrait": "p", "aliases": ["はな"], "gender": "女性"}, {"name": "太郎", "portrait": "q"}]}'
    characters = [n.to_character() for n in parse_json(data, Narrators).narrators]
    assert characters == [
        Character(name = "花子", portrait = "p", aliases = ["はな"], gender = Gender.FEMALE),
        Character(name = "太郎", portrait = "q"),
    ]


# --- Jev ---

class FakeSystemOne:
    def __init__(self, result):
        self.result = result
        self.calls: List[tuple] = []

    def __call__(self, state, questions):
        self.calls.append((state, questions))
        if isinstance(self.result, Exception):
            raise self.result
        return type("Response", (), {"choices": {"speaker": self.result} if self.result is not None else {}})()


def jev(result) -> tuple[JevSpeakerEstimator, FakeSystemOne]:
    estimator = JevSpeakerEstimator(api_key = "key")
    fake = FakeSystemOne(result)
    estimator.client.system_one = fake
    return estimator, fake


def jev_candidates() -> List[Character]:
    return [narrator(), Character(name = "花子", portrait = "p", gender = Gender.FEMALE), Character(name = "花子", portrait = "q")]


def test_jev_returns_index_and_confidence():
    estimator, fake = jev(ChoiceAnswer(choice = "花子 (2)", confidence = 0.75, probabilities = {}))
    previous = [Sentence(text = "地の文", speaker = narrator()), Sentence(text = "「あ」", speaker = jev_candidates()[1])]
    guess = estimator.estimate_speaker(jev_candidates(), previous, Sentence(text = "「い」"), [Sentence(text = "後")])
    assert guess == SpeakerGuess(2, 0.75)
    state, questions = fake.calls[0]
    assert state == {
        "今までの内容": [{"セリフ": "地の文"}, {"話者": "花子", "セリフ": "「あ」"}],
        "推測したいセリフの内容": "「い」",
        "後の内容": ["後"],
    }
    # 名前が重複した登場人物は番号を付けて区別する
    assert list(questions["speaker"].criteria) == ["ナレーター", "花子", "花子 (2)"]


@pytest.mark.parametrize("answer", [None, ChoiceAnswer(choice = "誰か", confidence = 0.9, probabilities = {})])
def test_jev_without_valid_answer(answer):
    estimator, _ = jev(answer)
    assert estimator.estimate_speaker(jev_candidates(), [], Sentence(text = "「あ」"), []) is None


@pytest.mark.parametrize(("error", "expected"), [
    (TypeSafeAPIError(413, None, {}, "too large"), ContextTooLongError),
    (TypeSafeAPIError(400, {"error": "context length exceeded"}, {}, "bad"), ContextTooLongError),
    (TypeSafeAPIError(400, {"error": "invalid"}, {}, "bad"), SpeakerEstimationError),
    (TypeSafeAPIError(500, {"error": "too long"}, {}, "server"), SpeakerEstimationError),
    (TypeSafeError("timeout"), SpeakerEstimationError),
])
def test_jev_maps_errors(error, expected):
    estimator, _ = jev(error)
    with pytest.raises(expected):
        estimator.estimate_speaker(jev_candidates(), [], Sentence(text = "「あ」"), [])
