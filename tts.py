import asyncio
import io
import os
import re
import unicodedata
import wave
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from openai import BadRequestError, OpenAI, UnprocessableEntityError
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.client import AsyncClient
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from models.gender import Gender
from models.narrator import Narrator
from models.sentence import Sentence

NARRATOR_NAME = "ナレーター"

# 入力起因の失敗がこの回数だけ続いた場合は、文ではなくサーバー側の問題とみなして中断する
MAX_CONSECUTIVE_FAILURES = 5


@dataclass
class Audio:
    """PCM音声データ"""
    rate: int
    width: int
    channels: int
    data: bytes


def validate_audio(audio: Audio) -> Audio:
    """空の音声やフレーム境界の合わない音声は、文の内容ではなくサーバーの不具合とみなして送出する"""
    if len(audio.data) == 0:
        raise RuntimeError("TTSサーバーから空の音声が返されました")
    if audio.width <= 0 or audio.channels <= 0 or len(audio.data) % (audio.width * audio.channels) != 0:
        raise RuntimeError(f"TTSサーバーから不正な音声が返されました(width={audio.width}, channels={audio.channels}, {len(audio.data)}バイト)")
    return audio


class TtsInputError(Exception):
    """文の内容が原因で合成できなかった。その文だけを読み飛ばして続行してよい"""


class TtsClient(ABC):
    # 一度でも合成に成功した音声と、入力起因の失敗が続いている回数。どちらもファイルをまたいで引き継ぐ
    succeeded_voices: set[str]
    consecutive_failures: int

    def __init__(self) -> None:
        self.succeeded_voices = set()
        self.consecutive_failures = 0

    @abstractmethod
    def list_voices(self) -> List[str]:
        """サーバーから取得できる音声の一覧。取得できない場合は空"""

    def unknown_voices(self, voices: Sequence[str]) -> List[str]:
        """指定された音声のうち、サーバーで使えないと分かっているもの"""
        return []

    @abstractmethod
    def synthesize(self, text: str, voice: str) -> Audio:
        """音声合成する。文の内容が原因で失敗した場合は TtsInputError を送出する"""


class OpenAiSpeechClient(TtsClient):
    """OpenAI互換の /v1/audio/speech で音声合成する"""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float):
        super().__init__()
        self.client = OpenAI(
            base_url = base_url,
            api_key = api_key,
            timeout = timeout,
        )
        self.model = model

    def list_voices(self) -> List[str]:
        # 音声一覧を取得する標準APIがないため、利用者が指定する
        return []

    def synthesize(self, text: str, voice: str) -> Audio:
        try:
            response = self.client.audio.speech.create(
                model = self.model,
                voice = voice,
                input = text,
                response_format = "wav",
            )
        except (BadRequestError, UnprocessableEntityError) as e:
            # 認証・接続・サーバー内部のエラーは続けても失敗するだけなので、そのまま送出して中断させる
            raise TtsInputError(str(e)) from e

        # PCM以外のWAVは wave モジュールが wave.Error を送出する
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            audio = Audio(
                rate = wav.getframerate(),
                width = wav.getsampwidth(),
                channels = wav.getnchannels(),
                data = wav.readframes(wav.getnframes()),
            )
        return validate_audio(audio)


class WyomingClient(TtsClient):
    """Wyomingプロトコルで音声合成する"""

    def __init__(self, uri: str, timeout: float, language: Optional[str] = None):
        super().__init__()
        self.uri = uri
        self.timeout = timeout
        self.language = language
        # 指定できる全ての音声。複数話者モデルは「音声名:話者名」と、既定の話者を使う「音声名」の両方で指定できる
        self._voices: Dict[str, SynthesizeVoice] = {}
        # 言語が一致し、音声一覧として使う音声
        self._listed: List[str] = []
        asyncio.run(self._describe())

    def _connect(self) -> AsyncClient:
        return AsyncClient.from_uri(self.uri, connect_timeout = self.timeout, read_timeout = self.timeout)

    def _match_language(self, languages: List[str]) -> bool:
        if not self.language or len(languages) == 0:
            return True
        requested = self.language.lower().replace("_", "-")
        # 「ja」と「ja-JP」のように、一方が他方の地域指定付きであれば一致とみなす
        return any(
            lang == requested or lang.startswith(requested + "-") or requested.startswith(lang + "-")
            for lang in (l.lower().replace("_", "-") for l in languages)
        )

    async def _describe(self) -> None:
        async with self._connect() as client:
            await client.write_event(Describe().event())
            while True:
                event = await client.read_event()
                if event is None:
                    raise RuntimeError("Wyomingサーバーとの接続が切断されました")
                if Info.is_type(event.type):
                    info = Info.from_event(event)
                    break
                if Error.is_type(event.type):
                    raise RuntimeError(f"Wyomingサーバーがエラーを返しました: {Error.from_event(event).text}")

        # Synthesize ではプログラムを選べず、サーバー既定のプログラムで合成されるため、最初のプログラムの音声だけを使う
        program = next((p for p in info.tts if p.installed), None)
        if program is not None:
            for voice in program.voices:
                if not voice.installed:
                    continue
                matched = self._match_language(voice.languages)
                self._voices[voice.name] = SynthesizeVoice(name = voice.name)
                if voice.speakers:
                    # 複数話者モデルは話者ごとに別の音声として扱う
                    for speaker in voice.speakers:
                        key = f"{voice.name}:{speaker.name}"
                        self._voices[key] = SynthesizeVoice(name = voice.name, speaker = speaker.name)
                        if matched:
                            self._listed.append(key)
                elif matched:
                    self._listed.append(voice.name)

    def list_voices(self) -> List[str]:
        return list(self._listed)

    def unknown_voices(self, voices: Sequence[str]) -> List[str]:
        return [v for v in voices if v not in self._voices]

    def synthesize(self, text: str, voice: str) -> Audio:
        return asyncio.run(self._synthesize(text, voice))

    async def _synthesize(self, text: str, voice: str) -> Audio:
        synthesize_voice = self._voices.get(voice, SynthesizeVoice(name = voice))
        audio: Optional[Audio] = None
        async with self._connect() as client:
            await client.write_event(Synthesize(text = text, voice = synthesize_voice).event())
            while True:
                event = await client.read_event()
                if event is None:
                    raise RuntimeError("Wyomingサーバーとの接続が切断されました")
                if AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    if audio is None:
                        audio = Audio(chunk.rate, chunk.width, chunk.channels, b"")
                    audio.data += AudioChunkConverter(audio.rate, audio.width, audio.channels).convert(chunk).audio
                elif AudioStop.is_type(event.type):
                    break
                elif Error.is_type(event.type):
                    # Wyomingのエラーは原因を区別できないため文の失敗として扱う。
                    # 設定やサーバーの問題であれば、未成功の音声での失敗や連続した失敗として中断される
                    raise TtsInputError(Error.from_event(event).text)

        if audio is None:
            # エラーも音声も返さないのは文の内容ではなくサーバーの不具合とみなす
            raise RuntimeError("Wyomingサーバーから音声が返されませんでした")
        return validate_audio(audio)


class VoiceAssigner:
    """登場人物ごとに音声を割り当てる。同じ登場人物には常に同じ音声を使う"""

    def __init__(
        self,
        voices: List[str],
        narrator_voice: Optional[str] = None,
        male_voices: Optional[List[str]] = None,
        female_voices: Optional[List[str]] = None,
    ):
        if len(voices) == 0:
            raise ValueError("利用できる音声がありません")
        male_voices = male_voices or []
        female_voices = female_voices or []
        unknown = [v for v in [narrator_voice or voices[0], *male_voices, *female_voices] if v not in voices]
        if len(unknown) > 0:
            raise ValueError(f"音声一覧にない音声が指定されています: {unknown}")

        self.voices = voices
        self.narrator_voice = narrator_voice or voices[0]
        self.gender_voices: Dict[Gender, List[str]] = {
            Gender.MALE: male_voices,
            Gender.FEMALE: female_voices,
        }
        self.assigned: Dict[str, str] = {NARRATOR_NAME: self.narrator_voice}
        self._alias_voices: Dict[str, str] = {}
        self.usage: Counter[str] = Counter([self.narrator_voice])

    def _find_voice(self, narrator: Narrator) -> Optional[str]:
        if narrator.name in self.assigned:
            return self.assigned[narrator.name]
        # ファイルごとの抽出で正式名が揺れても、以前の別名や正式名と一致すれば同じ人物とみなす。
        # 汎用的な呼び名で別人と混同しないよう、別名同士の一致は使わない
        if narrator.name in self._alias_voices:
            return self._alias_voices[narrator.name]
        return next((self.assigned[a] for a in narrator.aliases if a in self.assigned), None)

    def assign(self, narrator: Optional[Narrator]) -> str:
        if narrator is None or narrator.name == NARRATOR_NAME:
            return self.narrator_voice

        voice = self._find_voice(narrator)
        if voice is None:
            pool = self.voices
            if narrator.gender is not None and len(self.gender_voices.get(narrator.gender, [])) > 0:
                pool = self.gender_voices[narrator.gender]

            # ナレーターと聞き分けられるよう、他に候補があればナレーターの音声は避ける
            candidates = [v for v in pool if v != self.narrator_voice] or pool
            voice = min(candidates, key = lambda v: self.usage[v])
            self.usage[voice] += 1

        self.assigned[narrator.name] = voice
        for alias in narrator.aliases:
            self._alias_voices.setdefault(alias, voice)
        return voice


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


def _split_by_length(text: str, max_chars: int) -> List[str]:
    """文字数で区切る。句読点や括弧などの記号はできるだけ直前の文字と同じ塊に入れ、記号だけの塊ができないようにする"""
    units: List[str] = []
    for char in text:
        if units and (not is_readable(char) or not is_readable(units[-1])):
            units[-1] += char
        else:
            units.append(char)
    # 記号を含めた単位でも max_chars を超える場合は、上限を優先して単純に区切る
    return [c[i:i + max_chars] for c in _pack(units, max_chars) for i in range(0, len(c), max_chars)]


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


@dataclass
class SynthesisResult:
    segments: List[Audio] = field(default_factory = list)
    # 合成に失敗した行と理由
    failures: List[tuple[str, str]] = field(default_factory = list)
    # 読み上げる文字がないため送信しなかった文の数
    skipped: int = 0


def synthesize_sentences(
    sentences: List[Sentence],
    client: TtsClient,
    assigner: VoiceAssigner,
    max_chars: int = 0,
) -> SynthesisResult:
    """文ごとに音声合成する。文の内容による失敗は記録して続行し、それ以外の失敗は送出する"""
    result = SynthesisResult()
    for sentence in sentences:
        if not sentence.text:
            continue
        if not is_readable(sentence.text):
            result.skipped += 1
            continue

        voice = assigner.assign(sentence.narrator)
        # 行の途中で発話が切れないよう、分割した塊が全て合成できた行だけを使う
        segments: List[Audio] = []
        try:
            for chunk in split_for_tts(sentence.text, max_chars):
                segments.append(client.synthesize(chunk, voice))
                client.succeeded_voices.add(voice)
        except TtsInputError as e:
            # 400 などはモデル名や音声名の誤りでも返るため、一度も成功していない音声での失敗は設定の問題とみなす
            if voice not in client.succeeded_voices:
                raise RuntimeError(f"音声 {voice} での最初の合成に失敗しました。モデル名や音声名を確認してください: {e}") from e
            print(f"音声合成に失敗しました ({voice}): {e}\t{sentence.text}")
            result.failures.append((sentence.text, str(e)))
            client.consecutive_failures += 1
            if client.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                raise RuntimeError(f"音声合成が{client.consecutive_failures}回連続で失敗しました: {e}") from e
            continue

        result.segments.extend(segments)
        client.consecutive_failures = 0
    return result


def write_wav(path: Path, segments: List[Audio], pause: float = 0.0) -> None:
    """音声を順に連結して1つのWAVファイルに書き出す。形式が異なる音声は先頭に合わせて変換する"""
    if len(segments) == 0:
        return
    if pause < 0:
        raise ValueError("pause には0以上を指定してください")
    for segment in segments:
        validate_audio(segment)
    first = segments[0]
    # 8bit PCM は符号なしのため 0x80 が無音
    silence = (b"\x80" if first.width == 1 else b"\x00") * (int(first.rate * pause) * first.width * first.channels)

    # 途中で失敗しても書きかけのファイルが残らないよう、一時ファイルに書いてから置き換える
    path.parent.mkdir(parents = True, exist_ok = True)
    temp = path.with_name(path.name + ".tmp")
    try:
        with wave.open(str(temp), "wb") as wav:
            wav.setframerate(first.rate)
            wav.setsampwidth(first.width)
            wav.setnchannels(first.channels)
            for index, segment in enumerate(segments):
                if index > 0:
                    wav.writeframes(silence)
                # 変換器はリサンプリングの状態を持つため、音声ごとに作り直す
                chunk = AudioChunk(segment.rate, segment.width, segment.channels, segment.data)
                wav.writeframes(AudioChunkConverter(first.rate, first.width, first.channels).convert(chunk).audio)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok = True)
        raise
