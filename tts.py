import asyncio
import io
import wave
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from openai import OpenAI
from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.client import AsyncClient
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from models.gender import Gender
from models.narrator import Narrator

NARRATOR_NAME = "ナレーター"


@dataclass
class Audio:
    """PCM音声データ"""
    rate: int
    width: int
    channels: int
    data: bytes


class TtsClient(ABC):
    @abstractmethod
    def list_voices(self) -> List[str]:
        """サーバーから取得できる音声の一覧。取得できない場合は空"""

    @abstractmethod
    def synthesize(self, text: str, voice: str) -> Audio:
        pass


class OpenAiSpeechClient(TtsClient):
    """OpenAI互換の /v1/audio/speech で音声合成する"""

    def __init__(self, base_url: str, model: str, api_key: str, timeout: float):
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
        response = self.client.audio.speech.create(
            model = self.model,
            voice = voice,
            input = text,
            response_format = "wav",
        )
        with wave.open(io.BytesIO(response.content), "rb") as wav:
            return Audio(
                rate = wav.getframerate(),
                width = wav.getsampwidth(),
                channels = wav.getnchannels(),
                data = wav.readframes(wav.getnframes()),
            )


class WyomingClient(TtsClient):
    """Wyomingプロトコルで音声合成する"""

    def __init__(self, uri: str, timeout: float, language: Optional[str] = None):
        self.uri = uri
        self.timeout = timeout
        self.language = language
        self._voices: Dict[str, SynthesizeVoice] = asyncio.run(self._describe())

    def _connect(self) -> AsyncClient:
        return AsyncClient.from_uri(self.uri, connect_timeout = self.timeout, read_timeout = self.timeout)

    def _match_language(self, languages: List[str]) -> bool:
        if not self.language or len(languages) == 0:
            return True
        prefix = self.language.lower().replace("_", "-")
        return any(lang.lower().replace("_", "-").startswith(prefix) for lang in languages)

    async def _describe(self) -> Dict[str, SynthesizeVoice]:
        async with self._connect() as client:
            await client.write_event(Describe().event())
            while True:
                event = await client.read_event()
                if event is None:
                    raise RuntimeError("Wyomingサーバーとの接続が切断されました")
                if Info.is_type(event.type):
                    info = Info.from_event(event)
                    break

        voices: Dict[str, SynthesizeVoice] = {}
        for program in info.tts:
            for voice in program.voices:
                if not self._match_language(voice.languages):
                    continue
                if voice.speakers:
                    # 複数話者モデルは話者ごとに別の音声として扱う
                    for speaker in voice.speakers:
                        voices[f"{voice.name}:{speaker.name}"] = SynthesizeVoice(name = voice.name, speaker = speaker.name)
                else:
                    voices[voice.name] = SynthesizeVoice(name = voice.name)
        return voices

    def list_voices(self) -> List[str]:
        return list(self._voices.keys())

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
                    raise RuntimeError(Error.from_event(event).text)

        if audio is None:
            raise RuntimeError("Wyomingサーバーから音声が返されませんでした")
        return audio


class VoiceAssigner:
    """登場人物ごとに音声を割り当てる。同じ登場人物には常に同じ音声を使う"""

    def __init__(
        self,
        voices: List[str],
        narrator_voice: Optional[str] = None,
        male_voices: List[str] = [],
        female_voices: List[str] = [],
    ):
        if len(voices) == 0:
            raise ValueError("利用できる音声がありません")
        self.voices = voices
        self.narrator_voice = narrator_voice or voices[0]
        self.gender_voices: Dict[Gender, List[str]] = {
            Gender.MALE: male_voices,
            Gender.FEMALE: female_voices,
        }
        self.assigned: Dict[str, str] = {NARRATOR_NAME: self.narrator_voice}
        self.usage: Counter[str] = Counter([self.narrator_voice])

    def assign(self, narrator: Optional[Narrator]) -> str:
        name = narrator.name if narrator is not None else NARRATOR_NAME
        if name in self.assigned:
            return self.assigned[name]

        pool = self.voices
        if narrator is not None and narrator.gender is not None and len(self.gender_voices.get(narrator.gender, [])) > 0:
            pool = self.gender_voices[narrator.gender]

        # ナレーターと聞き分けられるよう、他に候補があればナレーターの音声は避ける
        candidates = [v for v in pool if v != self.narrator_voice] or pool
        voice = min(candidates, key = lambda v: self.usage[v])

        self.assigned[name] = voice
        self.usage[voice] += 1
        return voice


def write_wav(path: Path, segments: List[Audio], pause: float = 0.0) -> None:
    """音声を順に連結して1つのWAVファイルに書き出す。形式が異なる音声は先頭に合わせて変換する"""
    if len(segments) == 0:
        return
    first = segments[0]
    silence = bytes(int(first.rate * pause) * first.width * first.channels)

    path.parent.mkdir(parents = True, exist_ok = True)
    with wave.open(str(path), "wb") as wav:
        wav.setframerate(first.rate)
        wav.setsampwidth(first.width)
        wav.setnchannels(first.channels)
        for index, segment in enumerate(segments):
            if index > 0:
                wav.writeframes(silence)
            chunk = AudioChunk(segment.rate, segment.width, segment.channels, segment.data)
            wav.writeframes(AudioChunkConverter(first.rate, first.width, first.channels).convert(chunk).audio)
