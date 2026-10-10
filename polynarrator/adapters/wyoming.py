import asyncio
from typing import Dict, List, Optional, Sequence

from wyoming.audio import AudioChunk, AudioChunkConverter, AudioStop
from wyoming.client import AsyncClient
from wyoming.error import Error
from wyoming.info import Describe, Info
from wyoming.tts import Synthesize, SynthesizeVoice

from polynarrator.application.ports import SpeechSynthesizer, TtsInputError
from polynarrator.domain.audio import MAX_AUDIO_BYTES, Audio, validate_audio, validate_format


class WyomingClient(SpeechSynthesizer):
    """Wyomingプロトコルで音声合成する。内部で asyncio.run を使う同期APIのため、実行中のイベントループからは呼べない"""

    def __init__(self, uri: str, timeout: float, language: Optional[str] = None):
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
        if program is None:
            return
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
        data = bytearray()
        # 変換器はリサンプリングの状態を持つため、連続した同じ形式のチャンクでは使い回す
        converter: Optional[AudioChunkConverter] = None
        chunk_format: Optional[tuple[int, int, int]] = None
        async with self._connect() as client:
            await client.write_event(Synthesize(text = text, voice = synthesize_voice).event())
            while True:
                event = await client.read_event()
                if event is None:
                    raise RuntimeError("Wyomingサーバーとの接続が切断されました")
                if AudioChunk.is_type(event.type):
                    chunk = AudioChunk.from_event(event)
                    # 後続のチャンクの形式が変わる場合もあるため、変換する前にチャンクごとに確認する
                    validate_format(chunk.rate, chunk.width, chunk.channels, len(chunk.audio))
                    if audio is None:
                        audio = Audio(chunk.rate, chunk.width, chunk.channels, b"")
                    if converter is None or chunk_format != (chunk.rate, chunk.width, chunk.channels):
                        converter = AudioChunkConverter(audio.rate, audio.width, audio.channels)
                        chunk_format = (chunk.rate, chunk.width, chunk.channels)
                    # 変換で大きくなる場合も含め、変換して保持する前に上限を超えないかを見積もり(切り上げ)、
                    # リサンプリングの端数で見積もりを超える場合に備えて変換後の実際の大きさでも確認する
                    output_rate = audio.rate * audio.width * audio.channels
                    input_rate = chunk.rate * chunk.width * chunk.channels
                    converted_size = -(-len(chunk.audio) * output_rate // input_rate)
                    if len(data) + converted_size > MAX_AUDIO_BYTES:
                        raise RuntimeError(f"Wyomingサーバーから受け取った音声が上限({MAX_AUDIO_BYTES}バイト)を超えました")
                    data += converter.convert(chunk).audio
                    if len(data) > MAX_AUDIO_BYTES:
                        raise RuntimeError(f"Wyomingサーバーから受け取った音声が上限({MAX_AUDIO_BYTES}バイト)を超えました")
                elif AudioStop.is_type(event.type):
                    break
                elif Error.is_type(event.type):
                    # Wyomingのエラーは原因を区別できないため文の失敗として扱う。
                    # 設定やサーバーの問題であれば、未成功の音声での失敗や連続した失敗として中断される
                    raise TtsInputError(Error.from_event(event).text)

        if audio is None:
            # エラーも音声も返さないのは文の内容ではなくサーバーの不具合とみなす
            raise RuntimeError("Wyomingサーバーから音声が返されませんでした")
        audio.data = bytes(data)
        return validate_audio(audio)
