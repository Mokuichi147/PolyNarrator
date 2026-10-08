import asyncio
import threading
from typing import List, Optional

import pytest
from wyoming.audio import AudioChunk, AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import async_read_event, async_write_event
from wyoming.info import Attribution, Describe, Info, TtsProgram, TtsVoice, TtsVoiceSpeaker
from wyoming.tts import Synthesize


def attribution() -> Attribution:
    return Attribution(name = "test", url = "")


def tts_voice(name: str, languages: List[str], speakers: Optional[List[str]] = None, installed: bool = True) -> TtsVoice:
    return TtsVoice(
        name = name,
        attribution = attribution(),
        installed = installed,
        description = None,
        version = None,
        languages = languages,
        speakers = [TtsVoiceSpeaker(name = s) for s in speakers] if speakers else None,
    )


INFO = Info(tts = [TtsProgram(
    name = "tts",
    attribution = attribution(),
    installed = True,
    description = None,
    version = None,
    voices = [
        tts_voice("ja-a", ["ja_JP"]),
        tts_voice("ja-multi", ["ja"], ["s1", "s2"]),
        tts_voice("en-a", ["en_US"]),
        tts_voice("ja-missing", ["ja"], installed = False),
    ],
), TtsProgram(
    name = "not-installed",
    attribution = attribution(),
    installed = False,
    description = None,
    version = None,
    voices = [tts_voice("ja-other", ["ja"])],
), TtsProgram(
    name = "second",
    attribution = attribution(),
    installed = True,
    description = None,
    version = None,
    voices = [tts_voice("ja-second", ["ja"])],
)])


@pytest.fixture
def wyoming_server():
    state = {"describe": "info", "requests": []}
    loop = asyncio.new_event_loop()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        event = await async_read_event(reader)
        if event is not None and Describe.is_type(event.type):
            if state["describe"] == "error":
                await async_write_event(Error(text = "describe failed").event(), writer)
            elif state["describe"] == "info":
                await async_write_event(INFO.event(), writer)
        elif event is not None and Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            state["requests"].append(synthesize)
            if synthesize.text == "error":
                await async_write_event(Error(text = "synthesis failed").event(), writer)
            elif synthesize.text == "empty":
                await async_write_event(AudioStop().event(), writer)
            elif synthesize.text == "odd":
                await async_write_event(AudioChunk(rate = 22050, width = 2, channels = 1, audio = b"\x01").event(), writer)
                await async_write_event(AudioStop().event(), writer)
            elif synthesize.text == "empty-chunk":
                await async_write_event(AudioChunk(rate = 22050, width = 2, channels = 1, audio = b"").event(), writer)
                await async_write_event(AudioStop().event(), writer)
            elif synthesize.text != "disconnect":
                await async_write_event(AudioStart(rate = 22050, width = 2, channels = 1).event(), writer)
                for _ in range(3):
                    await async_write_event(AudioChunk(rate = 22050, width = 2, channels = 1, audio = b"\x01\x00" * 100).event(), writer)
                await async_write_event(AudioStop().event(), writer)
        writer.close()

    server = loop.run_until_complete(asyncio.start_server(handle, "127.0.0.1", 0))
    thread = threading.Thread(target = loop.run_forever, daemon = True)
    thread.start()
    yield f"tcp://127.0.0.1:{server.sockets[0].getsockname()[1]}", state
    loop.call_soon_threadsafe(server.close)
    loop.call_soon_threadsafe(loop.stop)
    thread.join()
    loop.close()
