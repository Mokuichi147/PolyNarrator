import asyncio
import io
import json
import threading
import wave
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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


def response_key(text: str) -> str:
    """応答を切り替えるキー。「名前:セリフ」の形式ではセリフ部分を使う"""
    return text.rsplit(":", 1)[-1].removesuffix("」")


@pytest.fixture
def wyoming_server():
    state = {"describe": "info", "info": INFO, "requests": []}
    loop = asyncio.new_event_loop()

    async def handle(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        event = await async_read_event(reader)
        if event is not None and Describe.is_type(event.type):
            if state["describe"] == "error":
                await async_write_event(Error(text = "describe failed").event(), writer)
            elif state["describe"] == "info":
                await async_write_event(state["info"].event(), writer)
        elif event is not None and Synthesize.is_type(event.type):
            synthesize = Synthesize.from_event(event)
            state["requests"].append(synthesize)
            synthesize_text = response_key(synthesize.text)
            if synthesize_text == "error":
                await async_write_event(Error(text = "synthesis failed").event(), writer)
            elif synthesize_text == "empty":
                await async_write_event(AudioStop().event(), writer)
            elif synthesize_text == "partial-error":
                await async_write_event(AudioChunk(rate = 22050, width = 2, channels = 1, audio = b"\x01\x00" * 100).event(), writer)
                await async_write_event(Error(text = "failed midway").event(), writer)
            elif synthesize_text == "mixed":
                await async_write_event(AudioChunk(rate = 22050, width = 2, channels = 1, audio = b"\x01\x00" * 2205).event(), writer)
                for _ in range(4):
                    await async_write_event(AudioChunk(rate = 11025, width = 2, channels = 2, audio = b"\x01\x00\x01\x00" * 1102).event(), writer)
                await async_write_event(AudioStop().event(), writer)
            elif synthesize_text == "bad-second":
                await async_write_event(AudioChunk(rate = 22050, width = 2, channels = 1, audio = b"\x01\x00" * 100).event(), writer)
                await async_write_event(AudioChunk(rate = 384001, width = 2, channels = 1, audio = b"\x01\x00" * 100).event(), writer)
                await async_write_event(AudioStop().event(), writer)
            elif synthesize_text == "odd":
                await async_write_event(AudioChunk(rate = 22050, width = 2, channels = 1, audio = b"\x01").event(), writer)
                await async_write_event(AudioStop().event(), writer)
            elif synthesize_text == "empty-chunk":
                await async_write_event(AudioChunk(rate = 22050, width = 2, channels = 1, audio = b"").event(), writer)
                await async_write_event(AudioStop().event(), writer)
            elif synthesize_text != "disconnect":
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
            status = statuses.get(response_key(body["input"]), 200)
            if status == 200:
                payload, content_type = wav_bytes(frames = 0 if body["input"] == "zero" else 240), "audio/wav"
                if body["input"] == "notwav":
                    payload = b"not a wav file"
                if body["input"] == "truncated":
                    payload = payload[:-100]
                elif body["input"] == "near-limit":
                    # 最大値の直前の(ストリーミング用ではない)データ長
                    payload = payload[:40] + b"\xfe\xff\xff\xff" + payload[44:]
                elif body["input"] == "streaming":
                    # ストリーミング用にRIFFとdataのサイズを最大値にしたヘッダー
                    payload = payload[:4] + b"\xff\xff\xff\xff" + payload[8:40] + b"\xff\xff\xff\xff" + payload[44:]
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
