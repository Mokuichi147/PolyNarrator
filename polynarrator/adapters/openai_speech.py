import io
import wave
from typing import List

from openai import BadRequestError, OpenAI, UnprocessableEntityError

from polynarrator.adapters.wav import WAV_UNKNOWN_SIZE, wav_data_size
from polynarrator.application.ports import SpeechSynthesizer, TtsInputError
from polynarrator.domain.audio import MAX_AUDIO_BYTES, Audio, validate_audio

# WAVのヘッダーなど音声以外の部分として許容する大きさ
WAV_HEADER_ALLOWANCE = 1 << 16


class OpenAiSpeechClient(SpeechSynthesizer):
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
        content = bytearray()
        try:
            with self.client.audio.speech.with_streaming_response.create(
                model = self.model,
                voice = voice,
                input = text,
                response_format = "wav",
            ) as response:
                # 巨大な応答でメモリを使い切らないよう、上限を超えた時点で読むのをやめる
                for block in response.iter_bytes():
                    content += block
                    if len(content) > MAX_AUDIO_BYTES + WAV_HEADER_ALLOWANCE:
                        raise RuntimeError(f"TTSサーバーから受け取った音声が上限({MAX_AUDIO_BYTES}バイト)を超えました")
        except (BadRequestError, UnprocessableEntityError) as e:
            # 認証・接続・サーバー内部のエラーは続けても失敗するだけなので、そのまま送出して中断させる
            raise TtsInputError(str(e)) from e
        return self._decode(bytes(content))

    def _decode(self, content: bytes) -> Audio:
        # PCM以外のWAVは wave モジュールが wave.Error を送出する
        with wave.open(io.BytesIO(content), "rb") as wav:
            audio = Audio(
                rate = wav.getframerate(),
                width = wav.getsampwidth(),
                channels = wav.getnchannels(),
                data = wav.readframes(wav.getnframes()),
            )
        if len(audio.data) > MAX_AUDIO_BYTES:
            raise RuntimeError(f"TTSサーバーから受け取った音声が上限({MAX_AUDIO_BYTES}バイト)を超えました")
        # ストリーミング用にデータ長を最大値にしたヘッダーを除き、ヘッダーより短いデータは途中で切れた応答とみなす。
        # 1フレーム未満の端数は聴感上の影響がないため、フレーム単位で比べる
        declared = wav_data_size(content)
        frame_size = audio.width * audio.channels
        if declared is not None and declared != WAV_UNKNOWN_SIZE and frame_size > 0:
            expected = declared // frame_size * frame_size
            if len(audio.data) < expected:
                raise RuntimeError(f"TTSサーバーから途中で切れた音声が返されました({len(audio.data)}/{expected}バイト)")
        return validate_audio(audio)
