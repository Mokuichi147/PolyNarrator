from dataclasses import dataclass

# 音声形式として受け付ける上限。異常な形式の音声で無音などに大量のメモリを使わないようにする
MAX_RATE = 384000
MAX_WIDTH = 4
# 形式の変換(AudioChunkConverter)がモノラルとステレオにしか対応していないため
MAX_CHANNELS = 2
# 1回の合成で受け取る音声の上限。音声を送り続ける異常なサーバーでメモリを使い切らないようにする
MAX_AUDIO_BYTES = 1 << 30


@dataclass
class Audio:
    """PCM音声データ"""
    rate: int
    width: int
    channels: int
    data: bytes


def validate_format(rate: int, width: int, channels: int, size: int) -> None:
    """扱えない形式やフレーム境界の合わない音声は、文の内容ではなくサーバーの不具合とみなして送出する"""
    valid_format = 0 < rate <= MAX_RATE and 0 < width <= MAX_WIDTH and 0 < channels <= MAX_CHANNELS
    if not valid_format or size % (width * channels) != 0:
        raise RuntimeError(f"TTSサーバーから不正な音声が返されました(rate={rate}, width={width}, channels={channels}, {size}バイト)")


def validate_audio(audio: Audio) -> Audio:
    """空の音声や不正な形式の音声は、文の内容ではなくサーバーの不具合とみなして送出する"""
    if len(audio.data) == 0:
        raise RuntimeError("TTSサーバーから空の音声が返されました")
    validate_format(audio.rate, audio.width, audio.channels, len(audio.data))
    return audio
