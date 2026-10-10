import os
import struct
import tempfile
import unicodedata
import wave
from pathlib import Path
from typing import List, Optional

from wyoming.audio import AudioChunk, AudioChunkConverter

from polynarrator.application.ports import AudioWriter
from polynarrator.domain.audio import Audio, validate_audio

# ストリーミングでデータ長が分からないWAVのヘッダーに入る値
WAV_UNKNOWN_SIZE = 0xFFFFFFFF


def wav_data_size(content: bytes) -> Optional[int]:
    """WAVの data チャンクに宣言されたバイト数。見つからない場合は None"""
    position = 12
    while position + 8 <= len(content):
        chunk_id, size = struct.unpack_from("<4sI", content, position)
        if chunk_id == b"data":
            return size
        position += 8 + size + (size & 1)
    return None


def _write_silence(wav: wave.Wave_write, audio_format: Audio, frames: int) -> None:
    """無音を一度に確保しないよう、最大1秒ずつ書き出す"""
    frame = (b"\x80" if audio_format.width == 1 else b"\x00") * (audio_format.width * audio_format.channels)  # 8bit PCM は符号なしのため 0x80 が無音
    block = frame * min(frames, audio_format.rate)
    while frames > 0:
        count = min(frames, audio_format.rate)
        wav.writeframes(block[:count * len(frame)])
        frames -= count


def write_wav(path: Path, segments: List[Audio], pause: float = 0.0) -> None:
    """音声を順に連結して1つのWAVファイルに書き出す。形式が異なる音声は先頭に合わせて変換する"""
    if pause < 0:
        raise ValueError("pause には0以上を指定してください")
    if len(segments) == 0:
        return
    for segment in segments:
        validate_audio(segment)
    first = segments[0]
    silence_frames = int(first.rate * pause)

    # 途中で失敗しても書きかけのファイルが残らないよう、同じフォルダに一意な一時ファイルを作ってから置き換える
    path.parent.mkdir(parents = True, exist_ok = True)
    # 長いファイル名でも名前の長さの上限を超えないよう、一時ファイルの名前は出力名によらない短いものにする
    # 一時ファイルはパスで開き直さず、作成時に開いたファイルにそのまま書き込む
    file = tempfile.NamedTemporaryFile(dir = path.parent, prefix = ".polynarrator-", suffix = ".wav.tmp", delete = False)
    temp = Path(file.name)
    try:
        with file, wave.open(file, "wb") as wav:
            wav.setframerate(first.rate)
            wav.setsampwidth(first.width)
            wav.setnchannels(first.channels)
            for index, segment in enumerate(segments):
                if index > 0:
                    _write_silence(wav, first, silence_frames)
                # 変換器はリサンプリングの状態を持つため、音声ごとに作り直す
                chunk = AudioChunk(segment.rate, segment.width, segment.channels, segment.data)
                wav.writeframes(AudioChunkConverter(first.rate, first.width, first.channels).convert(chunk).audio)
        os.replace(temp, path)
    except BaseException:
        temp.unlink(missing_ok = True)
        raise


class OutputDirectoryError(Exception):
    """出力先に書き出せない"""


class WavDirectory(AudioWriter):
    """入力ファイルごとに <入力ファイル名>.wav をフォルダに書き出す"""

    def __init__(self, directory: Path, pause: float = 0.0):
        self.directory = directory
        # 音声と音声の間に入れる無音の秒数
        self.pause = pause

    def path_for(self, source: str) -> Path:
        return self.directory / f"{Path(source).stem}.wav"

    def prepare(self, sources: List[str]) -> None:
        """書き出しを始める前に、出力先を作成して書き込めるか、出力ファイルが衝突しないかを確認する"""
        try:
            self.directory.mkdir(parents = True, exist_ok = True)
            with tempfile.TemporaryFile(dir = self.directory):
                pass
        except OSError as e:
            raise OutputDirectoryError(f"音声の出力先に書き込めません({self.directory}): {e}") from e
        outputs = [self.path_for(s) for s in sources]
        # 大文字小文字や Unicode 正規化を区別しないファイルシステムでも上書きし合わないよう、正規化した名前で比べる
        keys = [unicodedata.normalize("NFC", p.name).casefold() for p in outputs]
        duplicates = sorted({str(p) for p, key in zip(outputs, keys) if keys.count(key) > 1})
        if len(duplicates) > 0:
            raise OutputDirectoryError(f"複数の入力ファイルの出力先が同じになります: {duplicates}")
        # シンボリックリンクは置き換えるとリンク自体が通常のファイルに変わるため、通常のファイル以外は受け付けない
        conflicts = [str(p) for p in outputs if p.is_symlink() or (p.exists() and not p.is_file())]
        if len(conflicts) > 0:
            raise OutputDirectoryError(f"音声の出力先に通常のファイル以外のものがあります: {conflicts}")

    def write(self, source: str, segments: List[Audio]) -> str:
        path = self.path_for(source)
        write_wav(path, segments, self.pause)
        return str(path)

    def remove(self, source: str) -> None:
        self.path_for(source).unlink(missing_ok = True)
