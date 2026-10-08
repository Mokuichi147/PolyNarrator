import asyncio
import io
import os
import re
import struct
import tempfile
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

# ストリーミングでデータ長が分からないWAVのヘッダーに入る値
WAV_UNKNOWN_SIZE = 0xFFFFFFFF

# 音声形式として受け付ける上限。異常な形式の音声で無音などに大量のメモリを使わないようにする
MAX_RATE = 384000
MAX_WIDTH = 4
# 形式の変換(AudioChunkConverter)がモノラルとステレオにしか対応していないため
MAX_CHANNELS = 2
# 1回の合成で受け取る音声の上限。音声を送り続ける異常なサーバーでメモリを使い切らないようにする
MAX_AUDIO_BYTES = 1 << 30
# WAVのヘッダーなど音声以外の部分として許容する大きさ
WAV_HEADER_ALLOWANCE = 1 << 16

# 入力起因の失敗がこの回数だけ続いた場合は、文ではなくサーバー側の問題とみなして中断する
MAX_CONSECUTIVE_FAILURES = 5


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


def wav_data_size(content: bytes) -> Optional[int]:
    """WAVの data チャンクに宣言されたバイト数。見つからない場合は None"""
    position = 12
    while position + 8 <= len(content):
        chunk_id, size = struct.unpack_from("<4sI", content, position)
        if chunk_id == b"data":
            return size
        position += 8 + size + (size & 1)
    return None


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
        content = bytes(content)

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


class WyomingClient(TtsClient):
    """Wyomingプロトコルで音声合成する。内部で asyncio.run を使う同期APIのため、実行中のイベントループからは呼べない"""

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
        self.usage: Counter[str] = Counter([self.narrator_voice])
        # 人物ごとの音声と、名前・別名から人物への対応。複数の人物が持ち、どの人物か決められない別名は None にする
        self._person_voices: List[str] = []
        self._name_persons: Dict[str, int] = {}
        self._alias_persons: Dict[str, Optional[int]] = {}

    @staticmethod
    def _key(name: str) -> str:
        """前後の空白や Unicode の表記の違いで別人とみなさないよう、照合用に名前を正規化する"""
        return unicodedata.normalize("NFKC", name).strip()

    def _find_person(self, name: str, aliases: List[str]) -> Optional[int]:
        # 割り当て済みの名前は、後から別名で別の人物と結び付いても書き出し済みの音声と食い違わないよう元の人物のままにする
        if name in self._name_persons:
            return self._name_persons[name]
        # ファイルごとの抽出で正式名が揺れても、以前の別名や正式名と一致すれば同じ人物とみなす。
        # 汎用的な呼び名で別人と混同しないよう、別名同士の一致は使わない
        if self._alias_persons.get(name) is not None:
            return self._alias_persons[name]
        # 別名が複数の人物の正式名に一致する場合は、どの人物か決められないため新しい人物とする
        candidates = {self._name_persons[a] for a in aliases if a in self._name_persons}
        return candidates.pop() if len(candidates) == 1 else None

    def assign(self, narrator: Optional[Narrator]) -> str:
        # 名前のない登場人物は誰か区別できないため、ナレーターとして読む
        if narrator is None:
            return self.narrator_voice
        name = self._key(narrator.name)
        if name == NARRATOR_NAME or not name:
            return self.narrator_voice
        aliases = [a for a in (self._key(a) for a in narrator.aliases) if a]

        person = self._find_person(name, aliases)
        if person is None:
            pool = self.voices
            if narrator.gender is not None and len(self.gender_voices.get(narrator.gender, [])) > 0:
                pool = self.gender_voices[narrator.gender]

            # ナレーターと聞き分けられるよう、性別ごとの候補、全ての音声の順にナレーター以外の音声を探す
            candidates = (
                [v for v in pool if v != self.narrator_voice]
                or [v for v in self.voices if v != self.narrator_voice]
                or [self.narrator_voice]
            )
            voice = min(candidates, key = lambda v: self.usage[v])
            self.usage[voice] += 1
            person = len(self._person_voices)
            self._person_voices.append(voice)

        voice = self._person_voices[person]
        self._name_persons[name] = person
        self.assigned[name] = voice
        for alias in aliases:
            if alias not in self._alias_persons:
                self._alias_persons[alias] = person
            elif self._alias_persons[alias] != person:
                self._alias_persons[alias] = None
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


def _clusters(text: str) -> List[str]:
    """濁点などの結合文字を直前の文字と同じ単位にまとめる"""
    clusters: List[str] = []
    for char in text:
        if clusters and unicodedata.category(char).startswith("M"):
            clusters[-1] += char
        else:
            clusters.append(char)
    return clusters


def _split_by_length(text: str, max_chars: int) -> List[str]:
    """文字数で区切る。句読点や括弧などの記号はできるだけ直前の文字と同じ塊に入れ、記号だけの塊ができないようにする"""
    units: List[List[str]] = []
    for cluster in _clusters(text):
        if units and (not is_readable(cluster) or not is_readable("".join(units[-1]))):
            units[-1].append(cluster)
        else:
            units.append([cluster])
    # 記号を含めた単位でも max_chars を超える場合は、上限を優先して結合文字の単位で区切る。
    # 結合文字を含む1文字だけで max_chars を超える場合は、文字を壊さないよう区切らない
    pieces = [piece for unit in units for piece in (_pack(unit, max_chars) if len("".join(unit)) > max_chars else ["".join(unit)])]
    return _pack(pieces, max_chars)


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
    # 記号だけで読み上げる文字がないため送信しなかった行の数(空行は含まない)
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
