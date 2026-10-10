"""コマンドライン引数から各層の実装を組み立てて実行する"""

import argparse
import math
import os
from pathlib import Path
from typing import List, Optional, Sequence

from polynarrator.adapters.jev import JevSpeakerEstimator
from polynarrator.adapters.llm import OpenAiChatModel
from polynarrator.adapters.openai_speech import OpenAiSpeechClient
from polynarrator.adapters.text_files import list_text_files, read_novel
from polynarrator.adapters.wav import OutputDirectoryError, WavDirectory
from polynarrator.adapters.wyoming import WyomingClient
from polynarrator.application.narrate_novels import Chapter, NarrateNovels
from polynarrator.application.ports import SpeakerEstimator, SpeechSynthesizer
from polynarrator.application.speaker_estimation import SpeakerEstimation
from polynarrator.application.speech_synthesis import AudioExporter, SentenceSynthesizer, SynthesisAbortedError
from polynarrator.cli.console import ConsolePresenter
from polynarrator.domain.voice_assigner import VoiceAssigner

# 無音はメモリ上に作るため、極端な値で大量のメモリを確保しないよう上限を設ける
MAX_PAUSE = 60.0


def split_voices(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def non_negative_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError("0以上の数値を指定してください")
    return number


def positive_float(value: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("0より大きい数値を指定してください")
    return number


def pause_seconds(value: str) -> float:
    number = non_negative_float(value)
    if number > MAX_PAUSE:
        raise argparse.ArgumentTypeError(f"{MAX_PAUSE:g}秒以下を指定してください")
    return number


def non_negative_int(value: str) -> int:
    number = int(value)
    if number < 0:
        raise argparse.ArgumentTypeError("0以上の整数を指定してください")
    return number


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="1234")
    parser.add_argument("--model", default="qwen3.8-flash-next-iq3_s")
    parser.add_argument("--speaker-backend", choices=["llm", "jev"], default="llm", help="話者推測に使うバックエンド")
    parser.add_argument("--jev-model", default=None, help="Jev互換モデル名(未指定時は TYPESAFE_DEFAULT_MODEL または jev-latest)")
    parser.add_argument("--jev-base-url", default=None, help="Jev互換APIのURL(未指定時は TYPESAFE_BASE_URL または https://api.typesafe.ai)")
    parser.add_argument("--jev-api-key", default=None, help="Jev互換APIのキー(未指定時は TYPESAFE_API_KEY)")
    parser.add_argument("--tts-output", default=None, help="音声ファイルの出力先ディレクトリ(指定時のみ音声合成を行う)")
    parser.add_argument("--tts-backend", choices=["openai", "wyoming"], default="openai", help="音声合成に使うプロトコル")
    parser.add_argument("--tts-url", default=None, help="TTSサーバーのURL(未指定時は openai: http://localhost:8880/v1, wyoming: tcp://localhost:10200)")
    parser.add_argument("--tts-model", default="tts-1", help="OpenAI互換APIで使うモデル名")
    parser.add_argument("--tts-api-key", default=None, help="OpenAI互換APIのキー(未指定時は OPENAI_API_KEY)")
    parser.add_argument("--tts-language", default="ja", help="Wyomingで音声一覧を絞り込む言語(空文字で絞り込まない)")
    parser.add_argument("--tts-voices", default=None, help="使用する音声のカンマ区切り一覧(Wyomingでは未指定時にサーバーの一覧を使う)")
    parser.add_argument("--tts-narrator-voice", default=None, help="ナレーターの音声(未指定時は音声一覧の先頭)")
    parser.add_argument("--tts-male-voices", default=None, help="男性の登場人物に優先して使う音声のカンマ区切り一覧")
    parser.add_argument("--tts-female-voices", default=None, help="女性の登場人物に優先して使う音声のカンマ区切り一覧")
    parser.add_argument("--tts-pause", type=pause_seconds, default=0.3, help="行と行(長い行を分割した場合は分割した塊)の間に入れる無音の秒数")
    parser.add_argument("--tts-max-chars", type=non_negative_int, default=200, help="1回の合成に送る最大文字数。超える行は文末・読点・文字数の順で分割する(0で分割しない)")
    parser.add_argument("--tts-timeout", type=positive_float, default=60.0, help="TTSサーバーへのリクエストタイムアウト秒数")
    parser.add_argument("folder", help="data")
    return parser


def load_chapters(folder: str) -> List[Chapter]:
    """LLMやTTSサーバーでの処理を始める前に、全ての入力ファイルを読み込んでおく"""
    try:
        files = list_text_files(folder)
    except OSError as e:
        raise SystemExit(f"入力フォルダを読み込めませんでした({folder}): {e}")
    chapters: List[Chapter] = []
    for file in files:
        path = os.path.join(folder, file)
        try:
            chapters.append(Chapter(path, read_novel(path)))
        except (OSError, UnicodeDecodeError) as e:
            raise SystemExit(f"入力ファイルを読み込めませんでした({path}): {e}")
    return chapters


def prepare_output(directory: str, pause: float, chapters: List[Chapter]) -> WavDirectory:
    """LLMやTTSサーバーでの処理を始める前に、出力先に書き出せるかを確認する"""
    writer = WavDirectory(Path(directory), pause)
    try:
        writer.prepare([c.source for c in chapters])
    except OutputDirectoryError as e:
        raise SystemExit(str(e))
    return writer


def create_tts(args: argparse.Namespace) -> tuple[SpeechSynthesizer, VoiceAssigner]:
    if args.tts_backend == "wyoming":
        uri = args.tts_url or "tcp://localhost:10200"
        try:
            client: SpeechSynthesizer = WyomingClient(uri, args.tts_timeout, args.tts_language)
        except Exception as e:
            raise SystemExit(f"Wyomingサーバー({uri})から音声一覧を取得できませんでした: {e}")
    else:
        client = OpenAiSpeechClient(
            args.tts_url or "http://localhost:8880/v1",
            args.tts_model,
            args.tts_api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            args.tts_timeout,
        )

    voices = split_voices(args.tts_voices) or client.list_voices()
    male_voices = split_voices(args.tts_male_voices)
    female_voices = split_voices(args.tts_female_voices)
    role_voices = [*([args.tts_narrator_voice] if args.tts_narrator_voice else []), *male_voices, *female_voices]
    unknown = client.unknown_voices([*voices, *role_voices])
    if len(unknown) > 0:
        raise SystemExit(f"TTSサーバーにない音声が指定されています: {unknown}")
    if not args.tts_voices and args.tts_backend == "wyoming":
        # サーバーの一覧を使う場合、言語の絞り込みや複数話者モデルの既定話者で一覧に出ない音声も、明示されていれば使う
        voices.extend(v for v in dict.fromkeys(role_voices) if v not in voices)
    if len(voices) == 0:
        if args.tts_backend == "wyoming":
            raise SystemExit("TTSサーバーから利用できる音声を取得できませんでした。--tts-voices や --tts-language を確認してください")
        raise SystemExit("利用する音声を --tts-voices で指定してください")

    try:
        assigner = VoiceAssigner(
            voices,
            narrator_voice = args.tts_narrator_voice,
            male_voices = male_voices,
            female_voices = female_voices,
        )
    except ValueError as e:
        raise SystemExit(f"{e}。--tts-voices に含まれる音声を指定してください")
    print(f"TTS 初期化: backend={args.tts_backend}, 音声一覧={voices}")
    return client, assigner


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    # 出力先を作る前に入力フォルダを確認する(出力先の作成で入力フォルダが作られてしまわないように)
    chapters = load_chapters(args.folder)
    writer: Optional[WavDirectory] = None
    if args.tts_output is not None:
        if args.tts_output == "":
            parser.error("--tts-output に出力先のディレクトリを指定してください")
        writer = prepare_output(args.tts_output, args.tts_pause, chapters)

    presenter = ConsolePresenter()
    llm = OpenAiChatModel(args.host, args.port, args.model)
    estimator: SpeakerEstimator = llm
    if args.speaker_backend == "jev":
        estimator = JevSpeakerEstimator(args.jev_api_key, args.jev_model, args.jev_base_url)
    exporter: Optional[AudioExporter] = None
    if writer is not None:
        client, assigner = create_tts(args)
        exporter = AudioExporter(SentenceSynthesizer(client, assigner, presenter, args.tts_max_chars), writer, presenter)

    use_case = NarrateNovels(llm, SpeakerEstimation(estimator, presenter), presenter, exporter)
    try:
        report = use_case.run(chapters)
    except SynthesisAbortedError as e:
        raise SystemExit(f"音声合成を中断しました: {e}")

    if len(report.failed_sources) > 0:
        raise SystemExit(f"\n音声合成に失敗した文があります: {', '.join(report.failed_sources)}")
