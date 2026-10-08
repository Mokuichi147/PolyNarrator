import argparse
import math
import os
import tempfile
import unicodedata
from pathlib import Path
from typing import List, Optional

from natsort import natsorted
from llm import Ai
from jev import JevSpeakerEstimator
from models.novel import Novel
from tts import OpenAiSpeechClient, TtsClient, VoiceAssigner, WyomingClient, synthesize_sentences, write_wav


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


def create_tts(args: argparse.Namespace) -> tuple[TtsClient, VoiceAssigner]:
    if args.tts_backend == "wyoming":
        uri = args.tts_url or "tcp://localhost:10200"
        try:
            client: TtsClient = WyomingClient(uri, args.tts_timeout, args.tts_language)
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


def output_path(output_dir: Path, file: str) -> Path:
    return output_dir / f"{Path(file).stem}.wav"


def check_output_dir(path: Path, files: List[str]) -> None:
    """LLMやTTSサーバーでの処理を始める前に、出力先を作成して書き込めるかを確認する"""
    try:
        path.mkdir(parents = True, exist_ok = True)
        with tempfile.TemporaryFile(dir = path):
            pass
    except OSError as e:
        raise SystemExit(f"音声の出力先に書き込めません({path}): {e}")
    outputs = [output_path(path, f) for f in files]
    # 大文字小文字や Unicode 正規化を区別しないファイルシステムでも上書きし合わないよう、正規化した名前で比べる
    keys = [unicodedata.normalize("NFC", p.name).casefold() for p in outputs]
    duplicates = sorted({str(p) for p, key in zip(outputs, keys) if keys.count(key) > 1})
    if len(duplicates) > 0:
        raise SystemExit(f"複数の入力ファイルの出力先が同じになります: {duplicates}")
    # シンボリックリンクは置き換えるとリンク自体が通常のファイルに変わるため、通常のファイル以外は受け付けない
    conflicts = [str(p) for p in outputs if p.is_symlink() or (p.exists() and not p.is_file())]
    if len(conflicts) > 0:
        raise SystemExit(f"音声の出力先に通常のファイル以外のものがあります: {conflicts}")


def load_novels(folder: str, files: List[str]) -> List[Novel]:
    """LLMやTTSサーバーでの処理を始める前に、全ての入力ファイルを読み込んでおく"""
    novels: List[Novel] = []
    for file in files:
        novel = Novel()
        try:
            novel.load(os.path.join(folder, file))
        except (OSError, UnicodeDecodeError) as e:
            raise SystemExit(f"入力ファイルを読み込めませんでした({os.path.join(folder, file)}): {e}")
        novels.append(novel)
    return novels


def list_text_files(folder: str) -> List[str]:
    """入力フォルダ内のテキストファイル名を自然順で返す。.DS_Store や出力済みの音声などは対象外"""
    try:
        names = os.listdir(folder)
    except OSError as e:
        raise SystemExit(f"入力フォルダを読み込めませんでした({folder}): {e}")
    return natsorted(f for f in names if f.endswith(".txt") and os.path.isfile(os.path.join(folder, f)))


def synthesize_file(novel: Novel, outfile: Path, client: TtsClient, assigner: VoiceAssigner, max_chars: int, pause: float) -> bool:
    """小説を音声合成して WAV に書き出す。合成に失敗した文があれば True を返す"""
    try:
        result = synthesize_sentences(novel.sentences, client, assigner, max_chars)
        if len(result.segments) > 0:
            write_wav(outfile, result.segments, pause)
        else:
            # 以前の実行結果を今回の結果と取り違えないよう、古い音声は消しておく
            outfile.unlink(missing_ok = True)
    except Exception as e:
        # 認証・接続・設定・サーバー内部のエラーや、扱えない音声形式などは続けても失敗するため中断する
        raise SystemExit(f"音声合成を中断しました: {e}")

    if len(result.segments) == 0:
        if len(result.failures) > 0:
            print("全ての文の合成に失敗したため、音声を書き出しませんでした")
        else:
            print("読み上げられる文がないため、音声を書き出しませんでした")
    else:
        if len(result.failures) > 0:
            print(f"音声を書き出しました(合成に失敗した{len(result.failures)}文を除く): {outfile}")
        else:
            print(f"音声を書き出しました: {outfile}")
    if result.skipped > 0:
        print(f"記号だけで読み上げる文字がない{result.skipped}行は読み飛ばしました")
    return len(result.failures) > 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="1234")
    parser.add_argument("--model", default="granite4:small-h")
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


def main():
    parser = build_parser()
    args = parser.parse_args()
    # 出力先を作る前に入力フォルダを確認する(出力先の作成で入力フォルダが作られてしまわないように)
    files = list_text_files(args.folder)
    novels = load_novels(args.folder, files)
    if args.tts_output is not None:
        if args.tts_output == "":
            parser.error("--tts-output に出力先のディレクトリを指定してください")
        check_output_dir(Path(args.tts_output), files)

    ai = Ai(args.host, args.port, args.model)
    if args.speaker_backend == "jev":
        speaker_estimator = JevSpeakerEstimator(args.jev_api_key, args.jev_model, args.jev_base_url)
    else:
        speaker_estimator = ai
    tts: Optional[tuple[TtsClient, VoiceAssigner]] = create_tts(args) if args.tts_output is not None else None
    narrators = []
    failed_files: List[str] = []
    
    for index, (file, novel) in enumerate(zip(files, novels)):
        filepath = os.path.join(args.folder, file)
        
        response = ai.get_narrators(novel, narrators)
        if len(response) > 0:
            narrators = response
        
        print(index + 1, filepath)
        print("\n".join([f"  - {i.name} ({i.gender}) {i.aliases}" for i in narrators]))
        print()
        
        novel.narrators = narrators
        speaker_estimator.set_estimation_narrator(novel, 100, 0, True)

        if tts is not None:
            client, assigner = tts
            outfile = output_path(Path(args.tts_output), file)
            if synthesize_file(novel, outfile, client, assigner, args.tts_max_chars, args.tts_pause):
                failed_files.append(file)
    
    print("\n登場人物一覧")
    for narrator in narrators:
        print(f"- {narrator.name} ({narrator.gender}) {narrator.aliases}")

    if tts is not None:
        print("\n音声の割り当て")
        for name, voice in tts[1].assigned.items():
            print(f"- {name}: {voice}")

    if len(failed_files) > 0:
        raise SystemExit(f"\n音声合成に失敗した文があります: {', '.join(failed_files)}")
    


if __name__ == "__main__":
    main()
