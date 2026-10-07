import argparse
import os
from pathlib import Path
from typing import List, Optional

from natsort import natsorted
from llm import Ai
from jev import JevSpeakerEstimator
from models.novel import Novel
from tts import Audio, OpenAiSpeechClient, TtsClient, VoiceAssigner, WyomingClient, write_wav


def split_voices(value: Optional[str]) -> List[str]:
    if not value:
        return []
    return [v.strip() for v in value.split(",") if v.strip()]


def create_tts(args: argparse.Namespace) -> tuple[TtsClient, VoiceAssigner]:
    if args.tts_backend == "wyoming":
        client: TtsClient = WyomingClient(args.tts_url or "tcp://localhost:10200", args.tts_timeout, args.tts_language)
    else:
        client = OpenAiSpeechClient(
            args.tts_url or "http://localhost:8880/v1",
            args.tts_model,
            args.tts_api_key or os.environ.get("OPENAI_API_KEY", "not-needed"),
            args.tts_timeout,
        )

    voices = split_voices(args.tts_voices) or client.list_voices()
    if len(voices) == 0:
        raise SystemExit("利用する音声を --tts-voices で指定してください")
    print(f"TTS 初期化: backend={args.tts_backend}, 音声一覧={voices}")

    assigner = VoiceAssigner(
        voices,
        narrator_voice = args.tts_narrator_voice,
        male_voices = split_voices(args.tts_male_voices),
        female_voices = split_voices(args.tts_female_voices),
    )
    return client, assigner


def main():
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
    parser.add_argument("--tts-pause", type=float, default=0.3, help="文と文の間に入れる無音の秒数")
    parser.add_argument("--tts-timeout", type=float, default=60.0, help="TTSサーバーへのリクエストタイムアウト秒数")
    parser.add_argument("folder", help="data")
    args = parser.parse_args()

    ai = Ai(args.host, args.port, args.model)
    if args.speaker_backend == "jev":
        speaker_estimator = JevSpeakerEstimator(args.jev_api_key, args.jev_model, args.jev_base_url)
    else:
        speaker_estimator = ai
    tts: Optional[tuple[TtsClient, VoiceAssigner]] = create_tts(args) if args.tts_output else None
    narrators = []
    
    files: List[str] = os.listdir(args.folder)
    for index, file in enumerate(natsorted(files)):
        filepath = os.path.join(args.folder, file)
        
        novel = Novel()
        novel.load(filepath)
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
            segments: List[Audio] = []
            for sentence in novel.sentences:
                if not sentence.text:
                    continue
                voice = assigner.assign(sentence.narrator)
                try:
                    segments.append(client.synthesize(sentence.text, voice))
                except Exception as e:
                    print(f"音声合成に失敗しました ({voice}): {e}\t{sentence.text}")

            if len(segments) > 0:
                outfile = Path(args.tts_output) / f"{Path(file).stem}.wav"
                write_wav(outfile, segments, args.tts_pause)
                print(f"音声を書き出しました: {outfile}")
    
    print("\n登場人物一覧")
    for narrator in narrators:
        print(f"- {narrator.name} ({narrator.gender}) {narrator.aliases}")

    if tts is not None:
        print("\n音声の割り当て")
        for name, voice in tts[1].assigned.items():
            print(f"- {name}: {voice}")
    


if __name__ == "__main__":
    main()
