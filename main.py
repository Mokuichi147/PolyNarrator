import argparse
import os
from typing import List

from natsort import natsorted
from llm import Ai
from jev import JevSpeakerEstimator
from models.novel import Novel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="1234")
    parser.add_argument("--model", default="granite4:small-h")
    parser.add_argument("--speaker-backend", choices=["llm", "jev"], default="llm", help="話者推測に使うバックエンド")
    parser.add_argument("--jev-model", default=None, help="Jev互換モデル名(未指定時は TYPESAFE_DEFAULT_MODEL または jev-latest)")
    parser.add_argument("--jev-base-url", default=None, help="Jev互換APIのURL(未指定時は TYPESAFE_BASE_URL または https://api.typesafe.ai)")
    parser.add_argument("--jev-api-key", default=None, help="Jev互換APIのキー(未指定時は TYPESAFE_API_KEY)")
    parser.add_argument("folder", help="data")
    args = parser.parse_args()

    ai = Ai(args.host, args.port, args.model)
    if args.speaker_backend == "jev":
        speaker_estimator = JevSpeakerEstimator(args.jev_api_key, args.jev_model, args.jev_base_url)
    else:
        speaker_estimator = ai
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
    
    print("\n登場人物一覧")
    for narrator in narrators:
        print(f"- {narrator.name} ({narrator.gender}) {narrator.aliases}")
    


if __name__ == "__main__":
    main()
