import argparse
import os
from typing import List

from natsort import natsorted
from llm import Ai
from models.novel import Novel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="11434")
    parser.add_argument("--model", default="qwen3:30b-a3b-thinking-2507-q4_K_M")
    parser.add_argument("folder", help="data")
    args = parser.parse_args()
    
    ai = Ai(args.host, args.port, args.model)
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
        ai.set_estimation_narrator(novel, 40, 10, True)
    
    print("\n登場人物一覧")
    for narrator in narrators:
        print(f"- {narrator.name} ({narrator.gender}) {narrator.aliases}")
    


if __name__ == "__main__":
    main()
