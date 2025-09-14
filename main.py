import argparse
from llm import Ai
from models.novel import Novel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="11434")
    parser.add_argument("--model", default="qwen3:30b-a3b-thinking-2507-q4_K_M")
    parser.add_argument("filepath", help="novel.txt")
    args = parser.parse_args()
    
    novel = Novel()
    novel.load(args.filepath)
    print("\n".join([s.text for s in novel.sentences]))
    
    ai = Ai(args.host, args.port, args.model)
    narrators = ai.get_narrators(novel)
    
    print("\n登場人物一覧")
    for narrator in narrators:
        print(f"- {narrator.name}: {narrator.gender}")


if __name__ == "__main__":
    main()
