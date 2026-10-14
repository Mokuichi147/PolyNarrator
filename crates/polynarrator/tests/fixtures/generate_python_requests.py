"""Python 版が LLM・Jev 互換 API へ送る要求本文を python_requests.json に書き出す

Rust 版のテスト (tests/prompts.rs) は、同じ入力で同じ要求本文を送ることを確かめる。
リポジトリのルートで実行する: PYTHONPATH=. uv run python crates/polynarrator/tests/fixtures/generate_python_requests.py
"""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from polynarrator.adapters.jev import JevSpeakerEstimator
from polynarrator.adapters.llm import OpenAiChatModel
from polynarrator.domain.character import Character, Gender, narrator
from polynarrator.domain.novel import Novel, Sentence

requests = []


class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        requests.append({"path": self.path, "body": body})
        if self.path.endswith("/chat/completions"):
            content = '{"narrators": []}' if body["response_format"]["json_schema"]["name"] == "Narrators" else '{"narrator_index": 1}'
            payload = {"id": "x", "object": "chat.completion", "created": 0, "model": body["model"],
                       "choices": [{"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}
        else:
            payload = {"model": "jev", "usage": {}, "answers": {"speaker": {"type": "choice", "choice": "花子", "confidence": 0.5, "probabilities": {}}}}
        data = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args):
        pass


server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
threading.Thread(target = server.serve_forever, daemon = True).start()
port = str(server.server_address[1])

hanako = Character(name = "花子", portrait = "明るい'委員長\"", aliases = ["はな", "委員長'", "\\"], gender = Gender.FEMALE)
taro = Character(name = "太郎", portrait = "")
novel = Novel.from_lines(["地の文です。\n", "「こんにちは」\n", "\n", "「やあ」\n"])
known = [Character(name = "田中花子", portrait = "", aliases = ["花子"], gender = Gender.FEMALE), taro]
candidates = [narrator(), hanako, taro, Character(name = "花子", portrait = "別人", gender = Gender.OTHER)]
previous = [Sentence(text = "地の文", speaker = narrator()), Sentence(text = "「あ」", speaker = hanako), Sentence(text = "「い」")]
target = Sentence(text = "「う」")
following = [Sentence(text = "後"), Sentence(text = "「え」")]

llm = OpenAiChatModel("127.0.0.1", port, "test-model")
llm.extract_characters(novel, [])
llm.extract_characters(novel, known)
llm.estimate_speaker(candidates, previous, target, following)
llm.estimate_speaker(candidates, [], target, [])
JevSpeakerEstimator(api_key = "key", model = "jev-test", base_url = f"http://127.0.0.1:{port}").estimate_speaker(candidates, previous, target, following)
server.shutdown()

output = Path(__file__).with_name("python_requests.json")
output.write_text(json.dumps(requests, ensure_ascii = False, indent = 2) + "\n", encoding = "utf-8")
