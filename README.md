# Poly Narrator

ライトノベルを複数の話者で読み上げるためのツールです。


## プロジェクト構成

```
|- models
    |- sentence.py
    |- narrator.py
    |- novel.py
|- llm.py   (OpenAI API準拠サーバーでの登場人物抽出・話者推測)
|- jev.py   (Jev互換モデルでの話者推測)
|- main.py
```

## 使い方

登場人物の抽出は LM Studio などの OpenAI API 準拠サーバーで行います。
Ollama を使う場合は `--port 11434` を指定します。

```
uv run main.py --host localhost --port 1234 --model granite4:small-h data/
```

話者の推測を Jev(TypeSafe System One API)互換モデルで行う場合は `--speaker-backend jev` を指定します。
API キーは `--jev-api-key` または環境変数 `TYPESAFE_API_KEY` で指定します。

```
uv run main.py --speaker-backend jev --jev-model jev-latest --jev-base-url https://api.typesafe.ai data/
```
