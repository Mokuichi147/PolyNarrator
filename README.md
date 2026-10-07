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
|- tts.py   (OpenAI互換API / Wyomingプロトコルでの音声合成)
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

## 音声合成

`--tts-output` を指定すると、話者推測の結果をもとに登場人物ごとに音声を割り当てて読み上げ、入力ファイルごとに1つの WAV ファイルを書き出します。
TTS サーバーとは以下のいずれかのプロトコルで接続します。

### OpenAI互換API (`/v1/audio/speech`)

OpenAI 互換の音声合成 API を提供するサーバーを利用します。
音声一覧を取得する標準 API がないため、使用する音声を `--tts-voices` で指定します。

```
uv run main.py --tts-output out/ --tts-backend openai --tts-url http://localhost:8880/v1 --tts-model tts-1 --tts-voices voice1,voice2,voice3 data/
```

### Wyoming

Wyoming プロトコルに対応した TTS サーバーを利用します。
`--tts-voices` を省略するとサーバーから取得した音声のうち `--tts-language`(既定値 `ja`)に一致するものを使います。
複数話者の音声は `音声名:話者名` の形式で指定します。

```
uv run main.py --tts-output out/ --tts-backend wyoming --tts-url tcp://localhost:10200 data/
```

### 音声の割り当て

- ナレーターには `--tts-narrator-voice`(省略時は音声一覧の先頭)を使います。
- 登場人物には、ナレーターと別の音声のうち使用数が最も少ないものを割り当てます。一度割り当てた音声は以降のファイルでも同じものを使います。
- `--tts-male-voices` / `--tts-female-voices` を指定すると、性別が判明している登場人物にはその中から優先して割り当てます。
