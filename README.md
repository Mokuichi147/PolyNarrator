# Poly Narrator

ライトノベルなどの小説テキストを、登場人物ごとに声を変えて読み上げるためのツールです。

1. OpenAI API 互換の LLM サーバーで、小説から登場人物(名前・別名・性別・特徴)を抽出します。
2. 「」で囲まれたセリフごとに、誰の発言かを推測します。
3. 登場人物ごとに音声を割り当てて音声合成し、章ごとに WAV ファイルを書き出します(任意)。

## 必要なもの

- Python 3.12.10 以上と [uv](https://docs.astral.sh/uv/)
- OpenAI API 互換の LLM サーバー(JSON Schema による構造化出力に対応したモデル)
- 音声を書き出す場合: OpenAI 互換の `/v1/audio/speech` または Wyoming プロトコルに対応した TTS サーバー

```bash
uv sync
```

## 入力ファイル

章ごとに分けた UTF-8 の `.txt` ファイルを1つのフォルダに置きます。

- ファイル名の自然順に処理し、登場人物の一覧と音声の割り当ては後のファイルに引き継ぎます。
- 1行を1つの文として扱い、行全体が `「……」` で囲まれた行だけを話者推測の対象にします。それ以外の行はナレーターが読みます。

## 使い方

登場人物と話者を推測し、結果を標準出力に表示します。

```bash
uv run main.py --host localhost --port 1234 --model qwen3.8-flash-next-iq3_s data/
```

`--tts-output` を指定すると、音声合成して `<入力ファイル名>.wav` を書き出します。

```bash
uv run main.py --tts-output out/ --tts-backend openai --tts-url http://localhost:8880/v1 --tts-voices voice1,voice2,voice3 data/
```

```bash
uv run main.py --tts-output out/ --tts-backend wyoming --tts-url tcp://localhost:10200 data/
```

- OpenAI 互換 API には音声一覧を取得する標準 API がないため、`--tts-voices` が必須です。
- Wyoming で `--tts-voices` を省略すると、サーバーの音声のうち `--tts-language` に一致するものを使います。複数話者の音声は `音声名:話者名` の形式で指定します。

話者推測を Jev(TypeSafe System One API)互換モデルで行う場合は `--speaker-backend jev` を指定します(登場人物の抽出には引き続き LLM サーバーを使います)。

```bash
uv run main.py --speaker-backend jev --jev-api-key <API キー> data/
```

### 音声の割り当て

- ナレーターには `--tts-narrator-voice`(省略時は音声一覧の先頭)を使います。
- 登場人物には、ナレーター以外で使用数が最も少ない音声を割り当て、以降のファイルでも同じ音声を使います。
- `--tts-male-voices` / `--tts-female-voices` を指定すると、性別が分かっている登場人物にはその中から優先して割り当てます。

## オプション

| オプション | 既定値 | 説明 |
| --- | --- | --- |
| `--host` / `--port` | `localhost` / `1234` | LLM サーバー(`http://<host>:<port>/v1`) |
| `--model` | `qwen3.8-flash-next-iq3_s` | LLM のモデル名 |
| `--speaker-backend` | `llm` | 話者推測のバックエンド(`llm` / `jev`) |
| `--jev-model` | `TYPESAFE_DEFAULT_MODEL` または `jev-latest` | Jev 互換モデル名 |
| `--jev-base-url` | `TYPESAFE_BASE_URL` または `https://api.typesafe.ai` | Jev 互換 API の URL |
| `--jev-api-key` | `TYPESAFE_API_KEY` | Jev 互換 API のキー |
| `--tts-output` | なし | 音声の出力先フォルダ(指定時のみ音声合成) |
| `--tts-backend` | `openai` | TTS のプロトコル(`openai` / `wyoming`) |
| `--tts-url` | `http://localhost:8880/v1` / `tcp://localhost:10200` | TTS サーバーの URL |
| `--tts-model` | `tts-1` | OpenAI 互換 API のモデル名 |
| `--tts-api-key` | `OPENAI_API_KEY` | OpenAI 互換 API のキー |
| `--tts-language` | `ja` | Wyoming で音声を絞り込む言語(空文字で絞り込まない) |
| `--tts-voices` | Wyoming: サーバーの一覧 | 使用する音声(カンマ区切り) |
| `--tts-narrator-voice` | 音声一覧の先頭 | ナレーターの音声 |
| `--tts-male-voices` / `--tts-female-voices` | なし | 男性 / 女性に優先して使う音声(カンマ区切り) |
| `--tts-pause` | `0.3` | 行間に入れる無音の秒数(最大 60) |
| `--tts-max-chars` | `200` | 1回の合成の最大文字数。超える行は文末・読点・文字数の順で分割(0 で分割しない) |
| `--tts-timeout` | `60` | TTS サーバーへのリクエストのタイムアウト秒数 |

<details>
<summary>音声合成の詳しい挙動</summary>

- LLM や TTS サーバーでの処理を始める前に、入力ファイルを読み込めるか、出力先に書き込めるか、出力ファイル名が衝突しないか、指定した音声が一覧(Wyoming ではサーバー)にあるかを確認します。
- 空行や `＊　＊　＊` のように読み上げる文字がない行は読み飛ばします。
- 文の内容が原因で合成できなかった行(OpenAI 互換 API の 400/422、Wyoming のエラー応答)は除いて WAV を書き出し、最後に終了コード 1 で終了します。
- その音声で一度も合成に成功していない場合や、5回続けて失敗した場合は、設定やサーバーの問題とみなして中断します。接続・認証・サーバー内部のエラーでも中断し、そのファイルの WAV は書き出しません。
- 読み上げる文がないファイルでは WAV を書き出さず、以前の同名の WAV を削除します。
- 章によって登場人物の正式名が揺れても、以前の正式名・別名と一致すれば同じ音声を使います。複数の人物が持つ別名は一致に使いません。

</details>

## テスト

```bash
uv run pytest
```
