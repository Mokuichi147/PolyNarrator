# Poly Narrator

ライトノベルを複数の話者で読み上げるためのツールです。


## プロジェクト構成

```
|- models
    |- sentence.py
    |- narrator.py
    |- novel.py
|- tts.py
|- main.py
```

## セットアップ

1. 依存パッケージをインストールしてください。
   ```bash
   uv sync
   ```
   もしくは直接以下をインストールします。
   ```bash
   pip install natsort>=8.4.0 ollama>=0.5.3 requests>=2.31.0
   ```
2. 日本語高品質TTSには [VOICEVOX ENGINE](https://github.com/VOICEVOX/voicevox_engine) を利用します。事前にエンジンをダウンロードして起動してください。
   - macOS/Windows: 公式リリースからアプリを入手し、エンジンを起動。
   - Docker: `docker run -p 50021:50021 voicevox/voicevox_engine:cpu-ubuntu20.04-latest`
   - エンジンのライセンスは MIT、各話者（音声ライブラリ）の利用条件は [VOICEVOX キャラクターライセンス](https://voicevox.hiroshiba.jp/) を参照してください。

## 使い方

```bash
uv run main.py \
  --host localhost \
  --port 11434 \
  --model granite4:small-h \
  --tts-output audio_out \
  --tts-speaker-id 1 \
  --tts-timeout 45 \
  --tts-chunk-size 300 \
  data/novels
```

- `--tts-output` を指定すると、各テキストファイルごとに登場人物別の WAV ファイルを生成します。
- `--tts-host` / `--tts-port` で接続先の VOICEVOX エンジンを指定します（既定は `127.0.0.1:50021`）。
- プログラムは登場人物の `Gender` やプロフィール説明をもとに、適した VOICEVOX 話者スタイルを自動で選択します。`--tts-speaker-id` は全登場人物に共通のフォールバックIDを指定したい場合に利用できます。
- `--tts-speed-scale`, `--tts-pitch-scale`, `--tts-intonation-scale`, `--tts-volume-scale` で話速・ピッチ・抑揚・音量を調整できます。
- `--tts-disable-upspeak` を付けると疑問文語尾の自動上昇を無効化します。
- `--tts-timeout` で VOICEVOX からの応答待ち時間を調整できます。長文の合成には余裕を持った値を設定してください。
- `--tts-chunk-size` で1チャンクあたりの最大文字数を制御できます。文章が長い場合は値を小さくすると安定します。
- 1ファイル内で同じ登場人物が多数登場する場合、音声はチャンク単位に分割され、`<登場人物名>_NN_MM.wav` の形式で保存されます。

## 話者の自動割り当てについて

- 登場人物の `Gender`（男性・女性・その他）および人物紹介文から、幼年/青年/成人/高齢といった年齢区分を推定します。
- 区分ごとに優先する VOICEVOX の話者候補を定義しており、エンジンが提供するスタイルIDから利用可能なものを選びます。
- 候補に該当する話者が見つからない場合は、未使用の話者（同一キャラクターでもスタイルが異なる場合は同じ声として扱う）を優先的に割り当て、全キャラクターが同じ声になることを防ぎます。
- 既定で用意された候補に該当しない場合は、VOICEVOX が返す一覧を巡回して再利用します。
- 候補を調整したい場合は `main.py` の `VOICE_PREFERENCES` 定義を編集してください。
