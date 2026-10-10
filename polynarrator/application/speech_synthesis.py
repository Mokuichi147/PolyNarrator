from typing import List

from polynarrator.application.ports import AudioWriter, Presenter, SpeechSynthesizer, SynthesisResult, TtsInputError
from polynarrator.domain.audio import Audio
from polynarrator.domain.novel import Novel, Sentence
from polynarrator.domain.text import is_readable, split_for_tts
from polynarrator.domain.voice_assigner import VoiceAssigner

# 入力起因の失敗がこの回数だけ続いた場合は、文ではなくサーバー側の問題とみなして中断する
MAX_CONSECUTIVE_FAILURES = 5


class SynthesisAbortedError(RuntimeError):
    """続けても失敗するため音声合成を中断した"""


class SentenceSynthesizer:
    """文ごとに話者の音声で合成する。音声の割り当てと失敗の状況はファイルをまたいで引き継ぐ"""

    def __init__(self, synthesizer: SpeechSynthesizer, assigner: VoiceAssigner, presenter: Presenter, max_chars: int = 0):
        self.synthesizer = synthesizer
        self.assigner = assigner
        self.presenter = presenter
        self.max_chars = max_chars
        # 一度でも合成に成功した音声と、入力起因の失敗が続いている回数
        self.succeeded_voices: set[str] = set()
        self.consecutive_failures = 0

    def synthesize(self, sentences: List[Sentence]) -> SynthesisResult:
        """文の内容による失敗は記録して続行し、それ以外の失敗は送出する"""
        result = SynthesisResult()
        for sentence in sentences:
            if not sentence.text:
                continue
            if not is_readable(sentence.text):
                result.skipped += 1
                continue

            voice = self.assigner.assign(sentence.speaker)
            # 行の途中で発話が切れないよう、分割した塊が全て合成できた行だけを使う
            segments: List[Audio] = []
            try:
                for chunk in split_for_tts(sentence.text, self.max_chars):
                    segments.append(self.synthesizer.synthesize(chunk, voice))
                    self.succeeded_voices.add(voice)
            except TtsInputError as e:
                self._record_failure(voice, sentence.text, e)
                result.failures.append((sentence.text, str(e)))
                continue

            result.segments.extend(segments)
            self.consecutive_failures = 0
        return result

    def _record_failure(self, voice: str, text: str, error: TtsInputError) -> None:
        # 400 などはモデル名や音声名の誤りでも返るため、一度も成功していない音声での失敗は設定の問題とみなす
        if voice not in self.succeeded_voices:
            raise SynthesisAbortedError(f"音声 {voice} での最初の合成に失敗しました。モデル名や音声名を確認してください: {error}") from error
        self.presenter.sentence_synthesis_failed(voice, text, error)
        self.consecutive_failures += 1
        if self.consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
            raise SynthesisAbortedError(f"音声合成が{self.consecutive_failures}回連続で失敗しました: {error}") from error


class AudioExporter:
    """小説を音声合成して書き出す"""

    def __init__(self, synthesizer: SentenceSynthesizer, writer: AudioWriter, presenter: Presenter):
        self.synthesizer = synthesizer
        self.writer = writer
        self.presenter = presenter

    def export(self, source: str, novel: Novel) -> bool:
        """合成に失敗した文があれば True を返す。続けても失敗する場合は SynthesisAbortedError を送出する"""
        try:
            result = self.synthesizer.synthesize(novel.sentences)
            if len(result.segments) > 0:
                location = self.writer.write(source, result.segments)
            else:
                # 以前の実行結果を今回の結果と取り違えないよう、古い音声は消しておく
                location = None
                self.writer.remove(source)
        except SynthesisAbortedError:
            raise
        except Exception as e:
            # 認証・接続・設定・サーバー内部のエラーや、扱えない音声形式などは続けても失敗するため中断する
            raise SynthesisAbortedError(str(e)) from e

        self.presenter.audio_exported(location, result)
        return len(result.failures) > 0
