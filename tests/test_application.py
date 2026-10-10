from typing import Callable, Dict, List, Optional

import pytest

from polynarrator.application import speech_synthesis
from polynarrator.application.narrate_novels import Chapter, NarrateNovels
from polynarrator.application.ports import (
    AudioWriter,
    CharacterExtractionError,
    CharacterExtractor,
    ContextTooLongError,
    Presenter,
    SpeakerEstimationError,
    SpeakerEstimator,
    SpeakerGuess,
    SpeechSynthesizer,
    TtsInputError,
)
from polynarrator.application.speaker_estimation import SpeakerEstimation
from polynarrator.application.speech_synthesis import (
    MAX_CONSECUTIVE_FAILURES,
    AudioExporter,
    SentenceSynthesizer,
    SynthesisAbortedError,
)
from polynarrator.domain.audio import Audio
from polynarrator.domain.character import Character
from polynarrator.domain.novel import Novel, Sentence
from polynarrator.domain.voice_assigner import VoiceAssigner


def character(name: str) -> Character:
    return Character(name = name, portrait = "")


def pcm() -> Audio:
    return Audio(16000, 2, 1, b"\x01\x00" * 160)


def sentences(*texts: str, speaker: Optional[Character] = None) -> List[Sentence]:
    return [Sentence(text = t, speaker = speaker) for t in texts]


class FakeSynthesizer(SpeechSynthesizer):
    def __init__(self, fail: Callable[[str], Optional[Exception]] = lambda text: None):
        self.fail = fail
        self.requests: List[tuple[str, str]] = []

    def list_voices(self) -> List[str]:
        return []

    def synthesize(self, text: str, voice: str) -> Audio:
        self.requests.append((text, voice))
        error = self.fail(text)
        if error is not None:
            raise error
        return pcm()


def sentence_synthesizer(client: SpeechSynthesizer, voices: Optional[List[str]] = None, max_chars: int = 0) -> SentenceSynthesizer:
    return SentenceSynthesizer(client, VoiceAssigner(voices or ["n"]), Presenter(), max_chars)


# --- 文ごとの合成 ---

def test_synthesize_skips_unreadable_without_request():
    client = FakeSynthesizer()
    result = sentence_synthesizer(client).synthesize(sentences("", "あ", "＊　＊　＊", "「………………」", "い"))
    assert [t for t, _ in client.requests] == ["あ", "い"]
    assert len(result.segments) == 2
    assert result.skipped == 2
    assert result.failures == []


def test_synthesize_skipped_counts_symbol_lines_only():
    result = sentence_synthesizer(FakeSynthesizer()).synthesize(sentences("", "＊　＊　＊", ""))
    assert result.skipped == 1


def test_synthesize_records_input_errors_and_continues():
    client = FakeSynthesizer(lambda text: TtsInputError("bad") if text == "い" else None)
    result = sentence_synthesizer(client).synthesize(sentences("あ", "い", "う"))
    assert len(result.segments) == 2
    assert result.failures == [("い", "bad")]


def test_synthesize_aborts_after_consecutive_input_errors():
    client = FakeSynthesizer(lambda text: TtsInputError("bad") if text != "ok" else None)
    with pytest.raises(SynthesisAbortedError, match = "連続"):
        sentence_synthesizer(client).synthesize(sentences("ok", *["あ"] * 10))
    assert len(client.requests) == 1 + MAX_CONSECUTIVE_FAILURES


@pytest.mark.parametrize("failing", ["一文目。", "二文目。", "三文目。"])
def test_synthesize_drops_whole_line_when_a_chunk_fails(failing):
    client = FakeSynthesizer(lambda text: TtsInputError("bad") if text == failing else None)
    line = "一文目。二文目。三文目。"
    result = sentence_synthesizer(client, max_chars = 4).synthesize(sentences("ok", line, "ok"))
    assert len(result.segments) == 2
    assert result.failures == [(line, "bad")]


def test_synthesize_partial_line_success_counts_as_voice_success():
    # 先頭の塊が成功していれば、その音声は使えるとみなして行の失敗として続行する
    client = FakeSynthesizer(lambda text: TtsInputError("bad") if text == "二文目。" else None)
    result = sentence_synthesizer(client, max_chars = 4).synthesize(sentences("一文目。二文目。", "三文目。"))
    assert result.failures == [("一文目。二文目。", "bad")]
    assert len(result.segments) == 1


def test_synthesize_counts_split_line_as_one_failure():
    client = FakeSynthesizer(lambda text: TtsInputError("bad") if text.startswith("い") else None)
    lines = ["い。" * 4] * (MAX_CONSECUTIVE_FAILURES - 1)
    result = sentence_synthesizer(client, max_chars = 2).synthesize(sentences("ok", *lines, "ok"))
    assert len(result.failures) == MAX_CONSECUTIVE_FAILURES - 1


def test_synthesize_counts_consecutive_failures_across_files():
    synthesizer = sentence_synthesizer(FakeSynthesizer(lambda text: TtsInputError("bad") if text != "ok" else None))
    result = synthesizer.synthesize(sentences("ok", *["あ"] * (MAX_CONSECUTIVE_FAILURES - 1)))
    assert len(result.failures) == MAX_CONSECUTIVE_FAILURES - 1
    with pytest.raises(SynthesisAbortedError, match = "連続"):
        synthesizer.synthesize(sentences("い"))


def test_synthesize_aborts_when_voice_never_succeeded():
    # モデル名や音声名の誤りで 400 が返る場合を想定し、成功したことのない音声での失敗は中断する
    client = FakeSynthesizer(lambda text: TtsInputError("unknown voice") if text == "B" else None)
    speakers = [Sentence(text = "A", speaker = character("A")), Sentence(text = "B", speaker = character("B"))]
    with pytest.raises(SynthesisAbortedError, match = "最初の合成"):
        sentence_synthesizer(client, ["n", "a", "b"]).synthesize(speakers)


def test_synthesize_remembers_succeeded_voices_across_calls():
    synthesizer = sentence_synthesizer(FakeSynthesizer(lambda text: TtsInputError("bad") if text == "い" else None))
    synthesizer.synthesize(sentences("あ"))
    result = synthesizer.synthesize(sentences("い", "う"))
    assert result.failures == [("い", "bad")]


def test_synthesize_propagates_other_errors():
    client = FakeSynthesizer(lambda text: ConnectionError("down"))
    with pytest.raises(ConnectionError):
        sentence_synthesizer(client).synthesize(sentences("あ", "い"))
    assert len(client.requests) == 1


def test_synthesize_splits_long_text_with_same_voice():
    client = FakeSynthesizer()
    result = sentence_synthesizer(client, ["n", "a"], max_chars = 4).synthesize(sentences("一文目。二文目。", speaker = character("A")))
    assert client.requests == [("一文目。", "a"), ("二文目。", "a")]
    assert len(result.segments) == 2


def test_synthesize_reads_failure_limit_at_runtime(monkeypatch):
    monkeypatch.setattr(speech_synthesis, "MAX_CONSECUTIVE_FAILURES", 2)
    client = FakeSynthesizer(lambda text: TtsInputError("bad") if text != "ok" else None)
    with pytest.raises(SynthesisAbortedError, match = "2回連続"):
        sentence_synthesizer(client).synthesize(sentences("ok", "あ", "い", "う"))


# --- 章ごとの書き出し ---

class FakeWriter(AudioWriter):
    def __init__(self, fail: bool = False):
        self.fail = fail
        self.written: Dict[str, int] = {}
        self.removed: List[str] = []

    def write(self, source: str, segments: List[Audio]) -> str:
        if self.fail:
            raise OSError("disk full")
        self.written[source] = len(segments)
        return f"out/{source}.wav"

    def remove(self, source: str) -> None:
        if self.fail:
            raise OSError("is a directory")
        self.removed.append(source)


class RecordingPresenter(Presenter):
    def __init__(self):
        self.events: List[tuple] = []

    def chapter_started(self, index, source, characters):
        self.events.append(("chapter", source, [c.name for c in characters]))

    def character_extraction_failed(self, error):
        self.events.append(("extraction_failed", str(error)))

    def speaker_estimated(self, sentence, confidence):
        self.events.append(("estimated", sentence.text, sentence.speaker.name, confidence))

    def speaker_estimation_failed(self, sentence, error):
        self.events.append(("estimation_failed", sentence.text, str(error) if error else None))

    def context_shrunk(self, previous_count):
        self.events.append(("shrunk", previous_count))

    def audio_exported(self, location, result):
        self.events.append(("exported", location, len(result.failures)))

    def finished(self, characters, voices):
        self.events.append(("finished", [c.name for c in characters], voices))


def exporter(client: SpeechSynthesizer, writer: AudioWriter, presenter: Optional[Presenter] = None) -> AudioExporter:
    presenter = presenter or Presenter()
    return AudioExporter(SentenceSynthesizer(client, VoiceAssigner(["n"]), presenter), writer, presenter)


def test_export_writes_audio():
    writer, presenter = FakeWriter(), RecordingPresenter()
    failed = exporter(FakeSynthesizer(), writer, presenter).export("1.txt", Novel(sentences("あ", "い")))
    assert not failed
    assert writer.written == {"1.txt": 2}
    assert presenter.events == [("exported", "out/1.txt.wav", 0)]


def test_export_reports_partial_failure():
    writer = FakeWriter()
    failed = exporter(FakeSynthesizer(lambda text: TtsInputError("bad") if text == "い" else None), writer).export("1.txt", Novel(sentences("あ", "い")))
    assert failed
    assert writer.written == {"1.txt": 1}


def test_export_removes_stale_output_when_nothing_to_read():
    writer, presenter = FakeWriter(), RecordingPresenter()
    failed = exporter(FakeSynthesizer(), writer, presenter).export("1.txt", Novel(sentences("＊　＊　＊", "")))
    assert not failed
    assert writer.removed == ["1.txt"]
    assert presenter.events == [("exported", None, 0)]


@pytest.mark.parametrize("texts", [["あ"], ["＊　＊　＊"]])
def test_export_aborts_on_writer_error(texts):
    with pytest.raises(SynthesisAbortedError, match = "disk full|is a directory"):
        exporter(FakeSynthesizer(), FakeWriter(fail = True)).export("1.txt", Novel(sentences(*texts)))


def test_export_aborts_on_fatal_synthesis_error_without_writing():
    writer = FakeWriter()
    with pytest.raises(SynthesisAbortedError, match = "down"):
        exporter(FakeSynthesizer(lambda text: ConnectionError("down") if text == "い" else None), writer).export("1.txt", Novel(sentences("あ", "い")))
    assert writer.written == {} and writer.removed == []


# --- 話者の推測 ---

class FakeEstimator(SpeakerEstimator):
    """推測ごとに answers の結果を順に返す。例外は送出する"""

    def __init__(self, *answers):
        self.answers = list(answers)
        self.calls: List[tuple[List[str], str, List[str]]] = []

    def estimate_speaker(self, candidates, previous, target, following):
        self.calls.append(([s.text for s in previous], target.text, [s.text for s in following]))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


def dialogue_novel() -> Novel:
    return Novel(sentences("地の文", "「あ」", "「い」", "「う」"), [character("花子"), character("太郎")])


def test_speaker_estimation_assigns_candidates():
    novel = dialogue_novel()
    presenter = RecordingPresenter()
    estimator = FakeEstimator(SpeakerGuess(1, 0.9), SpeakerGuess(2), SpeakerGuess(0))
    SpeakerEstimation(estimator, presenter).run(novel)
    assert [s.speaker.name for s in novel.sentences] == ["ナレーター", "花子", "太郎", "ナレーター"]
    assert presenter.events[0] == ("estimated", "「あ」", "花子", 0.9)
    # セリフではない文は推測しない
    assert [target for _, target, _ in estimator.calls] == ["「あ」", "「い」", "「う」"]


@pytest.mark.parametrize("answer", [None, SpeakerGuess(3), SpeakerGuess(-1), SpeakerEstimationError("bad response")])
def test_speaker_estimation_falls_back_to_narrator(answer):
    novel = Novel(sentences("「あ」"), [character("花子"), character("太郎")])
    presenter = RecordingPresenter()
    SpeakerEstimation(FakeEstimator(answer), presenter).run(novel)
    assert novel.sentences[0].speaker.is_narrator
    error = "bad response" if isinstance(answer, Exception) else None
    assert presenter.events == [("estimation_failed", "「あ」", error)]


def test_speaker_estimation_passes_context():
    novel = dialogue_novel()
    estimator = FakeEstimator(SpeakerGuess(1), SpeakerGuess(2), SpeakerGuess(1))
    SpeakerEstimation(estimator, Presenter(), previous_count = 2, following_count = 1).run(novel)
    assert estimator.calls[2] == (["「あ」", "「い」"], "「う」", [])
    assert estimator.calls[0] == (["地の文"], "「あ」", ["「い」"])


def test_speaker_estimation_estimates_all_sentences_unless_dialogue_only():
    novel = dialogue_novel()
    estimator = FakeEstimator(*[SpeakerGuess(0)] * 4)
    SpeakerEstimation(estimator, Presenter(), dialogue_only = False).run(novel)
    assert len(estimator.calls) == 4


def test_speaker_estimation_shrinks_context_when_too_long():
    novel = Novel(sentences(*[f"文{i}" for i in range(8)], "「あ」"), [character("花子")])
    presenter = RecordingPresenter()
    estimator = FakeEstimator(ContextTooLongError("long"), ContextTooLongError("long"), SpeakerGuess(1))
    SpeakerEstimation(estimator, presenter, previous_count = 8).run(novel)
    assert [len(previous) for previous, _, _ in estimator.calls] == [8, 4, 2]
    assert novel.sentences[-1].speaker.name == "花子"
    assert presenter.events[:2] == [("shrunk", 4), ("shrunk", 2)]


def test_speaker_estimation_gives_up_when_too_long_without_context():
    novel = Novel(sentences("前", "「あ」"), [character("花子")])
    presenter = RecordingPresenter()
    estimator = FakeEstimator(ContextTooLongError("long"), ContextTooLongError("long"), ContextTooLongError("still long"))
    SpeakerEstimation(estimator, presenter, previous_count = 2).run(novel)
    assert [len(previous) for previous, _, _ in estimator.calls] == [1, 1, 0]
    assert novel.sentences[-1].speaker.is_narrator
    assert presenter.events[-1] == ("estimation_failed", "「あ」", "still long")


def test_speaker_estimation_propagates_other_errors():
    with pytest.raises(ConnectionError):
        SpeakerEstimation(FakeEstimator(ConnectionError("down")), Presenter()).run(Novel(sentences("「あ」")))


# --- 全体の流れ ---

class FakeExtractor(CharacterExtractor):
    def __init__(self, *results):
        self.results = list(results)
        self.known: List[List[str]] = []

    def extract_characters(self, novel, known):
        self.known.append([c.name for c in known])
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def test_narrate_novels_keeps_previous_characters_when_extraction_fails():
    extractor = FakeExtractor([character("花子")], CharacterExtractionError("broken"), [])
    presenter = RecordingPresenter()
    chapters = [Chapter(f"{i}.txt", Novel(sentences("地の文"))) for i in range(3)]
    report = NarrateNovels(extractor, SpeakerEstimation(FakeEstimator(), Presenter()), presenter).run(chapters)
    assert extractor.known == [[], ["花子"], ["花子"]]
    assert [c.name for c in report.characters] == ["花子"]
    assert all([c.name for c in chapter.novel.characters] == ["花子"] for chapter in chapters)
    assert ("extraction_failed", "broken") in presenter.events
    assert presenter.events[-1] == ("finished", ["花子"], None)


def test_narrate_novels_exports_each_chapter():
    extractor = FakeExtractor([character("花子")], [character("花子")])
    writer = FakeWriter()
    presenter = RecordingPresenter()
    client = FakeSynthesizer(lambda text: TtsInputError("bad") if text == "失敗" else None)
    chapters = [Chapter("1.txt", Novel(sentences("あ"))), Chapter("2.txt", Novel(sentences("い", "失敗")))]
    report = NarrateNovels(extractor, SpeakerEstimation(FakeEstimator(), Presenter()), presenter, exporter(client, writer, presenter)).run(chapters)
    assert writer.written == {"1.txt": 1, "2.txt": 1}
    assert report.failed_sources == ["2.txt"]
    assert presenter.events[-1] == ("finished", ["花子"], {"ナレーター": "n"})
