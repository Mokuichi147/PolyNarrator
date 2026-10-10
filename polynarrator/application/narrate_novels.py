from dataclasses import dataclass, field
from typing import List, Optional

from polynarrator.application.ports import CharacterExtractionError, CharacterExtractor, Presenter
from polynarrator.application.speaker_estimation import SpeakerEstimation
from polynarrator.application.speech_synthesis import AudioExporter
from polynarrator.domain.character import Character
from polynarrator.domain.novel import Novel


@dataclass
class Chapter:
    # 入力元を表す名前。表示と出力先の決定に使う
    source: str
    novel: Novel


@dataclass
class NarrationReport:
    characters: List[Character] = field(default_factory = list)
    # 合成に失敗した文がある章の入力元
    failed_sources: List[str] = field(default_factory = list)


class NarrateNovels:
    """章ごとに登場人物を抽出して話者を推測し、指定があれば音声を書き出す。登場人物の一覧は後の章に引き継ぐ"""

    def __init__(
        self,
        extractor: CharacterExtractor,
        speaker_estimation: SpeakerEstimation,
        presenter: Presenter,
        exporter: Optional[AudioExporter] = None,
    ):
        self.extractor = extractor
        self.speaker_estimation = speaker_estimation
        self.presenter = presenter
        self.exporter = exporter

    def run(self, chapters: List[Chapter]) -> NarrationReport:
        report = NarrationReport()
        for index, chapter in enumerate(chapters):
            try:
                extracted = self.extractor.extract_characters(chapter.novel, report.characters)
            except CharacterExtractionError as e:
                self.presenter.character_extraction_failed(e)
                extracted = []
            if len(extracted) > 0:
                report.characters = extracted
            self.presenter.chapter_started(index, chapter.source, report.characters)

            chapter.novel.characters = report.characters
            self.speaker_estimation.run(chapter.novel)

            if self.exporter is not None and self.exporter.export(chapter.source, chapter.novel):
                report.failed_sources.append(chapter.source)

        voices = self.exporter.synthesizer.assigner.assigned if self.exporter is not None else None
        self.presenter.finished(report.characters, voices)
        return report
