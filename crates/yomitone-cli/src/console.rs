use std::cell::RefCell;
use std::io::{self, Write};

use indexmap::IndexMap;
use yomitone::application::ports::{Presenter, SynthesisResult};
use yomitone::domain::character::Character;
use yomitone::domain::novel::Sentence;

fn speaker_name(sentence: &Sentence) -> &str {
    sentence.speaker.as_ref().map(|s| s.name.as_str()).unwrap_or_default()
}

fn character_line(character: &Character) -> String {
    let gender = character.gender.map(|g| g.label()).unwrap_or("不明");
    format!("{} ({gender}) [{}]", character.name, character.aliases.join(", "))
}

/// 処理の経過を表示する。既定では標準出力に書き出す
pub struct ConsolePresenter {
    out: RefCell<Box<dyn Write>>,
}

impl Default for ConsolePresenter {
    fn default() -> Self {
        Self::new(Box::new(io::stdout()))
    }
}

impl ConsolePresenter {
    pub fn new(out: Box<dyn Write>) -> Self {
        Self { out: RefCell::new(out) }
    }

    fn print(&self, text: impl AsRef<str>) {
        let mut out = self.out.borrow_mut();
        // 表示できなくても処理は続ける
        let _ = writeln!(out, "{}", text.as_ref());
        let _ = out.flush();
    }
}

impl Presenter for ConsolePresenter {
    fn chapter_started(&self, index: usize, source: &str, characters: &[Character]) {
        self.print(format!("{} {source}", index + 1));
        let lines: Vec<String> = characters.iter().map(|c| format!("  - {}", character_line(c))).collect();
        self.print(lines.join("\n"));
        self.print("");
    }

    fn character_extraction_failed(&self, error: &str) {
        self.print(format!("登場人物を抽出できませんでした: {error}"));
    }

    fn narration_assigned(&self, sentence: &Sentence) {
        self.print(format!("{}\t{}", speaker_name(sentence), sentence.text));
    }

    fn speaker_estimated(&self, sentence: &Sentence, confidence: Option<f64>) {
        match confidence {
            None => self.print(format!("{}\t{}", speaker_name(sentence), sentence.text)),
            Some(confidence) => self.print(format!("{}\t({confidence:.2})\t{}", speaker_name(sentence), sentence.text)),
        }
    }

    fn speaker_estimation_failed(&self, sentence: &Sentence, error: Option<&str>) {
        if let Some(error) = error {
            self.print(error);
        }
        self.print(format!("失敗\t{}\t{}", speaker_name(sentence), sentence.text));
    }

    fn context_shrunk(&self, previous_count: usize) {
        self.print(format!("入力長超過のため、履歴を直前{previous_count}文に縮小して再試行します"));
    }

    fn sentence_synthesis_failed(&self, voice: &str, text: &str, error: &str) {
        self.print(format!("音声合成に失敗しました ({voice}): {error}\t{text}"));
    }

    fn audio_exported(&self, location: Option<&str>, result: &SynthesisResult) {
        match location {
            None if !result.failures.is_empty() => {
                self.print("全ての文の合成に失敗したため、音声を書き出しませんでした")
            }
            None => self.print("読み上げられる文がないため、音声を書き出しませんでした"),
            Some(location) if !result.failures.is_empty() => {
                self.print(format!("音声を書き出しました(合成に失敗した{}文を除く): {location}", result.failures.len()))
            }
            Some(location) => self.print(format!("音声を書き出しました: {location}")),
        }
        if result.skipped > 0 {
            self.print(format!("記号だけで読み上げる文字がない{}行は読み飛ばしました", result.skipped));
        }
    }

    fn finished(&self, characters: &[Character], voices: Option<&IndexMap<String, String>>) {
        self.print("\n登場人物一覧");
        for character in characters {
            self.print(format!("- {}", character_line(character)));
        }

        if let Some(voices) = voices {
            self.print("\n音声の割り当て");
            for (name, voice) in voices {
                self.print(format!("- {name}: {voice}"));
            }
        }
    }
}
