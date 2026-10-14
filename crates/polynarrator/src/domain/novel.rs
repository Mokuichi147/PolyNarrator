use crate::domain::character::Character;
use crate::domain::text::strip;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Sentence {
    pub text: String,
    pub speaker: Option<Character>,
}

impl Sentence {
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into(), speaker: None }
    }

    pub fn spoken_by(text: impl Into<String>, speaker: Character) -> Self {
        Self { text: text.into(), speaker: Some(speaker) }
    }

    /// 行全体が「」で囲まれたセリフか
    pub fn is_dialogue(&self) -> bool {
        self.text.starts_with('「') && self.text.ends_with('」')
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Novel {
    pub sentences: Vec<Sentence>,
    pub characters: Vec<Character>,
}

impl Novel {
    pub fn new(sentences: Vec<Sentence>) -> Self {
        Self { sentences, characters: Vec::new() }
    }

    /// 1行を1つの文として扱う
    pub fn from_lines<S: AsRef<str>>(lines: impl IntoIterator<Item = S>) -> Self {
        Self::new(lines.into_iter().map(|line| Sentence::new(strip(line.as_ref()))).collect())
    }
}
