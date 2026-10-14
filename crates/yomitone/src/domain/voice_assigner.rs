use std::collections::{HashMap, HashSet};

use indexmap::IndexMap;
use thiserror::Error;
use unicode_normalization::UnicodeNormalization;

use crate::domain::character::{Character, Gender, NARRATOR_NAME};
use crate::domain::text::strip;

#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum VoiceAssignerError {
    #[error("利用できる音声がありません")]
    NoVoices,
    #[error("音声一覧にない音声が指定されています: {}", .0.join(", "))]
    UnknownVoices(Vec<String>),
}

/// 登場人物ごとに音声を割り当てる。同じ登場人物には常に同じ音声を使う
#[derive(Debug, Clone)]
pub struct VoiceAssigner {
    voices: Vec<String>,
    narrator_voice: String,
    male_voices: Vec<String>,
    female_voices: Vec<String>,
    assigned: IndexMap<String, String>,
    usage: HashMap<String, usize>,
    // 人物ごとの音声と、名前・別名から人物への対応。複数の人物が持ち、どの人物か決められない別名は None にする
    person_voices: Vec<String>,
    name_persons: HashMap<String, usize>,
    alias_persons: HashMap<String, Option<usize>>,
}

impl VoiceAssigner {
    pub fn new(
        voices: Vec<String>,
        narrator_voice: Option<String>,
        male_voices: Vec<String>,
        female_voices: Vec<String>,
    ) -> Result<Self, VoiceAssignerError> {
        let Some(first) = voices.first() else {
            return Err(VoiceAssignerError::NoVoices);
        };
        let narrator_voice = narrator_voice.unwrap_or_else(|| first.clone());
        let unknown: Vec<String> = std::iter::once(&narrator_voice)
            .chain(&male_voices)
            .chain(&female_voices)
            .filter(|v| !voices.contains(v))
            .cloned()
            .collect();
        if !unknown.is_empty() {
            return Err(VoiceAssignerError::UnknownVoices(unknown));
        }

        Ok(Self {
            assigned: IndexMap::from([(NARRATOR_NAME.to_string(), narrator_voice.clone())]),
            usage: HashMap::from([(narrator_voice.clone(), 1)]),
            voices,
            narrator_voice,
            male_voices,
            female_voices,
            person_voices: Vec::new(),
            name_persons: HashMap::new(),
            alias_persons: HashMap::new(),
        })
    }

    /// 全ての音声を一覧の先頭をナレーターとして使う
    pub fn with_voices<S: Into<String>>(voices: impl IntoIterator<Item = S>) -> Result<Self, VoiceAssignerError> {
        Self::new(voices.into_iter().map(Into::into).collect(), None, Vec::new(), Vec::new())
    }

    pub fn voices(&self) -> &[String] {
        &self.voices
    }

    pub fn narrator_voice(&self) -> &str {
        &self.narrator_voice
    }

    /// 割り当てた名前ごとの音声。割り当てた順に並ぶ
    pub fn assigned(&self) -> &IndexMap<String, String> {
        &self.assigned
    }

    /// 前後の空白や Unicode の表記の違いで別人とみなさないよう、照合用に名前を正規化する
    fn key(name: &str) -> String {
        strip(&name.nfkc().collect::<String>()).to_string()
    }

    fn gender_voices(&self, gender: Gender) -> &[String] {
        match gender {
            Gender::Male => &self.male_voices,
            Gender::Female => &self.female_voices,
            Gender::Other => &[],
        }
    }

    fn find_person(&self, name: &str, aliases: &[String]) -> Option<usize> {
        // 割り当て済みの名前は、後から別名で別の人物と結び付いても書き出し済みの音声と食い違わないよう元の人物のままにする
        if let Some(&person) = self.name_persons.get(name) {
            return Some(person);
        }
        // ファイルごとの抽出で正式名が揺れても、以前の別名や正式名と一致すれば同じ人物とみなす。
        // 汎用的な呼び名で別人と混同しないよう、別名同士の一致は使わない
        if let Some(&Some(person)) = self.alias_persons.get(name) {
            return Some(person);
        }
        // 別名が複数の人物の正式名に一致する場合は、どの人物か決められないため新しい人物とする
        let candidates: HashSet<usize> = aliases.iter().filter_map(|a| self.name_persons.get(a).copied()).collect();
        if candidates.len() == 1 {
            candidates.into_iter().next()
        } else {
            None
        }
    }

    fn new_person(&mut self, gender: Option<Gender>) -> usize {
        let pool = match gender {
            Some(gender) if !self.gender_voices(gender).is_empty() => self.gender_voices(gender),
            _ => &self.voices,
        };

        // ナレーターと聞き分けられるよう、性別ごとの候補、全ての音声の順にナレーター以外の音声を探す
        let others = |voices: &[String]| -> Vec<String> {
            voices.iter().filter(|v| **v != self.narrator_voice).cloned().collect()
        };
        let mut candidates = others(pool);
        if candidates.is_empty() {
            candidates = others(&self.voices);
        }
        if candidates.is_empty() {
            candidates = vec![self.narrator_voice.clone()];
        }
        let voice =
            candidates.into_iter().min_by_key(|v| self.usage.get(v).copied().unwrap_or(0)).expect("候補は空にならない");
        *self.usage.entry(voice.clone()).or_insert(0) += 1;
        self.person_voices.push(voice);
        self.person_voices.len() - 1
    }

    pub fn assign(&mut self, character: Option<&Character>) -> String {
        // 名前のない登場人物は誰か区別できないため、ナレーターとして読む
        let Some(character) = character else {
            return self.narrator_voice.clone();
        };
        let name = Self::key(&character.name);
        if name == NARRATOR_NAME || name.is_empty() {
            return self.narrator_voice.clone();
        }
        let aliases: Vec<String> = character.aliases.iter().map(|a| Self::key(a)).filter(|a| !a.is_empty()).collect();

        let person = match self.find_person(&name, &aliases) {
            Some(person) => person,
            None => self.new_person(character.gender),
        };

        let voice = self.person_voices[person].clone();
        self.name_persons.insert(name.clone(), person);
        self.assigned.insert(name, voice.clone());
        for alias in aliases {
            match self.alias_persons.get(&alias) {
                None => {
                    self.alias_persons.insert(alias, Some(person));
                }
                Some(&existing) if existing != Some(person) => {
                    self.alias_persons.insert(alias, None);
                }
                Some(_) => {}
            }
        }
        voice
    }
}
