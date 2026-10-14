use std::fmt;

pub const NARRATOR_NAME: &str = "ナレーター";

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Gender {
    Male,
    Female,
    Other,
}

impl Gender {
    pub const ALL: [Gender; 3] = [Gender::Male, Gender::Female, Gender::Other];

    /// LLM の応答や画面の表示に使う値
    pub fn label(self) -> &'static str {
        match self {
            Gender::Male => "男性",
            Gender::Female => "女性",
            Gender::Other => "その他",
        }
    }

    pub fn from_label(label: &str) -> Option<Gender> {
        Gender::ALL.into_iter().find(|g| g.label() == label)
    }
}

impl fmt::Display for Gender {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.label())
    }
}

/// 登場人物。これらの情報を元にセリフの話者を推測し、音声を割り当てる
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Character {
    pub name: String,
    pub portrait: String,
    pub aliases: Vec<String>,
    pub gender: Option<Gender>,
}

impl Character {
    pub fn new(name: impl Into<String>, portrait: impl Into<String>) -> Self {
        Self { name: name.into(), portrait: portrait.into(), ..Self::default() }
    }

    pub fn with_aliases<S: Into<String>>(mut self, aliases: impl IntoIterator<Item = S>) -> Self {
        self.aliases = aliases.into_iter().map(Into::into).collect();
        self
    }

    pub fn with_gender(mut self, gender: Option<Gender>) -> Self {
        self.gender = gender;
        self
    }

    pub fn is_narrator(&self) -> bool {
        self.name == NARRATOR_NAME
    }

    /// 誰のセリフでもない文を読むナレーター
    pub fn narrator() -> Self {
        Self::new(NARRATOR_NAME, "世界観の説明などキャラクターの発言ではない内容のナレーションを行う")
    }
}
