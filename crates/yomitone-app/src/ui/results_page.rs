//! 「結果」の画面。登場人物と音声の割り当てを表に出す

use naui::{Length, Result, Sizing, Stack, Table, TableColumn, TableRow, Ui};
use yomitone::domain::character::Character;

use super::parts::{heading, note, page, title};

#[derive(Clone)]
pub(crate) struct ResultsView {
    characters: Table,
    voices: Table,
}

impl ResultsView {
    pub(crate) fn set_characters(&self, characters: &[Character]) {
        let rows: Vec<TableRow> = characters
            .iter()
            .map(|c| {
                let gender = c.gender.map(|g| g.label()).unwrap_or("不明");
                TableRow::new([c.name.clone(), gender.to_string(), c.aliases.join("、"), c.portrait.clone()])
            })
            .collect();
        self.characters.set_rows(&rows);
    }

    pub(crate) fn set_voices(&self, voices: &[(String, String)]) {
        let rows: Vec<TableRow> =
            voices.iter().map(|(name, voice)| TableRow::new([name.as_str(), voice.as_str()])).collect();
        self.voices.set_rows(&rows);
    }

    pub(crate) fn clear(&self) {
        self.set_characters(&[]);
        self.set_voices(&[]);
    }
}

pub(crate) fn build(ui: &Ui) -> Result<(Stack, ResultsView)> {
    let page = page(ui)?;
    page.append(&title(ui, "結果")?);
    page.append(&note(ui, "最後に読み上げた内容です。登場人物の一覧は章ごとに更新し、後の章に引き継ぎます。")?);

    page.append(&heading(ui, "登場人物")?);
    let characters = ui.table()?;
    characters.set_columns(&[
        TableColumn::new("名前").width(160.0),
        TableColumn::new("性別").width(70.0),
        TableColumn::new("別名").width(180.0),
        TableColumn::new("特徴"),
    ]);
    characters.set_sizing(Sizing::new().width(Length::Fill).height(Length::Fixed(280.0)));
    page.append(&characters);

    page.append(&heading(ui, "音声の割り当て")?);
    page.append(&note(ui, "音声を書き出したときだけ表示します。")?);
    let voices = ui.table()?;
    voices.set_columns(&[TableColumn::new("名前").width(200.0), TableColumn::new("音声")]);
    voices.set_sizing(Sizing::new().width(Length::Fill).height(Length::Fixed(220.0)));
    page.append(&voices);

    Ok((page, ResultsView { characters, voices }))
}
