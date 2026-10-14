//! 画面の組み方をそろえるための部品

use std::cell::Cell;

use naui::{
    Align, FilePickerMode, Grid, GridCell, Label, Length, Orientation, Padding, PasswordInput, Result, Scroll,
    ScrollPolicy, Sizing, Stack, TextColor, TextInput, TextStyle, Track, Ui, Widget,
};

use super::store::Store;
use crate::settings::AppSettings;

/// 画面の土台。左端をそろえて上から積む
pub(crate) fn page(ui: &Ui) -> Result<Stack> {
    let page = ui.stack(Orientation::Vertical)?;
    page.set_spacing(12.0);
    page.set_padding(Padding::all(20.0));
    page.set_align(Align::Start);
    page.set_sizing(Sizing::fill_width());
    Ok(page)
}

/// 画面をスクロールに載せる。横はウィンドウの幅に合わせ、縦だけを送る
pub(crate) fn scrollable(ui: &Ui, page: &dyn Widget) -> Result<Scroll> {
    let scroll = ui.scroll()?;
    scroll.set_policy(ScrollPolicy::Never, ScrollPolicy::Auto);
    scroll.set_child(page);
    scroll.set_sizing(Sizing::fill());
    Ok(scroll)
}

pub(crate) fn title(ui: &Ui, text: &str) -> Result<Label> {
    let label = ui.label(text)?;
    label.set_style(TextStyle::Title);
    Ok(label)
}

pub(crate) fn heading(ui: &Ui, text: &str) -> Result<Label> {
    let label = ui.label(text)?;
    label.set_style(TextStyle::Heading);
    Ok(label)
}

/// 説明の文。小さく淡い文字にして、窓を狭めても読めるよう折り返す
pub(crate) fn note(ui: &Ui, text: &str) -> Result<Label> {
    let label = ui.label(text)?;
    label.set_style(TextStyle::Caption);
    label.set_color(TextColor::Secondary);
    label.set_wrap(true);
    label.set_sizing(Sizing::fill_width());
    Ok(label)
}

/// 左に項目名、右に入力欄を並べる表
pub(crate) struct Form {
    grid: Grid,
    rows: Cell<usize>,
}

impl Form {
    pub(crate) fn new(ui: &Ui) -> Result<Self> {
        let grid = ui.grid()?;
        grid.set_spacing(12.0, 8.0);
        grid.set_column_track(0, Track::Fixed(170.0));
        grid.set_column_track(1, Track::FILL);
        grid.set_sizing(Sizing::fill_width());
        Ok(Self { grid, rows: Cell::new(0) })
    }

    pub(crate) fn widget(&self) -> &Grid {
        &self.grid
    }

    fn next_row(&self) -> usize {
        let row = self.rows.get();
        self.rows.set(row + 1);
        row
    }

    pub(crate) fn add(&self, ui: &Ui, label: &str, field: &dyn Widget) -> Result<()> {
        let row = self.next_row();
        self.grid.attach(&ui.label(label)?, GridCell::new(0, row));
        self.grid.attach(field, GridCell::new(1, row));
        Ok(())
    }

    /// 直前の項目の説明
    pub(crate) fn note(&self, ui: &Ui, text: &str) -> Result<()> {
        let row = self.next_row();
        self.grid.attach(&note(ui, text)?, GridCell::new(1, row));
        Ok(())
    }
}

/// 設定の文字列を編集する入力欄。打つたびに保存する
pub(crate) fn text_field(
    ui: &Ui,
    store: &Store,
    get: impl Fn(&AppSettings) -> String,
    set: impl Fn(&mut AppSettings, String) + 'static,
    placeholder: &str,
) -> Result<TextInput> {
    let input = ui.text_input(&store.read(get))?;
    input.set_placeholder(placeholder);
    input.set_sizing(Sizing::fill_width());
    input.on_change({
        let store = store.clone();
        move |text| store.update(|s| set(s, text.to_string()))
    });
    Ok(input)
}

/// API キーの入力欄。打った文字は伏せ、ファイルには保存しない
pub(crate) fn secret_field(
    ui: &Ui,
    store: &Store,
    set: impl Fn(&mut AppSettings, String) + 'static,
    placeholder: &str,
) -> Result<PasswordInput> {
    let input = ui.password_input()?;
    input.set_placeholder(placeholder);
    input.set_sizing(Sizing::fill_width());
    input.on_change({
        let store = store.clone();
        move |text| store.update(|s| set(s, text.to_string()))
    });
    Ok(input)
}

/// フォルダのパスを入力する欄と、選ぶダイアログを開くボタン
pub(crate) fn folder_field(
    ui: &Ui,
    store: &Store,
    get: impl Fn(&AppSettings) -> String,
    set: impl Fn(&mut AppSettings, String) + Clone + 'static,
) -> Result<(Stack, TextInput, naui::FilePicker)> {
    let row = ui.stack(Orientation::Horizontal)?;
    row.set_spacing(8.0);
    row.set_sizing(Sizing::fill_width());
    let input = text_field(ui, store, get, set.clone(), "フォルダのパス")?;
    let picker = ui.file_picker("選択…")?;
    picker.set_mode(FilePickerMode::Folder);
    picker.on_select({
        let store = store.clone();
        let input = input.clone();
        move |entries| {
            let Some(path) = entries.first().and_then(|e| e.path()) else {
                return;
            };
            let path = path.display().to_string();
            // 入力欄の書き換えは通知されないので、設定もここで更新する
            input.set_text(&path);
            store.update(|s| set(s, path));
        }
    });
    row.append(&input);
    row.append(&picker);
    Ok((row, input, picker))
}

/// 数値の入力欄
pub(crate) fn number_field(
    ui: &Ui,
    value: f64,
    range: (Option<f64>, Option<f64>),
    step: f64,
    decimals: u32,
) -> Result<naui::NumberInput> {
    // 既定は整数なので、小数桁を決めてから値を入れる(先に入れると丸められる)
    let input = ui.number_input(0.0)?;
    input.set_decimals(decimals);
    input.set_step(step);
    input.set_range(range.0, range.1);
    input.set_value(value);
    input.set_sizing(Sizing::new().width(Length::Fixed(140.0)));
    Ok(input)
}
