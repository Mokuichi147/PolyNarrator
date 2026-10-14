//! 「読み上げ」の画面。入力と出力先を選び、読み上げを始めて経過を見る

use std::cell::{Cell, RefCell};
use std::collections::VecDeque;
use std::rc::Rc;
use std::time::Duration;

use naui::{
    Button, Grid, GridCell, Label, Orientation, Padding, ProgressBar, Result, Scroll, ScrollPolicy, Sizing, TextColor,
    Toggle, Track, Ui,
};
use yomitone::application::cancel::CancelFlag;

use super::parts::{folder_field, heading, note, title, Form};
use super::results_page::ResultsView;
use super::store::Store;
use crate::progress::Progress;
use crate::settings::{SpeakerBackend, TtsBackendSetting};
use crate::worker::{self, Event, Outcome};

/// 経過の表示に残す行数。長い小説でも表示が重くならないよう古い行から捨てる
const MAX_LOG_LINES: usize = 3000;
/// 経過の表示をまとめて書き換える間隔。1行ごとに書き換えると表示が追いつかない
const LOG_REFRESH: Duration = Duration::from_millis(200);

/// 経過の表示。届いた行をためておき、一定の間隔でまとめて表示する
#[derive(Clone)]
struct LogView {
    label: Label,
    scroll: Scroll,
    lines: Rc<RefCell<VecDeque<String>>>,
    dirty: Rc<Cell<bool>>,
}

impl LogView {
    fn push(&self, line: String) {
        let mut lines = self.lines.borrow_mut();
        lines.push_back(line);
        while lines.len() > MAX_LOG_LINES {
            lines.pop_front();
        }
        self.dirty.set(true);
    }

    fn clear(&self) {
        self.lines.borrow_mut().clear();
        self.dirty.set(true);
        self.flush();
    }

    fn flush(&self) {
        if !self.dirty.replace(false) {
            return;
        }
        let text = self.lines.borrow().iter().map(String::as_str).collect::<Vec<_>>().join("\n");
        self.label.set_text(&text);
        self.scroll.scroll_to_end();
    }
}

/// 読み上げの状態と、それを表すウィジェット
struct RunView {
    ui: Ui,
    store: Store,
    start: Button,
    stop: Button,
    bar: ProgressBar,
    status: Label,
    log: LogView,
    results: ResultsView,
    progress: RefCell<Progress>,
    /// 読み上げている間だけ持つ中止のフラグ
    running: RefCell<Option<CancelFlag>>,
}

impl RunView {
    fn set_status(&self, text: &str, color: TextColor) {
        self.status.set_text(text);
        self.status.set_color(color);
    }

    fn show_progress(&self) {
        let progress = self.progress.borrow();
        self.bar.set_value(progress.fraction());
        self.set_status(&progress.status(), TextColor::Default);
    }

    fn start(self: &Rc<Self>) {
        if self.running.borrow().is_some() {
            return;
        }
        let settings = self.store.snapshot();
        if let Err(message) = settings.validate() {
            self.set_status(&message, TextColor::Danger);
            return;
        }

        let cancel = CancelFlag::new();
        self.running.replace(Some(cancel.clone()));
        self.start.set_enabled(false);
        self.stop.set_enabled(true);
        self.log.clear();
        self.results.clear();
        self.progress.replace(Progress::default());
        self.bar.set_value(0.0);
        self.set_status("準備しています", TextColor::Default);

        // 受け取る側は UI スレッドに置き、送る側だけを処理のスレッドへ渡す
        let sender = self.ui.tasks().channel({
            let view = self.clone();
            move |event: Event| view.handle(event)
        });
        std::thread::spawn(move || {
            worker::run(settings, cancel, move |event| {
                // 画面を閉じた後に届いた経過は捨てる
                let _ = sender.send(event);
            })
        });
    }

    fn stop(&self) {
        if let Some(cancel) = self.running.borrow().as_ref() {
            cancel.cancel();
            self.stop.set_enabled(false);
            self.set_status("中止しています(今の文を終えたところで止まります)", TextColor::Secondary);
        }
    }

    fn handle(&self, event: Event) {
        match event {
            Event::Log(line) => self.log.push(line),
            Event::Loaded { counts, synthesize } => {
                self.progress.replace(Progress::new(counts, synthesize));
                self.show_progress();
            }
            Event::ChapterStarted { index, characters } => {
                self.progress.borrow_mut().chapter_started(index);
                self.results.set_characters(&characters);
                self.show_progress();
            }
            Event::SentenceEstimated => {
                self.progress.borrow_mut().sentence_estimated();
                self.show_progress();
            }
            Event::SynthesisProgressed { done, .. } => {
                self.progress.borrow_mut().synthesis_progressed(done);
                self.show_progress();
            }
            Event::Finished { characters, voices } => {
                self.results.set_characters(&characters);
                self.results.set_voices(&voices.unwrap_or_default());
            }
            Event::Done(outcome) => self.finish(outcome),
        }
    }

    fn finish(&self, outcome: Outcome) {
        self.running.replace(None);
        self.start.set_enabled(true);
        self.stop.set_enabled(false);
        let (message, color) = match outcome {
            Outcome::Completed => {
                self.bar.set_value(1.0);
                ("完了しました".to_string(), TextColor::Success)
            }
            Outcome::PartiallyFailed(sources) => {
                self.bar.set_value(1.0);
                let names: Vec<String> = sources
                    .iter()
                    .map(|s| {
                        std::path::Path::new(s)
                            .file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or(s.clone())
                    })
                    .collect();
                (format!("完了しました。音声合成に失敗した文がある章: {}", names.join(", ")), TextColor::Warning)
            }
            Outcome::Cancelled => ("中止しました".to_string(), TextColor::Secondary),
            Outcome::Failed(error) => (error, TextColor::Danger),
        };
        self.log.push(message.clone());
        self.log.flush();
        self.set_status(&message, color);
    }

    /// ウィンドウを閉じるときに、走っている処理を止める
    pub(crate) fn cancel_running(&self) {
        if let Some(cancel) = self.running.borrow().as_ref() {
            cancel.cancel();
        }
    }
}

/// 閉じるときに処理を止めるための取っ手
pub(crate) struct RunHandle(Rc<RunView>);

impl RunHandle {
    pub(crate) fn cancel_running(&self) {
        self.0.cancel_running();
    }
}

pub(crate) fn build(ui: &Ui, store: &Store, results: ResultsView) -> Result<(Grid, RunHandle)> {
    // 経過の表示を残りの高さいっぱいに広げるため、縦に並べるのにも Grid を使う
    let page = ui.grid()?;
    page.set_spacing(0.0, 12.0);
    page.set_padding(Padding::all(20.0));
    page.set_column_track(0, Track::FILL);
    page.set_sizing(Sizing::fill());
    let rows = Cell::new(0);
    // 次の行へ置き、置いた行を返す
    let append = |widget: &dyn naui::Widget| {
        let row = rows.get();
        page.attach(widget, GridCell::new(0, row));
        rows.set(row + 1);
        row
    };

    append(&title(ui, "読み上げ")?);
    append(&note(
        ui,
        "章ごとに分けた UTF-8 の .txt ファイルを置いたフォルダを選び、開始を押します。\
         ファイル名の順に処理し、行全体が「」で囲まれた行を誰のセリフか推測します。\
         接続先などは左の「設定」で変えられます。",
    )?);

    let form = Form::new(ui)?;
    let (input_row, _, _) = folder_field(
        ui,
        store,
        |s| s.input_folder.clone(),
        |s: &mut crate::settings::AppSettings, v| s.input_folder = v,
    )?;
    form.add(ui, "入力フォルダ", &input_row)?;

    let speakers = [SpeakerBackend::Llm, SpeakerBackend::Jev];
    let speaker = ui.combo_box()?;
    speaker.set_items(&speakers.map(SpeakerBackend::page_title));
    speaker.set_selected(speakers.iter().position(|&b| b == store.read(|s| s.speaker_backend)).unwrap_or(0));
    speaker.on_select({
        let store = store.clone();
        move |index| store.update(|s| s.speaker_backend = speakers[index.min(speakers.len() - 1)])
    });
    form.add(ui, "話者推測", &speaker)?;
    form.note(ui, "セリフの話者を推測するのに使います。登場人物の抽出には常に LLM を使います。")?;

    let export: Toggle = ui.toggle("音声合成して WAV を書き出す")?;
    export.set_on(store.read(|s| s.export_audio));
    form.add(ui, "音声", &export)?;
    let backends = [TtsBackendSetting::Openai, TtsBackendSetting::Wyoming];
    let tts = ui.combo_box()?;
    tts.set_items(&backends.map(TtsBackendSetting::page_title));
    tts.set_selected(backends.iter().position(|&b| b == store.read(|s| s.tts_backend)).unwrap_or(0));
    tts.on_select({
        let store = store.clone();
        move |index| store.update(|s| s.tts_backend = backends[index.min(backends.len() - 1)])
    });
    form.add(ui, "TTS", &tts)?;
    let (output_row, output_input, output_picker) = folder_field(
        ui,
        store,
        |s| s.output_folder.clone(),
        |s: &mut crate::settings::AppSettings, v| s.output_folder = v,
    )?;
    form.add(ui, "出力フォルダ", &output_row)?;
    form.note(ui, "章ごとに <入力ファイル名>.wav を書き出します。")?;
    let set_output_enabled = move |enabled: bool| {
        tts.set_enabled(enabled);
        output_input.set_enabled(enabled);
        output_picker.set_enabled(enabled);
    };
    set_output_enabled(store.read(|s| s.export_audio));
    export.on_toggle({
        let store = store.clone();
        move |on| {
            store.update(|s| s.export_audio = on);
            set_output_enabled(on);
        }
    });
    append(form.widget());

    let buttons = ui.stack(Orientation::Horizontal)?;
    buttons.set_spacing(8.0);
    let start = ui.button("開始")?;
    let stop = ui.button("中止")?;
    stop.set_enabled(false);
    buttons.append(&start);
    buttons.append(&stop);
    append(&buttons);

    let bar = ui.progress_bar()?;
    bar.set_sizing(Sizing::fill_width());
    append(&bar);
    let status = ui.label("")?;
    status.set_wrap(true);
    status.set_sizing(Sizing::fill_width());
    append(&status);

    append(&heading(ui, "経過")?);
    let log_label = ui.label("")?;
    log_label.set_wrap(true);
    log_label.set_selectable(true);
    log_label.set_sizing(Sizing::fill_width());
    let log_scroll = ui.scroll()?;
    log_scroll.set_policy(ScrollPolicy::Never, ScrollPolicy::Auto);
    log_scroll.set_child(&log_label);
    log_scroll.set_sizing(Sizing::fill().min_height(160.0));
    let log_row = append(&log_scroll);
    page.set_row_track(log_row, Track::FILL);

    let log = LogView { label: log_label, scroll: log_scroll, lines: Rc::default(), dirty: Rc::default() };
    ui.tasks().every(LOG_REFRESH, {
        let log = log.clone();
        move || log.flush()
    });

    let view = Rc::new(RunView {
        ui: ui.clone(),
        store: store.clone(),
        start: start.clone(),
        stop: stop.clone(),
        bar,
        status,
        log,
        results,
        progress: RefCell::default(),
        running: RefCell::default(),
    });
    start.on_click({
        let view = view.clone();
        move || view.start()
    });
    stop.on_click({
        let view = view.clone();
        move || view.stop()
    });
    Ok((page, RunHandle(view)))
}
