//! サイドバーと各画面。サイドバーで選んだ画面を右側に出す

mod parts;
mod results_page;
mod run_page;
mod settings_pages;
mod store;

use std::cell::Cell;
use std::rc::Rc;

use naui::{CloseResponse, GridCell, Result, SidebarItem, SidebarSection, ToolbarIcon, Track, Ui, Widget};

use self::parts::scrollable;
use self::store::Store;
use crate::settings::{SpeakerBackend, TtsBackendSetting};

/// サイドバーの項目。並びは画面の並びと同じで、通知される通し番号をそのまま画面の位置に使う
const MAIN_PAGES: [(&str, ToolbarIcon); 2] = [("読み上げ", ToolbarIcon::Open), ("結果", ToolbarIcon::Info)];
const SETTINGS_PAGES: [(&str, ToolbarIcon); 4] = [
    (SpeakerBackend::Llm.page_title(), ToolbarIcon::Settings),
    (SpeakerBackend::Jev.page_title(), ToolbarIcon::Search),
    (TtsBackendSetting::Openai.page_title(), ToolbarIcon::Share),
    (TtsBackendSetting::Wyoming.page_title(), ToolbarIcon::Share),
];
/// 項目名が省略されずに収まる幅
const SIDEBAR_WIDTH: f64 = 250.0;

fn sidebar_sections() -> Vec<SidebarSection> {
    let items = |pages: &[(&str, ToolbarIcon)]| {
        pages.iter().map(|(title, icon)| SidebarItem::new(*title).icon(*icon)).collect::<Vec<_>>()
    };
    vec![SidebarSection::untitled(items(&MAIN_PAGES)), SidebarSection::new("設定", items(&SETTINGS_PAGES))]
}

pub fn build(ui: &Ui) -> Result<()> {
    let store = Store::load();
    let window = ui.window("Yomitone", 980.0, 720.0)?;
    window.set_min_size(760.0, 540.0);

    // 選んだ画面だけを Fill の行へ置く
    let root = ui.grid()?;
    root.set_column_track(0, Track::FILL);
    root.set_row_track(0, Track::FILL);

    let (results, results_view) = results_page::build(ui)?;
    let (run, run_handle) = run_page::build(ui, &store, results_view)?;
    let screens: Vec<Box<dyn Widget>> = vec![
        Box::new(run),
        Box::new(scrollable(ui, &results)?),
        Box::new(scrollable(ui, &settings_pages::llm(ui, &store)?)?),
        Box::new(scrollable(ui, &settings_pages::jev(ui, &store)?)?),
        Box::new(scrollable(ui, &settings_pages::tts(ui, &store, TtsBackendSetting::Openai)?)?),
        Box::new(scrollable(ui, &settings_pages::tts(ui, &store, TtsBackendSetting::Wyoming)?)?),
    ];

    let sidebar = ui.sidebar()?;
    sidebar.set_sections(&sidebar_sections());
    sidebar.set_width(SIDEBAR_WIDTH);
    let go = Rc::new({
        let root = root.clone();
        let sidebar = sidebar.clone();
        let current = Cell::new(None);
        move |index: usize| {
            let Some(screen) = screens.get(index) else {
                return;
            };
            if current.replace(Some(index)) != Some(index) {
                root.replace(screen.as_ref(), GridCell::new(0, 0));
            }
            // 通知しないので、選び直しが回り続けない
            sidebar.set_selected(index);
        }
    });
    go(0);
    sidebar.on_select({
        let go = go.clone();
        move |index| go(index)
    });

    window.on_close_request(move || {
        run_handle.cancel_running();
        CloseResponse::Close
    });
    window.set_child(&root);
    window.set_sidebar(&sidebar);
    window.show();
    Ok(())
}
