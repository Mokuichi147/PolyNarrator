use std::cell::RefCell;
use std::path::PathBuf;
use std::rc::Rc;

use crate::settings::{self, AppSettings};

/// 画面で共有する設定。変えるたびにファイルへ保存する
#[derive(Clone)]
pub(crate) struct Store(Rc<Inner>);

struct Inner {
    settings: RefCell<AppSettings>,
    path: Option<PathBuf>,
}

impl Store {
    pub(crate) fn load() -> Self {
        let path = settings::default_path();
        let settings = path.as_deref().map(settings::load).unwrap_or_default();
        Self(Rc::new(Inner { settings: RefCell::new(settings), path }))
    }

    /// 今の設定の写し。読み上げを始めるときに渡す
    pub(crate) fn snapshot(&self) -> AppSettings {
        self.0.settings.borrow().clone()
    }

    pub(crate) fn read<R>(&self, f: impl FnOnce(&AppSettings) -> R) -> R {
        f(&self.0.settings.borrow())
    }

    pub(crate) fn update(&self, f: impl FnOnce(&mut AppSettings)) {
        f(&mut self.0.settings.borrow_mut());
        if let Some(path) = &self.0.path {
            // 保存できなくても、この起動の間は入力した設定で動かせる
            if let Err(error) = settings::save(path, &self.0.settings.borrow()) {
                eprintln!("設定を保存できませんでした({}): {error}", path.display());
            }
        }
    }
}
