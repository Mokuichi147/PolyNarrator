//! Yomitone のデスクトップアプリ。naui で各 OS のネイティブ UI を使う
//!
//! - [`settings`] … 画面で入力する設定と、その保存
//! - [`worker`] … 読み上げの処理を別のスレッドで動かし、経過を画面へ送る
//! - [`progress`] … 進捗バーと状況の表示に使う、処理の進み具合
//! - `ui` … サイドバーと各画面

pub mod progress;
pub mod settings;
mod ui;
pub mod worker;

pub use ui::build;
