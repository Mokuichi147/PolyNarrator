//! 小説を登場人物ごとの声で読み上げる
//!
//! 依存は外側から内側への一方向だけにする。
//!
//! - [`domain`] … 登場人物・文・音声などの概念と、外部に依存しない規則
//! - [`application`] … ユースケースと、ユースケースが外部に求める機能 (ports)
//! - [`adapters`] … ports を LLM・TTS サーバーやファイルで実装したもの
//! - [`setup`] … 設定から各層の実装を組み立てる。コマンドラインと GUI で共通に使う

pub mod adapters;
pub mod application;
pub mod domain;
pub mod setup;
