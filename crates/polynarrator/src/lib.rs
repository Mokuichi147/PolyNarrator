//! 小説を登場人物ごとの声で読み上げる
//!
//! 依存は外側から内側への一方向だけにする。
//!
//! - [`domain`] … 登場人物・文・音声などの概念と、外部に依存しない規則
//! - [`application`] … ユースケースと、ユースケースが外部に求める機能 (ports)
//! - [`adapters`] … ports を LLM・TTS サーバーやファイルで実装したもの

pub mod adapters;
pub mod application;
pub mod domain;
