use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;

/// 利用者が処理の中止を求めたか。別のスレッドから中止できるよう、複製しても同じ状態を共有する
#[derive(Debug, Clone, Default)]
pub struct CancelFlag(Arc<AtomicBool>);

impl CancelFlag {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    pub fn is_cancelled(&self) -> bool {
        self.0.load(Ordering::SeqCst)
    }
}
