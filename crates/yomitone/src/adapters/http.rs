//! HTTP のリクエストと、一時的な失敗の再試行

use std::io::Read;
use std::thread;
use std::time::{Duration, Instant};

/// HTTP サーバーへの送信に失敗した理由
#[derive(Debug)]
pub enum HttpFailure {
    /// 失敗を表すステータスコードが返された
    Status { status: u16, body: String, request_id: Option<String> },
    /// 接続できなかった、または応答を読めなかった
    Transport { message: String, timeout: bool },
}

/// 再試行の方針
pub struct RetryPolicy {
    /// 最初の送信のあとに再試行する回数
    pub max_retries: u32,
    pub retry_status: fn(u16) -> bool,
    pub backoff_initial: Duration,
    pub backoff_max: Duration,
    /// Retry-After をこの長さまで待つ。超える場合は通常の間隔で再試行する
    pub retry_after_max: Option<Duration>,
    /// 最初の送信から数えて、再試行を始められる時間の上限
    pub budget: Option<Duration>,
}

fn retry_after(response: &ureq::Response) -> Option<Duration> {
    if let Some(ms) = response.header("retry-after-ms").and_then(|v| v.trim().parse::<f64>().ok()) {
        if ms.is_finite() && ms >= 0.0 {
            return Some(Duration::from_secs_f64(ms / 1000.0));
        }
    }
    response
        .header("retry-after")
        .and_then(|v| v.trim().parse::<f64>().ok())
        .filter(|s| s.is_finite() && *s >= 0.0)
        .map(Duration::from_secs_f64)
}

fn is_timeout(error: &ureq::Transport) -> bool {
    let mut source: Option<&(dyn std::error::Error + 'static)> = std::error::Error::source(error);
    while let Some(error) = source {
        if let Some(io) = error.downcast_ref::<std::io::Error>() {
            if matches!(io.kind(), std::io::ErrorKind::TimedOut | std::io::ErrorKind::WouldBlock) {
                return true;
            }
        }
        source = error.source();
    }
    error.to_string().contains("timed out")
}

/// エラー応答の本文を読む。大きすぎる本文は途中までにする
fn read_error_body(response: ureq::Response) -> String {
    let mut body = Vec::new();
    let _ = response.into_reader().take(1 << 20).read_to_end(&mut body);
    String::from_utf8_lossy(&body).into_owned()
}

/// 送信し、失敗した送信を方針に従って再試行する。body があれば JSON の本文として送る。
/// request は送信のたびに呼ばれ、引数は何回目の再試行か(最初は 0)
pub fn send_with_retry(
    policy: &RetryPolicy,
    mut request: impl FnMut(u32) -> ureq::Request,
    body: Option<&serde_json::Value>,
) -> Result<ureq::Response, HttpFailure> {
    let started = Instant::now();
    let mut attempt = 0;
    loop {
        let sent = match body {
            Some(body) => request(attempt).send_json(body),
            None => request(attempt).call(),
        };
        let (failure, server_delay) = match sent {
            Ok(response) => return Ok(response),
            Err(ureq::Error::Status(status, response)) => {
                let delay = retry_after(&response);
                let request_id = response.header("x-typesafe-request-id").map(str::to_string);
                let retry = (policy.retry_status)(status);
                let failure = HttpFailure::Status { status, body: read_error_body(response), request_id };
                if !retry {
                    return Err(failure);
                }
                (failure, delay)
            }
            Err(ureq::Error::Transport(error)) => {
                let timeout = is_timeout(&error);
                (HttpFailure::Transport { message: error.to_string(), timeout }, None)
            }
        };
        if attempt >= policy.max_retries {
            return Err(failure);
        }

        let backoff = policy.backoff_initial.saturating_mul(1 << attempt.min(16)).min(policy.backoff_max);
        let delay = match (server_delay, policy.retry_after_max) {
            (Some(delay), Some(max)) if delay <= max => delay,
            (Some(delay), None) => delay,
            _ => backoff,
        };
        if let Some(budget) = policy.budget {
            if started.elapsed() + delay >= budget {
                return Err(failure);
            }
        }
        thread::sleep(delay);
        attempt += 1;
    }
}

/// OpenAI の公式 SDK と同じ再試行の方針
pub const OPENAI_RETRY: RetryPolicy = RetryPolicy {
    max_retries: 2,
    retry_status: |status| matches!(status, 408 | 409 | 429) || status >= 500,
    backoff_initial: Duration::from_millis(500),
    backoff_max: Duration::from_secs(8),
    retry_after_max: Some(Duration::from_secs(60)),
    budget: None,
};

/// OpenAI 互換 API のエラーを、公式 SDK と同じ「Error code: 400 - 本文」の形にする
pub fn openai_error_message(failure: &HttpFailure) -> String {
    match failure {
        HttpFailure::Status { status, body, .. } if body.is_empty() => format!("Error code: {status}"),
        HttpFailure::Status { status, body, .. } => format!("Error code: {status} - {body}"),
        HttpFailure::Transport { timeout: true, .. } => "Request timed out.".to_string(),
        HttpFailure::Transport { message, .. } => format!("Connection error: {message}"),
    }
}
