use std::cmp::Ordering;
use std::fs;
use std::io;
use std::path::Path;

use thiserror::Error;
use unicode_general_category::{get_general_category, GeneralCategory};

use crate::domain::novel::Novel;

#[derive(Debug, PartialEq, Eq)]
enum Part {
    Text(String),
    Number(Vec<u8>),
}

fn is_decimal(c: char) -> bool {
    get_general_category(c) == GeneralCategory::DecimalNumber
}

/// 10進数字の値。全角数字などの Unicode の数字は、0〜9 が連続して並ぶ範囲の位置から求める
fn digit_value(c: char) -> u8 {
    if let Some(d) = c.to_digit(10) {
        return d as u8;
    }
    let mut start = c as u32;
    while let Some(previous) = start.checked_sub(1).and_then(char::from_u32) {
        if !is_decimal(previous) {
            break;
        }
        start -= 1;
    }
    ((c as u32 - start) % 10) as u8
}

/// 自然順で比べるため、数字の並びとそれ以外に分ける
fn natural_key(name: &str) -> Vec<Part> {
    let mut parts = vec![Part::Text(String::new())];
    for c in name.chars() {
        match (parts.last_mut(), is_decimal(c)) {
            (Some(Part::Number(digits)), true) => digits.push(digit_value(c)),
            (Some(Part::Text(text)), false) => text.push(c),
            (_, true) => parts.push(Part::Number(vec![digit_value(c)])),
            (_, false) => parts.push(Part::Text(c.to_string())),
        }
    }
    parts
}

fn compare_numbers(a: &[u8], b: &[u8]) -> Ordering {
    let trim = |digits: &[u8]| -> Vec<u8> { digits.iter().copied().skip_while(|&d| d == 0).collect() };
    let (a, b) = (trim(a), trim(b));
    a.len().cmp(&b.len()).then_with(|| a.cmp(&b))
}

fn compare_natural(a: &str, b: &str) -> Ordering {
    let (ka, kb) = (natural_key(a), natural_key(b));
    for (pa, pb) in ka.iter().zip(&kb) {
        let ordering = match (pa, pb) {
            (Part::Text(x), Part::Text(y)) => x.cmp(y),
            (Part::Number(x), Part::Number(y)) => compare_numbers(x, y),
            (Part::Text(_), Part::Number(_)) => Ordering::Less,
            (Part::Number(_), Part::Text(_)) => Ordering::Greater,
        };
        if ordering != Ordering::Equal {
            return ordering;
        }
    }
    ka.len().cmp(&kb.len()).then_with(|| a.cmp(b))
}

/// フォルダ内のテキストファイル名を自然順で返す。.DS_Store や出力済みの音声などは対象外
pub fn list_text_files(folder: &Path) -> io::Result<Vec<String>> {
    let mut files = Vec::new();
    for entry in fs::read_dir(folder)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().into_owned();
        if name.ends_with(".txt") && entry.path().is_file() {
            files.push(name);
        }
    }
    files.sort_by(|a, b| compare_natural(a, b));
    Ok(files)
}

#[derive(Debug, Error)]
pub enum ReadNovelError {
    #[error("{0}")]
    Io(#[from] io::Error),
    #[error("UTF-8 として読み込めません({0}バイト目)")]
    Decode(usize),
}

/// UTF-8 のテキストファイルを1行1文の小説として読み込む。改行は \n・\r\n・\r のいずれも受け付ける
pub fn read_novel(path: &Path) -> Result<Novel, ReadNovelError> {
    let bytes = fs::read(path)?;
    let text = String::from_utf8(bytes).map_err(|e| ReadNovelError::Decode(e.utf8_error().valid_up_to()))?;
    let text = text.replace("\r\n", "\n").replace('\r', "\n");
    Ok(Novel::from_lines(text.split_inclusive('\n')))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn natural_order() {
        let mut names = vec!["10.txt", "2.txt", "1.txt", "a10.txt", "a9.txt", "１１.txt", "３.txt", "b.txt"];
        names.sort_by(|a, b| compare_natural(a, b));
        assert_eq!(names, ["1.txt", "2.txt", "３.txt", "10.txt", "１１.txt", "a9.txt", "a10.txt", "b.txt"]);
    }
}
