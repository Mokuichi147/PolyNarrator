use std::sync::LazyLock;

use regex::Regex;
use unicode_general_category::{get_general_category, GeneralCategory};

/// Python の `str.isspace` と同じ空白か。Unicode の White_Space に加え、区切り文字の制御コードも含む
pub fn is_space(c: char) -> bool {
    c.is_whitespace() || ('\u{1c}'..='\u{1f}').contains(&c)
}

/// 前後の空白を取り除く。全角空白なども空白として扱う
pub fn strip(text: &str) -> &str {
    text.trim_matches(is_space)
}

fn is_letter_or_number(c: char) -> bool {
    use GeneralCategory::*;
    matches!(
        get_general_category(c),
        UppercaseLetter
            | LowercaseLetter
            | TitlecaseLetter
            | ModifierLetter
            | OtherLetter
            | DecimalNumber
            | LetterNumber
            | OtherNumber
    )
}

fn is_mark(c: char) -> bool {
    use GeneralCategory::*;
    matches!(get_general_category(c), NonspacingMark | SpacingMark | EnclosingMark)
}

/// 読み上げられる文字(文字・数字)を含むか。「＊　＊　＊」や「………」のような記号だけの文は false
pub fn is_readable(text: &str) -> bool {
    text.chars().any(is_letter_or_number)
}

fn char_len(text: &str) -> usize {
    text.chars().count()
}

// 文末記号とその直後の閉じ括弧までを1文とする
static SENTENCE: LazyLock<Regex> =
    LazyLock::new(|| Regex::new(r"(?:[^。！？!?]+[。！？!?]*|[。！？!?]+)[」』）)]*").unwrap());
// 読点とその直後の閉じ括弧までを1区切りとする
static CLAUSE: LazyLock<Regex> = LazyLock::new(|| Regex::new(r"(?:[^、，,]+[、，,]*|[、，,]+)[」』）)]*").unwrap());

fn find_all<'a>(pattern: &Regex, text: &'a str) -> Vec<&'a str> {
    pattern.find_iter(text).map(|m| m.as_str()).collect()
}

/// 区切りを順に max_chars 以下の塊にまとめる。1つで max_chars を超える区切りはそのまま残す
fn pack<S: AsRef<str>>(pieces: &[S], max_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current = String::new();
    let mut current_len = 0;
    for piece in pieces {
        let piece = piece.as_ref();
        let piece_len = char_len(piece);
        if !current.is_empty() && current_len + piece_len > max_chars {
            chunks.push(std::mem::take(&mut current));
            current_len = 0;
        }
        current.push_str(piece);
        current_len += piece_len;
    }
    if !current.is_empty() {
        chunks.push(current);
    }
    chunks
}

/// 濁点などの結合文字を直前の文字と同じ単位にまとめる
fn clusters(text: &str) -> Vec<String> {
    let mut clusters: Vec<String> = Vec::new();
    for c in text.chars() {
        match clusters.last_mut() {
            Some(last) if is_mark(c) => last.push(c),
            _ => clusters.push(c.to_string()),
        }
    }
    clusters
}

/// 文字数で区切る。句読点や括弧などの記号はできるだけ直前の文字と同じ塊に入れ、記号だけの塊ができないようにする
fn split_by_length(text: &str, max_chars: usize) -> Vec<String> {
    let mut units: Vec<Vec<String>> = Vec::new();
    for cluster in clusters(text) {
        match units.last_mut() {
            Some(unit) if !is_readable(&cluster) || !is_readable(&unit.concat()) => unit.push(cluster),
            _ => units.push(vec![cluster]),
        }
    }
    // 記号を含めた単位でも max_chars を超える場合は、上限を優先して結合文字の単位で区切る。
    // 結合文字を含む1文字だけで max_chars を超える場合は、文字を壊さないよう区切らない
    let pieces: Vec<String> = units
        .iter()
        .flat_map(|unit| {
            let joined = unit.concat();
            if char_len(&joined) > max_chars {
                pack(unit, max_chars)
            } else {
                vec![joined]
            }
        })
        .collect();
    pack(&pieces, max_chars)
}

/// max_chars を超える文を、文末・読点・文字数の順に区切りを優先して max_chars 以下の塊に分ける。
/// max_chars が 0 の場合は分けない
pub fn split_for_tts(text: &str, max_chars: usize) -> Vec<String> {
    if max_chars == 0 || char_len(text) <= max_chars {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    for sentence in pack(&find_all(&SENTENCE, text), max_chars) {
        let clauses = if char_len(&sentence) > max_chars {
            pack(&find_all(&CLAUSE, &sentence), max_chars)
        } else {
            vec![sentence]
        };
        for clause in clauses {
            // 文末でも読点でも区切れない長さの場合は文字数で区切る
            if char_len(&clause) > max_chars {
                chunks.extend(split_by_length(&clause, max_chars));
            } else {
                chunks.push(clause);
            }
        }
    }
    chunks.retain(|c| is_readable(c));
    chunks
}
