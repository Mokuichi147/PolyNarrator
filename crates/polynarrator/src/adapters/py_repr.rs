//! LLM へのプロンプトを移植前の Python 版と同じ文字列にするため、Python の repr と同じ表記を作る

use std::fmt::Write;

use unicode_general_category::{get_general_category, GeneralCategory};

use crate::domain::character::Gender;

/// Python の str.isprintable と同じ判定
fn is_printable(c: char) -> bool {
    use GeneralCategory::*;
    if c == ' ' {
        return true;
    }
    !matches!(
        get_general_category(c),
        Control | Format | Surrogate | PrivateUse | Unassigned | LineSeparator | ParagraphSeparator | SpaceSeparator
    )
}

/// Python の repr(str)
pub fn str_repr(text: &str) -> String {
    let quote = if text.contains('\'') && !text.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(text.len() + 2);
    out.push(quote);
    for c in text.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c if is_printable(c) => out.push(c),
            c if (c as u32) < 0x100 => write!(out, "\\x{:02x}", c as u32).unwrap(),
            c if (c as u32) < 0x10000 => write!(out, "\\u{:04x}", c as u32).unwrap(),
            c => write!(out, "\\U{:08x}", c as u32).unwrap(),
        }
    }
    out.push(quote);
    out
}

/// Python の repr(list[str])
pub fn list_repr(items: &[String]) -> String {
    format!("[{}]", items.iter().map(|s| str_repr(s)).collect::<Vec<_>>().join(", "))
}

/// 移植前の Python 版の Gender 列挙型を文字列にしたときの表記
pub fn gender_repr(gender: Option<Gender>) -> &'static str {
    match gender {
        Some(Gender::Male) => "Gender.MALE",
        Some(Gender::Female) => "Gender.FEMALE",
        Some(Gender::Other) => "Gender.OTHER",
        None => "None",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_python_repr() {
        let items = ["a", "b'c", "d\n", "x\u{3000}y", "z\u{200b}w", "q\""].map(String::from);
        assert_eq!(list_repr(&items), "['a', \"b'c\", 'd\\n', 'x\\u3000y', 'z\\u200bw', 'q\"']");
        assert_eq!(str_repr("a'b\"c"), r#"'a\'b"c'"#);
        assert_eq!(list_repr(&[]), "[]");
    }
}
