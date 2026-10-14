use yomitone::domain::character::{Character, Gender};
use yomitone::domain::novel::{Novel, Sentence};
use yomitone::domain::text::{is_readable, split_for_tts};
use yomitone::domain::voice_assigner::{VoiceAssigner, VoiceAssignerError};

fn character(name: &str) -> Character {
    Character::new(name, "")
}

fn person(name: &str, gender: Option<Gender>, aliases: &[&str]) -> Character {
    Character::new(name, "").with_gender(gender).with_aliases(aliases.iter().copied())
}

fn voices(names: &[&str]) -> Vec<String> {
    names.iter().map(|s| s.to_string()).collect()
}

fn assigner(names: &[&str]) -> VoiceAssigner {
    VoiceAssigner::with_voices(names.iter().copied()).unwrap()
}

// --- 文と登場人物 ---

#[test]
fn novel_from_lines_strips_each_line() {
    let novel = Novel::from_lines(["あ\n", "\u{3000}「い」 \n", "\n"]);
    let texts: Vec<&str> = novel.sentences.iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["あ", "「い」", ""]);
    assert!(novel.sentences.iter().all(|s| s.speaker.is_none()));
}

#[test]
fn sentence_is_dialogue() {
    for (text, expected) in [("「はい」", true), ("「はい」と言った", false), ("はい", false), ("「", false)]
    {
        assert_eq!(Sentence::new(text).is_dialogue(), expected, "{text}");
    }
}

#[test]
fn narrator_character() {
    assert!(Character::narrator().is_narrator());
    assert!(!character("花子").is_narrator());
}

// --- 読み上げ可否と分割 ---

#[test]
fn is_readable_symbols() {
    for text in ["＊\u{3000}＊\u{3000}＊", "「………………」", "――――", "********", "\u{3000}", "「！？」"]
    {
        assert!(!is_readable(text), "{text}");
    }
}

#[test]
fn is_readable_text() {
    for text in ["こんにちは", "「はい」", "ＡＢＣ", "123", "「ああ……」", "ー"] {
        assert!(is_readable(text), "{text}");
    }
}

#[test]
fn split_for_tts_short_text_is_unchanged() {
    assert_eq!(split_for_tts("「はい。そうです」", 200), ["「はい。そうです」"]);
    assert_eq!(split_for_tts(&"あ".repeat(300), 0), ["あ".repeat(300)]);
}

#[test]
fn split_for_tts_splits_at_sentence_end() {
    let text = "一文目です。「二文目！」三文目です？四文目";
    let chunks = split_for_tts(text, 8);
    assert_eq!(chunks.concat(), text);
    assert_eq!(chunks, ["一文目です。", "「二文目！」", "三文目です？", "四文目"]);
}

#[test]
fn split_for_tts_packs_short_sentences() {
    assert_eq!(split_for_tts("あ。い。う。え。", 4), ["あ。い。", "う。え。"]);
}

#[test]
fn split_for_tts_splits_at_comma_then_by_length() {
    assert_eq!(split_for_tts("ああ、いいい、うう。", 6), ["ああ、", "いいい、", "うう。"]);
    let text = format!("{}、{}。う", "あ".repeat(7), "い".repeat(3));
    assert_eq!(split_for_tts(&text, 5), ["あああああ", "ああ、", "いいい。", "う"]);
}

#[test]
fn split_for_tts_never_exceeds_max_chars() {
    let texts =
        ["あ".repeat(23), "あいうえおかきくけこ、さしすせそたちつてと。".repeat(3), format!("「{}」", "あ".repeat(30))];
    for text in texts {
        let chunks = split_for_tts(&text, 7);
        assert!(chunks.iter().all(|c| c.chars().count() <= 7), "{chunks:?}");
        assert_eq!(chunks.concat(), text);
    }
}

#[test]
fn split_for_tts_strictly_limits_chunks_with_symbols() {
    let cases = [
        ("あ。".to_string(), 1),
        ("あ。」".to_string(), 2),
        (format!("あ{}い", "…".repeat(10)), 3),
        (format!("{}あ", "「".repeat(5)), 2),
    ];
    for (text, max_chars) in cases {
        let chunks = split_for_tts(&text, max_chars);
        assert!(chunks.iter().all(|c| c.chars().count() <= max_chars), "{chunks:?}");
        assert!(chunks.iter().all(|c| is_readable(c)), "{chunks:?}");
    }
}

#[test]
fn split_for_tts_keeps_combining_marks() {
    let ga = "\u{30ab}\u{3099}"; // NFD の「ガ」
                                 // 1文字で上限を超える結合文字列は区切らない
    assert_eq!(split_for_tts("a\u{301}\u{308}b", 2), ["a\u{301}\u{308}", "b"]);
    assert_eq!(split_for_tts(&ga.repeat(3), 3), [ga, ga, ga]);
    assert_eq!(split_for_tts(&format!("{}。", ga.repeat(3)), 4), [ga.repeat(2), format!("{ga}。")]);
}

#[test]
fn split_for_tts_keeps_symbols_at_boundary() {
    for text in ["あああああ。い", "あああああ、い", "あああああ。」い", "「ああああ」"] {
        let chunks = split_for_tts(text, 5);
        assert_eq!(chunks.concat(), text);
        assert!(chunks.iter().all(|c| is_readable(c)), "{chunks:?}");
    }
}

#[test]
fn split_for_tts_drops_unreadable_chunks() {
    assert_eq!(split_for_tts("あいうえお。……。", 6), ["あいうえお。"]);
}

// --- 音声の割り当て ---

#[test]
fn voice_assigner_narrator_and_least_used() {
    let mut assigner = assigner(&["n", "a", "b"]);
    assert_eq!(assigner.assign(None), "n");
    assert_eq!(assigner.assign(Some(&character("ナレーター"))), "n");
    assert_eq!(assigner.assign(Some(&character("A"))), "a");
    assert_eq!(assigner.assign(Some(&character("B"))), "b");
    assert_eq!(assigner.assign(Some(&character("C"))), "a");
    assert_eq!(assigner.assign(Some(&character("A"))), "a");
}

#[test]
fn voice_assigner_single_voice_is_shared_with_narrator() {
    assert_eq!(assigner(&["n"]).assign(Some(&character("A"))), "n");
}

#[test]
fn voice_assigner_gender_voices() {
    let mut assigner =
        VoiceAssigner::new(voices(&["n", "m", "f", "x"]), Some("x".into()), voices(&["m"]), voices(&["f"])).unwrap();
    assert_eq!(assigner.assign(None), "x");
    assert_eq!(assigner.assign(Some(&person("太郎", Some(Gender::Male), &[]))), "m");
    assert_eq!(assigner.assign(Some(&person("花子", Some(Gender::Female), &[]))), "f");
    assert_eq!(assigner.assign(Some(&character("不明"))), "n");
    // 性別ごとの指定がない性別は全音声から選ぶ(使用数が同じなら一覧の先頭)
    assert_eq!(assigner.assign(Some(&person("その他", Some(Gender::Other), &[]))), "n");
}

#[test]
fn voice_assigner_keeps_assigned_name_when_later_linked_by_alias() {
    let mut assigner = assigner(&["n", "a", "b"]);
    assert_eq!(assigner.assign(Some(&character("A"))), "a");
    assert_eq!(assigner.assign(Some(&character("B"))), "b");
    // 後から B が A の別名と分かっても、書き出し済みの音声と食い違わないよう両者とも元の音声を維持する
    assert_eq!(assigner.assign(Some(&person("B", None, &["A"]))), "b");
    assert_eq!(assigner.assign(Some(&character("A"))), "a");
}

#[test]
fn voice_assigner_normalizes_names() {
    let mut assigner = assigner(&["n", "a", "b"]);
    assert_eq!(assigner.assign(Some(&person("A", None, &["\u{30ab}\u{3099}"]))), "a");
    assert_eq!(assigner.assign(Some(&character(" A "))), "a");
    assert_eq!(assigner.assign(Some(&character("\u{30ac}"))), "a"); // NFD の別名と NFC の正式名
    assert_eq!(assigner.assign(Some(&character("Ａ"))), "a"); // 全角
    assert_eq!(assigner.assign(Some(&character(" ナレーター "))), "n");
    let names: Vec<&String> = assigner.assigned().keys().collect();
    assert_eq!(names, ["ナレーター", "A", "\u{30ac}"]);
}

#[test]
fn voice_assigner_blank_name_and_alias() {
    let mut assigner = assigner(&["n", "a", "b"]);
    assert_eq!(assigner.assign(Some(&person("", Some(Gender::Male), &[]))), "n");
    assert_eq!(assigner.assign(Some(&character("\u{3000}"))), "n");
    assert_eq!(assigner.assign(Some(&person("A", None, &["", " "]))), "a");
    assert_eq!(assigner.assign(Some(&person("B", None, &[""]))), "b");
}

#[test]
fn voice_assigner_ignores_ambiguous_alias() {
    let mut assigner = assigner(&["n", "a", "b", "c"]);
    assert_eq!(assigner.assign(Some(&person("A", None, &["先生"]))), "a");
    assert_eq!(assigner.assign(Some(&person("B", None, &["先生"]))), "b");
    // 複数の人物が持つ別名はどの人物か決められないため、新しい人物として扱う
    assert_eq!(assigner.assign(Some(&character("先生"))), "c");
}

#[test]
fn voice_assigner_ignores_alias_shared_by_people_with_same_voice() {
    let mut assigner = VoiceAssigner::new(voices(&["n", "a", "b"]), None, voices(&["a"]), voices(&["a"])).unwrap();
    assert_eq!(assigner.assign(Some(&person("A", Some(Gender::Male), &["X"]))), "a");
    assert_eq!(assigner.assign(Some(&person("B", Some(Gender::Female), &["X"]))), "a");
    assert_eq!(assigner.assign(Some(&person("X", Some(Gender::Other), &[]))), "b");
}

#[test]
fn voice_assigner_alias_of_same_person_with_varying_name() {
    let mut assigner = assigner(&["n", "a", "b"]);
    assert_eq!(assigner.assign(Some(&person("田中花子", None, &["花子", "委員長"]))), "a");
    // 正式名が揺れても同じ人物と分かっている名前から同じ別名が来た場合は、曖昧にしない
    assert_eq!(assigner.assign(Some(&person("花子", None, &["委員長"]))), "a");
    assert_eq!(assigner.assign(Some(&character("委員長"))), "a");
}

#[test]
fn voice_assigner_alias_matching_multiple_people_is_new_person() {
    let mut assigner = assigner(&["n", "a", "b", "c"]);
    assert_eq!(assigner.assign(Some(&character("A"))), "a");
    assert_eq!(assigner.assign(Some(&character("B"))), "b");
    assert_eq!(assigner.assign(Some(&person("C", None, &["A", "B"]))), "c");
}

#[test]
fn voice_assigner_gender_voices_only_narrator_falls_back() {
    let mut assigner = VoiceAssigner::new(voices(&["n", "x"]), None, voices(&["n"]), Vec::new()).unwrap();
    assert_eq!(assigner.assign(Some(&person("太郎", Some(Gender::Male), &[]))), "x");
}

#[test]
fn voice_assigner_matches_aliases_across_files() {
    let mut assigner = assigner(&["n", "a", "b", "c"]);
    let voice = assigner.assign(Some(&person("田中花子", None, &["花子", "委員長"])));
    // 後のファイルで正式名が別名の方に揺れた場合
    assert_eq!(assigner.assign(Some(&character("花子"))), voice);
    // 以前の正式名を別名に持つ場合
    assert_eq!(assigner.assign(Some(&person("委員長花子", None, &["田中花子"]))), voice);
    // 別名同士が一致するだけでは同一人物とみなさない
    assert_ne!(assigner.assign(Some(&person("次郎", None, &["委員長"]))), voice);
}

#[test]
fn voice_assigner_rejects_unknown_voices() {
    let base = voices(&["a", "b"]);
    let cases = [
        VoiceAssigner::new(base.clone(), Some("z".into()), Vec::new(), Vec::new()),
        VoiceAssigner::new(base.clone(), None, voices(&["z"]), Vec::new()),
        VoiceAssigner::new(base.clone(), None, Vec::new(), voices(&["a", "z"])),
    ];
    for result in cases {
        assert_eq!(result.unwrap_err(), VoiceAssignerError::UnknownVoices(voices(&["z"])));
    }
}

#[test]
fn voice_assigner_rejects_empty_voices() {
    assert_eq!(VoiceAssigner::new(Vec::new(), None, Vec::new(), Vec::new()).unwrap_err(), VoiceAssignerError::NoVoices);
}
