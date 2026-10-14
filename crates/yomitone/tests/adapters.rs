mod support;

use std::fs;
use std::path::Path;
use std::rc::Rc;

use serde_json::{json, Value};
use support::*;
use yomitone::adapters::openai_speech::OpenAiSpeechClient;
use yomitone::adapters::text_files::{list_text_files, read_novel};
use yomitone::adapters::wav::{wav_data_size, write_wav, WavDirectory, WriteWavError};
use yomitone::adapters::wyoming::WyomingClient;
use yomitone::application::narrate_novels::NarrationError;
use yomitone::application::ports::{AudioWriter, SilentPresenter, SpeechSynthesizer, TtsError};
use yomitone::application::speech_synthesis::SentenceSynthesizer;
use yomitone::domain::audio::Audio;
use yomitone::domain::novel::Sentence;
use yomitone::domain::voice_assigner::VoiceAssigner;

fn pcm(rate: u32, frames: usize, value: i16) -> Audio {
    Audio::new(rate, 2, 1, value.to_le_bytes().repeat(frames))
}

fn pcm_default() -> Audio {
    pcm(16000, 160, 1000)
}

fn sentences(texts: &[&str]) -> Vec<Sentence> {
    texts.iter().map(|t| Sentence::new(*t)).collect()
}

fn synthesizer(client: impl SpeechSynthesizer + 'static, voice: &str) -> SentenceSynthesizer {
    SentenceSynthesizer::new(
        Box::new(client),
        VoiceAssigner::with_voices([voice]).unwrap(),
        Rc::new(SilentPresenter),
        0,
    )
}

fn names(directory: &Path) -> Vec<String> {
    let mut names: Vec<String> =
        fs::read_dir(directory).unwrap().map(|e| e.unwrap().file_name().to_string_lossy().into_owned()).collect();
    names.sort();
    names
}

fn fatal_message<T: std::fmt::Debug>(result: Result<T, TtsError>) -> String {
    match result {
        Err(TtsError::Fatal(message)) => message,
        other => panic!("致命的なエラーではありません: {other:?}"),
    }
}

// --- WAV出力 ---

#[test]
fn write_wav_inserts_pause() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("out").join("a.wav");
    write_wav(&path, &[pcm(16000, 100, 1000), pcm(16000, 50, 1000)], 0.01).unwrap();
    let (rate, width, channels, data) = read_wav_file(&path);
    assert_eq!((rate, width, channels), (16000, 2, 1));
    assert_eq!(data.len(), (100 + 160 + 50) * 2);
    assert_eq!(data[200..200 + 320], [0; 320]);
    assert_eq!(names(&dir.path().join("out")), ["a.wav"]);
}

#[test]
fn write_wav_matches_python_wave_header() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.wav");
    write_wav(&path, &[pcm(24000, 240, 1)], 0.0).unwrap();
    assert_eq!(fs::read(&path).unwrap(), wav_bytes(24000, 240));
}

#[test]
fn write_wav_converts_to_first_format() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.wav");
    let stereo = Audio::new(16000, 2, 2, [1u8, 0, 2, 0].repeat(100));
    write_wav(&path, &[pcm(24000, 240, 1000), pcm(16000, 160, 1000), stereo], 0.0).unwrap();
    let (rate, width, channels, data) = read_wav_file(&path);
    assert_eq!((rate, width, channels), (24000, 2, 1));
    // 16kHz の160フレーム(0.01秒)は24kHzでおよそ240フレームになる
    assert!((data.len() as i64 / 2 - (240 + 240 + 150)).abs() <= 4, "{}", data.len());
}

#[test]
fn write_wav_8bit_silence_is_unsigned_center() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.wav");
    let segment = Audio::new(8000, 1, 1, vec![0x90; 80]);
    write_wav(&path, &[segment.clone(), segment], 0.01).unwrap();
    let (_, width, _, data) = read_wav_file(&path);
    assert_eq!(width, 1);
    assert_eq!(data, [vec![0x90; 80], vec![0x80; 80], vec![0x90; 80]].concat());
}

#[test]
fn write_wav_keeps_existing_file_on_invalid_audio() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.wav");
    fs::write(&path, b"old").unwrap();
    let broken = Audio::new(8000, 2, 1, vec![1, 0, 2]);
    let error = write_wav(&path, &[pcm_default(), broken], 0.0).unwrap_err();
    assert!(error.to_string().contains("不正な音声"), "{error}");
    assert_eq!(fs::read(&path).unwrap(), b"old");
    assert_eq!(names(dir.path()), ["a.wav"]);
}

#[test]
fn write_wav_with_long_file_name() {
    let dir = tempfile::tempdir().unwrap();
    let name = format!("{}.wav", "あ".repeat(80)); // UTF-8 で 244 バイト
    write_wav(&dir.path().join(&name), &[pcm_default()], 0.0).unwrap();
    assert_eq!(names(dir.path()), [name]);
}

#[test]
fn validate_audio_rejects_abnormal_format() {
    let cases = [
        Audio::new(0, 2, 1, vec![1, 0]),
        Audio::new(1_000_000_000, 2, 1, vec![1, 0]),
        Audio::new(16000, 8, 1, vec![0; 8]),
        Audio::new(16000, 2, 100, vec![0; 200]),
        Audio::new(16000, 2, 4, vec![0; 8]),
    ];
    for audio in cases {
        let dir = tempfile::tempdir().unwrap();
        let error = write_wav(&dir.path().join("a.wav"), &[audio.clone(), audio], 60.0).unwrap_err();
        assert!(error.to_string().contains("不正な音声"), "{error}");
        assert!(names(dir.path()).is_empty());
    }
}

#[test]
fn write_wav_long_pause_is_written_in_blocks() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.wav");
    write_wav(&path, &[pcm(16000, 10, 1000), pcm(16000, 10, 1000)], 2.5).unwrap();
    let (_, _, _, data) = read_wav_file(&path);
    assert_eq!(data.len(), (10 + 40000 + 10) * 2);
    assert!(data[20..data.len() - 20].iter().all(|&b| b == 0));
}

#[test]
fn write_wav_without_segments_writes_nothing() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("a.wav");
    write_wav(&path, &[], 0.3).unwrap();
    assert!(!path.exists());
}

#[test]
fn write_wav_rejects_negative_pause() {
    for segments in [vec![], vec![pcm_default()]] {
        let dir = tempfile::tempdir().unwrap();
        assert!(matches!(write_wav(&dir.path().join("a.wav"), &segments, -1.0), Err(WriteWavError::InvalidPause)));
        assert!(names(dir.path()).is_empty());
    }
}

// --- OpenAI互換API ---

fn speech_client(url: &str) -> OpenAiSpeechClient {
    OpenAiSpeechClient::new(url, "tts-1", "key", 10.0)
}

#[test]
fn openai_client_synthesize() {
    let (url, requests) = speech_server();
    let audio = speech_client(&url).synthesize("こんにちは", "alloy").unwrap();
    assert_eq!((audio.rate, audio.width, audio.channels, audio.data.len()), (24000, 2, 1, 480));
    let requests = requests.lock().unwrap();
    assert_eq!(requests[0].path, "/v1/audio/speech");
    assert_eq!(requests[0].header("authorization"), Some("Bearer key"));
    assert_eq!(
        requests[0].json(),
        json!({"model": "tts-1", "voice": "alloy", "input": "こんにちは", "response_format": "wav"})
    );
}

#[test]
fn openai_client_bad_request_is_input_error() {
    let (url, _) = speech_server();
    for text in ["bad", "unprocessable"] {
        let error = speech_client(&url).synthesize(text, "alloy").unwrap_err();
        assert!(matches!(&error, TtsError::Input(message) if message.contains(text)), "{error:?}");
    }
}

#[test]
fn wav_data_size_of_header() {
    let content = wav_bytes(24000, 10);
    assert_eq!(wav_data_size(&content), Some(20));
    assert_eq!(wav_data_size(&content[..20]), None);
}

#[test]
fn openai_client_broken_audio_is_not_input_error() {
    let (url, _) = speech_server();
    for (text, message) in
        [("truncated", "途中で切れた"), ("near-limit", "途中で切れた"), ("zero", "空の音声"), ("notwav", "RIFF")]
    {
        let error = fatal_message(speech_client(&url).synthesize(text, "alloy"));
        assert!(error.contains(message), "{text}: {error}");
    }
}

#[test]
fn openai_client_accepts_streaming_wav_header() {
    let (url, _) = speech_server();
    assert_eq!(speech_client(&url).synthesize("streaming", "alloy").unwrap().data.len(), 480);
}

#[test]
fn openai_client_rejects_too_large_audio() {
    let (url, _) = speech_server();
    let mut client = speech_client(&url);
    client.max_audio_bytes = 100;
    client.header_allowance = 44;
    assert!(fatal_message(client.synthesize("こんにちは", "alloy")).contains("上限"));
}

#[test]
fn openai_client_rejects_pcm_over_limit_within_header_allowance() {
    let (url, _) = speech_server();
    let mut client = speech_client(&url);
    // WAV全体(44 + 480バイト)はヘッダーの許容量の範囲に収まるが、PCM(480バイト)は上限を超える
    client.max_audio_bytes = 479;
    client.header_allowance = 100;
    assert!(fatal_message(client.synthesize("こんにちは", "alloy")).contains("上限"));
}

#[test]
fn openai_client_unknown_model_aborts() {
    let (url, _) = speech_server();
    let client = OpenAiSpeechClient::new(&url, "unknown-model", "key", 10.0);
    match synthesizer(client, "alloy").synthesize(&sentences(&["こんにちは"])) {
        Err(NarrationError::SynthesisAborted(message)) => assert!(message.contains("最初の合成"), "{message}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn openai_client_consecutive_bad_requests_across_files() {
    let (url, requests) = speech_server();
    let mut synthesizer = synthesizer(speech_client(&url), "alloy");
    synthesizer.max_consecutive_failures = 2;
    let result = synthesizer.synthesize(&sentences(&["こんにちは", "bad"])).unwrap();
    assert_eq!(result.failures.len(), 1);
    assert_eq!(result.failures[0].0, "bad");
    match synthesizer.synthesize(&sentences(&["bad", "さようなら"])) {
        Err(NarrationError::SynthesisAborted(message)) => assert!(message.contains("連続"), "{message}"),
        other => panic!("{other:?}"),
    }
    let inputs: Vec<Value> = requests.lock().unwrap().iter().map(|r| r.json()["input"].clone()).collect();
    assert_eq!(inputs, ["こんにちは", "bad", "bad"]);
}

#[test]
fn openai_client_auth_error_is_not_input_error() {
    let (url, _) = speech_server();
    assert!(fatal_message(speech_client(&url).synthesize("unauthorized", "alloy")).contains("401"));
}

// --- Wyoming ---

fn wyoming(uri: &str, language: &str) -> WyomingClient {
    WyomingClient::new(uri, 5.0, Some(language)).unwrap()
}

#[test]
fn wyoming_lists_voices_by_language() {
    let (uri, _) = wyoming_server();
    assert_eq!(wyoming(&uri, "ja").list_voices(), ["ja-a", "ja-multi:s1", "ja-multi:s2"]);
    assert_eq!(wyoming(&uri, "").list_voices(), ["ja-a", "ja-multi:s1", "ja-multi:s2", "en-a"]);
}

#[test]
fn wyoming_language_matches_region() {
    let (uri, _) = wyoming_server();
    let japanese = ["ja-a", "ja-multi:s1", "ja-multi:s2"];
    assert_eq!(wyoming(&uri, "ja-JP").list_voices(), japanese);
    assert_eq!(wyoming(&uri, "ja_jp").list_voices(), japanese);
    assert_eq!(wyoming(&uri, "en").list_voices(), ["en-a"]);
    assert!(wyoming(&uri, "j").list_voices().is_empty());
}

#[test]
fn wyoming_unknown_voices_ignores_language_filter() {
    let (uri, _) = wyoming_server();
    let voices: Vec<String> =
        ["ja-a", "en-a", "ja-multi", "ja-multi:s2", "ja-multi:s3", "ja-missing", "ja-other", "ja-second", "nope"]
            .map(String::from)
            .to_vec();
    // 未インストールの音声と、合成時に選べない2番目以降のプログラムの音声は使えない
    assert_eq!(
        wyoming(&uri, "ja").unknown_voices(&voices),
        ["ja-multi:s3", "ja-missing", "ja-other", "ja-second", "nope"]
    );
}

#[test]
fn wyoming_synthesize_with_speaker() {
    let (uri, state) = wyoming_server();
    let audio = wyoming(&uri, "ja").synthesize("こんにちは", "ja-multi:s2").unwrap();
    assert_eq!((audio.rate, audio.width, audio.channels, audio.data.len()), (22050, 2, 1, 600));
    assert_eq!(
        state.lock().unwrap().requests[0],
        json!({"text": "こんにちは", "voice": {"name": "ja-multi", "speaker": "s2"}})
    );
}

#[test]
fn wyoming_converts_chunks_to_first_format() {
    let (uri, _) = wyoming_server();
    let audio = wyoming(&uri, "ja").synthesize("mixed", "ja-a").unwrap();
    assert_eq!((audio.rate, audio.width, audio.channels), (22050, 2, 1));
    // 続く4つのチャンク(11025Hz ステレオ 各0.1秒)も 22050Hz モノラルに変換され、
    // 変換器の状態を引き継ぐためチャンク境界でフレーム数がずれない
    assert!((audio.data.len() as i64 / 2 - (2205 + 2204 * 4)).abs() <= 2, "{}", audio.data.len());
}

#[test]
fn wyoming_broken_audio_is_not_input_error() {
    let (uri, _) = wyoming_server();
    let cases = [
        ("bad-second", "不正な音声"),
        ("empty", "音声が返されませんでした"),
        ("empty-chunk", "空の音声"),
        ("odd", "不正な音声"),
        ("disconnect", "切断"),
    ];
    for (text, message) in cases {
        let error = fatal_message(wyoming(&uri, "ja").synthesize(text, "ja-a"));
        assert!(error.contains(message), "{text}: {error}");
    }
}

#[test]
fn wyoming_error_is_input_error() {
    let (uri, _) = wyoming_server();
    for (text, message) in [("partial-error", "failed midway"), ("error", "synthesis failed")] {
        assert_eq!(wyoming(&uri, "ja").synthesize(text, "ja-a"), Err(TtsError::Input(message.into())));
    }
}

fn disable_first_program(state: &std::sync::Mutex<WyomingState>) {
    state.lock().unwrap().info["tts"][0]["installed"] = json!(false);
}

#[test]
fn wyoming_uses_first_installed_program() {
    let (uri, state) = wyoming_server();
    disable_first_program(&state);
    let mut client = wyoming(&uri, "ja");
    assert_eq!(client.list_voices(), ["ja-second"]);
    client.synthesize("こんにちは", "ja-second").unwrap();
    assert_eq!(state.lock().unwrap().requests.last().unwrap()["voice"], json!({"name": "ja-second"}));
}

#[test]
fn wyoming_rejects_too_large_audio() {
    let (uri, _) = wyoming_server();
    let mut client = wyoming(&uri, "ja");
    client.max_audio_bytes = 500;
    assert!(fatal_message(client.synthesize("こんにちは", "ja-a")).contains("上限"));
    // 「mixed」の後続チャンク(11025Hz ステレオ)は 22050Hz モノラルへの変換で同じ大きさになる
    client.max_audio_bytes = 4410 + 4408 * 2;
    assert!(fatal_message(client.synthesize("mixed", "ja-a")).contains("上限"));
}

#[test]
fn wyoming_without_installed_program() {
    let (uri, state) = wyoming_server();
    state.lock().unwrap().info["tts"] = json!([]);
    assert!(wyoming(&uri, "ja").list_voices().is_empty());
}

#[test]
fn wyoming_error_after_success_is_recorded() {
    let (uri, _) = wyoming_server();
    let result = synthesizer(wyoming(&uri, "ja"), "ja-a")
        .synthesize(&sentences(&["こんにちは", "error", "さようなら"]))
        .unwrap();
    assert_eq!(result.segments.len(), 2);
    assert_eq!(result.failures, [("error".to_string(), "synthesis failed".to_string())]);
}

#[test]
fn wyoming_error_before_success_aborts() {
    let (uri, _) = wyoming_server();
    match synthesizer(wyoming(&uri, "ja"), "ja-a").synthesize(&sentences(&["error"])) {
        Err(NarrationError::SynthesisAborted(message)) => assert!(message.contains("最初の合成"), "{message}"),
        other => panic!("{other:?}"),
    }
}

#[test]
fn wyoming_describe_failures() {
    let (uri, state) = wyoming_server();
    state.lock().unwrap().describe = "error".into();
    assert!(WyomingClient::new(&uri, 5.0, Some("ja")).err().unwrap().contains("describe failed"));
    state.lock().unwrap().describe = "none".into();
    assert!(WyomingClient::new(&uri, 5.0, Some("ja")).err().unwrap().contains("切断"));
    let unreachable = format!("tcp://127.0.0.1:{}", unused_tcp_port());
    assert!(WyomingClient::new(&unreachable, 2.0, Some("ja")).is_err());
}

// --- 出力先フォルダ ---

#[test]
fn wav_directory_writes_and_removes_by_source_name() {
    let dir = tempfile::tempdir().unwrap();
    let mut writer = WavDirectory::new(dir.path().join("out"), 0.0);
    let location = writer.write("data/1.txt", &[pcm_default()]).unwrap();
    assert_eq!(location, dir.path().join("out").join("1.wav").display().to_string());
    writer.remove("other/1.txt").unwrap();
    writer.remove("data/2.txt").unwrap();
    assert!(names(&dir.path().join("out")).is_empty());
}

#[test]
fn wav_directory_uses_pause() {
    let dir = tempfile::tempdir().unwrap();
    WavDirectory::new(dir.path(), 0.01).write("a.txt", &[pcm(16000, 10, 1000), pcm(16000, 10, 1000)]).unwrap();
    let (_, _, _, data) = read_wav_file(&dir.path().join("a.wav"));
    assert_eq!(data.len() / 2, 10 + 160 + 10);
}

#[test]
fn wav_directory_rejects_duplicate_output_names() {
    // NFC の「ガ」と NFD の「カ + 濁点」
    for names in [[".txt", ".txt.txt"], ["A.txt", "a.txt"], ["\u{30ac}.txt", "\u{30ab}\u{3099}.txt"]] {
        let dir = tempfile::tempdir().unwrap();
        let error = WavDirectory::new(dir.path().join("out"), 0.0).prepare(&names).unwrap_err();
        assert!(error.to_string().contains("出力先が同じ"), "{error}");
    }
}

#[cfg(unix)]
#[test]
fn wav_directory_rejects_symlink() {
    let dir = tempfile::tempdir().unwrap();
    let target = dir.path().join("elsewhere.wav");
    fs::write(&target, b"").unwrap();
    fs::create_dir(dir.path().join("out")).unwrap();
    std::os::unix::fs::symlink(&target, dir.path().join("out").join("1.wav")).unwrap();
    let error = WavDirectory::new(dir.path().join("out"), 0.0).prepare(&["1.txt"]).unwrap_err();
    assert!(error.to_string().contains("通常のファイル以外"), "{error}");
}

#[test]
fn wav_directory_rejects_directory_at_output_path() {
    let dir = tempfile::tempdir().unwrap();
    fs::create_dir_all(dir.path().join("out").join("2.wav")).unwrap();
    let error = WavDirectory::new(dir.path().join("out"), 0.0).prepare(&["1.txt", "2.txt"]).unwrap_err();
    assert!(error.to_string().contains("2.wav"), "{error}");
}

#[test]
fn wav_directory_rejects_file_as_directory() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("file"), "").unwrap();
    let error = WavDirectory::new(dir.path().join("file"), 0.0).prepare(&["1.txt"]).unwrap_err();
    assert!(error.to_string().contains("書き込めません"), "{error}");
}

// --- 入力ファイル ---

#[test]
fn list_text_files_in_natural_order() {
    let dir = tempfile::tempdir().unwrap();
    for name in ["10.txt", "2.txt", "1.txt", ".DS_Store", "1.wav", "memo.md"] {
        fs::write(dir.path().join(name), "").unwrap();
    }
    fs::create_dir(dir.path().join("dir.txt")).unwrap();
    assert_eq!(list_text_files(dir.path()).unwrap(), ["1.txt", "2.txt", "10.txt"]);
}

#[test]
fn list_text_files_missing_folder() {
    let dir = tempfile::tempdir().unwrap();
    fs::write(dir.path().join("file.txt"), "").unwrap();
    assert!(list_text_files(&dir.path().join("missing")).is_err());
    assert!(list_text_files(&dir.path().join("file.txt")).is_err());
}

#[test]
fn read_novel_lines() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("1.txt");
    fs::write(&path, "あ\r\n「い」\r\n\r\nう\rえ").unwrap();
    let texts: Vec<String> = read_novel(&path).unwrap().sentences.into_iter().map(|s| s.text).collect();
    assert_eq!(texts, ["あ", "「い」", "", "う", "え"]);
}

#[test]
fn read_novel_rejects_undecodable_file() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("1.txt");
    fs::write(&path, [0x82, 0xa0]).unwrap(); // Shift_JIS の「あ」
    assert!(read_novel(&path).is_err());
}
