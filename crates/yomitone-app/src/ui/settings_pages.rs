//! 「設定」の画面。接続先ごとに画面を分け、それぞれで疎通を確かめられるようにする

use std::cell::RefCell;
use std::rc::Rc;

use naui::{Orientation, Result, Sizing, Stack, TextColor, Ui};
use yomitone::adapters::wyoming::WyomingClient;
use yomitone::application::ports::SpeechSynthesizer;
use yomitone::diagnostics::{check_jev, check_llm, check_tts};
use yomitone::setup::{self, DEFAULT_LLM_HOST, DEFAULT_LLM_MODEL, DEFAULT_LLM_PORT, MAX_TTS_PAUSE};

use super::parts::{check_section, heading, note, number_field, page, secret_field, text_field, title, Form};
use super::store::Store;
use crate::settings::{AppSettings, SpeakerBackend, TtsBackendSetting};
use crate::worker::describe_setup_error;

pub(crate) fn llm(ui: &Ui, store: &Store) -> Result<Stack> {
    let page = page(ui)?;
    page.append(&title(ui, "LLM")?);
    page.append(&note(
        ui,
        "登場人物の抽出と、「読み上げ」で話者推測に LLM を選んだときの推測に使う、OpenAI API 互換のサーバーです。\
         JSON Schema による構造化出力に対応したモデルが必要です。",
    )?);
    let form = Form::new(ui)?;
    form.add(ui, "ホスト", &text_field(ui, store, |s| s.llm_host.clone(), |s, v| s.llm_host = v, DEFAULT_LLM_HOST)?)?;
    form.add(ui, "ポート", &text_field(ui, store, |s| s.llm_port.clone(), |s, v| s.llm_port = v, DEFAULT_LLM_PORT)?)?;
    form.note(ui, "http://<ホスト>:<ポート>/v1 に接続します。")?;
    form.add(
        ui,
        "モデル名",
        &text_field(ui, store, |s| s.llm_model.clone(), |s, v| s.llm_model = v, DEFAULT_LLM_MODEL)?,
    )?;
    page.append(form.widget());

    check_section(
        ui,
        store,
        &page,
        "サーバーのモデル一覧を取得し、モデル名がその中にあるかを確かめます。",
        |settings| check_llm(&settings.llm_options()),
    )?;
    Ok(page)
}

pub(crate) fn jev(ui: &Ui, store: &Store) -> Result<Stack> {
    let page = page(ui)?;
    page.append(&title(ui, SpeakerBackend::Jev.page_title())?);
    page.append(&note(
        ui,
        "「読み上げ」で話者推測に Jev を選んだときに使う、TypeSafe System One API 互換のモデルです。\
         登場人物の抽出には LLM を使います。",
    )?);
    let form = Form::new(ui)?;
    let model = text_field(
        ui,
        store,
        |s| s.jev_model.clone(),
        |s, v| s.jev_model = v,
        "未入力時は TYPESAFE_DEFAULT_MODEL または jev-latest",
    )?;
    form.add(ui, "モデル名", &model)?;
    let base_url = text_field(
        ui,
        store,
        |s| s.jev_base_url.clone(),
        |s, v| s.jev_base_url = v,
        "未入力時は TYPESAFE_BASE_URL または https://api.typesafe.ai",
    )?;
    form.add(ui, "URL", &base_url)?;
    let api_key = secret_field(ui, store, |s, v| s.jev_api_key = v, "未入力時は環境変数 TYPESAFE_API_KEY")?;
    form.add(ui, "API キー", &api_key)?;
    form.note(ui, "API キーはファイルに保存しません。アプリを起動するたびに入力するか、環境変数を設定してください。")?;
    page.append(form.widget());

    check_section(
        ui,
        store,
        &page,
        "API キーでモデル一覧を取得し、モデル名がその中にあるかを確かめます。",
        |settings| check_jev(&settings.jev_options()),
    )?;
    Ok(page)
}

/// サーバーから取得した音声一覧。取得できなければ理由
type VoiceList = std::result::Result<Vec<String>, String>;

fn fetch_wyoming_voices(settings: &AppSettings) -> VoiceList {
    let options = settings.wyoming_tts.options(TtsBackendSetting::Wyoming.backend());
    let client = WyomingClient::new(options.url(), options.timeout, Some(&options.language))
        .map_err(|e| format!("Wyomingサーバー({})から音声一覧を取得できませんでした: {e}", options.url()))?;
    Ok(client.list_voices())
}

/// Wyoming サーバーの音声一覧を取得し、使用する音声に設定するボタン
fn wyoming_voice_list(ui: &Ui, store: &Store, page: &Stack, voices: naui::TextInput) -> Result<()> {
    page.append(&note(ui, "サーバーから、言語に一致する音声の一覧を取得できます。")?);
    let buttons = ui.stack(Orientation::Horizontal)?;
    buttons.set_spacing(8.0);
    let fetch = ui.button("音声一覧を取得")?;
    let apply = ui.button("使用する音声に設定")?;
    apply.set_enabled(false);
    buttons.append(&fetch);
    buttons.append(&apply);
    page.append(&buttons);
    let result = ui.label("")?;
    result.set_wrap(true);
    result.set_selectable(true);
    result.set_sizing(Sizing::fill_width());
    page.append(&result);

    let fetched: Rc<RefCell<Vec<String>>> = Rc::default();
    fetch.on_click({
        let ui = ui.clone();
        let store = store.clone();
        let (fetch, apply, result, fetched) = (fetch.clone(), apply.clone(), result.clone(), fetched.clone());
        move || {
            fetch.set_enabled(false);
            apply.set_enabled(false);
            result.set_text("取得しています…");
            result.set_color(TextColor::Secondary);
            let sender = ui.tasks().channel({
                let (fetch, apply, result, fetched) = (fetch.clone(), apply.clone(), result.clone(), fetched.clone());
                move |voices: VoiceList| {
                    fetch.set_enabled(true);
                    match voices {
                        Ok(voices) if voices.is_empty() => {
                            result.set_text("言語に一致する音声がありません。言語を確認してください");
                            result.set_color(TextColor::Warning);
                        }
                        Ok(voices) => {
                            result.set_text(&voices.join(", "));
                            result.set_color(TextColor::Default);
                            fetched.replace(voices);
                            apply.set_enabled(true);
                        }
                        Err(error) => {
                            result.set_text(&error);
                            result.set_color(TextColor::Danger);
                        }
                    }
                }
            });
            let settings = store.snapshot();
            std::thread::spawn(move || {
                let _ = sender.send(fetch_wyoming_voices(&settings));
            });
        }
    });
    apply.on_click({
        let store = store.clone();
        move || {
            let text = fetched.borrow().join(", ");
            // 入力欄の書き換えは通知されないので、設定もここで更新する
            voices.set_text(&text);
            store.update(|s| s.wyoming_tts.voices = text);
        }
    });
    Ok(())
}

/// TTS サーバー 1 つ分の設定。接続・声の割り当て・合成の設定を持つ
pub(crate) fn tts(ui: &Ui, store: &Store, backend: TtsBackendSetting) -> Result<Stack> {
    let page = page(ui)?;
    page.append(&title(ui, backend.page_title())?);
    page.append(&note(
        ui,
        match backend {
            TtsBackendSetting::Openai => {
                "「読み上げ」で TTS に OpenAI 互換 API を選んだときに使う、/v1/audio/speech に対応した TTS サーバーです。"
            }
            TtsBackendSetting::Wyoming => {
                "「読み上げ」で TTS に Wyoming を選んだときに使う、Wyoming プロトコルに対応した TTS サーバーです。"
            }
        },
    )?);

    page.append(&heading(ui, "接続")?);
    let connection = Form::new(ui)?;
    let placeholder = format!("未入力時は {}", backend.backend().default_url());
    let url = text_field(
        ui,
        store,
        move |s| s.tts_server(backend).url.clone(),
        move |s, v| s.tts_server_mut(backend).url = v,
        &placeholder,
    )?;
    connection.add(ui, "URL", &url)?;
    match backend {
        TtsBackendSetting::Openai => {
            let model = text_field(
                ui,
                store,
                |s| s.openai_tts.model.clone(),
                |s, v| s.openai_tts.model = v,
                setup::DEFAULT_OPENAI_TTS_MODEL,
            )?;
            connection.add(ui, "モデル名", &model)?;
            let api_key =
                secret_field(ui, store, |s, v| s.openai_tts.api_key = v, "未入力時は環境変数 OPENAI_API_KEY")?;
            connection.add(ui, "API キー", &api_key)?;
            connection.note(ui, "API キーはファイルに保存しません。")?;
        }
        TtsBackendSetting::Wyoming => {
            let language = text_field(
                ui,
                store,
                |s| s.wyoming_tts.language.clone(),
                |s, v| s.wyoming_tts.language = v,
                "空にすると絞り込みません",
            )?;
            connection.add(ui, "言語", &language)?;
            connection.note(
                ui,
                "サーバーの音声一覧から使う音声を、この言語で絞り込みます(ja と ja-JP は一致とみなします)。",
            )?;
        }
    }
    let timeout = number_field(ui, store.read(|s| s.tts_server(backend).timeout), (Some(1.0), None), 1.0, 0)?;
    timeout.on_change({
        let store = store.clone();
        move |value| store.update(|s| s.tts_server_mut(backend).timeout = value)
    });
    connection.add(ui, "タイムアウト(秒)", &timeout)?;
    page.append(connection.widget());

    page.append(&heading(ui, "声の割り当て")?);
    page.append(&note(
        ui,
        "ナレーターと登場人物に使う音声です。登場人物には、ナレーター以外で使用数が最も少ない音声を割り当て、\
         後の章でも同じ人物には同じ音声を使います。複数の音声はカンマで区切ります。",
    )?);
    let assignment = Form::new(ui)?;
    let voices = text_field(
        ui,
        store,
        move |s| s.tts_server(backend).voices.clone(),
        move |s, v| s.tts_server_mut(backend).voices = v,
        match backend {
            TtsBackendSetting::Openai => "必須",
            TtsBackendSetting::Wyoming => "空にするとサーバーの一覧(言語に一致する音声)",
        },
    )?;
    assignment.add(ui, "使用する音声", &voices)?;
    let narrator = text_field(
        ui,
        store,
        move |s| s.tts_server(backend).narrator_voice.clone(),
        move |s, v| s.tts_server_mut(backend).narrator_voice = v,
        "未入力時は使用する音声の先頭",
    )?;
    assignment.add(ui, "ナレーター", &narrator)?;
    let male = text_field(
        ui,
        store,
        move |s| s.tts_server(backend).male_voices.clone(),
        move |s, v| s.tts_server_mut(backend).male_voices = v,
        "",
    )?;
    assignment.add(ui, "男性に優先する音声", &male)?;
    let female = text_field(
        ui,
        store,
        move |s| s.tts_server(backend).female_voices.clone(),
        move |s, v| s.tts_server_mut(backend).female_voices = v,
        "",
    )?;
    assignment.add(ui, "女性に優先する音声", &female)?;
    assignment.note(ui, "性別が分かっている登場人物には、この中から優先して割り当てます。")?;
    page.append(assignment.widget());
    if backend == TtsBackendSetting::Wyoming {
        wyoming_voice_list(ui, store, &page, voices)?;
    }

    page.append(&heading(ui, "合成")?);
    let synthesis = Form::new(ui)?;
    let max_chars =
        number_field(ui, store.read(|s| s.tts_server(backend).max_chars) as f64, (Some(0.0), None), 10.0, 0)?;
    max_chars.on_change({
        let store = store.clone();
        move |value| store.update(|s| s.tts_server_mut(backend).max_chars = value.max(0.0) as usize)
    });
    synthesis.add(ui, "1回の最大文字数", &max_chars)?;
    synthesis.note(ui, "超える行は文末・読点・文字数の順で分割して合成します。0 で分割しません。")?;
    let pause =
        number_field(ui, store.read(|s| s.tts_server(backend).pause), (Some(0.0), Some(MAX_TTS_PAUSE)), 0.1, 1)?;
    pause.on_change({
        let store = store.clone();
        move |value| store.update(|s| s.tts_server_mut(backend).pause = value)
    });
    synthesis.add(ui, "行間の無音(秒)", &pause)?;
    page.append(synthesis.widget());

    let description = match backend {
        TtsBackendSetting::Openai => {
            "使用する音声のそれぞれで短い文を合成し、URL・モデル名・API キー・音声名が正しいかを確かめます。"
        }
        TtsBackendSetting::Wyoming => {
            "サーバーの音声一覧と照らして音声名を確かめ、ナレーターの音声で短い文を合成してみます。"
        }
    };
    check_section(ui, store, &page, description, move |settings| {
        check_tts(&settings.tts_server(backend).options(backend.backend()), |e| describe_setup_error(e, backend))
    })?;
    Ok(page)
}
