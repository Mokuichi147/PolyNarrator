//! 「設定」の画面。CLI のオプションと同じ項目を、内容ごとの画面に分けて並べる

use std::rc::Rc;

use naui::{Orientation, Result, Sizing, Stack, TextColor, Ui};
use yomitone::adapters::wyoming::WyomingClient;
use yomitone::application::ports::SpeechSynthesizer;
use yomitone::setup::{self, DEFAULT_LLM_HOST, DEFAULT_LLM_MODEL, DEFAULT_LLM_PORT, MAX_TTS_PAUSE};

use super::parts::{heading, note, number_field, page, secret_field, text_field, title, Form};
use super::store::Store;
use crate::settings::{AppSettings, SpeakerBackend, TtsBackendSetting};

pub(crate) fn llm(ui: &Ui, store: &Store) -> Result<Stack> {
    let page = page(ui)?;
    page.append(&title(ui, "LLM サーバー")?);
    page.append(&note(
        ui,
        "登場人物の抽出と、話者推測を LLM で行う場合の推測に使う、OpenAI API 互換のサーバーです。\
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
    Ok(page)
}

pub(crate) fn speaker(ui: &Ui, store: &Store) -> Result<Stack> {
    let page = page(ui)?;
    page.append(&title(ui, "話者推測")?);
    page.append(&note(ui, "行全体が「」で囲まれたセリフごとに、誰の発言かを推測する方法を選びます。")?);
    let form = Form::new(ui)?;
    let backend = ui.combo_box()?;
    backend.set_items(&["LLM サーバー", "Jev 互換 API"]);
    form.add(ui, "推測に使うもの", &backend)?;
    page.append(form.widget());

    page.append(&heading(ui, "Jev 互換 API")?);
    page.append(&note(
        ui,
        "TypeSafe System One API 互換のモデルで話者を推測します。登場人物の抽出には引き続き LLM サーバーを使います。",
    )?);
    let jev = Form::new(ui)?;
    let model = text_field(
        ui,
        store,
        |s| s.jev_model.clone(),
        |s, v| s.jev_model = v,
        "未入力時は TYPESAFE_DEFAULT_MODEL または jev-latest",
    )?;
    jev.add(ui, "モデル名", &model)?;
    let base_url = text_field(
        ui,
        store,
        |s| s.jev_base_url.clone(),
        |s, v| s.jev_base_url = v,
        "未入力時は TYPESAFE_BASE_URL または https://api.typesafe.ai",
    )?;
    jev.add(ui, "URL", &base_url)?;
    let api_key = secret_field(ui, store, |s, v| s.jev_api_key = v, "未入力時は環境変数 TYPESAFE_API_KEY")?;
    jev.add(ui, "API キー", &api_key)?;
    jev.note(ui, "API キーはファイルに保存しません。アプリを起動するたびに入力するか、環境変数を設定してください。")?;
    page.append(jev.widget());

    let set_backend = Rc::new(move |value: SpeakerBackend| {
        let enabled = value == SpeakerBackend::Jev;
        model.set_enabled(enabled);
        base_url.set_enabled(enabled);
        api_key.set_enabled(enabled);
    });
    let current = store.read(|s| s.speaker_backend);
    backend.set_selected(if current == SpeakerBackend::Jev { 1 } else { 0 });
    set_backend(current);
    backend.on_select({
        let store = store.clone();
        move |index| {
            let value = if index == 1 { SpeakerBackend::Jev } else { SpeakerBackend::Llm };
            store.update(|s| s.speaker_backend = value);
            set_backend(value);
        }
    });
    Ok(page)
}

pub(crate) fn tts(ui: &Ui, store: &Store) -> Result<Stack> {
    let page = page(ui)?;
    page.append(&title(ui, "音声合成")?);
    page.append(&note(
        ui,
        "「読み上げ」で音声を書き出すときに使う TTS サーバーです。\
         OpenAI 互換の /v1/audio/speech と Wyoming プロトコルに対応しています。",
    )?);

    let form = Form::new(ui)?;
    let backend = ui.combo_box()?;
    backend.set_items(&["OpenAI 互換 API", "Wyoming"]);
    form.add(ui, "プロトコル", &backend)?;
    let url = text_field(ui, store, |s| s.tts_url.clone(), |s, v| s.tts_url = v, "")?;
    form.add(ui, "URL", &url)?;
    form.note(ui, "未入力時はプロトコルごとの既定の URL に接続します。")?;
    page.append(form.widget());

    page.append(&heading(ui, "OpenAI 互換 API")?);
    let openai = Form::new(ui)?;
    let model =
        text_field(ui, store, |s| s.tts_model.clone(), |s, v| s.tts_model = v, setup::DEFAULT_OPENAI_TTS_MODEL)?;
    openai.add(ui, "モデル名", &model)?;
    let api_key = secret_field(ui, store, |s, v| s.tts_api_key = v, "未入力時は環境変数 OPENAI_API_KEY")?;
    openai.add(ui, "API キー", &api_key)?;
    openai.note(ui, "API キーはファイルに保存しません。")?;
    page.append(openai.widget());

    page.append(&heading(ui, "Wyoming")?);
    let wyoming = Form::new(ui)?;
    let language =
        text_field(ui, store, |s| s.tts_language.clone(), |s, v| s.tts_language = v, "空にすると絞り込みません")?;
    wyoming.add(ui, "言語", &language)?;
    wyoming.note(ui, "サーバーの音声一覧から使う音声を、この言語で絞り込みます(ja と ja-JP は一致とみなします)。")?;
    page.append(wyoming.widget());

    page.append(&heading(ui, "合成")?);
    let synthesis = Form::new(ui)?;
    let max_chars = number_field(ui, store.read(|s| s.tts_max_chars) as f64, (Some(0.0), None), 10.0, 0)?;
    max_chars.on_change({
        let store = store.clone();
        move |value| store.update(|s| s.tts_max_chars = value.max(0.0) as usize)
    });
    synthesis.add(ui, "1回の最大文字数", &max_chars)?;
    synthesis.note(ui, "超える行は文末・読点・文字数の順で分割して合成します。0 で分割しません。")?;
    let pause = number_field(ui, store.read(|s| s.tts_pause), (Some(0.0), Some(MAX_TTS_PAUSE)), 0.1, 1)?;
    pause.on_change({
        let store = store.clone();
        move |value| store.update(|s| s.tts_pause = value)
    });
    synthesis.add(ui, "行間の無音(秒)", &pause)?;
    let timeout = number_field(ui, store.read(|s| s.tts_timeout), (Some(1.0), None), 1.0, 0)?;
    timeout.on_change({
        let store = store.clone();
        move |value| store.update(|s| s.tts_timeout = value)
    });
    synthesis.add(ui, "タイムアウト(秒)", &timeout)?;
    page.append(synthesis.widget());

    let set_backend = Rc::new(move |value: TtsBackendSetting| {
        url.set_placeholder(&format!("未入力時は {}", value.backend().default_url()));
        let openai = value == TtsBackendSetting::Openai;
        model.set_enabled(openai);
        api_key.set_enabled(openai);
        language.set_enabled(!openai);
    });
    let current = store.read(|s| s.tts_backend);
    backend.set_selected(if current == TtsBackendSetting::Wyoming { 1 } else { 0 });
    set_backend(current);
    backend.on_select({
        let store = store.clone();
        move |index| {
            let value = if index == 1 { TtsBackendSetting::Wyoming } else { TtsBackendSetting::Openai };
            store.update(|s| s.tts_backend = value);
            set_backend(value);
        }
    });
    Ok(page)
}

/// サーバーから取得した音声一覧。取得できなければ理由
type VoiceList = std::result::Result<Vec<String>, String>;

fn fetch_voices(settings: &AppSettings) -> VoiceList {
    let options = settings.tts_options();
    if options.backend != setup::TtsBackend::Wyoming {
        return Err(
            "OpenAI 互換 API には音声一覧を取得する標準の API がないため、使用する音声を入力してください".into()
        );
    }
    let client = WyomingClient::new(options.url(), options.timeout, Some(&options.language))
        .map_err(|e| format!("Wyomingサーバー({})から音声一覧を取得できませんでした: {e}", options.url()))?;
    Ok(client.list_voices())
}

pub(crate) fn voices(ui: &Ui, store: &Store) -> Result<Stack> {
    let page = page(ui)?;
    page.append(&title(ui, "声の割り当て")?);
    page.append(&note(
        ui,
        "ナレーターと登場人物に使う音声です。登場人物には、ナレーター以外で使用数が最も少ない音声を割り当て、\
         後の章でも同じ人物には同じ音声を使います。複数の音声はカンマで区切ります。",
    )?);

    let form = Form::new(ui)?;
    let voices = text_field(
        ui,
        store,
        |s| s.tts_voices.clone(),
        |s, v| s.tts_voices = v,
        "OpenAI 互換 API では必須。Wyoming では空にするとサーバーの一覧",
    )?;
    form.add(ui, "使用する音声", &voices)?;
    let narrator = text_field(
        ui,
        store,
        |s| s.tts_narrator_voice.clone(),
        |s, v| s.tts_narrator_voice = v,
        "未入力時は使用する音声の先頭",
    )?;
    form.add(ui, "ナレーター", &narrator)?;
    let male = text_field(ui, store, |s| s.tts_male_voices.clone(), |s, v| s.tts_male_voices = v, "")?;
    form.add(ui, "男性に優先する音声", &male)?;
    let female = text_field(ui, store, |s| s.tts_female_voices.clone(), |s, v| s.tts_female_voices = v, "")?;
    form.add(ui, "女性に優先する音声", &female)?;
    form.note(ui, "性別が分かっている登場人物には、この中から優先して割り当てます。")?;
    page.append(form.widget());

    page.append(&heading(ui, "サーバーの音声")?);
    page.append(&note(ui, "Wyoming サーバーから、「音声合成」の言語に一致する音声の一覧を取得します。")?);
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

    let fetched: Rc<std::cell::RefCell<Vec<String>>> = Rc::default();
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
                            result.set_text("言語に一致する音声がありません。「音声合成」の言語を確認してください");
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
                let _ = sender.send(fetch_voices(&settings));
            });
        }
    });
    apply.on_click({
        let store = store.clone();
        move || {
            let text = fetched.borrow().join(", ");
            // 入力欄の書き換えは通知されないので、設定もここで更新する
            voices.set_text(&text);
            store.update(|s| s.tts_voices = text);
        }
    });
    Ok(page)
}
