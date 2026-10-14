// Windows でアプリを開いたときに、コンソールの窓が一緒に開かないようにする
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

use naui::Settings;

fn main() -> naui::Result<()> {
    naui::run(Settings::new("Yomitone").app_id("io.github.mokuichi147.yomitone"), yomitone_app::build)
}
