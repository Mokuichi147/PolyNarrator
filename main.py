import argparse
import os
from collections import Counter
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from natsort import natsorted
from llm import Ai
from models.gender import Gender
from models.novel import Novel
from models.narrator import Narrator
from tts import TextToSpeech

# 年齢層推定のためのキーワードリスト
CHILD_KEYWORDS = ("子供", "子ども", "幼い", "幼児", "少年", "少女", "赤ん坊", "幼少", "ちいさ", "小柄")
TEEN_KEYWORDS = ("学生", "高校", "中学", "十代", "思春期", "青年", "若者")
ELDER_KEYWORDS = ("老人", "老女", "老いた", "老齢", "年配", "高齢", "おじい", "おばあ", "老翁", "老婦", "老け")

# 性別と推定年齢に基づいて優先的に選択するVOICEVOX話者
VOICE_PREFERENCES: Dict[Gender, Dict[str, List[Tuple[str, Optional[str]]]]] = {
    Gender.FEMALE: {
        "child": [("ずんだもん", "あまあま"), ("春日部つむぎ", "ノーマル")],
        "teen": [("春日部つむぎ", "ノーマル"), ("冥鳴ひまり", "ノーマル")],
        "adult": [("冥鳴ひまり", "ノーマル"), ("四国めたん", "ノーマル")],
        "elder": [("四国めたん", "ツンツン"), ("冥鳴ひまり", "ノーマル")],
    },
    Gender.MALE: {
        "child": [("白上虎太郎", "ノーマル"), ("ずんだもん", "ノーマル")],
        "teen": [("青山龍星", "ノーマル"), ("玄野武宏", "ノーマル")],
        "adult": [("玄野武宏", "ノーマル"), ("青山龍星", "ノーマル")],
        "elder": [("ちび式じい", "ノーマル"), ("玄野武宏", "ノーマル")],
    },
    Gender.OTHER: {
        "child": [("ずんだもん", "ノーマル"), ("波音リツ", "ノーマル")],
        "teen": [("波音リツ", "ノーマル"), ("ずんだもん", "ノーマル")],
        "adult": [("波音リツ", "ノーマル"), ("ずんだもん", "セクシー")],
        "elder": [("ちび式じい", "ノーマル"), ("波音リツ", "ノーマル")],
    },
}

NEUTRAL_FALLBACK = [("ずんだもん", "ノーマル"), ("四国めたん", "ノーマル")]


@dataclass
class VoiceAssignment:
    speaker_id: int
    lines: List[str] = field(default_factory=list)


def classify_age_bucket(profile_text: str) -> str:
    text = profile_text or ""
    for keyword in CHILD_KEYWORDS:
        if keyword in text:
            return "child"
    for keyword in ELDER_KEYWORDS:
        if keyword in text:
            return "elder"
    for keyword in TEEN_KEYWORDS:
        if keyword in text:
            return "teen"
    return "adult"


def narrator_profile_text(narrator: Optional[Narrator]) -> str:
    if narrator is None:
        return ""
    parts: List[str] = [narrator.name]
    parts.extend(narrator.aliases or [])
    if narrator.portrait:
        parts.append(narrator.portrait)
    return " ".join(parts)


def narrator_key(narrator: Optional[Narrator]) -> str:
    if narrator is None:
        return "__NARRATOR__"

    aliases = sorted(narrator.aliases or [])
    alias_str = "|".join(aliases)
    return f"{narrator.name}|{alias_str}"


def resolve_gender(narrator: Optional[Narrator]) -> Gender:
    if narrator and narrator.gender:
        if isinstance(narrator.gender, Gender):
            return narrator.gender
        try:
            return Gender(narrator.gender)  # type: ignore[arg-type]
        except (ValueError, TypeError):
            pass
    return Gender.OTHER


def speaker_candidates(narrator: Optional[Narrator], age_bucket: str) -> List[Tuple[str, Optional[str]]]:
    candidates: List[Tuple[str, Optional[str]]] = []
    if narrator:
        direct_names = [narrator.name]
        direct_names.extend(narrator.aliases or [])
        for name in direct_names:
            name = name.strip()
            if name:
                candidates.append((name, None))

    gender = resolve_gender(narrator)
    candidates.extend(VOICE_PREFERENCES.get(gender, {}).get(age_bucket, []))

    # 性別が判明している場合でも中性的な選択肢を追加で検討
    if gender != Gender.OTHER:
        candidates.extend(VOICE_PREFERENCES[Gender.OTHER].get(age_bucket, []))

    candidates.extend(NEUTRAL_FALLBACK)

    # 重複排除しつつ順序保持
    unique: List[Tuple[str, Optional[str]]] = []
    seen = set()
    for item in candidates:
        if item not in seen:
            unique.append(item)
            seen.add(item)
    return unique


def select_speaker_id(
    narrator: Optional[Narrator],
    tts_engine: TextToSpeech,
    assigned: Dict[str, int],
    voice_cycle,
    voice_ids: List[int],
) -> int:
    key = narrator_key(narrator)
    if key in assigned:
        return assigned[key]

    if voice_cycle is None or not voice_ids:
        assigned[key] = tts_engine.default_speaker_id
        return assigned[key]

    profile = narrator_profile_text(narrator)
    age_bucket = classify_age_bucket(profile)
    used = set(assigned.values())
    pool_size = max(len(voice_ids), 1)

    for speaker_name, style_hint in speaker_candidates(narrator, age_bucket):
        speaker_id = tts_engine.find_style_id(speaker_name, style_hint)
        if speaker_id is not None:
            if speaker_id in used and len(used) < pool_size:
                continue
            assigned[key] = speaker_id
            return speaker_id

    # ラウンドロビンで未使用の話者を割り当てる
    for _ in range(len(voice_ids)):
        candidate = next(voice_cycle)
        if candidate not in used:
            assigned[key] = candidate
            return candidate

    # すべて使用済みの場合は最も少ない使用回数の話者を再利用
    usage_counts = Counter(assigned.values())
    best_candidate = min(voice_ids, key=lambda vid: (usage_counts.get(vid, 0), vid))
    assigned[key] = best_candidate
    return assigned[key]

    assigned[key] = tts_engine.default_speaker_id
    return assigned[key]


def chunk_lines(lines: List[str], max_chars: int = 300) -> List[str]:
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0

    for line in lines:
        normalized = line.strip()
        if not normalized:
            continue

        addition = len(normalized)
        if current and current_len + addition > max_chars:
            chunks.append("\n".join(current))
            current = [normalized]
            current_len = addition
        else:
            current.append(normalized)
            current_len += addition

    if current:
        chunks.append("\n".join(current))

    return chunks

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="11434")
    parser.add_argument("--model", default="granite4:small-h")
    parser.add_argument("--tts-output", help="音声ファイルを書き出すディレクトリ（指定された場合のみTTSを有効化）")
    parser.add_argument("--tts-host", default="127.0.0.1", help="VOICEVOXエンジンのホスト")
    parser.add_argument("--tts-port", type=int, default=50021, help="VOICEVOXエンジンのポート")
    parser.add_argument("--tts-speaker-id", type=int, default=1, help="VOICEVOXで利用する話者スタイルID")
    parser.add_argument("--tts-speed-scale", type=float, default=1.0, help="話速の倍率")
    parser.add_argument("--tts-pitch-scale", type=float, default=0.0, help="ピッチの調整量")
    parser.add_argument("--tts-intonation-scale", type=float, default=1.0, help="抑揚の倍率")
    parser.add_argument("--tts-volume-scale", type=float, default=1.0, help="音量の倍率")
    parser.add_argument("--tts-disable-upspeak", action="store_true", help="疑問文語尾の自動上昇を無効化する")
    parser.add_argument("--tts-timeout", type=float, default=30.0, help="VOICEVOXへのリクエストタイムアウト秒数")
    parser.add_argument("--tts-chunk-size", type=int, default=300, help="1チャンクあたりの最大文字数")
    parser.add_argument("folder", help="data")
    args = parser.parse_args()
    
    ai = Ai(args.host, args.port, args.model)
    narrators = []
    tts_engine = None
    tts_root: Path | None = None
    speaker_assignments: Dict[str, int] = {}
    voice_ids: List[int] = []
    voice_cycle = None

    if args.tts_output:
        tts_engine = TextToSpeech(
            host=args.tts_host,
            port=args.tts_port,
            speaker_id=args.tts_speaker_id,
            speed_scale=args.tts_speed_scale,
            pitch_scale=args.tts_pitch_scale,
            intonation_scale=args.tts_intonation_scale,
            volume_scale=args.tts_volume_scale,
            enable_interrogative_upspeak=not args.tts_disable_upspeak,
            timeout=args.tts_timeout,
        )
        tts_root = Path(args.tts_output)
        voice_ids = sorted({entry["id"] for entry in tts_engine.available_style_entries()})
        if not voice_ids:
            voice_ids = [tts_engine.default_speaker_id]
        voice_cycle = cycle(voice_ids)
        speakers = ", ".join(tts_engine.available_speakers()) or "N/A"
        print(
            "TTS 初期化: engine=voicevox, "
            f"endpoint={args.tts_host}:{args.tts_port}, speaker_id={args.tts_speaker_id}, "
            f"speed={args.tts_speed_scale}"
        )
        print(f"利用可能な話者ID一覧: {speakers}")
    
    files: List[str] = os.listdir(args.folder)
    for index, file in enumerate(natsorted(files)):
        filepath = os.path.join(args.folder, file)
        if not os.path.isfile(filepath):
            continue
        
        novel = Novel()
        novel.load(filepath)
        response = ai.get_narrators(novel, narrators)
        if len(response) > 0:
            narrators = response
        
        print(index + 1, filepath)
        print("\n".join([f"  - {i.name} ({i.gender}) {i.aliases}" for i in narrators]))
        print()

        novel.narrators = narrators
        ai.set_estimation_narrator(novel, 100, 0, True)

        if tts_engine and tts_root:
            audio_dir = tts_root / Path(file).stem
            assignments: Dict[str, VoiceAssignment] = {}
            for sentence in novel.sentences:
                narrator: Narrator | None = sentence.narrator
                key = narrator.name if narrator else "ナレーター"
                if key not in assignments:
                    speaker_id = select_speaker_id(
                        narrator,
                        tts_engine,
                        speaker_assignments,
                        voice_cycle,
                        voice_ids,
                    )
                    assignments[key] = VoiceAssignment(speaker_id=speaker_id)
                    style_desc = tts_engine.describe_style(speaker_id)
                    print(f"  -> 話者割当 {key}: {style_desc}")
                assignments[key].lines.append(sentence.text)

            for position, (narrator_name, assignment) in enumerate(assignments.items(), start=1):
                speaker_id = assignment.speaker_id
                chunks = chunk_lines(assignment.lines, max(args.tts_chunk_size, 50))
                for part_index, text in enumerate(chunks, start=1):
                    if not text:
                        continue
                    filename = TextToSpeech.safe_filename(narrator_name)
                    outfile = audio_dir / f"{index + 1:03d}_{position:02d}_{part_index:02d}_{filename}.wav"
                    try:
                        tts_engine.synthesize(text, outfile, speaker_id=speaker_id)
                    except RuntimeError as exc:
                        print(f"    ! 音声生成に失敗しました ({narrator_name}): {exc}")
                        continue
                    style_desc = tts_engine.describe_style(speaker_id)
                    print(f"    -> {outfile} ({style_desc})")
    
    print("\n登場人物一覧")
    for narrator in narrators:
        print(f"- {narrator.name} ({narrator.gender}) {narrator.aliases}")
    


if __name__ == "__main__":
    main()
