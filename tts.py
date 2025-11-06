from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import requests
from requests import Response


class TextToSpeech:
    """VOICEVOX エンジンを利用したローカルTTSラッパー"""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 50021,
        speaker_id: int = 1,
        speed_scale: float = 1.0,
        pitch_scale: float = 0.0,
        intonation_scale: float = 1.0,
        volume_scale: float = 1.0,
        timeout: float = 30.0,
        enable_interrogative_upspeak: bool = True,
    ) -> None:
        self.base_url = f"http://{host}:{port}"
        self.speed_scale = speed_scale
        self.pitch_scale = pitch_scale
        self.intonation_scale = intonation_scale
        self.volume_scale = volume_scale
        self.timeout = timeout
        self.enable_interrogative_upspeak = enable_interrogative_upspeak

        self._speakers = self._load_speaker_catalog()
        if not self._speakers:
            raise RuntimeError("VOICEVOXエンジンから話者一覧を取得できませんでした。エンジンが起動しているか確認してください。")

        self._style_lookup: Dict[int, Dict] = {}
        self._styles_by_name: Dict[str, List[Dict]] = {}
        for speaker in self._speakers:
            speaker_name = speaker.get("name", "")
            for style in speaker.get("styles", []):
                style_id = style.get("id")
                if style_id is None:
                    continue
                entry = {
                    "id": style_id,
                    "style_name": style.get("name", ""),
                    "speaker_name": speaker_name,
                    "speaker_uuid": speaker.get("speaker_uuid"),
                }
                self._style_lookup[style_id] = entry
                self._styles_by_name.setdefault(speaker_name, []).append(entry)

        for styles in self._styles_by_name.values():
            styles.sort(key=lambda item: item["id"])

        if speaker_id in self._style_lookup:
            self.default_speaker_id = speaker_id
        else:
            self.default_speaker_id = next(iter(self._style_lookup))
            print(f"警告: 指定された話者ID {speaker_id} が見つかりません。既定値 {self.default_speaker_id} を利用します。")

    def synthesize(self, text: str, destination: Path, speaker_id: Optional[int] = None) -> Path:
        destination = destination.with_suffix(".wav")
        destination.parent.mkdir(parents=True, exist_ok=True)

        normalized = text.strip()
        if not normalized:
            # 空のテキストは生成対象外とする
            return destination

        target_id = speaker_id if speaker_id is not None else self.default_speaker_id
        query = self._create_audio_query(normalized, target_id)
        query.update(
            {
                "speedScale": self.speed_scale,
                "pitchScale": self.pitch_scale,
                "intonationScale": self.intonation_scale,
                "volumeScale": self.volume_scale,
            }
        )

        audio = self._synthesize_audio(query, target_id)
        with open(destination, "wb") as wav:
            wav.write(audio)
        return destination

    def available_speakers(self) -> List[str]:
        return [
            f"{entry['id']}: {entry['speaker_name']} / {entry['style_name']}"
            for entry in sorted(self._style_lookup.values(), key=lambda item: item["id"])
        ]

    def describe_style(self, speaker_id: int) -> str:
        entry = self._style_lookup.get(speaker_id)
        if not entry:
            return str(speaker_id)
        return f"{entry['id']}: {entry['speaker_name']} / {entry['style_name']}"

    def styles_by_speaker(self, speaker_name: str) -> List[Dict]:
        return self._styles_by_name.get(speaker_name, [])

    def find_style_id(self, speaker_name: str, style_hint: Optional[str] = None) -> Optional[int]:
        styles = self.styles_by_speaker(speaker_name)
        if style_hint:
            for style in styles:
                if style_hint in style["style_name"]:
                    return style["id"]
        if styles:
            return styles[0]["id"]
        return None

    def available_style_entries(self) -> List[Dict]:
        return list(self._style_lookup.values())

    def get_style_entry(self, speaker_id: int) -> Optional[Dict]:
        return self._style_lookup.get(speaker_id)

    def speaker_names(self) -> List[str]:
        return list(self._styles_by_name.keys())

    def _create_audio_query(self, text: str, speaker_id: int) -> Dict:
        response = self._post(
            "/audio_query",
            params={"text": text, "speaker": speaker_id},
        )
        return response.json()

    def _synthesize_audio(self, query: Dict, speaker_id: int) -> bytes:
        response = self._post(
            "/synthesis",
            params={
                "speaker": speaker_id,
                "enable_interrogative_upspeak": str(self.enable_interrogative_upspeak).lower(),
            },
            json=query,
        )
        return response.content

    def _load_speaker_catalog(self) -> List[Dict]:
        response = self._request("GET", "/speakers")
        return response.json()

    def _post(self, path: str, *, params: Optional[Dict] = None, json: Optional[Dict] = None) -> Response:
        return self._request("POST", path, params=params, json=json)

    def _request(
        self,
        method: str,
        path: str,
        *,
        params: Optional[Dict] = None,
        json: Optional[Dict] = None,
    ) -> Response:
        url = f"{self.base_url}{path}"
        try:
            response = requests.request(
                method,
                url,
                params=params,
                json=json,
                timeout=(self.timeout, self.timeout),
            )
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            raise RuntimeError(f"VOICEVOXエンジンへのリクエストに失敗しました: {exc}") from exc

    @staticmethod
    def safe_filename(name: str) -> str:
        safe = re.sub(r"[^\w\-]+", "_", name)
        safe = re.sub(r"_+", "_", safe).strip("_")
        return safe or "narrator"
