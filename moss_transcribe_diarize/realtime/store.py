"""On-disk persistence for realtime sessions."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import soundfile as sf

from .stitch import Segment


class SessionStore:
    """把一个会话的元信息、录音和转写落到 ``runs_dir/<session-id>/``。

    ``transcript.jsonl`` 是崩溃恢复的权威来源——它逐条追加，任何时刻被中断都只
    影响最后一条。``audio.wav`` 是流式写入的，崩溃时 WAV 头里的长度字段会过期，
    音频内容还在但需要修复头部，属于尽力而为。
    """

    def __init__(
        self,
        runs_dir: str | Path,
        session_id: str | None = None,
        *,
        sample_rate: int = 16000,
        record_audio: bool = True,
        name: str = "",
    ):
        self.runs_dir = Path(runs_dir).expanduser()
        self.session_id = session_id or f"{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"
        self.dir = self.runs_dir / self.session_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.sample_rate = int(sample_rate)
        self.record_audio = bool(record_audio)
        self.name = name
        self.audio_path = self.dir / "audio.wav"
        self.transcript_path = self.dir / "transcript.jsonl"
        self.provisional_path = self.dir / "provisional.json"
        self.meta_path = self.dir / "session.json"
        self._audio = None
        self._started_at = time.time()
        self._closed = False
        self.write_meta(status="recording")

    def append_audio(self, pcm: np.ndarray) -> None:
        if not self.record_audio or self._closed:
            return
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        if self._audio is None:
            self._audio = sf.SoundFile(
                str(self.audio_path),
                mode="w",
                samplerate=self.sample_rate,
                channels=1,
                subtype="PCM_16",
            )
        self._audio.write(arr)

    def append_committed(self, segments: Iterable[Any]) -> None:
        with self.transcript_path.open("a", encoding="utf-8") as handle:
            for item in segments:
                handle.write(json.dumps(item.to_dict(), ensure_ascii=False) + "\n")

    def write_provisional(self, segments: Iterable[Segment]) -> None:
        payload = [
            {
                "start": round(seg.start, 3),
                "end": round(seg.end, 3),
                "speaker": seg.speaker,
                "text": seg.text,
            }
            for seg in segments
        ]
        self.provisional_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")

    def write_meta(self, **fields: Any) -> None:
        data = self.read_meta()
        data.update(fields)
        data.setdefault("session_id", self.session_id)
        data.setdefault("started_at", self._started_at)
        data["name"] = data.get("name") or self.name or self.session_id
        data["sample_rate"] = self.sample_rate
        data["record_audio"] = self.record_audio
        self.meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def read_meta(self) -> dict[str, Any]:
        return _read_json_object(self.meta_path)

    def finalize(
        self,
        speakers: Iterable[dict],
        *,
        status: str = "done",
    ) -> None:
        if self._closed:
            return
        self._closed = True
        if self._audio is not None:
            self._audio.close()
            self._audio = None
        self.write_meta(status=status, ended_at=time.time(), speakers=list(speakers))

    @staticmethod
    def session_dir(runs_dir: str | Path, session_id: str) -> Path:
        return Path(runs_dir).expanduser() / session_id

    @staticmethod
    def load_committed(runs_dir: str | Path, session_id: str) -> list[dict]:
        path = SessionStore.session_dir(runs_dir, session_id) / "transcript.jsonl"
        if not path.exists():
            return []
        rows: list[dict] = []
        # ``errors="replace"`` 而不是默认的严格解码：文件是逐条追加的，崩溃可能把最后
        # 一行切断在某个多字节字符中间。严格模式会因这一个坏字符让**整份转写**抛
        # ``UnicodeDecodeError``，而正确行为只是丢掉那一行。坏字节解成 U+FFFD 后，
        # 该行的 JSON 解析失败、被下面的 except 跳过，前面的行照常返回。
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return rows

    @staticmethod
    def list_sessions(runs_dir: str | Path) -> list[dict]:
        root = Path(runs_dir).expanduser()
        if not root.is_dir():
            return []
        sessions: list[dict] = []
        for path in root.iterdir():
            if not path.is_dir():
                continue
            meta = _read_json_object(path / "session.json")
            if not meta:
                continue
            meta.setdefault("session_id", path.name)
            sessions.append(meta)
        sessions.sort(key=lambda item: float(item.get("started_at") or 0.0), reverse=True)
        return sessions


def _read_json_object(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        # 同 ``load_committed``：被切断在字符中间的元信息解成 U+FFFD 后 JSON 解析失败，
        # 走 except 返回空字典，而不是让一个坏文件把整个 ``list_sessions`` 拖崩。
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}
