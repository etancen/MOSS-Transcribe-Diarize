"""Cross-window speaker identity via speaker embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol, runtime_checkable

import numpy as np

from .stitch import Segment

UNKNOWN_SPEAKER_ID = "U00"
UNKNOWN_SPEAKER_NAME = "未知"


@runtime_checkable
class SpeakerEmbedder(Protocol):
    """声纹嵌入。实现方负责把任意长度音频压成一个定长向量。"""

    embedding_dim: int

    def embed(self, audio: np.ndarray, sample_rate: int) -> np.ndarray | None:
        """返回嵌入，或在这段音频无法可靠嵌入时返回 ``None``。"""


@dataclass(frozen=True, slots=True)
class Assignment:
    segment: Segment
    speaker_id: str
    confident: bool


@dataclass(slots=True)
class _GalleryEntry:
    id: str
    centroid: np.ndarray
    samples: int = 1
    name: str = ""


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm <= 0.0 or not np.isfinite(norm):
        return arr
    return arr / norm


class SpeakerGallery:
    """把窗口内的局部说话人标签映射到全局一致的说话人编号。

    模型输出的 ``[S01]``/``[S02]`` 只在单个窗口内自洽。跨窗口的一致性是靠声纹
    嵌入的余弦相似度匹配做出来的，局部标签只是"同一窗口内这些段落属于同一个人"
    这个先验。
    """

    def __init__(
        self,
        embedder: SpeakerEmbedder | None,
        *,
        threshold: float = 0.55,
        min_segment_sec: float = 0.4,
        sample_rate: int = 16000,
    ):
        self._embedder = embedder
        self._threshold = float(threshold)
        self._sample_rate = int(sample_rate)
        self._min_samples = max(1, int(round(min_segment_sec * self._sample_rate)))
        self._entries: dict[str, _GalleryEntry] = {}
        self._next_index = 1
        self._unknown_assignments = 0

    @property
    def enabled(self) -> bool:
        return self._embedder is not None

    def assign(
        self,
        segments: list[Segment],
        audio_of: Callable[[Segment], np.ndarray | None],
    ) -> list[Assignment]:
        if not segments:
            return []
        if self._embedder is None:
            return [Assignment(seg, seg.speaker, False) for seg in segments]

        groups: dict[tuple[int, str], list[Segment]] = {}
        order: list[tuple[int, str]] = []
        for seg in segments:
            key = (seg.window_id, seg.speaker)
            if key not in groups:
                groups[key] = []
                order.append(key)
            groups[key].append(seg)

        decided: dict[tuple[int, str], tuple[str, bool]] = {}
        for key in order:
            decided[key] = self._decide_group(groups[key], audio_of)

        return [
            Assignment(seg, *decided[(seg.window_id, seg.speaker)]) for seg in segments
        ]

    def _decide_group(
        self,
        group: list[Segment],
        audio_of: Callable[[Segment], np.ndarray | None],
    ) -> tuple[str, bool]:
        best_audio: np.ndarray | None = None
        best_size = 0
        for seg in group:
            audio = audio_of(seg)
            if audio is None:
                continue
            size = int(np.asarray(audio).size)
            if size < self._min_samples:
                continue
            if size > best_size:
                best_audio, best_size = audio, size

        if best_audio is None:
            self._unknown_assignments += 1
            return UNKNOWN_SPEAKER_ID, False

        embedding = self._embedder.embed(best_audio, self._sample_rate)
        if embedding is None:
            self._unknown_assignments += 1
            return UNKNOWN_SPEAKER_ID, False

        return self._match(embedding), True

    def _match(self, embedding: np.ndarray) -> str:
        vec = _normalize(embedding)
        best_id: str | None = None
        best_score = -1.0
        for entry in self._entries.values():
            score = float(np.dot(entry.centroid, vec))
            if score > best_score:
                best_id, best_score = entry.id, score

        if best_id is not None and best_score >= self._threshold:
            entry = self._entries[best_id]
            entry.centroid = _normalize(entry.centroid * entry.samples + vec)
            entry.samples += 1
            return best_id

        new_id = f"S{self._next_index:02d}"
        self._next_index += 1
        self._entries[new_id] = _GalleryEntry(id=new_id, centroid=vec, samples=1)
        return new_id

    def rename(self, speaker_id: str, name: str) -> None:
        entry = self._entries.get(speaker_id)
        if entry is None:
            raise KeyError(speaker_id)
        entry.name = str(name).strip()

    def display_name(self, speaker_id: str) -> str:
        entry = self._entries.get(speaker_id)
        if entry is None:
            return UNKNOWN_SPEAKER_NAME if speaker_id == UNKNOWN_SPEAKER_ID else speaker_id
        return entry.name or entry.id

    def speakers(self) -> list[dict]:
        roster = [
            {"id": entry.id, "name": entry.name or entry.id, "samples": entry.samples}
            for entry in self._entries.values()
        ]
        if self._unknown_assignments:
            roster.append(
                {"id": UNKNOWN_SPEAKER_ID, "name": UNKNOWN_SPEAKER_NAME, "samples": self._unknown_assignments}
            )
        return roster
