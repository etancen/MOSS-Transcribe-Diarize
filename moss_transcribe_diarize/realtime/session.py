"""Orchestration: buffer audio, run windows, commit stable segments."""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

import numpy as np

from moss_transcribe_diarize.prompts import DEFAULT_PROMPT

from .buffer import AudioRingBuffer
from .config import RealtimeConfig
from .energy import is_silent
from .speaker import SpeakerEmbedder, SpeakerGallery
from .stitch import Segment, Stitcher
from .store import SessionStore
from .transcriber import WindowTranscriber
from .window import WindowPolicy


@dataclass(frozen=True, slots=True)
class CommittedSegment:
    id: str
    start: float
    end: float
    speaker_id: str
    speaker_name: str
    text: str
    confident: bool

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "start": round(self.start, 3),
            "end": round(self.end, 3),
            "speaker": self.speaker_id,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "speaker_confident": self.confident,
        }


def segment_audio(
    window_audio: np.ndarray,
    window_start: float,
    seg: Segment,
    sample_rate: int,
) -> np.ndarray | None:
    """按窗口内偏移切出段落音频。

    刻意不从环形缓冲重新取——缓冲可能已经回绕并淘汰了这段音频，而窗口副本
    一定是完整的。
    """
    total = int(window_audio.size)
    start = int(round((seg.start - window_start) * sample_rate))
    end = int(round((seg.end - window_start) * sample_rate))
    start = max(0, min(start, total))
    end = max(start, min(end, total))
    if end <= start:
        return None
    return window_audio[start:end]


class RealtimeSession:
    """一个会话的全部状态与行为。不负责网络，只吐事件。"""

    def __init__(
        self,
        config: RealtimeConfig,
        *,
        transcriber: WindowTranscriber,
        store: SessionStore,
        embedder: SpeakerEmbedder | None = None,
        prompt: str = DEFAULT_PROMPT,
    ):
        self.config = config
        self._transcriber = transcriber
        self._store = store
        self._prompt = prompt or DEFAULT_PROMPT
        self._buffer = AudioRingBuffer(config.buffer_capacity, config.sample_rate)
        self._policy = WindowPolicy(
            window=config.window,
            hop=config.hop,
            min_first_window=config.min_first_window,
        )
        self._stitcher = Stitcher(window=config.window, tail=config.tail)
        self._gallery = SpeakerGallery(
            embedder,
            threshold=config.speaker_threshold,
            min_segment_sec=config.min_segment_sec,
            sample_rate=config.sample_rate,
        )
        self._last_run_sec: float | None = None
        self._window_running = False
        self._failures = 0
        self._window_id = 0
        self._segment_counter = 0
        self._last_window_sec = 0.0
        self._closed = False
        self._committed: list[CommittedSegment] = []

    @property
    def committed(self) -> list[CommittedSegment]:
        return list(self._committed)

    @property
    def closed(self) -> bool:
        return self._closed

    def push_audio(self, pcm: np.ndarray) -> None:
        if self._closed:
            return
        arr = np.asarray(pcm, dtype=np.float32).reshape(-1)
        if arr.size == 0:
            return
        self._buffer.append(arr)
        self._store.append_audio(arr)

    async def run_pending(self) -> list[dict]:
        if self._closed or self._window_running:
            return []
        total = self._buffer.total_seconds
        decision = self._policy.decide(total_seconds=total, last_run_sec=self._last_run_sec)
        if not decision.should_run:
            return []
        audio = self._buffer.slice(decision.start_sec, decision.end_sec)
        if audio is None or audio.size == 0:
            # 没有可推理的音频。推进 last_run_sec，避免每次轮询都重试。
            self._last_run_sec = decision.end_sec
            return []
        if self.config.silence_gate and is_silent(
            audio,
            self.config.sample_rate,
            self.config.silence_rms_db,
            self.config.silence_frame_ratio,
        ):
            # 整窗静音：跳过推理，但必须推进 last_run_sec，否则下次轮询会重算
            # 同一个窗口，退化成忙等。
            self._last_run_sec = decision.end_sec
            return [self._status_event()]
        return await self._run_window(decision.start_sec, decision.end_sec, audio)

    async def close(self) -> list[dict]:
        if self._closed:
            return []
        events: list[dict] = []
        total = self._buffer.total_seconds
        window_audio: np.ndarray | None = None
        window_start = 0.0
        if total > 0:
            window_start = max(0.0, total - self.config.window)
            window_audio = self._buffer.slice(window_start, total)
            if window_audio is not None and window_audio.size:
                # 用正常的 tail 再跑一次，让临时区拿到最新推理结果，再整段定稿。
                events.extend(await self._run_window(window_start, total, window_audio))
        loop = asyncio.get_running_loop()
        # 收尾同样是阻塞工作：_commit 里的声纹嵌入要跑模型，finalize 要关 WAV 并重写元
        # 信息。整段放进工作线程，理由与逐窗路径相同——否则收尾期间事件循环被占住，在
        # WebSocket 传输下整个服务这段时间都不响应。
        had_remaining, newly = await loop.run_in_executor(
            None, self._teardown, window_audio, window_start
        )
        if had_remaining:
            if newly:
                events.append({"type": "committed", "segments": [seg.to_dict() for seg in newly]})
            events.append({"type": "provisional", "segments": []})
        events.append({"type": "speaker", "speakers": self._gallery.speakers()})
        self._closed = True
        return events

    # --- 内部 ---

    def _teardown(
        self,
        window_audio: np.ndarray | None,
        window_start: float,
    ) -> tuple[bool, list[CommittedSegment]]:
        """会话收尾：定稿临时区、落盘、关闭录音文件。

        与 ``_process_window`` 一样整个在工作线程里跑。返回的两个值正是事件构造需要
        的：本次 flush 是否有内容（决定要不要补一个"临时区已清空"事件），以及新定稿
        的段落。在工作线程里改 stitcher / gallery / store / 段落计数器是安全的，理由
        与 ``_process_window`` 相同——一次会话里这些状态只被串行触碰。
        """
        remaining = self._stitcher.flush()
        newly = self._commit(remaining, window_audio, window_start)
        self._store.finalize(self._gallery.speakers())
        return bool(remaining), newly

    async def _run_window(self, start: float, end: float, audio: np.ndarray) -> list[dict]:
        window_id = self._window_id
        self._window_id += 1
        self._window_running = True
        started = time.perf_counter()
        loop = asyncio.get_running_loop()
        try:
            committed, provisional = await loop.run_in_executor(
                None, self._process_window, start, end, window_id, audio
            )
        except Exception as exc:
            self._failures += 1
            return [
                {
                    "type": "error",
                    "code": "transcribe_failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                    "failures": self._failures,
                },
                self._status_event(),
            ]
        finally:
            self._window_running = False
            self._last_run_sec = end
        self._failures = 0
        self._last_window_sec = time.perf_counter() - started

        events: list[dict] = []
        if committed:
            events.append({"type": "committed", "segments": [seg.to_dict() for seg in committed]})
        events.append(
            {
                "type": "provisional",
                "segments": [
                    {
                        "start": round(seg.start, 3),
                        "end": round(seg.end, 3),
                        "speaker": seg.speaker,
                        "text": seg.text,
                    }
                    for seg in provisional
                ],
            }
        )
        events.append({"type": "speaker", "speakers": self._gallery.speakers()})
        events.append(self._status_event())
        return events

    def _process_window(
        self,
        start: float,
        end: float,
        window_id: int,
        audio: np.ndarray,
    ) -> tuple[list[CommittedSegment], list[Segment]]:
        raw = self._transcriber.transcribe_window(audio, prompt=self._prompt)
        result = self._stitcher.ingest(
            window_start=start,
            window_end=end,
            raw_text=raw,
            window_id=window_id,
        )
        committed = self._commit(result.committed, audio, start)
        self._store.write_provisional(result.provisional)
        return committed, result.provisional

    def _commit(
        self,
        segments: list[Segment],
        window_audio: np.ndarray | None,
        window_start: float,
    ) -> list[CommittedSegment]:
        if not segments:
            return []
        if window_audio is None:
            audio_of = lambda seg: None  # noqa: E731
        else:
            audio_of = lambda seg: segment_audio(  # noqa: E731
                window_audio, window_start, seg, self.config.sample_rate
            )
        assignments = self._gallery.assign(segments, audio_of)

        emitted: list[CommittedSegment] = []
        for item in assignments:
            self._segment_counter += 1
            seg = item.segment
            emitted.append(
                CommittedSegment(
                    id=f"seg-{self._segment_counter}",
                    start=seg.start,
                    end=seg.end,
                    speaker_id=item.speaker_id,
                    speaker_name=self._gallery.display_name(item.speaker_id),
                    text=seg.text,
                    confident=item.confident,
                )
            )
        self._store.append_committed(emitted)
        self._committed.extend(emitted)
        return emitted

    def _status_event(self) -> dict:
        hop = self.config.hop
        buffered = self._buffer.total_seconds
        # lag 用"字幕落后实时多少秒"来定义，而不是"缓冲里有多少秒音频没跑过推理"。
        # 后者恒等于零：_last_run_sec 每次都被设成窗口末尾，所以那个差值没有信息量。
        # 定稿水位线才是用户真正感受到的延迟，它天然包含 window + tail 的开销。
        lag = max(0.0, buffered - self._stitcher.committed_until)
        degraded = self._failures >= self.config.max_consecutive_failures
        return {
            "type": "status",
            "state": "degraded" if degraded else "running",
            "buffered_sec": round(buffered, 2),
            "rtf": round(self._last_window_sec / hop, 3) if hop > 0 else 0.0,
            "last_window_ms": int(round(self._last_window_sec * 1000)),
            "lag_sec": round(lag, 2),
            "degraded": degraded,
        }
