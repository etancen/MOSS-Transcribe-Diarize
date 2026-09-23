"""Window-level transcription backends."""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import soundfile as sf


@runtime_checkable
class WindowTranscriber(Protocol):
    """把一个音频窗口转成模型原始输出文本。"""

    def transcribe_window(self, audio: np.ndarray, *, prompt: str) -> str: ...


class HfWindowTranscriber:
    """本地 HF 后端：窗口写临时 wav，交给 ModelRunner。

    ``runner`` 只需要有 ``transcribe(path, *, prompt, max_new_tokens, decoding)``
    方法并返回带 ``.text`` 的对象，所以测试可以注入假实现，不必加载 torch。
    """

    def __init__(
        self,
        runner,
        scratch_dir: str | Path,
        *,
        sample_rate: int = 16000,
        max_new_tokens: int = 1020,
        decoding: str = "greedy",
    ):
        self._runner = runner
        self._scratch_dir = Path(scratch_dir).expanduser()
        self._scratch_dir.mkdir(parents=True, exist_ok=True)
        self._path = self._scratch_dir / "window.wav"
        self._sample_rate = int(sample_rate)
        self._max_new_tokens = int(max_new_tokens)
        self._decoding = decoding

    @property
    def window_path(self) -> Path:
        return self._path

    def transcribe_window(self, audio: np.ndarray, *, prompt: str) -> str:
        arr = np.asarray(audio, dtype=np.float32).reshape(-1)
        sf.write(str(self._path), arr, self._sample_rate, format="WAV", subtype="PCM_16")
        result = self._runner.transcribe(
            self._path,
            prompt=prompt,
            max_new_tokens=self._max_new_tokens,
            decoding=self._decoding,
        )
        return result.text
