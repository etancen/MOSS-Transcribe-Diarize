from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from moss_transcribe_diarize.realtime.transcriber import (
    HfWindowTranscriber,
    WindowTranscriber,
)


class _Result:
    def __init__(self, text: str):
        self.text = text


class FakeRunner:
    def __init__(self, text: str = "[0.5][S01]你好[1.5]"):
        self.text = text
        self.calls: list[dict] = []

    def transcribe(self, audio_path, *, prompt, max_new_tokens, decoding):
        self.calls.append(
            {
                "path": Path(audio_path),
                "prompt": prompt,
                "max_new_tokens": max_new_tokens,
                "decoding": decoding,
            }
        )
        return _Result(self.text)


class HfWindowTranscriberTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.scratch = Path(self._tmp.name) / "scratch"

    def test_is_a_window_transcriber(self):
        transcriber = HfWindowTranscriber(FakeRunner(), self.scratch)

        self.assertIsInstance(transcriber, WindowTranscriber)

    def test_returns_runner_text(self):
        transcriber = HfWindowTranscriber(FakeRunner("结果"), self.scratch)

        self.assertEqual(
            transcriber.transcribe_window(np.zeros(1600, dtype=np.float32), prompt="p"),
            "结果",
        )

    def test_creates_the_scratch_directory(self):
        HfWindowTranscriber(FakeRunner(), self.scratch)

        self.assertTrue(self.scratch.is_dir())

    def test_writes_a_readable_16k_wav(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch)
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)

        transcriber.transcribe_window(audio, prompt="p")

        written, rate = sf.read(str(runner.calls[0]["path"]), dtype="float32")
        self.assertEqual(rate, 16000)
        self.assertEqual(written.shape, (16000,))
        self.assertAlmostEqual(float(np.max(np.abs(written - audio))), 0.0, places=4)

    def test_overwrites_the_same_file_each_call(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch)
        transcriber.transcribe_window(np.zeros(1600, dtype=np.float32), prompt="p")
        first_size = runner.calls[0]["path"].stat().st_size

        transcriber.transcribe_window(np.zeros(800, dtype=np.float32), prompt="p")

        self.assertEqual(runner.calls[0]["path"], runner.calls[1]["path"])
        self.assertLess(runner.calls[1]["path"].stat().st_size, first_size)

    def test_forwards_prompt_and_budget(self):
        runner = FakeRunner()
        transcriber = HfWindowTranscriber(runner, self.scratch, max_new_tokens=777, decoding="sample")

        transcriber.transcribe_window(np.zeros(1600, dtype=np.float32), prompt="自定义")

        self.assertEqual(runner.calls[0]["prompt"], "自定义")
        self.assertEqual(runner.calls[0]["max_new_tokens"], 777)
        self.assertEqual(runner.calls[0]["decoding"], "sample")

    def test_accepts_an_empty_window_without_crashing(self):
        """空窗口在管线里应被上游拦住，但这里也不能因为写文件而崩。"""
        runner = FakeRunner("")
        transcriber = HfWindowTranscriber(runner, self.scratch)

        self.assertEqual(
            transcriber.transcribe_window(np.zeros(0, dtype=np.float32), prompt="p"), ""
        )


if __name__ == "__main__":
    unittest.main()
