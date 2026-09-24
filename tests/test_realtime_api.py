from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fastapi.testclient import TestClient

from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.store import SessionStore

SCRIPTED_REPLY = "[1.0][S01]你好[2.0]"


class ScriptedTranscriber:
    """按窗口返回预置文本的假转写器。"""

    def __init__(self, replies=None):
        self.replies = list(replies or [SCRIPTED_REPLY] * 8)
        self.windows = 0
        self.prompts: list[str] = []  # 让测试能观测 prompt 真的换了

    def transcribe_window(self, audio, *, prompt):
        self.windows += 1
        self.prompts.append(prompt)
        return self.replies.pop(0) if self.replies else ""


class _Row:
    """``append_committed`` 只需要有 ``to_dict()`` 的对象，这里省掉真的构造 CommittedSegment。"""

    def to_dict(self) -> dict:
        return {
            "id": "seg-1", "start": 1.0, "end": 2.0, "speaker": "S01",
            "speaker_name": "S01", "text": "你好", "speaker_confident": True,
        }


def _app(tmp: Path, **config_kwargs):
    from moss_transcribe_diarize.app.realtime_server import create_realtime_app

    created: list[ScriptedTranscriber] = []

    def factory() -> ScriptedTranscriber:
        instance = ScriptedTranscriber()
        created.append(instance)
        return instance

    config = RealtimeConfig(silence_gate=False, **config_kwargs)
    app = create_realtime_app(
        config=config,
        transcriber_factory=factory,
        embedder=None,
        runs_dir=tmp / "runs",
    )
    app.state.created_transcribers = created
    return app


class RuntimeRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.client = TestClient(_app(Path(self._tmp.name)))

    def test_reports_the_effective_config(self):
        payload = self.client.get("/api/runtime").json()

        self.assertEqual(payload["config"]["window"], 20.0)
        self.assertEqual(payload["config"]["hop"], 5.0)
        self.assertIn("backend", payload)

    def test_reports_speaker_state(self):
        payload = self.client.get("/api/runtime").json()
        self.assertIn("speaker", payload)
        self.assertIn("enabled", payload["speaker"])

    def test_advertises_the_export_formats(self):
        payload = self.client.get("/api/runtime").json()

        self.assertIn("txt", payload["export_formats"])


class SessionsRouteTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"
        # 先手工造一个已完成的会话，供读取路由使用
        store = SessionStore(self.runs, "s1", name="周会")
        store.append_committed([_Row()])
        store.write_meta(speakers=[{"id": "S01", "name": "张总", "samples": 2}])
        store.finalize([{"id": "S01", "name": "张总", "samples": 2}])
        self.client = TestClient(_app(Path(self._tmp.name)))

    def test_lists_sessions(self):
        payload = self.client.get("/api/sessions").json()
        self.assertEqual([s["session_id"] for s in payload["sessions"]], ["s1"])

    def test_returns_a_session_with_its_segments(self):
        payload = self.client.get("/api/sessions/s1").json()
        self.assertEqual(payload["session"]["name"], "周会")
        self.assertEqual(payload["segments"][0]["id"], "seg-1")

    def test_unknown_session_is_404_not_500(self):
        self.assertEqual(self.client.get("/api/sessions/nope").status_code, 404)

    def test_a_path_traversal_id_is_rejected_not_served(self):
        response = self.client.get("/api/sessions/..%2F..%2Fetc")
        self.assertIn(response.status_code, (400, 404))

    def test_missing_recording_is_404(self):
        self.assertEqual(self.client.get("/api/sessions/s1/audio").status_code, 404)

    def test_exports_every_advertised_format(self):
        for fmt in ("srt", "json", "ass", "md", "txt"):
            with self.subTest(fmt=fmt):
                response = self.client.get(f"/api/sessions/s1/export?format={fmt}")
                self.assertEqual(response.status_code, 200)
                self.assertTrue(response.text.strip())

    def test_export_uses_the_renamed_speaker(self):
        response = self.client.get("/api/sessions/s1/export?format=txt")
        self.assertIn("张总", response.text)

    def test_unknown_export_format_is_400(self):
        self.assertEqual(
            self.client.get("/api/sessions/s1/export?format=docx").status_code, 400
        )

    def test_exporting_an_unknown_session_is_404(self):
        self.assertEqual(
            self.client.get("/api/sessions/nope/export?format=srt").status_code, 404
        )


if __name__ == "__main__":
    unittest.main()
