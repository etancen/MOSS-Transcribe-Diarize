from __future__ import annotations

import json
import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np
import soundfile as sf

from moss_transcribe_diarize.realtime.stitch import Segment
from moss_transcribe_diarize.realtime.store import SessionStore


class FakeCommitted:
    """与 CommittedSegment 字段一致的鸭子类型替身。"""

    def __init__(self, id, start, end, speaker_id, speaker_name, text, confident):
        self.id = id
        self.start = start
        self.end = end
        self.speaker_id = speaker_id
        self.speaker_name = speaker_name
        self.text = text
        self.confident = confident

    def to_dict(self):
        return {
            "id": self.id,
            "start": self.start,
            "end": self.end,
            "speaker": self.speaker_id,
            "speaker_name": self.speaker_name,
            "text": self.text,
            "speaker_confident": self.confident,
        }


class SessionStoreTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _store(self, **kwargs) -> SessionStore:
        return SessionStore(self.runs, "sess-1", **kwargs)

    def test_creates_the_session_directory_and_meta(self):
        store = self._store()

        self.assertTrue(store.dir.is_dir())
        self.assertEqual(store.dir, self.runs / "sess-1")
        meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["session_id"], "sess-1")
        self.assertEqual(meta["status"], "recording")

    def test_audio_is_readable_after_finalize(self):
        store = self._store()
        audio = np.linspace(-0.5, 0.5, 16000, dtype=np.float32)
        store.append_audio(audio)

        store.finalize([])

        written, rate = sf.read(str(store.audio_path), dtype="float32")
        self.assertEqual(rate, 16000)
        self.assertEqual(written.shape, (16000,))

    def test_audio_appends_across_calls(self):
        store = self._store()
        store.append_audio(np.zeros(800, dtype=np.float32))
        store.append_audio(np.zeros(400, dtype=np.float32))

        store.finalize([])

        written, _ = sf.read(str(store.audio_path), dtype="float32")
        self.assertEqual(written.shape, (1200,))

    def test_finalize_leaves_a_valid_wav_header(self):
        # 只断言"能读回采样"抓不住"句柄没关闭"——libsndfile 读自己写的文件时会容忍
        # 过期的长度字段。直接断言 RIFF 的长度字段与文件实际长度一致。
        store = self._store()
        store.append_audio(np.zeros(16000, dtype=np.float32))
        store.finalize([])

        data = store.audio_path.read_bytes()
        riff_size = struct.unpack("<I", data[4:8])[0]

        self.assertEqual(riff_size, len(data) - 8)

    def test_no_audio_file_when_recording_is_disabled(self):
        store = self._store(record_audio=False)
        store.append_audio(np.zeros(800, dtype=np.float32))

        store.finalize([])

        self.assertFalse(store.audio_path.exists())

    def test_append_after_finalize_is_ignored(self):
        store = self._store()
        store.finalize([])

        store.append_audio(np.zeros(800, dtype=np.float32))

        self.assertFalse(store.audio_path.exists())

    def test_committed_segments_are_appended_as_jsonl(self):
        store = self._store()
        store.append_committed([FakeCommitted("seg-1", 0.0, 1.0, "S01", "张总", "你好", True)])
        store.append_committed([FakeCommitted("seg-2", 1.0, 2.0, "S02", "S02", "再见", False)])

        rows = SessionStore.load_committed(self.runs, "sess-1")

        self.assertEqual([row["id"] for row in rows], ["seg-1", "seg-2"])
        self.assertEqual(rows[0]["speaker_name"], "张总")
        self.assertEqual(rows[1]["speaker"], "S02")

    def test_load_committed_skips_corrupt_lines(self):
        store = self._store()
        store.append_committed([FakeCommitted("seg-1", 0.0, 1.0, "S01", "S01", "好", True)])
        with store.transcript_path.open("a", encoding="utf-8") as handle:
            handle.write("{ 这不是 JSON\n")

        rows = SessionStore.load_committed(self.runs, "sess-1")

        self.assertEqual([row["id"] for row in rows], ["seg-1"])

    def test_torn_multibyte_tail_does_not_lose_earlier_segments(self):
        # 尾部被切断在**多字节汉字中间**时，若实现是"先解码整个文件再切行"，
        # UnicodeDecodeError 会带走整份转写——而不是只丢最后一条。
        store = self._store()
        store.append_committed([FakeCommitted("seg-1", 0.0, 1.0, "S01", "S01", "你好", True)])
        raw = store.transcript_path.read_bytes()
        store.transcript_path.write_bytes(raw + "好".encode("utf-8")[:2])

        rows = SessionStore.load_committed(self.runs, "sess-1")

        self.assertEqual([row["id"] for row in rows], ["seg-1"])

    def test_torn_multibyte_meta_is_skipped_by_list_sessions(self):
        # Finding 1 的另一半：同样的截断发生在 session.json 上时，_read_json_object
        # 也必须走"替换坏字节再跳过"，而不是让 list_sessions 抛 UnicodeDecodeError。
        SessionStore(self.runs, "a").write_meta()
        (self.runs / "b").mkdir(parents=True)
        (self.runs / "b" / "session.json").write_bytes(b'{"session_id": "b"}' + "坏".encode("utf-8")[:2])

        listed = SessionStore.list_sessions(self.runs)

        self.assertEqual([item["session_id"] for item in listed], ["a"])

    def test_provisional_snapshot_round_trips(self):
        store = self._store()
        store.write_provisional([Segment(1.0, 2.0, "S03", "临时的", 4)])

        payload = json.loads(store.provisional_path.read_text(encoding="utf-8"))

        self.assertEqual(payload, [{"start": 1.0, "end": 2.0, "speaker": "S03", "text": "临时的"}])

    def test_finalize_records_speakers_and_end_time(self):
        store = self._store()
        store.finalize([{"id": "S01", "name": "张总", "samples": 3}], status="done")

        meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["status"], "done")
        self.assertEqual(meta["speakers"], [{"id": "S01", "name": "张总", "samples": 3}])
        self.assertIn("ended_at", meta)

    def test_finalize_is_idempotent(self):
        store = self._store()
        store.finalize([], status="done")

        store.finalize([], status="failed")

        meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["status"], "done")

    def test_write_meta_preserves_existing_fields(self):
        store = self._store()
        store.write_meta(name="周会")

        store.write_meta(status="running")

        meta = json.loads(store.meta_path.read_text(encoding="utf-8"))
        self.assertEqual(meta["name"], "周会")
        self.assertEqual(meta["status"], "running")

    def test_generated_session_id_is_unique(self):
        first = SessionStore(self.runs)
        second = SessionStore(self.runs)

        self.assertNotEqual(first.session_id, second.session_id)

    def test_list_sessions_returns_newest_first(self):
        SessionStore(self.runs, "a").write_meta(started_at=1.0)
        SessionStore(self.runs, "b").write_meta(started_at=2.0)

        listed = SessionStore.list_sessions(self.runs)

        self.assertEqual([item["session_id"] for item in listed], ["b", "a"])

    def test_list_sessions_on_missing_directory_is_empty(self):
        self.assertEqual(SessionStore.list_sessions(self.runs / "nope"), [])

    def test_corrupt_meta_is_skipped_by_list_sessions(self):
        SessionStore(self.runs, "a").write_meta()
        (self.runs / "b").mkdir(parents=True)
        (self.runs / "b" / "session.json").write_text("{ 坏了", encoding="utf-8")

        listed = SessionStore.list_sessions(self.runs)

        self.assertEqual([item["session_id"] for item in listed], ["a"])


if __name__ == "__main__":
    unittest.main()
