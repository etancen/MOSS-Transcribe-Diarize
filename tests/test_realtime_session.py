from __future__ import annotations

import asyncio
import tempfile
import threading
import unittest
import weakref
from pathlib import Path

import numpy as np

from moss_transcribe_diarize.realtime.config import RealtimeConfig
from moss_transcribe_diarize.realtime.session import RealtimeSession
from moss_transcribe_diarize.realtime.store import SessionStore

SILENCE = np.zeros(0, dtype=np.float32)  # 占位，实际音频由 _speech/_silence 生成


def _speech(seconds: float, *, amplitude: float = 0.3, sample_rate: int = 16000) -> np.ndarray:
    t = np.arange(int(round(seconds * sample_rate)), dtype=np.float32) / sample_rate
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def _silence(seconds: float, *, sample_rate: int = 16000) -> np.ndarray:
    return np.zeros(int(round(seconds * sample_rate)), dtype=np.float32)


class ScriptedTranscriber:
    """按调用次序返回预设文本，并记录每次收到的窗口时长。"""

    def __init__(self, replies: list[str]):
        self.replies = list(replies)
        self.window_seconds: list[float] = []

    def transcribe_window(self, audio, *, prompt):
        self.window_seconds.append(len(audio) / 16000.0)
        if not self.replies:
            return ""
        return self.replies.pop(0)


class FailingTranscriber:
    def __init__(self, message: str = "boom"):
        self.message = message
        self.calls = 0

    def transcribe_window(self, audio, *, prompt):
        self.calls += 1
        raise RuntimeError(self.message)


def _config(**kwargs) -> RealtimeConfig:
    params = {
        "window": 20.0,
        "hop": 5.0,
        "tail": 6.0,
        "min_first_window": 8.0,
        "silence_gate": False,
    }
    params.update(kwargs)
    return RealtimeConfig(**params)


class RealtimeSessionTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _session(self, transcriber, config=None, **store_kwargs) -> RealtimeSession:
        return RealtimeSession(
            config or _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1", **store_kwargs),
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def test_does_not_run_before_min_first_window(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(5.0))

        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_empty_buffer_returns_no_events(self):
        """缓冲里一帧音频都没有时，绝不能把空音频送进处理器。"""
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)

        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_empty_push_is_ignored(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)
        session.push_audio(np.zeros(0, dtype=np.float32))

        self.assertEqual(self._run(session.run_pending()), [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_first_window_runs_at_min_first_window(self):
        transcriber = ScriptedTranscriber(["[1][S01]开场[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        self.assertAlmostEqual(transcriber.window_seconds[0], 8.0, places=2)
        self.assertIn("committed", [event["type"] for event in events])

    def test_committed_segments_carry_ids_and_absolute_times(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual(committed["segments"][0]["id"], "seg-1")
        self.assertAlmostEqual(committed["segments"][0]["start"], 1.0, places=2)
        self.assertAlmostEqual(committed["segments"][0]["end"], 2.0, places=2)
        self.assertEqual(committed["segments"][0]["text"], "你好")

    def test_provisional_event_is_emitted_even_when_empty(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        provisional = next(event for event in events if event["type"] == "provisional")
        self.assertEqual(provisional["segments"], [])

    def test_late_segment_stays_provisional_then_commits_later(self):
        # 第一次窗口 [0,20]，段落落在 18-19 秒，在 tail=6 之内，所以只能是临时段。
        # 第二次窗口是 [10,30]，同一段内容此时落在窗口内的 8-9 秒处，换算回绝对
        # 时间仍是 18-19 秒，而 cutoff 是 30-6=24，于是被定稿。
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[8][S01]结尾[9]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        events = self._run(session.run_pending())
        self.assertEqual([e for e in events if e["type"] == "committed"], [])

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual([seg["text"] for seg in committed["segments"]], ["结尾"])
        self.assertAlmostEqual(committed["segments"][0]["start"], 18.0, places=2)
        self.assertAlmostEqual(committed["segments"][0]["end"], 19.0, places=2)

    def test_hop_throttles_repeated_runs(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 5)
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())

        session.push_audio(_speech(2.0))
        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(len(transcriber.window_seconds), 1)

    def test_window_covers_at_most_window_seconds(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 5)
        session = self._session(transcriber)
        session.push_audio(_speech(60.0))

        self._run(session.run_pending())
        session.push_audio(_speech(10.0))
        self._run(session.run_pending())

        self.assertAlmostEqual(transcriber.window_seconds[1], 20.0, places=2)

    def test_records_audio_to_store(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())

        events = self._run(session.close())
        del events

        audio_path = self.runs / "s1" / "audio.wav"
        self.assertTrue(audio_path.exists())
        import soundfile as sf

        written, _ = sf.read(str(audio_path), dtype="float32")
        self.assertEqual(written.shape[0], 8 * 16000)

    def test_close_flushes_provisional_and_finalizes(self):
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[18][S01]结尾[19]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        self._run(session.run_pending())

        events = self._run(session.close())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual([seg["text"] for seg in committed["segments"]], ["结尾"])
        self.assertTrue(session.closed)

    def test_close_is_idempotent(self):
        transcriber = ScriptedTranscriber([])
        session = self._session(transcriber)

        self._run(session.close())

        self.assertEqual(self._run(session.close()), [])

    def test_push_after_close_is_ignored(self):
        transcriber = ScriptedTranscriber([])
        session = self._session(transcriber)
        self._run(session.close())

        session.push_audio(_speech(8.0))

        self.assertEqual(session.committed, [])

    def test_status_event_reports_transcript_lag(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 5)
        session = self._session(transcriber)
        session.push_audio(_speech(30.0))

        events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        # 首次运行的窗口是 [0, 30]：Ruling 15 的读法 B 规定首窗左边界恒为 0，不能套用
        # 后续窗口的 max(0, total - window)。段落局部 1-2 秒即绝对 1-2 秒，低于
        # cutoff = 30 - 6 = 24，故被定稿、水位线推到 2.0。字幕落后 = 缓冲 30 - 定稿 2 = 28。
        self.assertAlmostEqual(status["lag_sec"], 28.0, places=1)
        self.assertAlmostEqual(status["buffered_sec"], 30.0, places=1)
        self.assertIn("rtf", status)

    def test_single_failure_reports_error_and_keeps_going(self):
        transcriber = FailingTranscriber()
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        error = next(event for event in events if event["type"] == "error")
        self.assertEqual(error["code"], "transcribe_failed")
        self.assertIn("boom", error["detail"])

    def test_repeated_failures_mark_degraded(self):
        transcriber = FailingTranscriber()
        session = self._session(transcriber)
        for _ in range(3):
            session.push_audio(_speech(10.0))
            events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertTrue(status["degraded"])

    def test_a_success_resets_the_failure_counter(self):
        class FlakyTranscriber:
            def __init__(self):
                self.calls = 0

            def transcribe_window(self, audio, *, prompt):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("第一次失败")
                return "[1][S01]恢复[2]"

        session = self._session(FlakyTranscriber())
        session.push_audio(_speech(10.0))
        self._run(session.run_pending())

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertFalse(status["degraded"])

    def test_speaker_event_lists_the_roster(self):
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))

        events = self._run(session.run_pending())

        speaker = next(event for event in events if event["type"] == "speaker")
        self.assertEqual(speaker["speakers"], [])

    def test_audio_of_uses_window_offsets_not_the_buffer(self):
        """段落音频按窗口内偏移切取，避免环形缓冲回绕后取错位置。"""
        class RecordingSpeakerEmbedder:
            embedding_dim = 2
            seen_sizes: list[int] = []

            def embed(self, audio, sample_rate):
                RecordingSpeakerEmbedder.seen_sizes.append(int(np.asarray(audio).size))
                return np.array([1.0, 0.0], dtype=np.float32)

        RecordingSpeakerEmbedder.seen_sizes = []
        transcriber = ScriptedTranscriber(["[1][S01]你好[2]"])
        session = RealtimeSession(
            _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
            embedder=RecordingSpeakerEmbedder(),
        )
        session.push_audio(_speech(8.0))

        self._run(session.run_pending())

        self.assertEqual(RecordingSpeakerEmbedder.seen_sizes, [16000])

    def test_hop_measures_from_the_last_run_not_the_last_poll(self):
        """被跳过的轮询不得推进 _last_run_sec。

        跳过一次之后必须还能按"距上次真正跑过的窗口"重新触发；若在未运行的分支里
        顺手把 last_run 推到当前缓冲末尾，节流会把下一次本该跑的窗口也吃掉。
        """
        transcriber = ScriptedTranscriber(["[1][S01]a[2]", "[3][S01]b[4]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())  # 窗口 [0, 8]，last_run = 8

        session.push_audio(_speech(2.0))  # 总长 10，距 8 只有 2 秒 → 跳过
        self.assertEqual(self._run(session.run_pending()), [])

        session.push_audio(_speech(4.0))  # 总长 14，距 8 已 6 秒 ≥ hop → 必须跑
        events = self._run(session.run_pending())

        # 第三窗是 [0, 14]，段落 3-4 秒越过水位线 2.0 且低于 cutoff 8，故被定稿。
        self.assertIn("committed", [event["type"] for event in events])
        self.assertEqual(len(transcriber.window_seconds), 2)

    def test_segment_audio_comes_from_the_window_copy_after_the_buffer_evicts_it(self):
        """推理期间缓冲会继续写入并淘汰窗口音频，段落音频必须取自窗口副本。

        真实场景里一次窗口推理要花几秒到几十秒，网络层在此期间仍在推流。这里让转写器
        在 transcribe_window 里推入超过容量的音频，把窗口覆盖的那段从环形缓冲中淘汰，
        于是"改用缓冲重新切取"会取不到音频（slice 返回 None），声纹嵌入根本不会被调用。
        """
        embedder_sizes: list[int] = []

        class RecordingSpeakerEmbedder:
            embedding_dim = 2

            def embed(self, audio, sample_rate):
                embedder_sizes.append(int(np.asarray(audio).size))
                return np.array([1.0, 0.0], dtype=np.float32)

        class PushingTranscriber:
            """转写期间继续推流的转写器，模拟推理期间网络层仍在写入。"""

            def __init__(self):
                self.session_ref = None
                self.window_seconds: list[float] = []

            def transcribe_window(self, audio, *, prompt):
                self.window_seconds.append(len(audio) / 16000.0)
                session = self.session_ref() if self.session_ref is not None else None
                if session is not None:
                    session.push_audio(_speech(40.0))
                return "[1][S01]你好[2]"

        transcriber = PushingTranscriber()
        session = RealtimeSession(
            _config(buffer_capacity=30.0),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
            embedder=RecordingSpeakerEmbedder(),
        )
        # 弱引用：转写器若强引用会话就形成环，WAV 句柄会活过 TemporaryDirectory.cleanup。
        transcriber.session_ref = weakref.ref(session)
        session.push_audio(_speech(8.0))

        self._run(session.run_pending())

        self.assertEqual(len(transcriber.window_seconds), 1)
        self.assertEqual(embedder_sizes, [16000])

    def test_recovery_after_degrading_clears_the_counter(self):
        """连续失败到 degraded 之后，一次成功必须把 degraded 清掉。

        只做"一次失败 + 一次成功"是测不出计数未复位的：那时计数只有 1，远低于阈值，
        两种实现都给 False。必须先真的跨过阈值。
        """
        class EventuallyWorkingTranscriber:
            def __init__(self):
                self.calls = 0

            def transcribe_window(self, audio, *, prompt):
                self.calls += 1
                if self.calls <= 3:
                    raise RuntimeError("连续失败")
                return "[1][S01]恢复[2]"

        session = self._session(EventuallyWorkingTranscriber())
        for _ in range(3):
            session.push_audio(_speech(10.0))
            events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertTrue(status["degraded"])

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        status = next(event for event in events if event["type"] == "status")
        self.assertFalse(status["degraded"])

    def test_provisional_event_replaces_rather_than_accumulates(self):
        """provisional 是整体替换：第二窗没有临时段时事件必须是空表。"""
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[8][S01]结尾[9]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        first = self._run(session.run_pending())

        first_provisional = next(event for event in first if event["type"] == "provisional")
        self.assertEqual([seg["text"] for seg in first_provisional["segments"]], ["结尾"])

        session.push_audio(_speech(10.0))
        second = self._run(session.run_pending())

        second_provisional = next(event for event in second if event["type"] == "provisional")
        self.assertEqual(second_provisional["segments"], [])

    def test_close_uses_a_fresh_window_result_when_promoting(self):
        """close() 必须重跑一次推理再定稿，不能直接提升上一窗的临时快照。

        两个窗口给出不同文本，定稿内容因此能区分"新推理结果"与"旧快照"。
        """
        transcriber = ScriptedTranscriber(["[18][S01]旧[19]", "[18][S01]新[19]"])
        session = self._session(transcriber)
        session.push_audio(_speech(20.0))
        self._run(session.run_pending())

        events = self._run(session.close())

        committed = next(event for event in events if event["type"] == "committed")
        self.assertEqual([seg["text"] for seg in committed["segments"]], ["新"])
        self.assertEqual(len(transcriber.window_seconds), 2)

    def test_close_runs_the_blocking_teardown_off_the_event_loop_thread(self):
        """close() 的收尾（声纹嵌入、关 WAV、写元信息）必须在工作线程里跑。

        事件内容在两种实现下完全一样，所以"跑在哪个线程上"是这条路径唯一可观测的差
        异，只能记录线程标识。两个窗口都只产出临时段，逐窗路径不会调用嵌入器，于是嵌
        入器唯一的一次调用必然来自 close() 的 flush；若收尾退回事件循环线程，嵌入器
        看到的 ident 就等于循环线程的 ident。
        """
        class ThreadRecordingEmbedder:
            embedding_dim = 2
            idents: list[int] = []

            def embed(self, audio, sample_rate):
                ThreadRecordingEmbedder.idents.append(threading.get_ident())
                return np.array([1.0, 0.0], dtype=np.float32)

        ThreadRecordingEmbedder.idents = []
        transcriber = ScriptedTranscriber(["[18][S01]结尾[19]", "[18][S01]结尾[19]"])
        session = RealtimeSession(
            _config(),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
            embedder=ThreadRecordingEmbedder(),
        )
        session.push_audio(_speech(20.0))
        self._run(session.run_pending())

        async def close_inside_the_loop():
            loop_ident = threading.get_ident()
            events = await session.close()
            return loop_ident, events

        loop_ident, events = self._run(close_inside_the_loop())

        self.assertIn("committed", [event["type"] for event in events])
        self.assertEqual(len(ThreadRecordingEmbedder.idents), 1)
        self.assertNotEqual(ThreadRecordingEmbedder.idents[0], loop_ident)

    def test_push_after_close_retains_no_audio(self):
        """关闭后再推流不得被会话保留。

        这个守卫没有公开可观测的症状：run_pending/close 都以 _closed 短路，committed
        也不变。它唯一的后果是会话不再持有音频（内存），所以只能断言内部缓冲。
        """
        transcriber = ScriptedTranscriber([])
        session = self._session(transcriber)
        self._run(session.close())

        session.push_audio(_speech(8.0))

        self.assertEqual(session.committed, [])
        self.assertEqual(session._buffer.total_seconds, 0.0)


class SilenceGateTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.runs = Path(self._tmp.name) / "runs"

    def _session(self, transcriber, **kwargs) -> RealtimeSession:
        # 注意不要把 silence_gate 同时写死再通过 **kwargs 传一次——那会是重复关键字。
        params = {"silence_gate": True}
        params.update(kwargs)
        return RealtimeSession(
            _config(**params),
            transcriber=transcriber,
            store=SessionStore(self.runs, "s1"),
        )

    def _run(self, coro):
        return asyncio.run(coro)

    def test_silent_window_skips_inference(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))

        events = self._run(session.run_pending())

        self.assertEqual(transcriber.window_seconds, [])
        status = next(event for event in events if event["type"] == "status")
        self.assertEqual(status["state"], "running")

    def test_silent_window_still_advances_the_schedule(self):
        """跳过后必须推进 last_run_sec，否则每次轮询都在重算同一个窗口。"""
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))
        self._run(session.run_pending())

        session.push_audio(_silence(2.0))
        events = self._run(session.run_pending())

        self.assertEqual(events, [])
        self.assertEqual(transcriber.window_seconds, [])

    def test_speech_after_silence_is_transcribed(self):
        transcriber = ScriptedTranscriber(["[1][S01]恢复[2]"] * 3)
        session = self._session(transcriber)
        session.push_audio(_silence(10.0))
        self._run(session.run_pending())

        session.push_audio(_speech(10.0))
        events = self._run(session.run_pending())

        self.assertEqual(len(transcriber.window_seconds), 1)
        self.assertIn("committed", [event["type"] for event in events])

    def test_close_runs_the_final_window_even_when_it_is_silent(self):
        # close() 刻意**不**经过静音门控：收尾那一窗是为了让临时区拿到最新结果。
        # 构造一个整段安静的收尾窗口，就能从转写器的调用次数上直接看出来——门控若
        # 在这里生效，最后这一窗根本不会被推理。
        #
        # 时序：前 8 秒有语音（首窗 [0,8] 出一段临时段），之后 22 秒全静音，于是
        # total = 30 时收尾窗口是 [10, 30]，整段落在这段静音里。
        transcriber = ScriptedTranscriber(["[6][S01]旧[7]", "[1][S01]新[2]"])
        session = self._session(transcriber)
        session.push_audio(_speech(8.0))
        self._run(session.run_pending())
        calls_after_first_run = len(transcriber.window_seconds)

        session.push_audio(_silence(22.0))
        self._run(session.close())

        self.assertEqual(len(transcriber.window_seconds), calls_after_first_run + 1)

    def test_gate_can_be_disabled(self):
        transcriber = ScriptedTranscriber(["[1][S01]a[2]"] * 3)
        session = self._session(transcriber, silence_gate=False)
        session.push_audio(_silence(10.0))

        self._run(session.run_pending())

        self.assertEqual(len(transcriber.window_seconds), 1)


if __name__ == "__main__":
    unittest.main()
