from __future__ import annotations

import unittest

from moss_transcribe_diarize.realtime.stitch import (
    MIN_SEGMENT_DURATION,
    Segment,
    clamp_to_window,
    parse_window_segments,
)


class ParseWindowSegmentsTest(unittest.TestCase):
    def test_shifts_local_times_to_absolute(self):
        segments = parse_window_segments("[0.5][S01]你好[1.5]", window_start=40.0, window_id=3)

        self.assertEqual(
            segments,
            [Segment(start=40.5, end=41.5, speaker="S01", text="你好", window_id=3)],
        )

    def test_parses_multiple_segments(self):
        raw = "[0.48][S01]Welcome[1.66][12.26][S02]Ready[13.81]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.speaker for s in segments], ["S01", "S02"])
        self.assertEqual([s.text for s in segments], ["Welcome", "Ready"])

    def test_whitespace_separated_noise_does_not_block_later_segments(self):
        # 杂散文本由空白分隔时走 _after_end 的 _pending_after_end 分支：下一个 [ 到来
        # 即干净地 emit 本段，后续片段照常产出。这是"偏离格式不作废整次推理"的良性
        # 一侧。
        raw = "垃圾开头 [0.5][S01]有效[1.5] [2.0][S02]也好[3.0]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["有效", "也好"])

    def test_unseparated_noise_pollutes_a_segment_and_is_dropped(self):
        # 没有空白分隔时解析器会把 [1.5] 折回正文，再把 [2.0] 当成该段的结束时间，
        # 于是本段文字含方括号、结束时间被顶到下一段起点，而 [S02] 解析失败被丢弃。
        # 污染段整段丢弃，所以这里期望空——宁可丢，也不把污染的文字写进定稿。
        raw = "[0.5][S01]有效[1.5]垃圾[2.0][S02]也好[3.0]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual(segments, [])

    def test_trailing_prose_after_the_last_segment_drops_that_segment(self):
        # 流尾的杂散文本（_after_end 的第 1 种例外）：[end] 之后遇到非空白字符会把
        # [end] 折回正文并退回"读取正文"状态，于是该片段永不闭合、close() 也不吐出
        # 它。窗口的最后一段因此会丢；下一个窗口会重新覆盖这段音频，所以是暂时性丢失。
        raw = "这是模型的解释性文字 [0.5][S01]有效内容[1.5] 更多废话"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual(segments, [])

    def test_trailing_segment_without_end_is_dropped(self):
        raw = "[0.5][S01]完整[1.5][2.0][S02]缺尾巴"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual([s.text for s in segments], ["完整"])

    def test_inverted_timestamps_drop_the_segment(self):
        # 解析器只接受 end >= start；顺序颠倒时它把 [end] 折回正文、该片段不闭合，
        # 于是整段被丢弃。所以 parse_window_segments 里不需要交换分支——那会是一段
        # 永远执行不到的死代码。
        raw = "[3.0][S01]顺序反了[1.0]"

        segments = parse_window_segments(raw, window_start=0.0, window_id=0)

        self.assertEqual(segments, [])

    def test_empty_text_yields_nothing(self):
        self.assertEqual(parse_window_segments("", window_start=0.0, window_id=0), [])


class ClampToWindowTest(unittest.TestCase):
    def test_segment_inside_window_is_unchanged(self):
        seg = Segment(start=5.0, end=7.0, speaker="S01", text="a", window_id=0)

        self.assertEqual(clamp_to_window(seg, 0.0, 20.0), seg)

    def test_segment_crossing_left_edge_is_truncated(self):
        seg = Segment(start=-1.0, end=3.0, speaker="S01", text="a", window_id=0)

        clamped = clamp_to_window(seg, 0.0, 20.0)

        self.assertEqual(clamped.start, 0.0)
        self.assertEqual(clamped.end, 3.0)
        self.assertEqual(clamped.text, "a")

    def test_segment_crossing_right_edge_is_dropped(self):
        seg = Segment(start=19.0, end=21.0, speaker="S01", text="a", window_id=0)

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))

    def test_segment_entirely_before_window_is_dropped(self):
        seg = Segment(start=-5.0, end=-1.0, speaker="S01", text="a", window_id=0)

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))

    def test_segment_entirely_after_window_is_dropped(self):
        seg = Segment(start=25.0, end=27.0, speaker="S01", text="a", window_id=0)

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))

    def test_segment_shorter_than_minimum_after_clamping_is_dropped(self):
        seg = Segment(
            start=-1.0,
            end=MIN_SEGMENT_DURATION / 2.0,
            speaker="S01",
            text="a",
            window_id=0,
        )

        self.assertIsNone(clamp_to_window(seg, 0.0, 20.0))


if __name__ == "__main__":
    unittest.main()
