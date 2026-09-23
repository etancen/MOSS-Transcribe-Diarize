from __future__ import annotations

import unittest

from moss_transcribe_diarize.realtime.config import RealtimeConfig


class RealtimeConfigTest(unittest.TestCase):
    def test_defaults_are_valid(self):
        config = RealtimeConfig()

        self.assertEqual(config.window, 20.0)
        self.assertEqual(config.hop, 5.0)
        self.assertEqual(config.tail, 6.0)
        self.assertEqual(config.sample_rate, 16000)
        self.assertTrue(config.silence_gate)

    def test_effective_max_new_tokens_scales_with_window(self):
        self.assertEqual(RealtimeConfig(window=20.0).effective_max_new_tokens(), 1020)
        self.assertEqual(RealtimeConfig(window=40.0).effective_max_new_tokens(), 2040)

    def test_effective_max_new_tokens_has_a_floor(self):
        self.assertEqual(
            RealtimeConfig(window=1.0, hop=1.0, tail=0.0).effective_max_new_tokens(), 256
        )

    def test_explicit_max_new_tokens_wins(self):
        config = RealtimeConfig(window=20.0, max_new_tokens=77)

        self.assertEqual(config.effective_max_new_tokens(), 77)

    def test_rejects_hop_longer_than_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(window=10.0, hop=20.0)

    def test_rejects_tail_at_or_past_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(window=10.0, hop=5.0, tail=10.0)

    def test_rejects_buffer_too_small_for_window(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(window=20.0, hop=5.0, buffer_capacity=25.0)

    def test_rejects_out_of_range_similarity_threshold(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(speaker_threshold=1.5)

    def test_rejects_out_of_range_silence_ratio(self):
        with self.assertRaises(ValueError):
            RealtimeConfig(silence_frame_ratio=0.0)

    def test_config_is_frozen(self):
        config = RealtimeConfig()

        with self.assertRaises(Exception):
            config.window = 30.0


if __name__ == "__main__":
    unittest.main()
