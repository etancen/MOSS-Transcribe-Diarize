from __future__ import annotations

import subprocess
import sys
import unittest


def _run_with_blocked(module: str, blocked: tuple[str, ...]) -> subprocess.CompletedProcess:
    """在子进程里导入 module，并让 blocked 里的顶层包不可用。"""
    code = (
        "import sys\n"
        f"for name in {blocked!r}:\n"
        "    sys.modules[name] = None\n"
        f"import {module}\n"
        "print('IMPORT_OK')\n"
    )
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


class LazyPackageImportTest(unittest.TestCase):
    def test_realtime_modules_import_without_torch(self):
        result = _run_with_blocked(
            "moss_transcribe_diarize.realtime.prompts_probe",
            ("torch", "transformers"),
        )
        # 该模块不存在，期望 ImportError 而不是 torch 相关的错误，
        # 说明父包 __init__ 没有把 torch 拉进来。
        self.assertNotIn("IMPORT_OK", result.stdout)
        self.assertIn("ModuleNotFoundError", result.stderr)
        self.assertNotIn("torch", result.stderr.lower())

    def test_prompts_module_imports_without_torch(self):
        result = _run_with_blocked(
            "moss_transcribe_diarize.prompts",
            ("torch", "transformers"),
        )
        self.assertIn("IMPORT_OK", result.stdout, result.stderr)
        self.assertNotIn("torch", result.stderr.lower())

    def test_lazy_attribute_still_resolves(self):
        import moss_transcribe_diarize as mtd
        from moss_transcribe_diarize.transcript_parser import TranscriptSegment

        self.assertIs(mtd.TranscriptSegment, TranscriptSegment)
        self.assertIsInstance(mtd.DEFAULT_PROMPT, str)

    def test_unknown_attribute_raises(self):
        import moss_transcribe_diarize as mtd

        with self.assertRaises(AttributeError):
            mtd.definitely_not_a_real_name

    def test_all_names_are_importable(self):
        import moss_transcribe_diarize as mtd

        for name in mtd.__all__:
            self.assertTrue(hasattr(mtd, name), name)


if __name__ == "__main__":
    unittest.main()
