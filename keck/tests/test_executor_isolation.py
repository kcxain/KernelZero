import importlib.util
import os
import tempfile
import unittest
from pathlib import Path


_EXECUTOR_PATH = Path(__file__).parents[1] / "app" / "services" / "executor.py"
_SPEC = importlib.util.spec_from_file_location("keck_executor_under_test", _EXECUTOR_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_EXECUTOR = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_EXECUTOR)

_extract_error = _EXECUTOR._extract_error
_run_test_process = _EXECUTOR._run_test_process


class ExecutorIsolationTest(unittest.TestCase):
    def test_successful_process(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text("print('ok')\n", encoding="utf-8")
            returncode, output, error = _run_test_process(tmpdir, timeout=5)

        self.assertEqual(returncode, 0)
        self.assertIn("ok", output)
        self.assertIsNone(error)

    def test_aborted_process_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text(
                "import os\nos.abort()\n", encoding="utf-8"
            )
            returncode, output, error = _run_test_process(tmpdir, timeout=5)

        self.assertIsNotNone(returncode)
        self.assertNotEqual(returncode, 0)
        self.assertIsNone(error)
        message = _extract_error(output, returncode)
        self.assertIn("failed: test process", message)
        if os.name == "posix":
            self.assertIn("SIGABRT", message)

    def test_timeout_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            Path(tmpdir, "test.py").write_text(
                "import time\ntime.sleep(30)\n", encoding="utf-8"
            )
            returncode, _, error = _run_test_process(tmpdir, timeout=1)

        self.assertIsNone(returncode)
        self.assertIsNotNone(error)
        self.assertIn("timed out", error)


if __name__ == "__main__":
    unittest.main()
