import os
import subprocess
import sys
import unittest

from bench_support import bounded_run


class ProcessLimits(unittest.TestCase):
    def test_normal_output_and_exit(self):
        result = bounded_run([sys.executable, "-c", "print('ok'); raise SystemExit(7)"], timeout=5)
        self.assertEqual(result.stdout, b"ok\n")
        self.assertEqual(result.returncode, 7)
        self.assertIsNone(result.limit_reason)

    def test_exit125_is_distinct_from_capture_overflow(self):
        result = bounded_run([sys.executable, "-c", "raise SystemExit(125)"], timeout=5)
        self.assertEqual(result.returncode, 125)
        self.assertIsNone(result.limit_reason)

    def test_unlimited_diagnostic_output_reports_the_limit(self):
        code = "import os\nwhile True: os.write(2, b'x'*65536)"
        result = bounded_run([sys.executable, "-c", code], timeout=5)
        self.assertEqual(result.returncode, 125)
        self.assertIn("output limit exceeded", result.limit_reason)
        self.assertLessEqual(len(result.stderr), 8 * 1024 * 1024 + 23)

    def test_unlimited_output_is_bounded(self):
        code = "import os\nwhile True: os.write(1, b'x'*65536)"
        result = bounded_run([sys.executable, "-c", code], timeout=5)
        self.assertNotEqual(result.returncode, 0)
        self.assertLessEqual(len(result.stdout), 8 * 1024 * 1024)

    def test_silent_loop_has_deadline(self):
        with self.assertRaises(subprocess.TimeoutExpired):
            bounded_run([sys.executable, "-c", "while True: pass"], timeout=0.2)

    def test_build_artifacts_have_an_independent_budget(self):
        code = "import tempfile\nwith tempfile.TemporaryFile() as f: f.write(b'x'*(9*1024*1024))\nprint('built')"
        result = bounded_run([sys.executable, "-c", code], file_mib=16, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"built\n")

    def test_build_budget_does_not_relax_output_limit(self):
        code = "import os,time; os.write(1, b'x'*(9*1024*1024)); time.sleep(60)"
        result = bounded_run([sys.executable, "-c", code], file_mib=16, timeout=5)
        self.assertEqual(result.returncode, 125)
        self.assertLessEqual(len(result.stdout), 8 * 1024 * 1024)
        self.assertIn(b"output limit exceeded", result.stderr)

    def test_input_survives_output_watchdog_polls(self):
        code = "import sys,time; time.sleep(0.3); data=sys.stdin.buffer.read(); print(len(data))"
        result = bounded_run([sys.executable, "-c", code], input=b'x'*131072, timeout=5)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, b"131072\n")

    def test_descendant_is_stopped_on_deadline(self):
        code = "import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); print(p.pid,flush=True); time.sleep(60)"
        with self.assertRaises(subprocess.TimeoutExpired) as caught:
            bounded_run([sys.executable, "-c", code], timeout=0.3)
        pid = int(caught.exception.stdout)
        # A zombie may remain briefly until the container init reaps it; it
        # cannot run or allocate. A living descendant is a failed kill contract.
        state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], capture_output=True, text=True)
        self.assertTrue(not state.stdout.strip() or state.stdout.strip().startswith("Z"))


if __name__ == "__main__":
    unittest.main()
