import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import opcensus
from bench_support import BoundedCompletedProcess


class Census(unittest.TestCase):
    def measure(self, stderr, *, code=0, reason=None):
        result = BoundedCompletedProcess([], code, b"answer", stderr)
        result.limit_reason = reason
        with patch("bench_support.bounded_run", return_value=result) as run:
            fields = opcensus.census(Path("nonexistent.aheui"))
        self.assertNotIn("MAJIT_LOG", run.call_args.kwargs["env"])
        self.assertEqual(run.call_args.kwargs["env"]["MAJIT_LOG_OPS"], "1")
        return fields

    def test_counts_only_emission_events(self):
        fields = self.measure(b"""[dynasm] op-log-version=1
[dynasm] _assemble: 4 ops \xe2\x86\x92 5 ra_ops, frame_depth=10
[dynasm] emit[0]: IntAdd args=[] result=x
[dynasm] guard[1]: GuardTrue args=[] faillocs=0
[dynasm] discard[2]: GcStore args=[]
[dynasm] _assemble done: fail_index=1
[jit-stats] loops_compiled=1 bridges_compiled=0
""")
        self.assertEqual(fields, {"traces": 1, "total_ops": 3, "op.IntAdd": 1,
                                  "op.GuardTrue": 1, "op.GcStore": 1,
                                  "out_bytes": 6, "exit": 0})

    def test_real_exit125_is_not_a_watchdog_failure(self):
        self.assertEqual(self.measure(b"[jit-stats] loops_compiled=0\n", code=125)["exit"], 125)

    def test_limit_failure_is_not_a_measurement(self):
        with self.assertRaisesRegex(opcensus.CensusError, "output limit"):
            self.measure(b"", code=125, reason="output limit exceeded")

    def test_timeout_is_reported(self):
        with patch("bench_support.bounded_run", side_effect=subprocess.TimeoutExpired([], 60)):
            with self.assertRaisesRegex(opcensus.CensusError, "timed out"):
                opcensus.census(Path("missing"))

    def test_unsupported_binary_cannot_look_like_an_improvement(self):
        with self.assertRaisesRegex(opcensus.CensusError, "rebuild"):
            self.measure(b"[jit-stats] loops_compiled=1\n")

    def test_missing_stats_and_crashes_fail(self):
        with self.assertRaisesRegex(opcensus.CensusError, "missing JIT statistics"):
            self.measure(b"")
        with self.assertRaisesRegex(opcensus.CensusError, "signal 11"):
            self.measure(b"", code=-11)

    def test_record_does_not_write_partial_measurements(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "aheui"
            binary.touch()
            with patch.object(opcensus, "BINARY", binary), \
                 patch.object(opcensus, "selected", return_value=[("a", Path("a")), ("b", Path("b"))]), \
                 patch.object(opcensus, "census", side_effect=[{}, opcensus.CensusError("timeout")]), \
                 patch.object(opcensus, "write_baseline") as write:
                self.assertEqual(opcensus.main(["opcensus", "record"]), 1)
                write.assert_not_called()


if __name__ == "__main__":
    unittest.main()
