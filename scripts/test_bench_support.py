import tempfile
import unittest
from pathlib import Path

from bench_support import format_fields, parse_fields, read_int_fields, write_fields
from jitstats import field_verdict, floor_regression


class Baselines(unittest.TestCase):
    def test_text_roundtrip(self):
        fields = {"z": "value=with=equals", "a": "1"}
        self.assertEqual(format_fields(fields), "a=1\nz=value=with=equals\n")
        self.assertEqual(parse_fields(format_fields(fields)), fields)
        self.assertEqual(parse_fields("ignored\na=1\na=2\n"), {"a": "2"})

    def test_integer_file_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nested" / "fixture.opcensus"
            write_fields(path, {"exit": -1, "traces": 2})
            self.assertEqual(read_int_fields(path), {"exit": -1, "traces": 2})

    def test_gate_and_display_agree(self):
        for field in ("guard_failures", "loops_compiled", "loops_aborted"):
            for base in (0, 1, 8, 100):
                for current in (0, 1, 2, 3, 8, 10, 11, 100, 125, 126):
                    verdict, _ = field_verdict(field, base, current)
                    failures = floor_regression({field: str(base)}, {field: str(current)})
                    self.assertEqual(bool(failures), verdict, (field, base, current))

    def test_missing_counter_policy(self):
        self.assertEqual(floor_regression({}, {"guard_failures": "20"}), [])
        self.assertEqual(floor_regression({}, {"loops_compiled": "20"}), [])
        self.assertEqual(floor_regression({}, {"loops_aborted": "1"}), ["loops_aborted 0 -> 1"])


if __name__ == "__main__":
    unittest.main()
