import unittest
from types import SimpleNamespace
from unittest.mock import patch

from backend.app import system


class SystemTelemetryTests(unittest.TestCase):
    def tearDown(self):
        system._memory_hardware.cache_clear()

    def test_telemetry_includes_memory_usage_and_hardware(self):
        fake_psutil = SimpleNamespace(
            cpu_percent=lambda interval=None: 12.5,
            virtual_memory=lambda: SimpleNamespace(total=32_000, available=9_000),
        )
        with (
            patch.object(system, "psutil", fake_psutil),
            patch.object(system, "query_gpus", return_value=[]),
            patch.object(system, "_memory_hardware", return_value={"type": "DDR5", "speed_mts": 6000}),
        ):
            result = system.telemetry()

        self.assertEqual(result["memory_total_bytes"], 32_000)
        self.assertEqual(result["memory_available_bytes"], 9_000)
        self.assertEqual(result["memory_used_bytes"], 23_000)
        self.assertEqual(result["memory_type"], "DDR5")
        self.assertEqual(result["memory_speed_mts"], 6000)

    def test_telemetry_uses_memory_fallback_without_psutil(self):
        with (
            patch.object(system, "psutil", None),
            patch.object(system, "query_gpus", return_value=[]),
            patch.object(system, "_memory_hardware", return_value={"type": None, "speed_mts": None}),
            patch.object(system, "_fallback_memory", return_value=(16_000, 4_000)),
        ):
            result = system.telemetry()

        self.assertEqual(result["memory_total_bytes"], 16_000)
        self.assertEqual(result["memory_available_bytes"], 4_000)
        self.assertEqual(result["memory_used_bytes"], 12_000)

    @patch.object(system.sys, "platform", "win32")
    @patch.object(system.subprocess, "run")
    def test_memory_hardware_reads_windows_dimm_details(self, run):
        run.return_value = SimpleNamespace(
            stdout='[{"SMBIOSMemoryType":34,"ConfiguredClockSpeed":6000,"Speed":6000}]',
            check_returncode=lambda: None,
        )

        self.assertEqual(system._memory_hardware(), {"type": "DDR5", "speed_mts": 6000})


if __name__ == "__main__":
    unittest.main()
