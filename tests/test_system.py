import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from backend.app import system


class SystemTelemetryTests(unittest.TestCase):
    def tearDown(self):
        system._memory_hardware.cache_clear()
        system._runtime_speed_cache.clear()

    def test_telemetry_includes_memory_usage_and_hardware(self):
        fake_psutil = SimpleNamespace(
            cpu_percent=lambda interval=None: 12.5,
            virtual_memory=lambda: SimpleNamespace(total=32_000, available=9_000),
        )
        with (
            patch.object(system, "psutil", fake_psutil),
            patch.object(system, "query_gpus", return_value=[]),
            patch.object(system, "cpu_package_power_w", return_value=None),
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
            patch.object(system, "cpu_package_power_w", return_value=None),
            patch.object(system, "_memory_hardware", return_value={"type": None, "speed_mts": None}),
            patch.object(system, "_fallback_memory", return_value=(16_000, 4_000)),
        ):
            result = system.telemetry()

        self.assertEqual(result["memory_total_bytes"], 16_000)
        self.assertEqual(result["memory_available_bytes"], 4_000)
        self.assertEqual(result["memory_used_bytes"], 12_000)

    @patch.object(system.sys, "platform", "win32")
    @patch.object(system.subprocess, "run")
    def test_cpu_package_power_reads_windows_energy_meter(self, run):
        run.return_value = SimpleNamespace(
            stdout="45.75\n",
            check_returncode=lambda: None,
        )

        self.assertEqual(system.cpu_package_power_w(), 45.75)

    def test_power_tracker_integrates_session_and_lifetime_energy(self):
        saved = []
        saved_seconds = []
        tracker = system.PowerTracker(
            total_ever_wh=2000,
            total_ever_seconds=3600,
            save_total_ever_wh=saved.append,
            save_total_ever_seconds=saved_seconds.append,
        )

        first = tracker.sample(300, 100)
        second = tracker.sample(300, 110)

        self.assertEqual(first, {
            "session_kwh": 0.0,
            "session_measured_seconds": 0.0,
            "total_ever_kwh": 2.0,
            "total_ever_measured_seconds": 3600,
        })
        self.assertAlmostEqual(second["session_kwh"], 300 * 10 / 3_600_000)
        self.assertEqual(second["session_measured_seconds"], 10)
        self.assertAlmostEqual(second["total_ever_kwh"], 2 + second["session_kwh"])
        self.assertEqual(second["total_ever_measured_seconds"], 3610)
        self.assertAlmostEqual(saved[-1], 2000 + (300 * 10 / 3600))
        self.assertEqual(saved_seconds[-1], 3610)

    def test_power_tracker_does_not_fill_long_sampling_gaps(self):
        tracker = system.PowerTracker()
        tracker.sample(300, 100)

        result = tracker.sample(300, 120)

        self.assertEqual(result["session_kwh"], 0)
        self.assertEqual(result["session_measured_seconds"], 0)

    def test_telemetry_reports_estimated_system_power_breakdown(self):
        fake_psutil = SimpleNamespace(
            cpu_percent=lambda interval=None: 12.5,
            virtual_memory=lambda: SimpleNamespace(total=32_000, available=9_000),
        )
        tracker = system.PowerTracker()
        with (
            patch.object(system, "psutil", fake_psutil),
            patch.object(system, "query_gpus", return_value=[{"power_draw_w": 225.0}]),
            patch.object(system, "cpu_package_power_w", return_value=45.0),
            patch.object(system, "_memory_hardware", return_value={"type": "DDR5", "speed_mts": 6000}),
        ):
            result = system.telemetry(power_tracker=tracker)

        self.assertEqual(result["power"]["gpu_w"], 225.0)
        self.assertEqual(result["power"]["cpu_w"], 45.0)
        self.assertAlmostEqual(result["power"]["current_system_w"], (225 + 45 + 35) / 0.9)
        self.assertEqual(result["power"]["session_kwh"], 0.0)
        self.assertEqual(result["power"]["session_measured_seconds"], 0.0)
        self.assertEqual(result["power"]["total_ever_measured_seconds"], 0.0)

    @patch.object(system.sys, "platform", "win32")
    @patch.object(system.subprocess, "run")
    def test_memory_hardware_reads_windows_dimm_details(self, run):
        run.return_value = SimpleNamespace(
            stdout='[{"SMBIOSMemoryType":34,"ConfiguredClockSpeed":6000,"Speed":6000}]',
            check_returncode=lambda: None,
        )

        self.assertEqual(system._memory_hardware(), {"type": "DDR5", "speed_mts": 6000})

    def test_runtime_token_speed_reports_latest_and_session_average(self):
        response = Mock()
        response.read.return_value = b"""
llamacpp:prompt_tokens_total 1200
llamacpp:prompt_seconds_total 0.8
llamacpp:tokens_predicted_total 500
llamacpp:tokens_predicted_seconds_total 5
llamacpp:prompt_tokens_seconds 1750
llamacpp:predicted_tokens_seconds 105
llamacpp:requests_processing 1
"""
        response.__enter__ = Mock(return_value=response)
        response.__exit__ = Mock(return_value=False)
        with patch.object(system.urllib.request, "urlopen", return_value=response) as urlopen:
            result = system.runtime_token_speed([("0.0.0.0", 8080)])

        urlopen.assert_called_once_with("http://127.0.0.1:8080/metrics", timeout=0.75)
        self.assertEqual(result["preprocessing"]["current_tps"], 1750)
        self.assertEqual(result["preprocessing"]["session_average_tps"], 1500)
        self.assertEqual(result["decode"]["current_tps"], 105)
        self.assertEqual(result["decode"]["session_average_tps"], 100)
        self.assertEqual(result["active_requests"], 1)

    def test_runtime_token_speed_keeps_last_nonzero_sample(self):
        payloads = [
            b"""llamacpp:prompt_tokens_total 100\nllamacpp:prompt_seconds_total 1\nllamacpp:tokens_predicted_total 100\nllamacpp:tokens_predicted_seconds_total 2\nllamacpp:prompt_tokens_seconds 100\nllamacpp:predicted_tokens_seconds 50\n""",
            b"""llamacpp:prompt_tokens_total 100\nllamacpp:prompt_seconds_total 1\nllamacpp:tokens_predicted_total 100\nllamacpp:tokens_predicted_seconds_total 2\nllamacpp:prompt_tokens_seconds 0\nllamacpp:predicted_tokens_seconds 0\n""",
        ]

        def open_metrics(*args, **kwargs):
            response = Mock()
            response.read.return_value = payloads.pop(0)
            response.__enter__ = Mock(return_value=response)
            response.__exit__ = Mock(return_value=False)
            return response

        with patch.object(system.urllib.request, "urlopen", side_effect=open_metrics):
            system.runtime_token_speed([("127.0.0.1", 8080)])
            result = system.runtime_token_speed([("127.0.0.1", 8080)])

        self.assertEqual(result["preprocessing"]["current_tps"], 100)
        self.assertEqual(result["decode"]["current_tps"], 50)


if __name__ == "__main__":
    unittest.main()
