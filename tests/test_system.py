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
