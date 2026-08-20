import unittest

from backend.app.scripts import detect_quantization, parse_script
from backend.app.runs import _with_ninfer_defaults

NINFER_SCRIPT = """& wsl.exe -d "Ubuntu-24.04" -- bash -lc 'NINFER_PORT=8081 NINFER_CONCURRENCY=3 NINFER_MAX_CONTEXT=262144 NINFER_MIN_CONTEXT=163840 NINFER_MODEL_FILE=qwen3_8_27b_nvfp4.ninfer ~/ninfer-qwen38/run-qwen38-nvfp4.sh --model-id qwen3.8-27b'"""


class NInferScriptTests(unittest.TestCase):
    def test_parse_script_recognizes_ninfer_runtime(self):
        info = parse_script(NINFER_SCRIPT)
        self.assertEqual(info.runtime, "ninfer")
        self.assertEqual(info.executable, "wsl.exe")
        self.assertEqual(info.wsl_distro, "Ubuntu-24.04")
        self.assertEqual(info.wsl_launcher, "~/ninfer-qwen38/run-qwen38-nvfp4.sh")
        self.assertEqual(info.host, "127.0.0.1")
        self.assertEqual(info.port, 8081)
        self.assertEqual(info.ctx_size, 163840)
        self.assertEqual(info.concurrency, 3)
        self.assertEqual(info.model_ref, "qwen3.8-27b")
        self.assertEqual(info.alias, None)
        self.assertEqual(info.quantization, "NVFP4")
        self.assertFalse(info.flash_attention)
        self.assertTrue(info.mtp)

    def test_parse_script_defaults_ninfer_host_and_port(self):
        info = parse_script('& wsl.exe -d "Ubuntu" -- bash -lc "NINFER_MODEL_FILE=x.ninfer ~/ninfer-qwen38/run-qwen38-nvfp4.sh"')
        self.assertEqual(info.runtime, "ninfer")
        self.assertEqual(info.wsl_distro, "Ubuntu")
        self.assertEqual(info.host, "127.0.0.1")
        self.assertEqual(info.port, 8081)

    def test_parse_script_ninfer_explicit_host(self):
        info = parse_script('& wsl.exe -d "Ubuntu" -- bash -lc "NINFER_HOST=0.0.0.0 NINFER_PORT=9090 ~/ninfer-qwen38/run-qwen38-nvfp4.sh"')
        self.assertEqual(info.host, "0.0.0.0")
        self.assertEqual(info.port, 9090)

    def test_ninfer_quantization_detected_from_model_file(self):
        self.assertEqual(detect_quantization(None, None, "qwen3_8_27b_nvfp4.ninfer", NINFER_SCRIPT), "NVFP4")

    def test_ninfer_context_uses_min_when_both_are_set(self):
        info = parse_script(
            '& wsl.exe -d "Ubuntu" -- bash -lc "NINFER_MAX_CONTEXT=131072 NINFER_MIN_CONTEXT=163840 ~/ninfer-qwen38/run-qwen38-nvfp4.sh"'
        )
        self.assertEqual(info.ctx_size, 163840)

    def test_ninfer_context_uses_max_when_min_is_absent(self):
        info = parse_script(
            '& wsl.exe -d "Ubuntu" -- bash -lc "NINFER_MAX_CONTEXT=131072 ~/ninfer-qwen38/run-qwen38-nvfp4.sh"'
        )
        self.assertEqual(info.ctx_size, 131072)

    def test_with_ninfer_defaults_injects_lan_host(self):
        args = ["-lc", "NINFER_PORT=8081 ~/ninfer-qwen38/run-qwen38-nvfp4.sh"]
        result = _with_ninfer_defaults(args)
        self.assertEqual(result[1], "NINFER_HOST=0.0.0.0 NINFER_PORT=8081 ~/ninfer-qwen38/run-qwen38-nvfp4.sh")

    def test_with_ninfer_defaults_preserves_explicit_host(self):
        args = ["-lc", "NINFER_HOST=127.0.0.1 ~/ninfer-qwen38/run-qwen38-nvfp4.sh"]
        self.assertEqual(_with_ninfer_defaults(args), args)

    def test_with_ninfer_defaults_ignores_non_payload_scripts(self):
        args = ["-m", "model.gguf", "--host", "0.0.0.0"]
        self.assertEqual(_with_ninfer_defaults(args), args)


if __name__ == "__main__":
    unittest.main()