import unittest

from backend.app.scripts import autosuggest_name, can_fit_vram, detect_quantization, estimate_vram_mib, parse_script


SCRIPT = """& "$env:USERPROFILE\\AI\\qwen36-35b-llamacpp\\llama.cpp\\llama-server.exe" `
  -hf "unsloth/Qwen3.6-27B-MTP-GGUF:UD-Q5_K_XL" `
  --alias "Qwen3.6-27B-MTP-UD-Q5_K_XL" `
  --host 127.0.0.1 `
  --port 8080 `
  -c 131072 `
  -n 32768 `
  -ngl 99 `
  -fa on `
  -ctk q8_0 `
  -ctv q8_0 `
  --spec-type draft-mtp `
  --log-verbosity 1"""


class ScriptTests(unittest.TestCase):
    def test_parse_script_extracts_key_fields(self):
        info = parse_script(SCRIPT)
        self.assertEqual(info.host, "127.0.0.1")
        self.assertEqual(info.port, 8080)
        self.assertEqual(info.ctx_size, 131072)
        self.assertEqual(info.quantization, "UD-Q5_K_XL")
        self.assertTrue(info.flash_attention)
        self.assertTrue(info.mtp)

    def test_parse_script_strips_quoted_executable_from_args(self):
        raw = '& "C:\\Users\\Roy\\AI\\llama.cpp\\llama-server.exe" `\n  -m "C:\\models\\model-Q4_K_M.gguf" `\n  --host 127.0.0.1'
        info = parse_script(raw)
        self.assertEqual(info.executable, "C:\\Users\\Roy\\AI\\llama.cpp\\llama-server.exe")
        self.assertEqual(info.args[0], "-m")
        self.assertNotIn("llama-server.exe", info.args[0])

    def test_autosuggest_name(self):
        name = autosuggest_name("Qwen3.6 27B", SCRIPT)
        self.assertIn("Qwen3.6 27B", name)
        self.assertIn("128kctx", name)
        self.assertIn("UD-Q5_K_XL", name)
        self.assertIn("Flash", name)
        self.assertIn("MTP", name)

    def test_quantization_detection(self):
        self.assertEqual(detect_quantization("model-Q4_K_M.gguf"), "Q4_K_M")
        self.assertEqual(detect_quantization("repo/name:UD-Q5_K_XL"), "UD-Q5_K_XL")

    def test_vram_gate(self):
        estimate = estimate_vram_mib(8 * 1024 * 1024 * 1024, 32768)
        ok, _ = can_fit_vram(16384, estimate)
        self.assertTrue(ok)
        ok, reason = can_fit_vram(4096, estimate)
        self.assertFalse(ok)
        self.assertIn("Needs", reason)


if __name__ == "__main__":
    unittest.main()
