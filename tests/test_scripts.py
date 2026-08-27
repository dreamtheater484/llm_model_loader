import unittest

from backend.app.scripts import autosuggest_name, can_fit_vram, detect_quantization, estimate_vram_mib, is_fit_managed, parse_script


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

    def test_parse_script_does_not_mislabel_dspark_as_mtp(self):
        info = parse_script("--spec-type draft-dspark")
        self.assertFalse(info.mtp)

    def test_parse_script_extracts_moe_cpu_experts(self):
        raw = '-m "C:\\models\\Qwen3.6-35B-A3B-Q4_K_M.gguf" --ctx-size 65536 --n-cpu-moe 34'
        info = parse_script(raw)
        self.assertEqual(info.n_cpu_moe, 34)

    def test_parse_script_strips_single_quoted_json_arg(self):
        raw = """--chat-template-kwargs '{"preserve_thinking":false}'"""
        info = parse_script(raw)
        self.assertEqual(info.args, ["--chat-template-kwargs", '{"preserve_thinking":false}'])

    def test_parse_script_strips_quoted_executable_from_args(self):
        raw = '& "D:\\Documents\\AI\\llama.cpp\\llama-server.exe" `\n  -m "C:\\models\\model-Q4_K_M.gguf" `\n  --host 127.0.0.1'
        info = parse_script(raw)
        self.assertEqual(info.executable, "D:\\Documents\\AI\\llama.cpp\\llama-server.exe")
        self.assertEqual(info.args[0], "-m")
        self.assertNotIn("llama-server.exe", info.args[0])

    def test_parse_script_extracts_fit_and_cache_settings(self):
        raw = "--fit on -ngl auto -ctk q4_0 -ctv q4_0 -np 1"
        info = parse_script(raw)
        self.assertTrue(info.fit)
        self.assertEqual(info.gpu_layers, "auto")
        self.assertEqual(info.cache_type_k, "q4_0")
        self.assertEqual(info.cache_type_v, "q4_0")
        self.assertEqual(info.parallel, 1)
        self.assertTrue(is_fit_managed(info))

    def test_fixed_gpu_layers_are_not_fit_managed(self):
        self.assertFalse(is_fit_managed(parse_script("--fit on -ngl 99")))

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

    def test_moe_cpu_experts_skip_dense_estimate(self):
        estimate = estimate_vram_mib(22 * 1024 * 1024 * 1024, 65536, n_cpu_moe=34)
        self.assertIsNone(estimate)
        ok, reason = can_fit_vram(7948, estimate, allow_unknown=True)
        self.assertTrue(ok)
        self.assertIn("MoE", reason)

    def test_manual_vram_overrides_automatic_estimate(self):
        ok, _ = can_fit_vram(7948, estimated_mib=23611, manual_mib=6500)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
