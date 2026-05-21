import unittest

from backend.app.gpu import fallback_vram_mib, parse_nvidia_smi_csv


class GpuTests(unittest.TestCase):
    def test_parse_nvidia_smi_csv(self):
        text = """index, name, uuid, memory.total [MiB], memory.used [MiB], memory.free [MiB], power.draw [W], power.limit [W], utilization.gpu [%]
0, NVIDIA GeForce RTX 5090, GPU-1, 32607 MiB, 1024 MiB, 31583 MiB, 44.12 W, 575.00 W, 7 %
"""
        [gpu] = parse_nvidia_smi_csv(text)
        self.assertEqual(gpu.index, 0)
        self.assertEqual(gpu.memory_total_mib, 32607)
        self.assertEqual(gpu.memory_free_mib, 31583)
        self.assertEqual(gpu.power_draw_w, 44.12)
        self.assertTrue(gpu.supported)

    def test_vram_fallbacks_include_requested_cards(self):
        self.assertEqual(fallback_vram_mib("NVIDIA GeForce RTX 5090"), 32768)
        self.assertEqual(fallback_vram_mib("NVIDIA GeForce RTX 4070 Laptop GPU"), 8192)
        self.assertEqual(fallback_vram_mib("NVIDIA GeForce RTX 5070 Laptop GPU"), 12288)


if __name__ == "__main__":
    unittest.main()

