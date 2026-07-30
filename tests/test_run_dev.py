import unittest
from unittest.mock import patch

import run_dev


class RunDevTests(unittest.TestCase):
    def test_select_port_uses_configured_port_when_available(self):
        with patch("run_dev.port_available", return_value=True):
            self.assertEqual(run_dev.select_port("127.0.0.1", 8174), 8174)

    def test_select_port_refuses_to_start_a_second_backend(self):
        with patch("run_dev.port_available", return_value=False):
            with self.assertRaisesRegex(SystemExit, "already in use"):
                run_dev.select_port("127.0.0.1", 8174)


if __name__ == "__main__":
    unittest.main()
