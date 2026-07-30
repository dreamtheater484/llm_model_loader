import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_dev


class FrontendBuildTests(unittest.TestCase):
    def make_frontend(self, directory):
        frontend = Path(directory) / "frontend"
        (frontend / "src").mkdir(parents=True)
        (frontend / "dist").mkdir()
        (frontend / "node_modules").mkdir()
        (frontend / "src" / "main.jsx").write_text("const version = 1;\n", encoding="utf-8")
        (frontend / "package.json").write_text('{"scripts":{"build":"vite build"}}\n', encoding="utf-8")
        (frontend / "dist" / "index.html").write_text("<html>old build</html>\n", encoding="utf-8")
        return frontend

    def test_source_fingerprint_ignores_generated_directories(self):
        with tempfile.TemporaryDirectory() as directory:
            frontend = self.make_frontend(directory)
            original = run_dev.frontend_source_fingerprint(frontend)

            (frontend / "dist" / "index.html").write_text("<html>different build</html>\n", encoding="utf-8")
            (frontend / "node_modules" / "dependency.js").write_text("generated\n", encoding="utf-8")

            self.assertEqual(run_dev.frontend_source_fingerprint(frontend), original)

    def test_current_frontend_build_is_not_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            frontend = self.make_frontend(directory)
            fingerprint = run_dev.frontend_source_fingerprint(frontend)
            marker = frontend / "dist" / ".source-fingerprint"
            marker.write_text(fingerprint + "\n", encoding="utf-8")

            with patch.multiple(
                run_dev,
                FRONTEND=frontend,
                FRONTEND_DIST_INDEX=frontend / "dist" / "index.html",
                FRONTEND_BUILD_FINGERPRINT=marker,
            ):
                with patch("run_dev.subprocess.run") as run:
                    run_dev.ensure_frontend_build()

            run.assert_not_called()

    def test_stale_frontend_build_is_rebuilt_and_marked_current(self):
        with tempfile.TemporaryDirectory() as directory:
            frontend = self.make_frontend(directory)
            marker = frontend / "dist" / ".source-fingerprint"
            marker.write_text("old-source\n", encoding="utf-8")

            with patch.multiple(
                run_dev,
                FRONTEND=frontend,
                FRONTEND_DIST_INDEX=frontend / "dist" / "index.html",
                FRONTEND_BUILD_FINGERPRINT=marker,
            ):
                with patch("run_dev.shutil.which", return_value="npm.cmd"):
                    with patch("run_dev.subprocess.run") as run:
                        run_dev.ensure_frontend_build()

            run.assert_called_once_with(["npm.cmd", "run", "build"], cwd=frontend, check=True)
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), run_dev.frontend_source_fingerprint(frontend))

    def test_legacy_build_without_fingerprint_is_rebuilt(self):
        with tempfile.TemporaryDirectory() as directory:
            frontend = self.make_frontend(directory)
            marker = frontend / "dist" / ".source-fingerprint"

            with patch.multiple(
                run_dev,
                FRONTEND=frontend,
                FRONTEND_DIST_INDEX=frontend / "dist" / "index.html",
                FRONTEND_BUILD_FINGERPRINT=marker,
            ):
                with patch("run_dev.shutil.which", return_value="npm.cmd"):
                    with patch("run_dev.subprocess.run") as run:
                        run_dev.ensure_frontend_build()

            run.assert_called_once_with(["npm.cmd", "run", "build"], cwd=frontend, check=True)
            self.assertTrue(marker.exists())

    def test_source_change_makes_existing_build_stale(self):
        with tempfile.TemporaryDirectory() as directory:
            frontend = self.make_frontend(directory)
            marker = frontend / "dist" / ".source-fingerprint"
            marker.write_text(run_dev.frontend_source_fingerprint(frontend), encoding="utf-8")
            (frontend / "src" / "main.jsx").write_text("const version = 2;\n", encoding="utf-8")

            with patch.multiple(
                run_dev,
                FRONTEND=frontend,
                FRONTEND_DIST_INDEX=frontend / "dist" / "index.html",
                FRONTEND_BUILD_FINGERPRINT=marker,
            ):
                with patch("run_dev.shutil.which", return_value="npm.cmd"):
                    with patch("run_dev.subprocess.run") as run:
                        run_dev.ensure_frontend_build()

            run.assert_called_once()
            self.assertEqual(marker.read_text(encoding="utf-8").strip(), run_dev.frontend_source_fingerprint(frontend))


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
