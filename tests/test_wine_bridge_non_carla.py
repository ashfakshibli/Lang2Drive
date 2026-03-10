import tempfile
import unittest
from pathlib import Path

from carla_wine_bridge import (
    WineRuntimeConfig,
    WineRuntimeError,
    build_wine_python_command,
    convert_script_args_for_wine,
    posix_to_windows_path,
    validate_wine_runtime,
)


class TestWineBridge(unittest.TestCase):
    def test_posix_to_windows_path_under_drive_c(self):
        with tempfile.TemporaryDirectory() as tmp:
            wineprefix = Path(tmp) / "prefix"
            script_path = (
                wineprefix
                / "drive_c"
                / "Program Files"
                / "WindowsNoEditor"
                / "VLM-AV"
                / "generated_code"
                / "demo.py"
            )
            script_path.parent.mkdir(parents=True, exist_ok=True)
            script_path.write_text("print('ok')", encoding="utf-8")

            converted = posix_to_windows_path(script_path, wineprefix)
            self.assertEqual(
                converted,
                r"C:\Program Files\WindowsNoEditor\VLM-AV\generated_code\demo.py",
            )

    def test_convert_script_args_for_wine_converts_output_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "CARLA.app"
            wineprefix = app / "Contents" / "SharedSupport" / "prefix"
            output_dir = wineprefix / "drive_c" / "Program Files" / "WindowsNoEditor" / "VLM-AV" / "scenes"
            output_dir.mkdir(parents=True, exist_ok=True)

            config = WineRuntimeConfig(
                carla_app_path=app,
                wineprefix=wineprefix,
                wine_binary=app / "Contents" / "SharedSupport" / "wine" / "bin" / "wine",
                python_exe_windows=r"C:\Program Files\Python310\python.exe",
            )

            args = ["--duration", "25", "--output-dir", str(output_dir)]
            converted = convert_script_args_for_wine(args, config)

            self.assertEqual(converted[0], "--duration")
            self.assertEqual(converted[2], "--output-dir")
            self.assertTrue(converted[3].startswith("C:\\"))
            self.assertIn(r"\scenes", converted[3])

    def test_build_wine_python_command(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "CARLA.app"
            wineprefix = app / "Contents" / "SharedSupport" / "prefix"
            script = wineprefix / "drive_c" / "Program Files" / "WindowsNoEditor" / "VLM-AV" / "scenario.py"
            script.parent.mkdir(parents=True, exist_ok=True)
            script.write_text("print('ok')", encoding="utf-8")

            config = WineRuntimeConfig(
                carla_app_path=app,
                wineprefix=wineprefix,
                wine_binary=app / "Contents" / "SharedSupport" / "wine" / "bin" / "wine",
                python_exe_windows=r"C:\Program Files\Python310\python.exe",
            )

            command = build_wine_python_command(
                script_path=script,
                script_args=["--duration", "10", "--output-dir", str(script.parent / "out")],
                config=config,
            )

            self.assertEqual(command[0], str(config.wine_binary))
            self.assertEqual(command[1], config.python_exe_windows)
            self.assertEqual(command[2], r"C:\Program Files\WindowsNoEditor\VLM-AV\scenario.py")
            self.assertIn("--output-dir", command)

    def test_validate_wine_runtime_fails_when_runtime_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            app = Path(tmp) / "MissingCARLA.app"
            wineprefix = app / "Contents" / "SharedSupport" / "prefix"
            config = WineRuntimeConfig(
                carla_app_path=app,
                wineprefix=wineprefix,
                wine_binary=app / "Contents" / "SharedSupport" / "wine" / "bin" / "wine",
                python_exe_windows=r"C:\Program Files\Python310\python.exe",
            )

            with self.assertRaises(WineRuntimeError):
                validate_wine_runtime(config)


if __name__ == "__main__":
    unittest.main()
