import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agentic_wine_handoff_runner import (
    build_script_args,
    detect_output_flag,
    load_manifest,
    resolve_manifest_path,
    run_manifest,
)


class FakeCompletedProcess:
    def __init__(self, returncode: int = 0, stdout: str = "", stderr: str = ""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestWineHandoffRunnerNonCarla(unittest.TestCase):
    def test_resolve_manifest_path_uses_latest_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff_dir = Path(tmp)
            manifest_path = handoff_dir / "run" / "scene" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text("{}", encoding="utf-8")

            pointer = handoff_dir / "LATEST_MANIFEST_POSIX.txt"
            pointer.write_text(str(manifest_path), encoding="utf-8")

            resolved = resolve_manifest_path(None, handoff_dir)
            self.assertEqual(resolved, manifest_path.resolve())

    def test_resolve_manifest_path_fails_when_pointer_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff_dir = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                resolve_manifest_path(None, handoff_dir)

    def test_output_flag_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code_file = tmp_path / "scene.py"
            code_file.write_text("parser.add_argument('--output-dir')", encoding="utf-8")
            self.assertEqual(detect_output_flag(code_file), "--output-dir")

            code_file.write_text("print('fallback output')", encoding="utf-8")
            self.assertEqual(detect_output_flag(code_file), "--output")

    def test_build_script_args_uses_detected_output_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code_file = tmp_path / "scene.py"
            output_dir = tmp_path / "out"
            code_file.write_text("parser.add_argument('--output-dir')", encoding="utf-8")
            args = build_script_args(code_file, output_dir, duration_seconds=20)
            self.assertEqual(args[0], "--duration")
            self.assertEqual(args[2], "--output-dir")

    def test_run_manifest_success_and_failure_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code_file = tmp_path / "scene.py"
            output_dir = tmp_path / "out"
            output_dir.mkdir(parents=True, exist_ok=True)
            code_file.write_text("parser.add_argument('--output-dir')", encoding="utf-8")

            manifest = {
                "code_file_posix": str(code_file),
                "code_file_windows": str(code_file),
                "output_dir_posix": str(output_dir),
                "output_dir_windows": str(output_dir),
                "duration_seconds": 10,
            }

            with patch(
                "agentic_wine_handoff_runner.run_python_script",
                return_value=FakeCompletedProcess(returncode=0, stdout="ok"),
            ):
                result_ok = run_manifest(
                    manifest,
                    timeout=None,
                    dry_run=False,
                    cwd=tmp_path,
                    min_success_ratio=0.5,
                )
            self.assertTrue(result_ok["success"])
            self.assertEqual(result_ok["returncode"], 0)
            self.assertIn("--output-dir", result_ok["script_args"])
            self.assertTrue(result_ok["process_success"])

            with patch(
                "agentic_wine_handoff_runner.run_python_script",
                return_value=FakeCompletedProcess(returncode=1, stderr="boom"),
            ):
                result_fail = run_manifest(
                    manifest,
                    timeout=None,
                    dry_run=False,
                    cwd=tmp_path,
                    min_success_ratio=0.5,
                )
            self.assertFalse(result_fail["success"])
            self.assertEqual(result_fail["returncode"], 1)
            self.assertFalse(result_fail["process_success"])

    def test_run_manifest_accepts_frames_when_process_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code_file = tmp_path / "scene.py"
            output_dir = tmp_path / "out"
            for view in ["front", "front_left", "front_right", "rear"]:
                view_dir = output_dir / view
                view_dir.mkdir(parents=True, exist_ok=True)
                for idx in range(100):
                    (view_dir / f"{view}_frame_{idx:08d}.png").write_text("x", encoding="utf-8")
            code_file.write_text("parser.add_argument('--output-dir')", encoding="utf-8")

            manifest = {
                "code_file_posix": str(code_file),
                "code_file_windows": str(code_file),
                "output_dir_posix": str(output_dir),
                "output_dir_windows": str(output_dir),
                "duration_seconds": 10,
            }

            with patch(
                "agentic_wine_handoff_runner.run_python_script",
                return_value=FakeCompletedProcess(returncode=1, stderr="runtime issue"),
            ):
                result = run_manifest(
                    manifest,
                    timeout=None,
                    dry_run=False,
                    cwd=tmp_path,
                    min_success_ratio=0.5,
                )

            self.assertFalse(result["process_success"])
            self.assertTrue(result["frames_ok"])
            self.assertTrue(result["success"])
            self.assertIn("note", result)

    def test_writes_simulation_result_json_contract(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff_dir = Path(tmp)
            manifest_path = handoff_dir / "run" / "demo" / "manifest.json"
            code_file = handoff_dir / "run" / "demo" / "scene.py"
            output_dir = handoff_dir / "run" / "demo" / "frames"
            output_dir.mkdir(parents=True, exist_ok=True)
            code_file.write_text("print('no output-dir')", encoding="utf-8")
            manifest_payload = {
                "code_file_posix": str(code_file),
                "code_file_windows": str(code_file),
                "output_dir_posix": str(output_dir),
                "output_dir_windows": str(output_dir),
                "duration_seconds": 5,
            }
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")

            manifest = load_manifest(manifest_path)
            with patch(
                "agentic_wine_handoff_runner.run_python_script",
                return_value=FakeCompletedProcess(returncode=1, stderr="bad"),
            ):
                result = run_manifest(
                    manifest,
                    timeout=20,
                    dry_run=False,
                    cwd=handoff_dir,
                    min_success_ratio=0.5,
                )
            result["manifest_path"] = str(manifest_path)

            result_path = manifest_path.parent / "simulation_result.json"
            result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

            written = json.loads(result_path.read_text(encoding="utf-8"))
            self.assertIn("success", written)
            self.assertIn("script_args", written)
            self.assertFalse(written["success"])


if __name__ == "__main__":
    unittest.main()
