import json
import tempfile
import unittest
from pathlib import Path

from agentic_visual_evaluator import (
    apply_majority_pass_rule,
    build_key_frames,
    build_codex_command,
    collect_frames,
    parse_json_response,
    resolve_manifest_path,
    sample_evenly,
)


class TestAgenticVisualEvaluatorNonCarla(unittest.TestCase):
    def test_collect_frames_flat_and_recursive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            flat_dir = root / "flat"
            flat_dir.mkdir(parents=True, exist_ok=True)
            (flat_dir / "frame_0001.png").write_text("x", encoding="utf-8")
            (flat_dir / "frame_0002.png").write_text("x", encoding="utf-8")

            flat_frames = collect_frames(flat_dir)
            self.assertEqual(len(flat_frames), 2)

            nested_dir = root / "nested"
            (nested_dir / "front").mkdir(parents=True, exist_ok=True)
            (nested_dir / "front_right").mkdir(parents=True, exist_ok=True)
            (nested_dir / "front" / "front_frame_00000001.png").write_text("x", encoding="utf-8")
            (nested_dir / "front_right" / "front_right_frame_00000001.png").write_text(
                "x",
                encoding="utf-8",
            )

            nested_frames = collect_frames(nested_dir)
            self.assertEqual(len(nested_frames), 2)

    def test_sample_evenly_deterministic(self):
        frames = [Path(f"/tmp/frame_{idx:04d}.png") for idx in range(30)]
        sampled = sample_evenly(frames, 12)
        self.assertEqual(len(sampled), 12)
        expected_indices = [int(i * (len(frames) - 1) / (12 - 1)) for i in range(12)]
        self.assertEqual(sampled, [frames[i] for i in expected_indices])

    def test_build_key_frames_is_deterministic(self):
        frames = [Path(f"/tmp/front_frame_{idx:08d}.png") for idx in range(100)]
        key_frames = build_key_frames(frames, 9)
        self.assertEqual(len(key_frames), 9)
        self.assertEqual(key_frames[0], frames[0])
        self.assertEqual(key_frames[-1], frames[-1])

    def test_build_codex_command_contains_images_and_model(self):
        images = [Path("/tmp/f1.png"), Path("/tmp/f2.png")]
        command = build_codex_command(
            codex_bin="codex",
            output_file=Path("/tmp/out.txt"),
            schema_file=Path("/tmp/schema.json"),
            image_paths=images,
            model="gpt-5.2-codex",
        )

        self.assertEqual(command[0], "codex")
        self.assertIn("--model", command)
        self.assertIn("gpt-5.2-codex", command)
        self.assertEqual(command.count("--image"), 2)
        self.assertEqual(command[-1], "-")

    def test_parse_json_response_accepts_plain_and_fenced_json(self):
        parsed_plain = parse_json_response('{"criteria_results": [], "summary": "ok", "suggested_fix_prompt": ""}')
        self.assertIn("criteria_results", parsed_plain)

        parsed_fenced = parse_json_response(
            "```json\n"
            '{"criteria_results": [], "summary": "ok", "suggested_fix_prompt": ""}\n'
            "```"
        )
        self.assertIn("summary", parsed_fenced)

    def test_majority_pass_rule(self):
        pass_case = apply_majority_pass_rule(
            [{"pass": True}, {"pass": False}, {"pass": True}, {"pass": True}]
        )
        self.assertTrue(pass_case["overall_pass"])
        self.assertEqual(pass_case["criteria_passed"], 3)

        fail_case = apply_majority_pass_rule(
            [{"pass": True}, {"pass": False}, {"pass": False}, {"pass": False}]
        )
        self.assertFalse(fail_case["overall_pass"])
        self.assertEqual(fail_case["criteria_total"], 4)

        tie_case = apply_majority_pass_rule(
            [{"pass": True}, {"pass": True}, {"pass": False}, {"pass": False}]
        )
        self.assertFalse(tie_case["overall_pass"])

    def test_resolve_manifest_path_uses_latest_pointer(self):
        with tempfile.TemporaryDirectory() as tmp:
            handoff_dir = Path(tmp)
            manifest_path = handoff_dir / "run" / "scene" / "manifest.json"
            manifest_path.parent.mkdir(parents=True, exist_ok=True)
            manifest_path.write_text(json.dumps({"ok": True}), encoding="utf-8")

            pointer = handoff_dir / "LATEST_MANIFEST_POSIX.txt"
            pointer.write_text(str(manifest_path), encoding="utf-8")

            resolved = resolve_manifest_path(None, handoff_dir)
            self.assertEqual(resolved, manifest_path.resolve())


if __name__ == "__main__":
    unittest.main()
