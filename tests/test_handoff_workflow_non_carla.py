import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import openpyxl

import agent_skill_scene_loop as loop


def _write_excel(path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Scenes"
    sheet.append(["Keyword", "Prompt"])
    sheet.append(["Red Light Violation", "Vehicle runs a red signal at intersection"])
    sheet.append(["Sudden Heavy Rain", "Rain suddenly reduces visibility"])
    workbook.save(path)


def _write_excel_with_scene_specs(path: Path) -> None:
    workbook = openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Scenes"
    sheet.append(["Keyword", "Prompt", "Scene Specifications"])
    sheet.append(
        [
            "Overhead Sign Collapse",
            "Overhead hazard appears while ego vehicle approaches the structure.",
            (
                "Road Context: urban\n"
                "Camera Contract: Save synchronized front, front_left, front_right, rear, "
                "and drone_follow streams using matched frame ids.\n"
                "Event Contract: Overhead object collapses ahead."
            ),
        ]
    )
    workbook.save(path)


class TestHandoffWorkflowNonCarla(unittest.TestCase):
    def test_prepare_creates_manifest_and_latest_pointers(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            excel = base / "scenes.xlsx"
            _write_excel(excel)
            prompt_template = base / "prompt.txt"
            prompt_template.write_text(
                "Scenario: {SCENE_KEYWORD}\nPrompt: {SCENE_PROMPT}\n",
                encoding="utf-8",
            )

            with patch.multiple(
                loop,
                BASE_DIR=base,
                EXCEL_PATH=excel,
                PROMPT_TEMPLATE_PATH=prompt_template,
                PROMPTS_DIR=base / "generated_prompts",
                CODE_DIR=base / "generated_code",
                SCENES_DIR=base / "scenes",
                HANDOFF_DIR_DEFAULT=base / "handoffs",
            ):
                args = argparse.Namespace(
                    excel=str(excel),
                    handoff_dir=str(base / "handoffs"),
                    scene_serial=1,
                    duration=25,
                    seed_from_latest=False,
                )
                rc = loop.cmd_prepare(args)
                self.assertEqual(rc, 0)

                pointer_posix = base / "handoffs" / "LATEST_MANIFEST_POSIX.txt"
                pointer_windows = base / "handoffs" / "LATEST_MANIFEST_WINDOWS.txt"
                self.assertTrue(pointer_posix.exists())
                self.assertTrue(pointer_windows.exists())

                manifest_path = Path(pointer_posix.read_text(encoding="utf-8").strip())
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

                for field in [
                    "run_id",
                    "scene_keyword",
                    "scenario_keyword",
                    "shot_index",
                    "code_file_posix",
                    "code_file_windows",
                    "output_dir_posix",
                    "output_dir_windows",
                    "duration_seconds",
                    "prompt_file",
                    "generated_at",
                ]:
                    self.assertIn(field, manifest)

                self.assertEqual(manifest["scene_serial"], 1)
                self.assertEqual(manifest["duration_seconds"], 25)
                self.assertTrue(str(manifest["code_file_posix"]).endswith(".py"))
                self.assertTrue(str(manifest["output_dir_posix"]).endswith("shot_0"))

    def test_prepare_without_scene_serial_uses_and_updates_next_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            excel = base / "scenes.xlsx"
            _write_excel(excel)
            prompt_template = base / "prompt.txt"
            prompt_template.write_text("{SCENE_PROMPT}", encoding="utf-8")

            with patch.multiple(
                loop,
                BASE_DIR=base,
                EXCEL_PATH=excel,
                PROMPT_TEMPLATE_PATH=prompt_template,
                PROMPTS_DIR=base / "generated_prompts",
                CODE_DIR=base / "generated_code",
                SCENES_DIR=base / "scenes",
                HANDOFF_DIR_DEFAULT=base / "handoffs",
            ):
                args = argparse.Namespace(
                    excel=str(excel),
                    handoff_dir=str(base / "handoffs"),
                    scene_serial=None,
                    duration=20,
                    seed_from_latest=False,
                )

                rc_first = loop.cmd_prepare(args)
                self.assertEqual(rc_first, 0)
                state_file = base / "handoffs" / "NEXT_SCENE_SERIAL.txt"
                self.assertEqual(state_file.read_text(encoding="utf-8").strip(), "2")

                rc_second = loop.cmd_prepare(args)
                self.assertEqual(rc_second, 0)
                self.assertEqual(state_file.read_text(encoding="utf-8").strip(), "1")

    def test_select_scene_by_serial_guardrail(self):
        scenes = [{"serial": 1, "keyword": "A"}]
        with self.assertRaises(ValueError):
            loop.select_scene_by_serial(scenes, 0)

    def test_prepare_normalizes_legacy_camera_contract_to_front_only_in_manifest_and_prompts(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            excel = base / "scenes.xlsx"
            _write_excel_with_scene_specs(excel)
            prompt_template = base / "prompt.txt"
            prompt_template.write_text(
                "Scenario: {SCENE_KEYWORD}\nSpecs:\n{SCENE_SPECIFICATIONS}\n",
                encoding="utf-8",
            )

            with patch.multiple(
                loop,
                BASE_DIR=base,
                EXCEL_PATH=excel,
                PROMPT_TEMPLATE_PATH=prompt_template,
                PROMPTS_DIR=base / "generated_prompts",
                CODE_DIR=base / "generated_code",
                SCENES_DIR=base / "scenes",
                HANDOFF_DIR_DEFAULT=base / "handoffs",
            ):
                args = argparse.Namespace(
                    excel=str(excel),
                    handoff_dir=str(base / "handoffs"),
                    scene_serial=1,
                    duration=20,
                    seed_from_latest=False,
                    shot_index=0,
                )
                rc = loop.cmd_prepare(args)
                self.assertEqual(rc, 0)

                manifest_path = Path((base / "handoffs" / "LATEST_MANIFEST_POSIX.txt").read_text(encoding="utf-8").strip())
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                self.assertIn("single front camera", manifest["scene_specifications"])
                self.assertNotIn("front_left, front_right, rear, and drone_follow", manifest["scene_specifications"])

                prompt_file = Path(manifest["prompt_file"])
                full_prompt_file = Path(manifest["full_prompt_file"])
                self.assertIn("single front camera", prompt_file.read_text(encoding="utf-8"))
                self.assertIn("single front camera", full_prompt_file.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
