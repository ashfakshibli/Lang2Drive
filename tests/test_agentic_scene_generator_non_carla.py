import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent_backends import AgentBackend
from agentic_scene_generator import AgenticSceneGenerator, parse_args
from carla_wine_bridge import resolve_runtime_mode


class DummyBackend(AgentBackend):
    def enhance_prompt(self, keyword: str, original_prompt: str, max_tokens: int = 1024) -> str:
        del keyword, max_tokens
        return original_prompt

    def generate_code(
        self,
        full_prompt: str,
        base_max_tokens: int,
        retry_max_tokens: int,
        context_label: str,
    ) -> str:
        del full_prompt, base_max_tokens, retry_max_tokens, context_label
        return "def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n"

    def fix_code(
        self,
        fix_prompt: str,
        base_max_tokens: int,
        retry_max_tokens: int,
        context_label: str,
    ) -> str:
        del fix_prompt, base_max_tokens, retry_max_tokens, context_label
        return "def main():\n    pass\n\nif __name__ == '__main__':\n    main()\n"


class FakeResult:
    def __init__(self, returncode=0, stdout="ok", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


class TestAgenticSceneGeneratorNonCarla(unittest.TestCase):
    def test_parse_args_includes_handoff_options(self):
        with patch(
            "sys.argv",
            [
                "agentic_scene_generator.py",
                "--mode",
                "generate-only",
                "--scene-keyword",
                "red_light_violation",
                "--prepare-handoff",
                "--handoff-dir",
                "/tmp/handoffs",
                "--handoff-duration",
                "30",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.mode, "generate-only")
        self.assertEqual(args.scene_keyword, "red_light_violation")
        self.assertTrue(args.prepare_handoff)
        self.assertEqual(args.handoff_dir, "/tmp/handoffs")
        self.assertEqual(args.handoff_duration, 30)

    def test_parse_args_accepts_scene_serial(self):
        with patch(
            "sys.argv",
            [
                "agentic_scene_generator.py",
                "--mode",
                "generate-only",
                "--scene-serial",
                "7",
                "--force-regenerate",
                "--prepare-handoff",
            ],
        ):
            args = parse_args()

        self.assertEqual(args.scene_serial, 7)
        self.assertTrue(args.force_regenerate)
        self.assertTrue(args.prepare_handoff)

    def test_runtime_mode_auto_resolution(self):
        self.assertEqual(resolve_runtime_mode("auto", platform_name="darwin"), "wine-bridge")
        self.assertEqual(resolve_runtime_mode("auto", platform_name="linux"), "local")

    def test_init_keeps_expected_output_and_report_paths(self):
        generator = AgenticSceneGenerator(
            test_mode=True,
            generate_only=True,
            scenario_limit=1,
            execution_mode="interactive",
            max_attempts=2,
            runtime="local",
            backend=DummyBackend(),
        )

        self.assertEqual(generator.prompts_dir.name, "generated_prompts")
        self.assertEqual(generator.code_dir.name, "generated_code")
        self.assertEqual(generator.scenes_dir.name, "scenes")
        self.assertTrue(generator.intervention_report_file.name.startswith("intervention_report_"))
        self.assertEqual(generator.intervention_summary_csv.name, "intervention_summary.csv")
        self.assertEqual(generator.scenario_intervention_csv.name, "scenario_intervention_data.csv")

    def test_run_simulation_detects_output_dir_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code_file = tmp_path / "scenario.py"
            output_dir = tmp_path / "out"
            code_file.write_text("parser.add_argument('--output-dir')", encoding="utf-8")

            generator = AgenticSceneGenerator.__new__(AgenticSceneGenerator)
            generator.runtime_mode = "local"
            generator.base_dir = tmp_path

            with patch("agentic_scene_generator.run_python_script", return_value=FakeResult()) as mock_run:
                success, error = AgenticSceneGenerator.run_simulation(
                    generator,
                    code_file=code_file,
                    output_dir=output_dir,
                    duration_seconds=10,
                )

            self.assertTrue(success)
            self.assertIsNone(error)
            called_args = mock_run.call_args.kwargs["script_args"]
            self.assertIn("--output-dir", called_args)

    def test_run_simulation_detects_output_flag_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            code_file = tmp_path / "scenario.py"
            output_dir = tmp_path / "out"
            code_file.write_text("print('no output-dir flag in script')", encoding="utf-8")

            generator = AgenticSceneGenerator.__new__(AgenticSceneGenerator)
            generator.runtime_mode = "local"
            generator.base_dir = tmp_path

            with patch("agentic_scene_generator.run_python_script", return_value=FakeResult()) as mock_run:
                success, error = AgenticSceneGenerator.run_simulation(
                    generator,
                    code_file=code_file,
                    output_dir=output_dir,
                    duration_seconds=10,
                )

            self.assertTrue(success)
            self.assertIsNone(error)
            called_args = mock_run.call_args.kwargs["script_args"]
            self.assertIn("--output", called_args)


if __name__ == "__main__":
    unittest.main()
