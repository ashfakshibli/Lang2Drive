import unittest
from pathlib import Path
from unittest.mock import patch

from agent_backends import AgentBackendError, CodexCliBackend, extract_code_from_response


class TestAgentBackends(unittest.TestCase):
    def test_codex_command_composition(self):
        backend = CodexCliBackend(workspace_dir=Path("/tmp/project"), model="o3")
        command = backend._build_codex_command(Path("/tmp/codex_output.txt"))

        self.assertEqual(command[0], "codex")
        self.assertEqual(command[1], "exec")
        self.assertIn("--skip-git-repo-check", command)
        self.assertIn("--sandbox", command)
        self.assertIn("workspace-write", command)
        self.assertIn("--model", command)
        self.assertIn("o3", command)
        self.assertEqual(command[-1], "-")

    def test_extract_code_from_response_prefers_code_fence(self):
        response = """Text before\n```python\ndef main():\n    pass\n```\nText after"""
        code = extract_code_from_response(response)
        self.assertIn("def main", code)
        self.assertNotIn("Text before", code)

    def test_generate_code_retries_when_first_output_looks_incomplete(self):
        backend = CodexCliBackend(workspace_dir=Path("."))
        responses = [
            "```python\nprint('hi')\n```",
            "```python\ndef main():\n    pass\n\nif __name__ == '__main__':\n    main()\n```",
        ]

        def fake_invoke(*_args, **_kwargs):
            return responses.pop(0)

        backend._invoke_prompt_with_retries = fake_invoke  # type: ignore[method-assign]

        generated = backend.generate_code(
            full_prompt="Generate code",
            base_max_tokens=200,
            retry_max_tokens=400,
            context_label="unit-test",
        )

        self.assertIn("def main", generated)

    def test_retry_failure_has_clear_error_message(self):
        backend = CodexCliBackend(workspace_dir=Path("."), max_retries=2)

        def always_fail(*_args, **_kwargs):
            raise AgentBackendError("simulated failure")

        backend._run_codex_once = always_fail  # type: ignore[method-assign]

        with patch("agent_backends.time.sleep", return_value=None):
            with self.assertRaises(AgentBackendError) as ctx:
                backend._invoke_prompt_with_retries(
                    prompt="Hello",
                    purpose="test purpose",
                    max_tokens=256,
                )

        self.assertIn("Failed to complete test purpose", str(ctx.exception))
        self.assertIn("simulated failure", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
