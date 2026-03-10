#!/usr/bin/env python3
"""Agent backend abstractions and Codex CLI implementation."""

from __future__ import annotations

import re
import subprocess
import tempfile
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional


DEFAULT_CODEX_TIMEOUT_SECONDS = 420
DEFAULT_MAX_RETRIES = 3


class AgentBackendError(RuntimeError):
    """Raised when an agent backend call fails."""


def extract_code_from_response(text: str) -> str:
    """Extract Python source code from an agent response."""
    code_block_pattern = re.compile(r"```(?:python)?\s*\n?(.*?)\n?```", re.DOTALL)
    match = code_block_pattern.search(text)
    if match:
        return match.group(1).strip()

    stripped = text.strip()
    if stripped.endswith("```"):
        stripped = stripped[:-3].strip()
    if stripped.startswith("```python"):
        stripped = stripped[len("```python") :].strip()
    elif stripped.startswith("```"):
        stripped = stripped[3:].strip()
    return stripped


class AgentBackend(ABC):
    """Provider-agnostic interface for scene prompt/code generation."""

    @abstractmethod
    def enhance_prompt(self, keyword: str, original_prompt: str, max_tokens: int = 1024) -> str:
        """Enhance a scene prompt for CARLA code generation."""

    @abstractmethod
    def generate_code(
        self,
        full_prompt: str,
        base_max_tokens: int,
        retry_max_tokens: int,
        context_label: str,
    ) -> str:
        """Generate runnable CARLA Python code."""

    @abstractmethod
    def fix_code(
        self,
        fix_prompt: str,
        base_max_tokens: int,
        retry_max_tokens: int,
        context_label: str,
    ) -> str:
        """Generate fixed CARLA Python code."""


class CodexCliBackend(AgentBackend):
    """Agent backend that runs Codex via `codex exec`."""

    def __init__(
        self,
        workspace_dir: Path,
        codex_bin: str = "codex",
        model: Optional[str] = None,
        timeout_seconds: int = DEFAULT_CODEX_TIMEOUT_SECONDS,
        max_retries: int = DEFAULT_MAX_RETRIES,
    ) -> None:
        self.workspace_dir = Path(workspace_dir)
        self.codex_bin = codex_bin
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def _build_codex_command(self, output_file: Path) -> list[str]:
        """Build `codex exec` command for non-interactive prompt execution."""
        cmd = [
            self.codex_bin,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "workspace-write",
            "-o",
            str(output_file),
        ]
        if self.model:
            cmd.extend(["--model", self.model])

        # Use stdin for large prompts and stable escaping.
        cmd.append("-")
        return cmd

    def _run_codex_once(self, prompt: str, timeout_seconds: int) -> str:
        """Run a single Codex CLI prompt and return final assistant text."""
        with tempfile.NamedTemporaryFile("w+", suffix=".txt", delete=False) as handle:
            output_file = Path(handle.name)

        cmd = self._build_codex_command(output_file)
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                cwd=self.workspace_dir,
                capture_output=True,
                text=True,
                timeout=timeout_seconds,
            )
        except FileNotFoundError as err:
            raise AgentBackendError(
                "Codex CLI not found. Install Codex CLI and verify `codex` is on PATH."
            ) from err
        except subprocess.TimeoutExpired as err:
            raise AgentBackendError(
                f"Codex CLI timed out after {timeout_seconds} seconds"
            ) from err

        try:
            final_text = output_file.read_text(encoding="utf-8", errors="ignore").strip()
        finally:
            output_file.unlink(missing_ok=True)

        if result.returncode != 0:
            stderr_excerpt = (result.stderr or "").strip()[-2000:]
            stdout_excerpt = (result.stdout or "").strip()[-1000:]
            details = stderr_excerpt or stdout_excerpt or "Unknown Codex failure"
            raise AgentBackendError(f"Codex CLI failed (exit {result.returncode}): {details}")

        if final_text:
            return final_text

        stdout_fallback = (result.stdout or "").strip()
        if stdout_fallback:
            return stdout_fallback

        raise AgentBackendError("Codex CLI returned no output text")

    def _invoke_prompt_with_retries(
        self,
        prompt: str,
        purpose: str,
        max_tokens: int,
        max_retries: Optional[int] = None,
        timeout_seconds: Optional[int] = None,
    ) -> str:
        """Run Codex prompt with retry policy."""
        del max_tokens  # Codex CLI does not expose direct max_tokens control.

        attempts = max_retries if max_retries is not None else self.max_retries
        timeout = timeout_seconds if timeout_seconds is not None else self.timeout_seconds

        last_error: Optional[Exception] = None
        for attempt in range(1, attempts + 1):
            try:
                return self._run_codex_once(prompt=prompt, timeout_seconds=timeout)
            except Exception as err:  # pragma: no cover - branch exercised in tests via mocks
                last_error = err
                if attempt < attempts:
                    time.sleep(attempt * 2)

        message = f"Failed to complete {purpose} after {attempts} attempts"
        if last_error:
            raise AgentBackendError(f"{message}: {last_error}") from last_error
        raise AgentBackendError(message)

    @staticmethod
    def _code_looks_complete(code: str) -> bool:
        """Heuristic check for complete, runnable CARLA script."""
        normalized = (code or "").strip()
        if not normalized:
            return False
        if "def main" not in normalized:
            return False
        if "__name__" not in normalized or "main()" not in normalized:
            return False
        return True

    def _request_code(
        self,
        prompt: str,
        base_max_tokens: int,
        retry_max_tokens: int,
        context_label: str,
    ) -> str:
        """Request code and retry once with stricter instruction if output looks truncated."""
        response = self._invoke_prompt_with_retries(
            prompt=prompt,
            purpose=context_label,
            max_tokens=base_max_tokens,
        )
        code = extract_code_from_response(response)
        if self._code_looks_complete(code):
            return code

        if retry_max_tokens <= base_max_tokens:
            return code

        retry_prompt = (
            f"{prompt}\n\n"
            "Your previous answer looked incomplete. Return complete runnable Python code only, "
            "including `def main()` and `if __name__ == '__main__': main()`."
        )
        response_retry = self._invoke_prompt_with_retries(
            prompt=retry_prompt,
            purpose=f"{context_label} retry",
            max_tokens=retry_max_tokens,
            max_retries=1,
        )
        code_retry = extract_code_from_response(response_retry)
        return code_retry or code

    def enhance_prompt(self, keyword: str, original_prompt: str, max_tokens: int = 1024) -> str:
        """Generate a stronger scene prompt from a short scenario description."""
        prompt = (
            "Enhance this CARLA simulation prompt for reliable Python code generation.\n\n"
            f"Original Keyword: {keyword}\n"
            f"Original Prompt: {original_prompt}\n\n"
            "Requirements:\n"
            "1. Keep the exact intent of the original prompt.\n"
            "2. Add concrete technical details about actor placement, timing, behavior, and camera-visible outcomes.\n"
            "3. Include universal baseline constraints: busy intersection context, dense multi-direction traffic, "
            "and many realistic roadside/construction objects visible concurrently with the key event.\n"
            "4. Default to FRONT camera only unless the prompt explicitly requests multi-angle capture.\n"
            "5. Keep the ego lane clear by default unless the scenario explicitly requires a lead vehicle or lane obstacle.\n"
            "6. Require dense traffic beside ego, behind ego, and in crossing/oncoming lanes when map topology allows.\n"
            "7. Include varied pedestrians/human actors in scene-appropriate locations when the scenario supports them.\n"
            "8. Prefer realistic static meshes via static.prop.mesh when dedicated blueprints are missing.\n"
            "9. Assume the first acceptance candidate is a single clear, sunny daytime baseline run.\n"
            "10. Return plain text only (no markdown or code fences)."
        )
        response = self._invoke_prompt_with_retries(
            prompt=prompt,
            purpose=f"prompt enhancement for {keyword}",
            max_tokens=max_tokens,
        )
        return response.strip()

    def generate_code(
        self,
        full_prompt: str,
        base_max_tokens: int,
        retry_max_tokens: int,
        context_label: str,
    ) -> str:
        """Generate CARLA code from a full prompt template."""
        return self._request_code(
            prompt=full_prompt,
            base_max_tokens=base_max_tokens,
            retry_max_tokens=retry_max_tokens,
            context_label=f"code generation ({context_label})",
        )

    def fix_code(
        self,
        fix_prompt: str,
        base_max_tokens: int,
        retry_max_tokens: int,
        context_label: str,
    ) -> str:
        """Generate CARLA fix code from fix prompt context."""
        return self._request_code(
            prompt=fix_prompt,
            base_max_tokens=base_max_tokens,
            retry_max_tokens=retry_max_tokens,
            context_label=f"code fix ({context_label})",
        )


__all__ = [
    "AgentBackend",
    "AgentBackendError",
    "CodexCliBackend",
    "extract_code_from_response",
]
