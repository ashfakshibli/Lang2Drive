#!/usr/bin/env python3
"""Runtime adapter for executing CARLA Python scripts locally or via Wine bridge."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


def _default_carla_app_path() -> Path:
    configured = os.getenv("CARLA_APP_PATH", "").strip()
    if configured:
        return Path(configured).expanduser()
    return (Path.home() / "Applications" / "CARLA.app").expanduser()


DEFAULT_CARLA_APP_PATH = _default_carla_app_path()
DEFAULT_WINE_PYTHON_EXE = os.getenv(
    "CARLA_WINE_PYTHON_EXE", r"C:\Program Files\Python310\python.exe"
)
PATH_ARGUMENT_FLAGS = {
    "--output",
    "--output-dir",
    "--log",
    "--log-file",
    "--report",
    "--report-file",
    "--output-subdir",
    "--texture",
}
RUNTIME_CHOICES = {"auto", "wine-bridge", "local"}


@dataclass(frozen=True)
class WineRuntimeConfig:
    """Resolved filesystem paths for Wine Python execution."""

    carla_app_path: Path
    wineprefix: Path
    wine_binary: Path
    python_exe_windows: str


class WineRuntimeError(RuntimeError):
    """Raised when Wine runtime is unavailable or misconfigured."""


def resolve_runtime_mode(mode: str, platform_name: Optional[str] = None) -> str:
    """Resolve runtime mode. `auto` maps to `wine-bridge` on macOS, else `local`."""
    normalized = (mode or "auto").strip().lower()
    if normalized not in RUNTIME_CHOICES:
        raise ValueError(f"Invalid runtime mode: {mode}")

    if normalized != "auto":
        return normalized

    platform_value = platform_name if platform_name is not None else sys.platform
    return "wine-bridge" if platform_value == "darwin" else "local"


def build_wine_runtime_config(carla_app_path: Optional[Path] = None) -> WineRuntimeConfig:
    """Build Wine runtime config from defaults/environment."""
    app_path = Path(carla_app_path) if carla_app_path is not None else DEFAULT_CARLA_APP_PATH
    wineprefix = app_path / "Contents" / "SharedSupport" / "prefix"
    wine_binary = app_path / "Contents" / "SharedSupport" / "wine" / "bin" / "wine"
    return WineRuntimeConfig(
        carla_app_path=app_path,
        wineprefix=wineprefix,
        wine_binary=wine_binary,
        python_exe_windows=DEFAULT_WINE_PYTHON_EXE,
    )


def windows_to_host_path(windows_path: str, wineprefix: Path) -> Path:
    """Convert Windows path (e.g. C:\\foo\\bar) to host path in Wine prefix."""
    match = re.match(r"^([A-Za-z]):\\(.*)$", windows_path)
    if not match:
        raise ValueError(f"Unsupported Windows path format: {windows_path}")

    drive = match.group(1).lower()
    suffix = match.group(2)
    suffix_parts = [part for part in suffix.split("\\") if part]

    if drive == "z":
        return Path("/").joinpath(*suffix_parts)

    drive_root = wineprefix / f"drive_{drive}"
    return drive_root.joinpath(*suffix_parts)


def posix_to_windows_path(path: Path, wineprefix: Path) -> str:
    """Convert host POSIX path to Wine Windows path."""
    abs_path = Path(path).expanduser().resolve()
    drive_c_root = (wineprefix / "drive_c").resolve()

    try:
        rel_path = abs_path.relative_to(drive_c_root)
        return "C:\\" + "\\".join(rel_path.parts)
    except ValueError:
        return "Z:" + str(abs_path).replace("/", "\\")


def _looks_like_path(value: str) -> bool:
    """Heuristic detection for path-like CLI argument values."""
    if not value:
        return False
    if value.startswith("-") and not value.startswith("./"):
        return False
    return (
        "/" in value
        or "\\" in value
        or value.startswith(".")
        or value.startswith("~")
        or value.endswith(".py")
        or value.endswith(".json")
        or value.endswith(".csv")
        or value.endswith(".txt")
        or value.endswith(".log")
    )


def _convert_path_value(value: str, config: WineRuntimeConfig) -> str:
    """Convert path-like value to Windows path when needed."""
    if not _looks_like_path(value):
        return value
    return posix_to_windows_path(Path(value), config.wineprefix)


def convert_script_args_for_wine(
    script_args: Iterable[str],
    config: WineRuntimeConfig,
) -> list[str]:
    """Convert known path-valued script args to Windows paths."""
    args = list(script_args)
    converted: list[str] = []
    idx = 0

    while idx < len(args):
        arg = args[idx]

        matched_equals_flag = next(
            (flag for flag in PATH_ARGUMENT_FLAGS if arg.startswith(f"{flag}=")),
            None,
        )
        if matched_equals_flag is not None:
            _, raw_value = arg.split("=", 1)
            converted.append(f"{matched_equals_flag}={_convert_path_value(raw_value, config)}")
            idx += 1
            continue

        if arg in PATH_ARGUMENT_FLAGS and idx + 1 < len(args):
            converted.append(arg)
            converted.append(_convert_path_value(args[idx + 1], config))
            idx += 2
            continue

        converted.append(arg)
        idx += 1

    return converted


def validate_wine_runtime(config: WineRuntimeConfig) -> None:
    """Fail fast when Wine runtime dependencies are missing."""
    if not config.carla_app_path.exists():
        raise WineRuntimeError(
            f"CARLA app path not found: {config.carla_app_path}. "
            "Set CARLA_APP_PATH to your CARLA.app location."
        )

    if not config.wine_binary.exists():
        raise WineRuntimeError(
            f"Wine binary not found: {config.wine_binary}. "
            "Verify CARLA.app bundle contains Wine runtime."
        )

    wine_root = config.wine_binary.parent.parent
    shared_support_root = config.carla_app_path / "Contents" / "SharedSupport"
    libinotify_candidates = [
        shared_support_root / "libinotify.0.dylib",
        wine_root / "libinotify.0.dylib",
        wine_root / "lib" / "libinotify.0.dylib",
        Path("/opt/homebrew/lib/libinotify.0.dylib"),
        Path("/usr/local/lib/libinotify.0.dylib"),
    ]
    if not any(path.exists() for path in libinotify_candidates):
        locations = "\n  - ".join(str(path) for path in libinotify_candidates)
        raise WineRuntimeError(
            "Missing libinotify.0.dylib required by wineserver.\n"
            "Expected one of:\n"
            f"  - {locations}\n"
            "Install libinotify (for example via Homebrew) and ensure "
            "libinotify.0.dylib is discoverable by Wine."
        )

    python_host_path = windows_to_host_path(config.python_exe_windows, config.wineprefix)
    if not python_host_path.exists():
        raise WineRuntimeError(
            "Wine Python executable not found at "
            f"{python_host_path} (configured as {config.python_exe_windows}). "
            "Install Python inside the Wine prefix or set CARLA_WINE_PYTHON_EXE."
        )


def build_wine_python_command(
    script_path: Path,
    script_args: Iterable[str],
    config: WineRuntimeConfig,
) -> list[str]:
    """Build command to execute a Python script through Wine Python."""
    script_windows = posix_to_windows_path(Path(script_path), config.wineprefix)
    args_windows = convert_script_args_for_wine(script_args, config)
    return [
        str(config.wine_binary),
        config.python_exe_windows,
        script_windows,
        *args_windows,
    ]


def run_python_script(
    script_path: Path,
    script_args: Optional[Iterable[str]] = None,
    runtime_mode: str = "auto",
    timeout: Optional[int] = None,
    capture_output: bool = True,
    text: bool = True,
    cwd: Optional[Path] = None,
    env: Optional[dict[str, str]] = None,
    config: Optional[WineRuntimeConfig] = None,
    python_executable: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    """Run script in selected runtime mode and return subprocess result."""
    resolved_mode = resolve_runtime_mode(runtime_mode)
    args = list(script_args or [])

    if resolved_mode == "local":
        cmd = [python_executable or sys.executable, str(script_path), *args]
        return subprocess.run(
            cmd,
            capture_output=capture_output,
            text=text,
            timeout=timeout,
            cwd=cwd,
            env=env,
        )

    runtime_config = config if config is not None else build_wine_runtime_config()
    validate_wine_runtime(runtime_config)

    cmd = build_wine_python_command(script_path=Path(script_path), script_args=args, config=runtime_config)
    merged_env = os.environ.copy()
    merged_env.update(env or {})
    merged_env.setdefault("WINEPREFIX", str(runtime_config.wineprefix))
    merged_env.setdefault("WINEDEBUG", "-all")

    result = subprocess.run(
        cmd,
        capture_output=capture_output,
        text=text,
        timeout=timeout,
        cwd=cwd,
        env=merged_env,
    )

    stderr_text = result.stderr if isinstance(result.stderr, str) else ""
    if result.returncode != 0 and (
        "Can't check in server_mach_port" in stderr_text
        or "bind: Operation not permitted" in stderr_text
    ):
        raise WineRuntimeError(
            "Wine bridge could not attach to wineserver from this terminal session.\n"
            "Use Wineskin CMD (`WSS-cmd`) for PythonAPI execution, or relaunch CARLA.app "
            "and retry from a terminal environment that can access the wineserver mach port."
        )

    return result


def parse_args() -> argparse.Namespace:
    """Parse bridge CLI arguments."""
    parser = argparse.ArgumentParser(
        description="Run CARLA Python scripts either locally or through Wine bridge"
    )
    parser.add_argument("--script", required=True, help="Path to the Python script to run")
    parser.add_argument("--runtime", choices=sorted(RUNTIME_CHOICES), default="auto")
    parser.add_argument("--duration", type=int, help="Simulation duration in seconds")
    parser.add_argument("--output-dir", help="Output directory path")
    parser.add_argument("--timeout", type=int, default=240, help="Process timeout in seconds")
    parser.add_argument(
        "--passthrough-arg",
        action="append",
        default=[],
        help="Extra argument forwarded to the target script (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    """CLI entrypoint."""
    args = parse_args()

    script_args: list[str] = []
    if args.duration is not None:
        script_args.extend(["--duration", str(args.duration)])
    if args.output_dir:
        script_args.extend(["--output-dir", args.output_dir])
    script_args.extend(args.passthrough_arg)

    try:
        result = run_python_script(
            script_path=Path(args.script),
            script_args=script_args,
            runtime_mode=args.runtime,
            timeout=args.timeout,
            capture_output=True,
            text=True,
            cwd=Path(__file__).parent,
        )
    except WineRuntimeError as err:
        print(f"[RUNTIME ERROR] {err}")
        return 2
    except Exception as err:
        print(f"[ERROR] Failed running script: {err}")
        return 3

    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr, file=sys.stderr)

    return int(result.returncode)


if __name__ == "__main__":
    sys.exit(main())
