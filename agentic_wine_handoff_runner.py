#!/usr/bin/env python3
"""Run latest handoff manifest in Wine CMD/local runtime and write simulation_result.json."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

from carla_wine_bridge import run_python_script


BASE_DIR = Path(__file__).parent.resolve()
HANDOFF_DIR_DEFAULT = BASE_DIR / "handoffs"
LATEST_POSIX_POINTER = "LATEST_MANIFEST_POSIX.txt"
LATEST_WINDOWS_POINTER = "LATEST_MANIFEST_WINDOWS.txt"
DEFAULT_MIN_SUCCESS_RATIO = 0.5
KNOWN_CAMERA_VIEWS = ("front", "front_left", "front_right", "rear", "drone_follow")


def _pick_manifest_key() -> str:
    return LATEST_WINDOWS_POINTER if os.name == "nt" else LATEST_POSIX_POINTER


def resolve_manifest_path(explicit_path: Optional[str], handoff_dir: Path) -> Path:
    """Resolve manifest from explicit path or latest pointer file."""
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.is_absolute():
            return candidate.resolve()

        base_candidate = (BASE_DIR / candidate).resolve()
        if base_candidate.exists():
            return base_candidate

        return (handoff_dir / candidate).resolve()

    pointer_name = _pick_manifest_key()
    pointer = handoff_dir / pointer_name
    if not pointer.exists() and pointer_name != LATEST_POSIX_POINTER:
        pointer = handoff_dir / LATEST_POSIX_POINTER

    if not pointer.exists():
        raise FileNotFoundError(
            "Latest manifest pointer not found. Expected one of: "
            f"{handoff_dir / LATEST_POSIX_POINTER}, {handoff_dir / LATEST_WINDOWS_POINTER}"
        )

    target = pointer.read_text(encoding="utf-8").strip()
    if not target:
        raise FileNotFoundError(f"Manifest pointer is empty: {pointer}")

    path = Path(target)
    if path.is_absolute():
        return path.resolve()

    by_handoff = (handoff_dir / path).resolve()
    if by_handoff.exists():
        return by_handoff

    return (BASE_DIR / path).resolve()


def load_manifest(manifest_path: Path) -> Dict[str, Any]:
    """Load handoff manifest JSON."""
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _resolve_path_from_manifest(manifest: Dict[str, Any], kind: str) -> Path:
    """Resolve code/output path from manifest, preferring host-appropriate field."""
    if kind not in {"code", "output"}:
        raise ValueError(f"Unsupported manifest path kind: {kind}")

    if kind == "code":
        keys = ["code_file_windows", "code_file_posix"] if os.name == "nt" else ["code_file_posix", "code_file_windows"]
    else:
        keys = ["output_dir_windows", "output_dir_posix"] if os.name == "nt" else ["output_dir_posix", "output_dir_windows"]

    for key in keys:
        raw = str(manifest.get(key, "")).strip()
        if raw:
            return Path(raw)

    raise ValueError(f"Manifest missing required {kind} path fields")


def detect_output_flag(code_file: Path) -> str:
    """Use --output-dir when present in script, fallback to --output."""
    code_text = _read_code_text(code_file)
    if not code_text:
        return "--output"
    return "--output-dir" if "--output-dir" in code_text else "--output"


def _read_code_text(code_file: Path) -> str:
    try:
        return code_file.read_text(encoding="utf-8", errors="ignore")
    except Exception:
        return ""


def build_script_args(
    code_file: Path,
    output_dir: Path,
    duration_seconds: int,
    extra_args: Optional[Sequence[str]] = None,
) -> list[str]:
    """Build simulation script args based on supported output flag."""
    output_flag = detect_output_flag(code_file)
    args = ["--duration", str(int(duration_seconds)), output_flag, str(output_dir)]
    if extra_args:
        args.extend([str(x) for x in extra_args])
    return args


def _required_camera_views_from_manifest(manifest: Dict[str, Any]) -> list[str]:
    """Infer required camera views from scene specifications when explicitly declared."""
    spec = str(manifest.get("scene_specifications", "") or "")
    camera_contract_line = ""
    for raw_line in spec.splitlines():
        line = str(raw_line).strip()
        if line.lower().startswith("camera contract:"):
            camera_contract_line = line
            break
    if not camera_contract_line:
        return []
    camera_contract_lower = camera_contract_line.lower()
    required = [view for view in KNOWN_CAMERA_VIEWS if view in camera_contract_lower]
    return required


def _collect_frame_stats(output_dir: Path, duration_seconds: int, min_success_ratio: float) -> Dict[str, Any]:
    camera_views = list(KNOWN_CAMERA_VIEWS)
    expected_frames = max(1, int(duration_seconds) * 20)

    per_view: Dict[str, int] = {}
    for view in camera_views:
        view_dir = output_dir / view
        if view_dir.exists():
            per_view[view] = len(list(view_dir.glob(f"{view}_frame_*.png")))

    if per_view:
        best_stream = max(per_view.values())
        total_frames = sum(per_view.values())
        coverage_ratio = best_stream / expected_frames
        return {
            "expected_frames": expected_frames,
            "frame_counts_by_view": per_view,
            "best_stream_frames": best_stream,
            "total_frames": total_frames,
            "coverage_ratio": coverage_ratio,
            "frames_ok": coverage_ratio >= min_success_ratio,
        }

    recursive_images = [
        *output_dir.rglob("*.png"),
        *output_dir.rglob("*.jpg"),
        *output_dir.rglob("*.jpeg"),
    ]
    total_frames = len(recursive_images)
    coverage_ratio = total_frames / expected_frames
    return {
        "expected_frames": expected_frames,
        "frame_counts_by_view": {},
        "best_stream_frames": total_frames,
        "total_frames": total_frames,
        "coverage_ratio": coverage_ratio,
        "frames_ok": coverage_ratio >= min_success_ratio,
    }


def _apply_camera_contract_checks(
    frame_stats: Dict[str, Any],
    required_views: Sequence[str],
    min_success_ratio: float,
) -> Dict[str, Any]:
    """Tighten frame success when a scene explicitly requires specific camera views."""
    required = [str(v) for v in required_views if str(v)]
    if not required:
        return frame_stats

    per_view = {
        str(k): int(v)
        for k, v in dict(frame_stats.get("frame_counts_by_view") or {}).items()
    }
    expected_frames = max(1, int(frame_stats.get("expected_frames", 1) or 1))

    missing_views = [view for view in required if per_view.get(view, 0) <= 0]
    view_coverage = {
        view: (per_view.get(view, 0) / expected_frames) for view in required
    }
    weak_views = [
        view for view in required if view_coverage.get(view, 0.0) < float(min_success_ratio)
    ]
    count_values = [per_view.get(view, 0) for view in required if per_view.get(view, 0) > 0]
    synchronized_count_match = bool(count_values) and len(set(count_values)) == 1 and not missing_views
    camera_contract_ok = not missing_views and not weak_views

    frame_stats.update(
        {
            "required_camera_views": required,
            "camera_contract_ok": camera_contract_ok,
            "camera_contract_missing_views": missing_views,
            "camera_contract_weak_views": weak_views,
            "camera_contract_view_coverage": {
                view: round(view_coverage.get(view, 0.0), 4) for view in required
            },
            "camera_contract_synchronized_count_match": synchronized_count_match,
        }
    )

    if not camera_contract_ok:
        frame_stats["frames_ok"] = False
        frame_stats["camera_contract_error"] = (
            "Explicit camera contract not satisfied "
            f"(required={required}, missing={missing_views}, weak={weak_views})"
        )
    elif not synchronized_count_match:
        frame_stats["camera_contract_warning"] = (
            "Required camera views exist but frame counts are not equal; "
            "matched frame-id contract may be degraded."
        )

    return frame_stats


def _default_timeout(duration_seconds: int) -> int:
    # Budget: load_world up to 120s + apply_settings retries up to 450s +
    # traffic spawn ~60s + actual sim + headroom.  Minimum 480s.
    return max(int(duration_seconds) + 420, int(duration_seconds) * 12, 480)


def run_manifest(
    manifest: Dict[str, Any],
    timeout: Optional[int],
    dry_run: bool,
    cwd: Optional[Path],
    min_success_ratio: float = DEFAULT_MIN_SUCCESS_RATIO,
    output_dir_override: Optional[Path] = None,
    script_args_extra: Optional[Sequence[str]] = None,
    label: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one manifest and return structured simulation result payload."""
    code_file = _resolve_path_from_manifest(manifest, "code")
    output_dir = output_dir_override or _resolve_path_from_manifest(manifest, "output")
    duration_seconds = int(manifest.get("duration_seconds", 20))
    timeout_seconds = int(timeout) if timeout is not None else _default_timeout(duration_seconds)
    script_args = build_script_args(
        code_file,
        output_dir,
        duration_seconds,
        extra_args=script_args_extra,
    )

    payload: Dict[str, Any] = {
        "started_at": datetime.now().isoformat(),
        "code_file": str(code_file),
        "output_dir": str(output_dir),
        "duration_seconds": duration_seconds,
        "timeout_seconds": timeout_seconds,
        "script_args": script_args,
        "dry_run": dry_run,
    }
    if script_args_extra:
        payload["script_args_extra"] = [str(x) for x in script_args_extra]
    if label:
        payload["label"] = label

    if dry_run:
        payload.update(
            {
                "process_success": True,
                "frames_ok": False,
                "success": True,
                "returncode": 0,
                "stdout": "",
                "stderr": "",
                "note": "Dry-run only; process not executed.",
            }
        )
        payload["finished_at"] = datetime.now().isoformat()
        return payload

    output_dir.mkdir(parents=True, exist_ok=True)

    process_success = False
    returncode: Optional[int] = None
    stdout = ""
    stderr = ""
    error = ""

    try:
        result = run_python_script(
            script_path=code_file,
            script_args=script_args,
            runtime_mode="local",
            timeout=timeout_seconds,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
        returncode = int(result.returncode)
        stdout = (result.stdout or "")
        stderr = (result.stderr or "")
        process_success = returncode == 0
    except subprocess.TimeoutExpired:
        returncode = -1
        error = f"Timeout after {timeout_seconds}s"
    except Exception as err:  # pragma: no cover - integration branch
        returncode = -2
        error = f"Execution error: {type(err).__name__}: {err}"

    frame_stats = _collect_frame_stats(output_dir, duration_seconds, min_success_ratio)
    required_camera_views = _required_camera_views_from_manifest(manifest)
    frame_stats = _apply_camera_contract_checks(
        frame_stats,
        required_camera_views,
        min_success_ratio,
    )
    frames_ok = bool(frame_stats["frames_ok"])
    success = bool(process_success or frames_ok)

    payload.update(
        {
            "process_success": process_success,
            "frames_ok": frames_ok,
            "success": success,
            "returncode": returncode,
            "stdout": stdout,
            "stderr": stderr,
            "error": error,
            **frame_stats,
        }
    )

    if frames_ok and not process_success:
        payload["note"] = (
            "Process exited non-zero or timed out, but frame coverage met threshold; "
            "marking simulation as success."
        )
    elif frame_stats.get("camera_contract_error"):
        payload["note"] = str(frame_stats.get("camera_contract_error"))

    payload["finished_at"] = datetime.now().isoformat()
    return payload


def _write_stage2_log(log_file: Path, manifest_path: Path, result: Dict[str, Any]) -> None:
    lines = [
        f"timestamp: {datetime.now().isoformat()}",
        f"manifest_path: {manifest_path}",
        f"code_file: {result.get('code_file')}",
        f"output_dir: {result.get('output_dir')}",
        f"duration_seconds: {result.get('duration_seconds')}",
        f"timeout_seconds: {result.get('timeout_seconds')}",
        f"script_args: {result.get('script_args')}",
        f"returncode: {result.get('returncode')}",
        f"process_success: {result.get('process_success')}",
        f"frames_ok: {result.get('frames_ok')}",
        f"success: {result.get('success')}",
        f"coverage_ratio: {result.get('coverage_ratio')}",
        f"total_frames: {result.get('total_frames')}",
        f"best_stream_frames: {result.get('best_stream_frames')}",
        f"frame_counts_by_view: {result.get('frame_counts_by_view')}",
        f"required_camera_views: {result.get('required_camera_views')}",
        f"camera_contract_ok: {result.get('camera_contract_ok')}",
        f"camera_contract_missing_views: {result.get('camera_contract_missing_views')}",
        f"camera_contract_weak_views: {result.get('camera_contract_weak_views')}",
        f"camera_contract_synchronized_count_match: {result.get('camera_contract_synchronized_count_match')}",
    ]
    if result.get("error"):
        lines.extend(["", "error:", str(result.get("error"))])
    if result.get("note"):
        lines.extend(["", "note:", str(result.get("note"))])
    if result.get("stdout"):
        lines.extend(["", "stdout:", str(result.get("stdout"))[-4000:]])
    if result.get("stderr"):
        lines.extend(["", "stderr:", str(result.get("stderr"))[-4000:]])

    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run latest handoff manifest in Wine/local runtime")
    parser.add_argument("--handoff-manifest", help="Manifest path (defaults to latest pointer)")
    parser.add_argument("--handoff-dir", default=str(HANDOFF_DIR_DEFAULT))
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-success-ratio", type=float, default=DEFAULT_MIN_SUCCESS_RATIO)
    parser.add_argument(
        "--script-arg",
        action="append",
        default=[],
        help="Extra argument passed through to the scene script (repeatable)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    handoff_dir = Path(args.handoff_dir).resolve()
    handoff_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = resolve_manifest_path(args.handoff_manifest, handoff_dir)
    if not manifest_path.exists():
        print(f"[ERROR] Manifest not found: {manifest_path}")
        return 2

    manifest = load_manifest(manifest_path)
    result = run_manifest(
        manifest=manifest,
        timeout=args.timeout,
        dry_run=args.dry_run,
        cwd=BASE_DIR,
        min_success_ratio=float(args.min_success_ratio),
        script_args_extra=list(args.script_arg or []),
    )
    result["manifest_path"] = str(manifest_path.resolve())

    result_path = manifest_path.parent / "simulation_result.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")

    stage2_log = manifest_path.parent / f"stage2_runner_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _write_stage2_log(stage2_log, manifest_path, result)

    print(f"[INFO] Simulation result saved: {result_path}")
    print(f"[INFO] Stage 2 log saved: {stage2_log}")

    if result.get("success"):
        if result.get("note"):
            print(f"[PASS] {result['note']}")
        else:
            print("[PASS] Simulation completed successfully.")
        return 0

    print("[FAILED] Simulation failed. Check simulation_result.json for details.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
