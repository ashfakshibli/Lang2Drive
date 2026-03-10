#!/usr/bin/env python3
"""Run a standalone CARLA scene across the shared time x weather matrix (20 variants)."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from agentic_wine_handoff_runner import (
    BASE_DIR,
    DEFAULT_MIN_SUCCESS_RATIO,
    HANDOFF_DIR_DEFAULT,
    _default_timeout,
    _read_code_text,
    _resolve_path_from_manifest,
    load_manifest,
    resolve_manifest_path,
    run_manifest,
)
from scene_utils.time_weather_matrix import (
    TimeWeatherSpecError,
    build_scene_env_cli_args,
    iter_time_weather_combinations,
    load_time_weather_spec,
)


REQUIRED_SCENE_ENV_FLAGS = (
    "--time-preset",
    "--weather-preset",
    "--sun-altitude",
    "--sun-azimuth",
    "--streetlights",
    "--cloudiness",
    "--precipitation",
    "--precipitation-deposits",
    "--wind-intensity",
    "--fog-density",
    "--fog-distance",
    "--wetness",
)


def _scene_support_issues(code_file: Path) -> List[str]:
    code_text = _read_code_text(code_file)
    if not code_text:
        return [f"Could not read scene script: {code_file}"]
    missing = [flag for flag in REQUIRED_SCENE_ENV_FLAGS if flag not in code_text]
    return [f"Scene script missing required env CLI flags: {', '.join(missing)}"] if missing else []


def _variant_attempt_summary(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "returncode": result.get("returncode"),
        "success": bool(result.get("success")),
        "process_success": bool(result.get("process_success")),
        "frames_ok": bool(result.get("frames_ok")),
        "coverage_ratio": float(result.get("coverage_ratio", 0.0)),
        "best_stream_frames": int(result.get("best_stream_frames", 0)),
        "total_frames": int(result.get("total_frames", 0)),
        "error": str(result.get("error", "") or ""),
        "note": str(result.get("note", "") or ""),
    }


def _write_matrix_log(log_file: Path, manifest_path: Path, result: Dict[str, Any]) -> None:
    lines = [
        f"timestamp: {datetime.now().isoformat()}",
        f"manifest_path: {manifest_path}",
        "mode: time_weather_matrix",
        f"code_file: {result.get('code_file')}",
        f"output_dir: {result.get('output_dir')}",
        f"matrix_output_root: {result.get('matrix_output_root')}",
        f"matrix_spec_path: {result.get('matrix_spec_path')}",
        f"duration_seconds: {result.get('duration_seconds')}",
        f"timeout_seconds: {result.get('timeout_seconds')}",
        f"max_attempts_per_variant: {result.get('max_attempts_per_variant')}",
        f"cooldown_seconds: {result.get('cooldown_seconds')}",
        f"dry_run: {result.get('dry_run')}",
        f"variant_count_expected: {result.get('variant_count_expected')}",
        f"variant_count_completed: {result.get('variant_count_completed')}",
        f"variant_success_count: {result.get('variant_success_count')}",
        f"variant_process_success_count: {result.get('variant_process_success_count')}",
        f"variant_frames_ok_count: {result.get('variant_frames_ok_count')}",
        f"success: {result.get('success')}",
        f"coverage_ratio_min: {result.get('coverage_ratio_min')}",
        f"coverage_ratio_avg: {result.get('coverage_ratio_avg')}",
        f"best_stream_frames_min: {result.get('best_stream_frames_min')}",
        f"best_stream_frames_max: {result.get('best_stream_frames_max')}",
        f"failed_variants: {result.get('failed_variants')}",
        "",
        "variants:",
    ]
    for variant in result.get("matrix_variants", []):
        lines.append(
            "  "
            + f"{variant.get('variant_name')}: "
            + f"success={variant.get('success')} "
            + f"attempts_used={variant.get('attempts_used')} "
            + f"coverage={variant.get('coverage_ratio')} "
            + f"best_stream={variant.get('best_stream_frames')} "
            + f"output_dir={variant.get('output_dir')}"
        )
        for idx, attempt in enumerate(variant.get("attempt_history", []), start=1):
            lines.append(
                "    "
                + f"attempt {idx}: returncode={attempt.get('returncode')} "
                + f"success={attempt.get('success')} "
                + f"process_success={attempt.get('process_success')} "
                + f"frames_ok={attempt.get('frames_ok')} "
                + f"coverage={attempt.get('coverage_ratio')} "
                + f"best_stream={attempt.get('best_stream_frames')}"
            )
            if attempt.get("error"):
                lines.append("      error: " + str(attempt.get("error")))
            elif attempt.get("note"):
                lines.append("      note: " + str(attempt.get("note")))
    log_file.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _run_variant_with_retries(
    manifest: Dict[str, Any],
    variant_output_dir: Path,
    variant_name: str,
    script_args_extra: List[str],
    timeout: Optional[int],
    dry_run: bool,
    min_success_ratio: float,
    max_attempts_per_variant: int,
    cooldown_seconds: int,
) -> Dict[str, Any]:
    attempt_history: List[Dict[str, Any]] = []
    final_result: Optional[Dict[str, Any]] = None

    for attempt in range(1, max_attempts_per_variant + 1):
        print(f"[MATRIX] {variant_name} attempt {attempt}/{max_attempts_per_variant}")
        result = run_manifest(
            manifest=manifest,
            timeout=timeout,
            dry_run=dry_run,
            cwd=BASE_DIR,
            min_success_ratio=min_success_ratio,
            output_dir_override=variant_output_dir,
            script_args_extra=script_args_extra,
            label=variant_name,
        )
        attempt_history.append(_variant_attempt_summary(result))
        final_result = result
        if result.get("success"):
            break
        if attempt < max_attempts_per_variant and not dry_run and cooldown_seconds > 0:
            print(f"[MATRIX] {variant_name} failed, cooling down {cooldown_seconds}s before retry...")
            time.sleep(float(cooldown_seconds))

    assert final_result is not None
    final_result["attempts_used"] = len(attempt_history)
    final_result["attempt_history"] = attempt_history
    return final_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run finalized scene manifest across shared time x weather matrix"
    )
    parser.add_argument("--handoff-manifest", help="Manifest path (defaults to latest pointer)")
    parser.add_argument("--handoff-dir", default=str(HANDOFF_DIR_DEFAULT))
    parser.add_argument("--spec", help="Path to time/weather matrix JSON spec")
    parser.add_argument("--timeout", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--min-success-ratio", type=float, default=DEFAULT_MIN_SUCCESS_RATIO)
    parser.add_argument(
        "--matrix-output-subdir",
        help="Override matrix output subfolder (defaults to spec value)",
    )
    parser.add_argument("--max-attempts-per-variant", type=int, default=2)
    parser.add_argument("--cooldown-seconds", type=int, default=8)
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

    try:
        spec = load_time_weather_spec(args.spec)
    except TimeWeatherSpecError as err:
        print(f"[ERROR] Invalid matrix spec: {err}")
        return 3

    code_file = _resolve_path_from_manifest(manifest, "code")
    output_dir = _resolve_path_from_manifest(manifest, "output")
    duration_seconds = int(manifest.get("duration_seconds", 20))
    timeout_seconds = int(args.timeout) if args.timeout is not None else _default_timeout(duration_seconds)
    matrix_output_subdir = str(args.matrix_output_subdir or spec["matrix_output_subdir"]).strip()
    matrix_root = output_dir / matrix_output_subdir

    support_issues = _scene_support_issues(code_file)
    if support_issues:
        error = "; ".join(support_issues)
        print(f"[ERROR] {error}")
        result = {
            "started_at": datetime.now().isoformat(),
            "finished_at": datetime.now().isoformat(),
            "matrix_mode": True,
            "success": False,
            "process_success": False,
            "frames_ok": False,
            "returncode": 1,
            "error": error,
            "manifest_path": str(manifest_path.resolve()),
            "code_file": str(code_file),
            "output_dir": str(output_dir),
            "matrix_output_root": str(matrix_root),
            "matrix_spec_path": str(spec.get("spec_path", "")),
            "duration_seconds": duration_seconds,
            "timeout_seconds": timeout_seconds,
            "dry_run": bool(args.dry_run),
            "max_attempts_per_variant": int(args.max_attempts_per_variant),
            "cooldown_seconds": int(args.cooldown_seconds),
            "variant_count_expected": len(spec["time_presets"]) * len(spec["weather_presets"]),
            "variant_count_completed": 0,
            "variant_success_count": 0,
            "variant_process_success_count": 0,
            "variant_frames_ok_count": 0,
            "failed_variants": [],
            "coverage_ratio_min": 0.0,
            "coverage_ratio_avg": 0.0,
            "best_stream_frames_min": 0,
            "best_stream_frames_max": 0,
            "total_frames_across_variants": 0,
            "matrix_variants": [],
        }
        result_path = manifest_path.parent / "simulation_result_matrix20.json"
        result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
        stage2_log = manifest_path.parent / f"stage2_runner_matrix20_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        _write_matrix_log(stage2_log, manifest_path, result)
        print(f"[INFO] Simulation result saved: {result_path}")
        print(f"[INFO] Stage 2 log saved: {stage2_log}")
        return 1

    started_at = datetime.now().isoformat()
    variants: List[Dict[str, Any]] = []
    combinations = list(iter_time_weather_combinations(spec))
    total_combos = len(combinations)

    for idx, combo in enumerate(combinations, start=1):
        time_preset = combo["time_preset"]
        weather_preset = combo["weather_preset"]
        variant_name = str(combo["variant_name"])
        variant_output_dir = matrix_root / variant_name
        print(
            f"[MATRIX] ({idx}/{total_combos}) "
            f"{time_preset['label']} x {weather_preset['label']} -> {variant_output_dir}"
        )

        cli_args = build_scene_env_cli_args(time_preset=time_preset, weather_preset=weather_preset)
        variant_result = _run_variant_with_retries(
            manifest=manifest,
            variant_output_dir=variant_output_dir,
            variant_name=variant_name,
            script_args_extra=cli_args,
            timeout=args.timeout,
            dry_run=bool(args.dry_run),
            min_success_ratio=float(args.min_success_ratio),
            max_attempts_per_variant=max(1, int(args.max_attempts_per_variant)),
            cooldown_seconds=max(0, int(args.cooldown_seconds)),
        )
        variant_result.update(
            {
                "variant_name": variant_name,
                "matrix_index": idx,
                "matrix_total": total_combos,
                "time_preset": str(time_preset["key"]),
                "time_preset_spec": time_preset,
                "weather_preset": str(weather_preset["key"]),
                "weather_preset_spec": weather_preset,
            }
        )
        variants.append(variant_result)
        print(
            "[MATRIX] "
            + ("PASS" if variant_result.get("success") else "FAIL")
            + f" {variant_name} attempts={variant_result.get('attempts_used')} "
            + f"frames={variant_result.get('best_stream_frames', 0)} "
            + f"coverage={float(variant_result.get('coverage_ratio', 0.0)):.2f}"
        )

    success_count = sum(1 for v in variants if v.get("success"))
    process_success_count = sum(1 for v in variants if v.get("process_success"))
    frames_ok_count = sum(1 for v in variants if v.get("frames_ok"))
    coverages = [float(v.get("coverage_ratio", 0.0)) for v in variants]
    best_stream_counts = [int(v.get("best_stream_frames", 0)) for v in variants]
    total_frames_across = sum(int(v.get("total_frames", 0)) for v in variants)
    failed_variants = [str(v.get("variant_name")) for v in variants if not v.get("success")]
    all_success = success_count == total_combos

    result = {
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
        "matrix_mode": True,
        "manifest_path": str(manifest_path.resolve()),
        "code_file": str(code_file),
        "output_dir": str(output_dir),
        "matrix_output_root": str(matrix_root),
        "matrix_spec_path": str(spec.get("spec_path", "")),
        "matrix_output_subdir": matrix_output_subdir,
        "duration_seconds": duration_seconds,
        "timeout_seconds": timeout_seconds,
        "dry_run": bool(args.dry_run),
        "min_success_ratio": float(args.min_success_ratio),
        "max_attempts_per_variant": max(1, int(args.max_attempts_per_variant)),
        "cooldown_seconds": max(0, int(args.cooldown_seconds)),
        "variant_count_expected": total_combos,
        "variant_count_completed": len(variants),
        "variant_success_count": success_count,
        "variant_process_success_count": process_success_count,
        "variant_frames_ok_count": frames_ok_count,
        "failed_variants": failed_variants,
        "coverage_ratio_min": min(coverages) if coverages else 0.0,
        "coverage_ratio_avg": (sum(coverages) / len(coverages)) if coverages else 0.0,
        "best_stream_frames_min": min(best_stream_counts) if best_stream_counts else 0,
        "best_stream_frames_max": max(best_stream_counts) if best_stream_counts else 0,
        "total_frames_across_variants": total_frames_across,
        "process_success": process_success_count == total_combos,
        "frames_ok": frames_ok_count == total_combos,
        "success": all_success,
        "returncode": 0 if all_success else 1,
        "error": "",
        "time_presets": spec["time_presets"],
        "weather_presets": spec["weather_presets"],
        "matrix_variants": variants,
    }

    result_path = manifest_path.parent / "simulation_result_matrix20.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    stage2_log = manifest_path.parent / f"stage2_runner_matrix20_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    _write_matrix_log(stage2_log, manifest_path, result)

    print(f"[INFO] Simulation result saved: {result_path}")
    print(f"[INFO] Stage 2 log saved: {stage2_log}")
    print(f"[INFO] Matrix summary: {success_count}/{total_combos} variants succeeded.")
    print(f"[INFO] Matrix outputs root: {matrix_root}")

    if all_success:
        print("[PASS] Matrix simulation completed successfully.")
        return 0

    print("[FAILED] One or more matrix variants failed. Check simulation_result_matrix20.json for details.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
