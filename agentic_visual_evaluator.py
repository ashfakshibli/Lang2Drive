#!/usr/bin/env python3
"""Evaluate generated scene frames from handoff manifests.

This evaluator supports two modes:
- Codex CLI image evaluation (when available)
- Manual strict fallback with key-frame and intent-window criteria
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


BASE_DIR = Path(__file__).parent.resolve()
HANDOFF_DIR_DEFAULT = BASE_DIR / "handoffs"
LATEST_POSIX_POINTER = "LATEST_MANIFEST_POSIX.txt"
DEFAULT_MODEL = os.getenv("SCENE_GENERATOR_CODEX_MODEL", "gpt-5.2-codex")

KEYFRAME_PERCENTILES = [0.05, 0.2, 0.35, 0.5, 0.65, 0.8, 0.95]
DEFAULT_EVENT_WINDOW = (0.3, 0.7)
EVENT_WINDOW_BY_SCENARIO: Dict[str, Tuple[float, float]] = {
    "red_light_violation": (0.2, 0.6),
    "sudden_pedestrian_crossing": (0.25, 0.65),
}
LOG_TOKENS_BY_SCENARIO: Dict[str, List[str]] = {
    "red_light_violation": [
        "[violation]",
        "red light",
        "crossed stop line",
        "crossed stop-line",
        "target signal remains red",
        "tl_state=red",
    ],
    "sudden_pedestrian_crossing": ["pedestrian", "walker", "crossing"],
}
INTENT_CHECKLIST_BY_SCENARIO: Dict[str, List[str]] = {
    "red_light_violation": [
        "A signalized intersection context is visible.",
        "The violating vehicle advances through/into junction during red phase.",
        "Temporal progression shows approach then violation, not static frames.",
    ],
    "sudden_pedestrian_crossing": [
        "Pedestrian appears and enters crossing trajectory near ego path.",
        "Ego context and crossing interaction are visible in front-view timeline.",
        "Temporal progression shows pedestrian onset and crossing movement.",
    ],
}


def resolve_manifest_path(explicit_path: Optional[str], handoff_dir: Path) -> Path:
    """Resolve manifest from explicit path or latest POSIX pointer."""
    if explicit_path:
        candidate = Path(explicit_path)
        if candidate.is_absolute():
            return candidate.resolve()

        base_candidate = (BASE_DIR / candidate).resolve()
        if base_candidate.exists():
            return base_candidate

        return (handoff_dir / candidate).resolve()

    pointer = handoff_dir / LATEST_POSIX_POINTER
    if not pointer.exists():
        raise FileNotFoundError(f"Latest manifest pointer not found: {pointer}")

    target = pointer.read_text(encoding="utf-8").strip()
    if not target:
        raise FileNotFoundError(f"Latest manifest pointer is empty: {pointer}")

    path = Path(target)
    if path.is_absolute():
        return path.resolve()
    return (handoff_dir / path).resolve()


def collect_frames(output_dir: Path) -> List[Path]:
    """Collect frame files from flat or nested camera directories."""
    if not output_dir.exists():
        return []

    patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp"]
    root_hits: List[Path] = []
    for pattern in patterns:
        root_hits.extend(output_dir.glob(pattern))

    if root_hits:
        return sorted(set(path.resolve() for path in root_hits))

    recursive_hits: List[Path] = []
    for pattern in patterns:
        recursive_hits.extend(output_dir.rglob(pattern))
    return sorted(set(path.resolve() for path in recursive_hits))


def sample_evenly(frames: List[Path], sample_count: int) -> List[Path]:
    """Return deterministic evenly spaced samples across the frame list."""
    if sample_count <= 0 or not frames:
        return []
    if len(frames) <= sample_count:
        return list(frames)

    if sample_count == 1:
        return [frames[0]]

    last = len(frames) - 1
    indices = [int(i * last / (sample_count - 1)) for i in range(sample_count)]
    return [frames[idx] for idx in indices]


def _frame_number(path: Path) -> Optional[int]:
    """Extract trailing frame number from filename if present."""
    match = re.search(r"(\d+)(?=\.[^.]+$)", path.name)
    if not match:
        return None
    return int(match.group(1))


def _primary_stream_frames(output_dir: Path, all_frames: List[Path]) -> List[Path]:
    """Prefer front camera stream for key-frame extraction."""
    front_dir = output_dir / "front"
    if front_dir.exists():
        front_frames = sorted(front_dir.glob("front_frame_*.png"))
        if front_frames:
            return [path.resolve() for path in front_frames]
    return all_frames


def build_key_frames(frames: List[Path], keyframe_count: int) -> List[Path]:
    """Build deterministic key frames focused on timeline anchors."""
    if not frames:
        return []

    ordered = sorted(frames, key=lambda p: (_frame_number(p) is None, _frame_number(p), str(p)))

    count = min(max(1, keyframe_count), len(ordered))
    percentiles = KEYFRAME_PERCENTILES
    if count < len(percentiles):
        percentiles = [i / (count - 1) if count > 1 else 0.0 for i in range(count)]

    last = len(ordered) - 1
    indices = {int(round(p * last)) for p in percentiles}
    if 0 not in indices:
        indices.add(0)
    if last not in indices:
        indices.add(last)

    selected = [ordered[idx] for idx in sorted(indices)]
    if len(selected) > count:
        selected = sample_evenly(selected, count)
    return selected


def _event_window_for_scenario(scenario_keyword: str) -> Tuple[float, float]:
    return EVENT_WINDOW_BY_SCENARIO.get(scenario_keyword, DEFAULT_EVENT_WINDOW)


def _event_window_coverage(key_frames: List[Path], full_frames: List[Path], event_window: Tuple[float, float]) -> Tuple[bool, str]:
    """Check that key frames capture the expected event timeline window."""
    if not key_frames or not full_frames:
        return False, "No frames available for event-window check."

    full_count = len(full_frames)
    start_idx = int(event_window[0] * (full_count - 1))
    end_idx = int(event_window[1] * (full_count - 1))

    position_map = {path: idx for idx, path in enumerate(full_frames)}
    in_window = 0
    for path in key_frames:
        idx = position_map.get(path)
        if idx is not None and start_idx <= idx <= end_idx:
            in_window += 1

    passed = in_window >= 2
    notes = (
        f"Event window {event_window[0]:.2f}-{event_window[1]:.2f}, "
        f"key frames in window: {in_window}/{len(key_frames)}"
    )
    return passed, notes


def _load_simulation_result(manifest_path: Path) -> Dict[str, Any]:
    sim_result_path = manifest_path.parent / "simulation_result.json"
    if not sim_result_path.exists():
        return {}
    try:
        return json.loads(sim_result_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _intent_checklist(scenario_keyword: str, scene_prompt: str) -> List[str]:
    checklist = INTENT_CHECKLIST_BY_SCENARIO.get(scenario_keyword)
    if checklist:
        return checklist
    return [
        "Primary scenario event appears in key frames.",
        "Temporal progression from setup to event is visible.",
        f"Intent alignment check against prompt: {scene_prompt[:160]}",
    ]


def _intent_log_signal(
    sim_result: Dict[str, Any],
    scenario_keyword: str,
    output_dir: Optional[Path] = None,
) -> Tuple[bool, str]:
    tokens = LOG_TOKENS_BY_SCENARIO.get(scenario_keyword, [])
    if not tokens:
        return True, "No scenario-specific log token requirements configured."

    combined_parts = [
        str(sim_result.get("stdout", "") or ""),
        str(sim_result.get("stderr", "") or ""),
    ]

    if output_dir is not None:
        scenario_log = output_dir / f"{scenario_keyword}_simulation.log"
        if scenario_log.exists():
            try:
                combined_parts.append(
                    scenario_log.read_text(encoding="utf-8", errors="ignore")
                )
            except Exception:
                pass

    combined = "\n".join(combined_parts).lower()
    matched = [token for token in tokens if token in combined]
    passed = len(matched) > 0
    notes = f"Matched log tokens: {matched}" if matched else f"No intent log tokens found. Expected one of: {tokens}"
    return passed, notes


def build_codex_command(
    codex_bin: str,
    output_file: Path,
    schema_file: Path,
    image_paths: List[Path],
    model: str,
) -> List[str]:
    """Build `codex exec` command for image evaluation."""
    command = [
        codex_bin,
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "workspace-write",
        "-o",
        str(output_file),
        "--schema-file",
        str(schema_file),
        "--model",
        model,
    ]
    for image_path in image_paths:
        command.extend(["--image", str(image_path)])
    command.append("-")
    return command


def parse_json_response(raw_text: str) -> Dict[str, Any]:
    """Parse plain JSON or fenced JSON response."""
    text = (raw_text or "").strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    return json.loads(text)


def apply_majority_pass_rule(criteria_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute strict-majority pass over criterion-level booleans."""
    total = len(criteria_results)
    passed = sum(1 for criterion in criteria_results if bool(criterion.get("pass")))
    overall_pass = passed > (total / 2) if total > 0 else False
    return {
        "overall_pass": overall_pass,
        "criteria_total": total,
        "criteria_passed": passed,
    }


def _manual_evaluation(
    scenario_keyword: str,
    scene_prompt: str,
    output_dir: Path,
    full_frames: List[Path],
    sampled_frames: List[Path],
    key_frames: List[Path],
    sim_result: Dict[str, Any],
    strict_intent: bool,
    human_intent_verdict: str,
    human_intent_notes: str,
) -> Dict[str, Any]:
    coverage_ratio = float(sim_result.get("coverage_ratio", 0.0) or 0.0)
    process_success = bool(sim_result.get("process_success")) if sim_result else False
    frames_ok = bool(sim_result.get("frames_ok")) if sim_result else bool(full_frames)

    event_window = _event_window_for_scenario(scenario_keyword)
    event_pass, event_notes = _event_window_coverage(key_frames, full_frames, event_window)
    log_pass, log_notes = _intent_log_signal(sim_result, scenario_keyword, output_dir=output_dir)

    if human_intent_verdict == "pass":
        human_pass = True
    elif human_intent_verdict == "fail":
        human_pass = False
    else:
        human_pass = not strict_intent

    human_notes_combined = human_intent_notes.strip() or (
        "Human intent verdict not provided."
        if strict_intent
        else "Human intent verdict optional in non-strict mode."
    )

    criteria_results = [
        {
            "criterion": "runtime_completed_cleanly",
            "pass": process_success,
            "notes": f"process_success={process_success}",
        },
        {
            "criterion": "frame_coverage_threshold",
            "pass": frames_ok,
            "notes": f"frames_ok={frames_ok}, coverage_ratio={coverage_ratio:.3f}",
        },
        {
            "criterion": "key_frames_available",
            "pass": len(key_frames) >= max(4, min(8, len(full_frames))) if full_frames else False,
            "notes": f"key_frames={len(key_frames)}, total_frames={len(full_frames)}",
        },
        {
            "criterion": "sampled_frames_available",
            "pass": len(sampled_frames) >= min(6, len(full_frames)) if full_frames else False,
            "notes": f"sampled_frames={len(sampled_frames)}, total_frames={len(full_frames)}",
        },
        {
            "criterion": "intent_event_window_captured",
            "pass": event_pass,
            "notes": event_notes,
        },
        {
            "criterion": "intent_log_signal_present",
            "pass": log_pass,
            "notes": log_notes,
        },
        {
            "criterion": "human_intent_verdict",
            "pass": human_pass,
            "notes": f"verdict={human_intent_verdict}; {human_notes_combined}",
        },
    ]

    summary = apply_majority_pass_rule(criteria_results)
    checklist = _intent_checklist(scenario_keyword, scene_prompt)

    strict_gate_pending = bool(strict_intent and human_intent_verdict == "unknown")
    strict_gate_failed = bool(strict_intent and human_intent_verdict == "fail")
    if strict_gate_pending or strict_gate_failed:
        summary["overall_pass"] = False

    return {
        "criteria_results": criteria_results,
        "summary": (
            "Manual strict evaluator used. Review key frames and checklist, then set "
            "--human-intent-verdict pass|fail for final sign-off."
        ),
        "strict_gate_pending_human_verdict": strict_gate_pending,
        "strict_gate_failed_human_verdict": strict_gate_failed,
        "intent_checklist": checklist,
        "event_window": {
            "start_percent": event_window[0],
            "end_percent": event_window[1],
        },
        "suggested_fix_prompt": (
            "Update scenario so key event clearly appears in front-view timeline within the event window, "
            "and improve actor timing/placement for intent fidelity."
        ),
        **summary,
    }


def _run_codex_evaluation(
    image_paths: List[Path],
    scenario_keyword: str,
    scene_prompt: str,
    model: str,
    codex_bin: str,
) -> Dict[str, Any]:
    schema_payload = {
        "type": "object",
        "required": ["criteria_results", "summary", "suggested_fix_prompt"],
        "properties": {
            "criteria_results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["criterion", "pass", "notes"],
                    "properties": {
                        "criterion": {"type": "string"},
                        "pass": {"type": "boolean"},
                        "notes": {"type": "string"},
                    },
                },
            },
            "summary": {"type": "string"},
            "suggested_fix_prompt": {"type": "string"},
        },
    }

    checklist = _intent_checklist(scenario_keyword, scene_prompt)
    event_window = _event_window_for_scenario(scenario_keyword)
    checklist_lines = "\n".join(f"- {item}" for item in checklist)

    prompt = (
        "You are evaluating whether CARLA simulation key/sampled frames match scenario intent.\n"
        "Return JSON only that conforms to schema.\n\n"
        f"Scenario keyword: {scenario_keyword}\n"
        f"Scenario intent prompt:\n{scene_prompt}\n\n"
        f"Expected event window (timeline fraction): {event_window[0]:.2f} to {event_window[1]:.2f}\n"
        "Intent checklist:\n"
        f"{checklist_lines}\n\n"
        "Evaluate criteria for: intent visibility, temporal progression, camera usability, and likely event occurrence in event window."
    )

    temp_dir = BASE_DIR / "handoffs" / ".tmp_eval"
    temp_dir.mkdir(parents=True, exist_ok=True)
    schema_file = temp_dir / f"schema_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.json"
    output_file = temp_dir / f"codex_eval_{datetime.now().strftime('%Y%m%d_%H%M%S%f')}.txt"
    schema_file.write_text(json.dumps(schema_payload, indent=2), encoding="utf-8")

    command = build_codex_command(
        codex_bin=codex_bin,
        output_file=output_file,
        schema_file=schema_file,
        image_paths=image_paths,
        model=model,
    )

    result = subprocess.run(
        command,
        input=prompt,
        cwd=BASE_DIR,
        capture_output=True,
        text=True,
        timeout=300,
    )

    if result.returncode != 0:
        stderr_excerpt = (result.stderr or "").strip()[-2000:]
        stdout_excerpt = (result.stdout or "").strip()[-1000:]
        details = stderr_excerpt or stdout_excerpt or "unknown codex failure"
        raise RuntimeError(f"Codex evaluation failed (exit {result.returncode}): {details}")

    response_text = output_file.read_text(encoding="utf-8", errors="ignore").strip()
    parsed = parse_json_response(response_text)
    summary = apply_majority_pass_rule(parsed.get("criteria_results", []))
    parsed.update(summary)
    parsed["intent_checklist"] = checklist
    parsed["event_window"] = {"start_percent": event_window[0], "end_percent": event_window[1]}
    return parsed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate latest handoff frames")
    parser.add_argument("--handoff-manifest", help="Manifest path (default: latest POSIX pointer)")
    parser.add_argument("--handoff-dir", default=str(HANDOFF_DIR_DEFAULT))
    parser.add_argument("--sample-count", type=int, default=12)
    parser.add_argument("--keyframe-count", type=int, default=9)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output-json", help="Optional explicit output JSON path")
    parser.add_argument("--codex-bin", default="codex")
    parser.add_argument(
        "--manual-only",
        action="store_true",
        help="Skip Codex CLI and emit manual-evaluation payload",
    )
    parser.add_argument(
        "--strict-intent",
        action="store_true",
        help="Require explicit human intent verdict for strict pass/fail gating",
    )
    parser.add_argument(
        "--human-intent-verdict",
        choices=["pass", "fail", "unknown"],
        default="unknown",
        help="Human verdict after key-frame review",
    )
    parser.add_argument(
        "--human-intent-notes",
        default="",
        help="Optional notes for manual intent verdict",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    handoff_dir = Path(args.handoff_dir).resolve()
    manifest_path = resolve_manifest_path(args.handoff_manifest, handoff_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    output_dir_raw = str(manifest.get("output_dir_posix") or manifest.get("output_dir_windows") or "").strip()
    if not output_dir_raw:
        print("[ERROR] Manifest missing output directory path.")
        return 2

    output_dir = Path(output_dir_raw)
    scene_prompt = str(manifest.get("scene_prompt_original", ""))
    scenario_keyword = str(manifest.get("scenario_keyword", "scenario"))

    frames = collect_frames(output_dir)
    sampled = sample_evenly(frames, int(args.sample_count))
    primary_frames = _primary_stream_frames(output_dir, frames)
    key_frames = build_key_frames(primary_frames, int(args.keyframe_count))
    sim_result = _load_simulation_result(manifest_path)

    evaluation: Dict[str, Any]
    codex_available = shutil.which(args.codex_bin) is not None
    if args.manual_only or not codex_available:
        evaluation = _manual_evaluation(
            scenario_keyword=scenario_keyword,
            scene_prompt=scene_prompt,
            output_dir=output_dir,
            full_frames=primary_frames,
            sampled_frames=sampled,
            key_frames=key_frames,
            sim_result=sim_result,
            strict_intent=bool(args.strict_intent),
            human_intent_verdict=str(args.human_intent_verdict),
            human_intent_notes=str(args.human_intent_notes),
        )
        evaluation["evaluation_mode"] = "manual_fallback"
        if not codex_available and not args.manual_only:
            evaluation["summary"] += " Codex CLI not found on PATH."
    else:
        try:
            image_paths = list(dict.fromkeys([*key_frames, *sampled]))
            evaluation = _run_codex_evaluation(
                image_paths=image_paths,
                scenario_keyword=scenario_keyword,
                scene_prompt=scene_prompt,
                model=args.model,
                codex_bin=args.codex_bin,
            )
            evaluation["evaluation_mode"] = "codex_cli"
        except Exception as err:
            evaluation = _manual_evaluation(
                scenario_keyword=scenario_keyword,
                scene_prompt=scene_prompt,
                full_frames=primary_frames,
                sampled_frames=sampled,
                key_frames=key_frames,
                sim_result=sim_result,
                strict_intent=bool(args.strict_intent),
                human_intent_verdict=str(args.human_intent_verdict),
                human_intent_notes=str(args.human_intent_notes),
            )
            evaluation["evaluation_mode"] = "manual_fallback"
            evaluation["summary"] += f" Codex evaluation failed: {err}"

    payload: Dict[str, Any] = {
        "generated_at": datetime.now().isoformat(),
        "manifest_path": str(manifest_path.resolve()),
        "scenario_keyword": scenario_keyword,
        "scene_keyword": manifest.get("scene_keyword"),
        "output_dir": str(output_dir),
        "sample_count_requested": int(args.sample_count),
        "keyframe_count_requested": int(args.keyframe_count),
        "frame_count_total": len(frames),
        "frame_count_primary_stream": len(primary_frames),
        "strict_intent": bool(args.strict_intent),
        "human_intent_verdict": str(args.human_intent_verdict),
        "sampled_frames": [str(path) for path in sampled],
        "key_frames": [str(path) for path in key_frames],
        "simulation_result_snapshot": {
            "success": sim_result.get("success"),
            "process_success": sim_result.get("process_success"),
            "frames_ok": sim_result.get("frames_ok"),
            "coverage_ratio": sim_result.get("coverage_ratio"),
            "returncode": sim_result.get("returncode"),
        },
        **evaluation,
    }

    output_json = (
        Path(args.output_json).resolve()
        if args.output_json
        else (manifest_path.parent / f"{scenario_keyword}_visual_eval.json")
    )
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"[INFO] Visual evaluation saved: {output_json}")
    print(f"[INFO] Overall pass: {payload.get('overall_pass')}")
    print(f"[INFO] Criteria passed: {payload.get('criteria_passed')}/{payload.get('criteria_total')}")

    return 0 if bool(payload.get("overall_pass")) else 1


if __name__ == "__main__":
    raise SystemExit(main())
