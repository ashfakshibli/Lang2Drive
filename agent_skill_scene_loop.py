#!/usr/bin/env python3
"""Manual scene-loop manager for Codex app workflows (no Codex CLI dependency)."""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from carla_wine_bridge import build_wine_runtime_config, posix_to_windows_path
from scene_excel_utils import (
    normalize_scene_specifications_for_generation,
    read_unique_scenes_from_excel as read_scenes_excel_shared,
)


BASE_DIR = Path(__file__).parent.resolve()
EXCEL_PATH = BASE_DIR / "Keyword Prompt Verification.xlsx"
PROMPT_TEMPLATE_PATH = BASE_DIR / "prompt.txt"
PROMPTS_DIR = BASE_DIR / "generated_prompts"
CODE_DIR = BASE_DIR / "generated_code"
SCENES_DIR = BASE_DIR / "scenes"
HANDOFF_DIR_DEFAULT = BASE_DIR / "handoffs"
LATEST_POSIX_POINTER = "LATEST_MANIFEST_POSIX.txt"
LATEST_WINDOWS_POINTER = "LATEST_MANIFEST_WINDOWS.txt"
NEXT_SCENE_SERIAL_FILE = "NEXT_SCENE_SERIAL.txt"


def generate_scenario_keyword(keyword: str) -> str:
    scenario_keyword = keyword.lower().replace(" ", "_")
    scenario_keyword = "".join(c for c in scenario_keyword if c.isalnum() or c == "_")
    return scenario_keyword


def read_unique_scenes_from_excel(excel_path: Path) -> List[Dict[str, Any]]:
    scenes = read_scenes_excel_shared(excel_path)
    for scene in scenes:
        scene["scenario_keyword"] = generate_scenario_keyword(scene["keyword"])
    return scenes


def select_scene_by_serial(scenes: List[Dict[str, Any]], scene_serial: int) -> Dict[str, Any]:
    if not scenes:
        raise ValueError("No scenes found in Excel file.")
    if scene_serial <= 0:
        raise ValueError("scene serial must be >= 1")
    normalized = ((scene_serial - 1) % len(scenes)) + 1
    selected = scenes[normalized - 1]
    selected = dict(selected)
    selected["requested_serial"] = scene_serial
    selected["normalized_serial"] = normalized
    selected["wrapped"] = normalized != scene_serial
    return selected


def shot_paths(run_id: str, scenario_keyword: str, shot_index: int = 0) -> Dict[str, Path]:
    prompt_dir = PROMPTS_DIR / scenario_keyword / run_id / f"shot_{shot_index}"
    code_dir = CODE_DIR / scenario_keyword / run_id / f"shot_{shot_index}"
    output_dir = SCENES_DIR / scenario_keyword / run_id / f"shot_{shot_index}"
    prompt_dir.mkdir(parents=True, exist_ok=True)
    code_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "prompt_dir": prompt_dir,
        "code_dir": code_dir,
        "output_dir": output_dir,
        "prompt_file": prompt_dir / "enhanced_prompt.txt",
        "full_prompt_file": prompt_dir / "full_prompt.txt",
        "code_file": code_dir / f"{scenario_keyword}.py",
    }


def load_next_scene_serial(handoff_dir: Path) -> int:
    state_file = handoff_dir / NEXT_SCENE_SERIAL_FILE
    if not state_file.exists():
        return 1
    raw = state_file.read_text(encoding="utf-8").strip()
    if not raw:
        return 1
    try:
        value = int(raw)
        return value if value > 0 else 1
    except ValueError:
        return 1


def save_next_scene_serial(handoff_dir: Path, serial: int) -> None:
    state_file = handoff_dir / NEXT_SCENE_SERIAL_FILE
    state_file.write_text(str(max(1, int(serial))), encoding="utf-8")


def find_latest_seed_code(scenario_keyword: str, exclude_run_id: str) -> Optional[Path]:
    candidates = list((CODE_DIR / scenario_keyword).glob(f"**/{scenario_keyword}.py"))
    if not candidates:
        return None
    filtered = [
        path
        for path in candidates
        if f"/{exclude_run_id}/" not in str(path).replace("\\", "/")
    ]
    if not filtered:
        return None
    return max(filtered, key=lambda p: p.stat().st_mtime)


def load_prompt_template() -> str:
    if not PROMPT_TEMPLATE_PATH.exists():
        return "{SCENE_PROMPT}"
    return PROMPT_TEMPLATE_PATH.read_text(encoding="utf-8")


def build_full_prompt(
    template: str,
    scenario_keyword: str,
    scene_prompt: str,
    scene_specifications: str,
) -> str:
    full_prompt = template.replace("{SCENE_KEYWORD}", scenario_keyword).replace(
        "{SCENE_PROMPT}",
        scene_prompt,
    )
    if "{SCENE_SPECIFICATIONS}" in full_prompt:
        full_prompt = full_prompt.replace("{SCENE_SPECIFICATIONS}", scene_specifications)
    elif scene_specifications.strip():
        full_prompt = (
            f"{full_prompt}\n\nSCENE_SPECIFICATIONS:\n{scene_specifications.strip()}\n"
        )
    return full_prompt


def _to_windows_path(path: Path) -> str:
    runtime_config = build_wine_runtime_config()
    return posix_to_windows_path(path.resolve(), runtime_config.wineprefix)


def write_stage1_log(log_path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        f"timestamp: {datetime.now().isoformat()}",
        f"run_id: {payload.get('run_id')}",
        f"scene_keyword: {payload.get('scene_keyword')}",
        f"scenario_keyword: {payload.get('scenario_keyword')}",
        f"scene_serial: {payload.get('scene_serial')}",
        f"generation_mode: {payload.get('generation_mode')}",
        f"generation_status: {payload.get('generation_status')}",
        f"code_seed_source: {payload.get('code_seed_source')}",
        f"code_file_posix: {payload.get('code_file_posix')}",
        f"output_dir_posix: {payload.get('output_dir_posix')}",
        "",
        "scene_prompt_original:",
        str(payload.get("scene_prompt_original", "")),
        "",
        "scene_specifications:",
        str(payload.get("scene_specifications", "")),
        "",
    ]
    log_path.write_text("\n".join(lines), encoding="utf-8")


def write_scene_generator_log(log_path: Path, payload: Dict[str, Any]) -> None:
    lines = [
        f"timestamp: {datetime.now().isoformat()}",
        f"run_id: {payload.get('run_id')}",
        f"scene_serial: {payload.get('scene_serial')}",
        f"scene_keyword: {payload.get('scene_keyword')}",
        f"scenario_keyword: {payload.get('scenario_keyword')}",
        f"generation_mode: {payload.get('generation_mode')}",
        f"generation_status: {payload.get('generation_status')}",
        f"manifest_path: {payload.get('manifest_path')}",
        f"prompt_file: {payload.get('prompt_file')}",
        f"code_file_posix: {payload.get('code_file_posix')}",
        f"output_dir_posix: {payload.get('output_dir_posix')}",
    ]
    if payload.get("code_seed_source"):
        lines.append(f"code_seed_source: {payload.get('code_seed_source')}")
    lines.extend(
        [
            "",
            "scene_prompt_original:",
            str(payload.get("scene_prompt_original", "")),
            "",
            "scene_specifications:",
            str(payload.get("scene_specifications", "")),
            "",
        ]
    )
    log_path.write_text("\n".join(lines), encoding="utf-8")


def cmd_list_scenes(args: argparse.Namespace) -> int:
    scenes = read_unique_scenes_from_excel(Path(args.excel))
    limit = args.limit if args.limit is not None else len(scenes)
    for scene in scenes[:limit]:
        print(
            f"{scene['serial']:>2}. {scene['keyword']} "
            f"({scene['sheet']}!A{scene['row']}) -> {scene['scenario_keyword']}"
        )
    print(f"Total scenes: {len(scenes)}")
    return 0


def cmd_prepare(args: argparse.Namespace) -> int:
    handoff_dir = Path(args.handoff_dir).resolve()
    handoff_dir.mkdir(parents=True, exist_ok=True)

    scenes = read_unique_scenes_from_excel(Path(args.excel))
    requested_serial = args.scene_serial if args.scene_serial is not None else load_next_scene_serial(handoff_dir)
    scene = select_scene_by_serial(scenes, requested_serial)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    scenario_keyword = str(scene["scenario_keyword"])
    shot_index = int(getattr(args, "shot_index", 0) or 0)
    if shot_index < 0:
        raise ValueError("shot index must be >= 0")
    paths = shot_paths(run_id=run_id, scenario_keyword=scenario_keyword, shot_index=shot_index)

    template = load_prompt_template()
    scene_specifications = normalize_scene_specifications_for_generation(
        str(scene.get("scene_specifications", "") or ""),
        scene_prompt=str(scene["prompt"]),
    )
    full_prompt = build_full_prompt(
        template=template,
        scenario_keyword=scenario_keyword,
        scene_prompt=str(scene["prompt"]),
        scene_specifications=scene_specifications,
    )
    paths["prompt_file"].write_text(
        (
            f"KEYWORD: {scene['keyword']}\n"
            f"SCENARIO_KEYWORD: {scenario_keyword}\n"
            f"SHOT_INDEX: {shot_index}\n\n"
            f"ORIGINAL PROMPT:\n{scene['prompt']}\n\n"
            f"SCENE SPECIFICATIONS:\n{scene_specifications}\n\n"
            "ENHANCED PROMPT:\n"
            f"{scene['prompt']}\n"
        ),
        encoding="utf-8",
    )
    paths["full_prompt_file"].write_text(full_prompt, encoding="utf-8")

    seed_code = find_latest_seed_code(scenario_keyword, exclude_run_id=run_id)
    if seed_code and args.seed_from_latest:
        shutil.copy2(seed_code, paths["code_file"])
        seed_source = str(seed_code.resolve())
    else:
        paths["code_file"].write_text(
            (
                "import argparse\n\n\n"
                "def main() -> None:\n"
                "    parser = argparse.ArgumentParser(description='CARLA scenario script')\n"
                "    parser.add_argument('--duration', type=float, default=20.0)\n"
                "    parser.add_argument('--output-dir', default='./scenes/output')\n"
                "    parser.add_argument('--time-preset', default='scene_default')\n"
                "    parser.add_argument('--weather-preset', default='scene_default')\n"
                "    parser.add_argument('--sun-altitude', type=float, default=None)\n"
                "    parser.add_argument('--sun-azimuth', type=float, default=None)\n"
                "    parser.add_argument('--streetlights', default='auto')\n"
                "    parser.add_argument('--cloudiness', type=float, default=None)\n"
                "    parser.add_argument('--precipitation', type=float, default=None)\n"
                "    parser.add_argument('--precipitation-deposits', type=float, default=None)\n"
                "    parser.add_argument('--wind-intensity', type=float, default=None)\n"
                "    parser.add_argument('--fog-density', type=float, default=None)\n"
                "    parser.add_argument('--fog-distance', type=float, default=None)\n"
                "    parser.add_argument('--wetness', type=float, default=None)\n"
                "    args = parser.parse_args()\n"
                "    del args\n"
                "    raise NotImplementedError('TODO_IMPLEMENT_SCENARIO_LOGIC')\n\n\n"
                "if __name__ == '__main__':\n"
                "    main()\n"
            ),
            encoding="utf-8",
        )
        seed_source = ""

    handoff_run_dir = handoff_dir / run_id / scenario_keyword
    handoff_run_dir.mkdir(parents=True, exist_ok=True)

    stage1_log_path = handoff_run_dir / "stage1_agent_prepare.log"
    scene_generator_log_path = BASE_DIR / f"scene_generator_{run_id}.log"
    manifest_path = handoff_run_dir / "manifest.json"
    manifest = {
        "run_id": run_id,
        "scene_keyword": scene["keyword"],
        "scenario_keyword": scenario_keyword,
        "scene_serial": int(scene["normalized_serial"]),
        "shot_index": shot_index,
        "shot_status": "pending_agent_code",
        "generation_mode": "agent_skill_manual",
        "generation_status": "pending_agent_code",
        "code_seed_source": seed_source,
        "scene_prompt_original": str(scene["prompt"]),
        "scene_specifications": scene_specifications,
        "code_file_posix": str(paths["code_file"].resolve()),
        "code_file_windows": _to_windows_path(paths["code_file"].resolve()),
        "output_dir_posix": str(paths["output_dir"].resolve()),
        "output_dir_windows": _to_windows_path(paths["output_dir"].resolve()),
        "duration_seconds": int(args.duration),
        "prompt_file": str(paths["prompt_file"].resolve()),
        "full_prompt_file": str(paths["full_prompt_file"].resolve()),
        "generated_at": datetime.now().isoformat(),
        "scene_list_total": len(scenes),
        "requested_scene_serial": int(scene["requested_serial"]),
        "serial_wrapped": bool(scene["wrapped"]),
        "stage1_log_file": str(stage1_log_path.resolve()),
        "scene_generator_log_file": str(scene_generator_log_path.resolve()),
    }
    write_stage1_log(stage1_log_path, manifest)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    write_scene_generator_log(
        scene_generator_log_path,
        {
            **manifest,
            "manifest_path": str(manifest_path.resolve()),
        },
    )

    (handoff_dir / LATEST_POSIX_POINTER).write_text(str(manifest_path.resolve()), encoding="utf-8")
    (handoff_dir / LATEST_WINDOWS_POINTER).write_text(
        _to_windows_path(manifest_path.resolve()),
        encoding="utf-8",
    )

    next_serial = scene["normalized_serial"] + 1
    if next_serial > len(scenes):
        next_serial = 1
    save_next_scene_serial(handoff_dir, next_serial)

    payload = {
        "manifest_path": str(manifest_path.resolve()),
        "run_id": run_id,
        "scene_serial": scene["normalized_serial"],
        "scene_keyword": scene["keyword"],
        "scenario_keyword": scenario_keyword,
        "code_file": str(paths["code_file"].resolve()),
        "prompt_file": str(paths["prompt_file"].resolve()),
        "output_dir": str(paths["output_dir"].resolve()),
        "shot_index": shot_index,
        "next_scene_serial": next_serial,
        "generation_status": manifest["generation_status"],
        "code_seed_source": seed_source,
        "scene_generator_log_file": str(scene_generator_log_path.resolve()),
    }
    print(json.dumps(payload, indent=2))
    return 0


def resolve_manifest_path(explicit_path: Optional[str], handoff_dir: Path) -> Path:
    if explicit_path:
        p = Path(explicit_path)
        if p.is_absolute():
            return p.resolve()
        candidate = (BASE_DIR / p).resolve()
        if candidate.exists():
            return candidate
        return (handoff_dir / p).resolve()

    pointer = handoff_dir / LATEST_POSIX_POINTER
    if not pointer.exists():
        raise FileNotFoundError(f"Latest manifest pointer not found: {pointer}")
    path = Path(pointer.read_text(encoding="utf-8").strip())
    if path.is_absolute():
        return path.resolve()
    return (handoff_dir / path).resolve()


def validate_code_readiness(code_file: Path) -> List[str]:
    issues: List[str] = []
    if not code_file.exists():
        return [f"Code file missing: {code_file}"]
    text = code_file.read_text(encoding="utf-8", errors="ignore")
    if "from scene_utils.fresh_scene_runtime import run_scene" in text:
        issues.append(
            "Code still depends on scene_utils.fresh_scene_runtime; replace with standalone per-scene CARLA script."
        )
    if "def main" not in text:
        issues.append("Missing `def main`")
    if "__name__" not in text or "main()" not in text:
        issues.append("Missing executable guard `if __name__ == '__main__': main()`")
    if "TODO_IMPLEMENT_SCENARIO_LOGIC" in text:
        issues.append("Code still contains placeholder TODO_IMPLEMENT_SCENARIO_LOGIC")
    if "--output-dir" not in text and "--output" not in text:
        issues.append("Script does not expose --output-dir or --output argument")
    for flag in (
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
    ):
        if flag not in text:
            issues.append(f"Script missing required environment CLI argument {flag} (matrix capture contract)")
    return issues


def cmd_mark_ready(args: argparse.Namespace) -> int:
    handoff_dir = Path(args.handoff_dir).resolve()
    manifest_path = resolve_manifest_path(args.manifest, handoff_dir)
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    code_file = Path(str(manifest.get("code_file_posix", "")))
    issues = validate_code_readiness(code_file)

    if issues:
        payload = {
            "ready": False,
            "manifest_path": str(manifest_path),
            "code_file": str(code_file),
            "issues": issues,
        }
        print(json.dumps(payload, indent=2))
        return 1

    manifest["generation_status"] = "ready_for_wine"
    manifest["shot_status"] = "ready_for_wine"
    manifest["generation_ready_at"] = datetime.now().isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    stage1_log_value = str(manifest.get("stage1_log_file", "") or "").strip()
    stage1_log_file = Path(stage1_log_value) if stage1_log_value else None
    if stage1_log_file is not None:
        if stage1_log_file.exists():
            existing = stage1_log_file.read_text(encoding="utf-8")
        else:
            existing = ""
        stage1_log_file.write_text(
            existing
            + f"\nstatus_update: ready_for_wine at {manifest['generation_ready_at']}\n",
            encoding="utf-8",
        )

    scene_generator_log_value = str(manifest.get("scene_generator_log_file", "") or "").strip()
    scene_generator_log_file = Path(scene_generator_log_value) if scene_generator_log_value else None
    if scene_generator_log_file is not None:
        if scene_generator_log_file.exists():
            existing_log = scene_generator_log_file.read_text(encoding="utf-8")
        else:
            existing_log = ""
        scene_generator_log_file.write_text(
            existing_log + f"\nstatus_update: ready_for_wine at {manifest['generation_ready_at']}\n",
            encoding="utf-8",
        )

    payload = {
        "ready": True,
        "manifest_path": str(manifest_path.resolve()),
        "run_id": manifest.get("run_id"),
        "scene_serial": manifest.get("scene_serial"),
        "scene_keyword": manifest.get("scene_keyword"),
        "scenario_keyword": manifest.get("scenario_keyword"),
        "code_file": str(code_file.resolve()),
        "wine_command": 'cd "C:\\Program Files\\WindowsNoEditor\\VLM-AV" && 02_run_latest_scene.cmd',
        "wine_command_matrix20": 'cd "C:\\Program Files\\WindowsNoEditor\\VLM-AV" && 04_run_scene_matrix20.cmd',
    }
    print(json.dumps(payload, indent=2))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Manual scene-loop manager for Codex app workflows"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    list_parser = sub.add_parser("list-scenes", help="List scene serials from Excel")
    list_parser.add_argument("--excel", default=str(EXCEL_PATH))
    list_parser.add_argument("--limit", type=int)

    prepare_parser = sub.add_parser("prepare", help="Prepare one scene handoff for manual coding")
    prepare_parser.add_argument("--excel", default=str(EXCEL_PATH))
    prepare_parser.add_argument("--handoff-dir", default=str(HANDOFF_DIR_DEFAULT))
    prepare_parser.add_argument("--scene-serial", type=int)
    prepare_parser.add_argument("--shot-index", type=int, default=0)
    prepare_parser.add_argument("--duration", type=int, default=20)
    prepare_parser.add_argument(
        "--seed-from-latest",
        action="store_true",
        default=True,
        help="Seed new run with latest existing code for the same scenario (default)",
    )
    prepare_parser.add_argument(
        "--no-seed-from-latest",
        action="store_false",
        dest="seed_from_latest",
        help="Do not seed from existing code",
    )

    ready_parser = sub.add_parser("mark-ready", help="Validate code and mark latest/selected manifest ready")
    ready_parser.add_argument("--manifest", help="Manifest path (defaults to latest pointer)")
    ready_parser.add_argument("--handoff-dir", default=str(HANDOFF_DIR_DEFAULT))

    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "list-scenes":
        return cmd_list_scenes(args)
    if args.command == "prepare":
        return cmd_prepare(args)
    if args.command == "mark-ready":
        return cmd_mark_ready(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
