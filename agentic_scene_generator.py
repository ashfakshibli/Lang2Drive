#!/usr/bin/env python3
"""Agentic scene generator v2.

This module supports workbook-driven scene generation, handoff preparation, and
optional Codex CLI integration without requiring external orchestration.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from agent_backends import AgentBackend, AgentBackendError, CodexCliBackend
from carla_wine_bridge import resolve_runtime_mode, run_python_script
from scene_excel_utils import (
    normalize_scene_specifications_for_generation,
    read_unique_scenes_from_excel as read_scenes_excel_shared,
)


RUNTIME_CHOICES = ["auto", "wine-bridge", "local"]
MODE_CHOICES = ["generate-only", "test", "full"]
EXECUTION_CHOICES = ["interactive", "autonomous"]
BACKEND_CHOICES = ["none", "codex"]


@dataclass
class ShotArtifacts:
    shot_index: int
    prompt_text: str
    user_adjustment_prompt: str
    prompt_file: str
    full_prompt_file: str
    code_file: str
    scene_output_dir: str
    sim_success: Any
    frames_ok: Any
    accepted_by_user: Any
    status: str
    timestamp: str


class AgenticSceneGenerator:
    """Generate CARLA scene code and optional handoff manifests."""

    def __init__(
        self,
        test_mode: bool,
        generate_only: bool,
        scenario_limit: Optional[int],
        execution_mode: str,
        max_attempts: int,
        runtime: str,
        backend: Optional[AgentBackend] = None,
        scene_keyword: Optional[str] = None,
        scene_serial: Optional[int] = None,
        prepare_handoff: bool = False,
        handoff_dir: Optional[Path] = None,
        handoff_duration: int = 20,
        force_regenerate: bool = False,
    ) -> None:
        self.base_dir = Path(__file__).parent.resolve()
        self.excel_path = self.base_dir / "Keyword Prompt Verification.xlsx"
        self.prompt_template_path = self.base_dir / "prompt.txt"

        self.prompts_dir = self.base_dir / "generated_prompts"
        self.code_dir = self.base_dir / "generated_code"
        self.scenes_dir = self.base_dir / "scenes"

        self.prompts_dir.mkdir(parents=True, exist_ok=True)
        self.code_dir.mkdir(parents=True, exist_ok=True)
        self.scenes_dir.mkdir(parents=True, exist_ok=True)

        self.test_mode = test_mode
        self.generate_only = generate_only
        self.scenario_limit = scenario_limit
        self.execution_mode = execution_mode
        self.max_attempts = max_attempts
        self.scene_keyword_filter = scene_keyword
        self.scene_serial = scene_serial
        self.prepare_handoff = prepare_handoff
        self.handoff_dir = (handoff_dir or (self.base_dir / "handoffs")).resolve()
        self.handoff_duration = int(handoff_duration)
        self.force_regenerate = force_regenerate

        self.runtime_requested = runtime
        self.runtime_mode = resolve_runtime_mode(runtime)

        self.backend = backend

        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.intervention_report_file = self.base_dir / f"intervention_report_{self.run_id}.json"
        self.intervention_summary_csv = self.base_dir / "intervention_summary.csv"
        self.scenario_intervention_csv = self.base_dir / "scenario_intervention_data.csv"

        self.prompt_template = self._load_prompt_template()

    def _load_prompt_template(self) -> str:
        if not self.prompt_template_path.exists():
            return "{SCENE_PROMPT}"
        return self.prompt_template_path.read_text(encoding="utf-8")

    @staticmethod
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

    @staticmethod
    def generate_scenario_keyword(keyword: str) -> str:
        scenario_keyword = keyword.lower().replace(" ", "_")
        scenario_keyword = "".join(ch for ch in scenario_keyword if ch.isalnum() or ch == "_")
        return scenario_keyword

    def read_unique_scenes_from_excel(self) -> List[Dict[str, Any]]:
        scenes = read_scenes_excel_shared(self.excel_path)
        for scene in scenes:
            scene["scenario_keyword"] = self.generate_scenario_keyword(scene["keyword"])
        return scenes

    def _select_scenes(self, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        selected = list(scenes)

        if self.scene_keyword_filter:
            needle = self.generate_scenario_keyword(self.scene_keyword_filter)
            matched = [
                scene
                for scene in selected
                if needle in scene["scenario_keyword"]
                or needle in self.generate_scenario_keyword(scene["keyword"])
            ]
            if len(matched) != 1 and self.generate_only:
                raise ValueError(
                    f"--scene-keyword must match exactly one scenario in generate-only mode; matched {len(matched)}"
                )
            selected = matched

        if self.scene_serial is not None:
            if not selected:
                raise ValueError("No scenes available for --scene-serial selection")
            if self.scene_serial <= 0:
                raise ValueError("--scene-serial must be >= 1")

            normalized = ((self.scene_serial - 1) % len(selected)) + 1
            scene = dict(selected[normalized - 1])
            scene["requested_serial"] = self.scene_serial
            scene["normalized_serial"] = normalized
            scene["serial_wrapped"] = normalized != self.scene_serial
            selected = [scene]

        if self.test_mode:
            selected = selected[:2]
        elif self.scenario_limit is not None:
            selected = selected[: self.scenario_limit]

        return selected

    def _shot_paths(self, scenario_keyword: str, shot_index: int = 0) -> Dict[str, Path]:
        prompt_dir = self.prompts_dir / scenario_keyword / self.run_id / f"shot_{shot_index}"
        code_dir = self.code_dir / scenario_keyword / self.run_id / f"shot_{shot_index}"
        output_dir = self.scenes_dir / scenario_keyword / self.run_id / f"shot_{shot_index}"

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

    def _latest_seed_code(self, scenario_keyword: str) -> Optional[Path]:
        candidates = list((self.code_dir / scenario_keyword).glob(f"**/{scenario_keyword}.py"))
        if not candidates:
            legacy = self.code_dir / f"{scenario_keyword}.py"
            return legacy if legacy.exists() else None
        return max(candidates, key=lambda path: path.stat().st_mtime)

    def _build_fallback_code(self, scenario_keyword: str) -> str:
        is_highway = any(
            token in scenario_keyword
            for token in ("highway", "freeway", "expressway", "underpass", "tunnel")
        )
        default_town = "Town04" if is_highway else "Town03"
        return f"""#!/usr/bin/env python3

import argparse
import random
import sys
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import carla
except ImportError:
    print("[ERROR] CARLA Python API not found.")
    sys.exit(1)


CAMERA_KEYS = ["front"]
images_received = {{"front": None}}
images_lock = threading.Lock()
frame_ready = threading.Event()


def log(message: str, handle) -> None:
    line = str(message).rstrip()
    print(line)
    try:
        handle.write(line + "\\n")
        handle.flush()
    except Exception:
        pass


def save_image_to_disk(image, output_path: Path) -> bool:
    try:
        if image is None:
            return False
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]
        Image.fromarray(array).save(output_path, "PNG")
        return True
    except Exception as err:
        print(f"[ERROR] Failed to save image: {{err}}")
        return False


def make_camera_callback(camera_name: str):
    def callback(image):
        with images_lock:
            images_received[camera_name] = image
            frame_ready.set()
    return callback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fallback CARLA scene scaffold for {scenario_keyword}")
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output-dir", default="./scenes/output")
    parser.add_argument("--time-preset", default="noon")
    parser.add_argument("--weather-preset", default="clear")
    parser.add_argument("--sun-altitude", type=float, default=None)
    parser.add_argument("--sun-azimuth", type=float, default=None)
    parser.add_argument("--streetlights", default="off")
    parser.add_argument("--cloudiness", type=float, default=10.0)
    parser.add_argument("--precipitation", type=float, default=0.0)
    parser.add_argument("--precipitation-deposits", type=float, default=0.0)
    parser.add_argument("--wind-intensity", type=float, default=5.0)
    parser.add_argument("--fog-density", type=float, default=0.0)
    parser.add_argument("--fog-distance", type=float, default=120.0)
    parser.add_argument("--wetness", type=float, default=0.0)
    return parser.parse_args()


def pick_ego_blueprint(bp_lib):
    try:
        return bp_lib.find("vehicle.tesla.model3")
    except Exception:
        vehicles = bp_lib.filter("vehicle.*")
        if not vehicles:
            raise RuntimeError("No vehicle blueprints available for ego spawn")
        return random.choice(vehicles)


def choose_spawn_points(world):
    spawn_points = world.get_map().get_spawn_points()
    if not spawn_points:
        raise RuntimeError("No spawn points available")
    return spawn_points


def apply_environment(world, args) -> None:
    weather = world.get_weather()
    weather.cloudiness = float(args.cloudiness)
    weather.precipitation = float(args.precipitation)
    weather.precipitation_deposits = float(args.precipitation_deposits)
    weather.wind_intensity = float(args.wind_intensity)
    weather.fog_density = float(args.fog_density)
    weather.fog_distance = float(args.fog_distance)
    weather.wetness = float(args.wetness)
    if args.sun_altitude is not None:
        weather.sun_altitude_angle = float(args.sun_altitude)
    elif str(args.time_preset).lower() == "noon":
        weather.sun_altitude_angle = 75.0
    if args.sun_azimuth is not None:
        weather.sun_azimuth_angle = float(args.sun_azimuth)
    world.set_weather(weather)


def spawn_background_traffic(world, traffic_manager, ego_spawn, handle):
    actors = []
    spawn_points = choose_spawn_points(world)
    bp_lib = world.get_blueprint_library()
    vehicle_bps = [bp for bp in bp_lib.filter("vehicle.*") if bp.has_attribute("number_of_wheels")]
    random.shuffle(spawn_points)
    target = min(28, max(12, len(spawn_points) // 3))
    for spawn in spawn_points:
        if len(actors) >= target:
            break
        if spawn.location.distance(ego_spawn.location) < 12.0:
            continue
        bp = random.choice(vehicle_bps)
        if bp.has_attribute("color"):
            bp.set_attribute("color", random.choice(bp.get_attribute("color").recommended_values))
        if bp.has_attribute("driver_id"):
            bp.set_attribute("driver_id", random.choice(bp.get_attribute("driver_id").recommended_values))
        actor = world.try_spawn_actor(bp, spawn)
        if actor is None:
            continue
        actor.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.vehicle_percentage_speed_difference(actor, random.uniform(-35.0, -5.0))
        actors.append(actor)
    log(f"[INFO] Spawned {{len(actors)}} background vehicles", handle)
    return actors


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    front_dir = output_dir / "front"
    front_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "{scenario_keyword}_simulation.log"

    actors = []
    sensors = []
    original_settings = None

    with log_path.open("w", encoding="utf-8") as log_handle:
        log("[INFO] Starting fallback scaffold for {scenario_keyword}", log_handle)
        log("[INFO] This scaffold preserves Stage 1/Wine readiness while awaiting scene-specific refinement.", log_handle)
        client = carla.Client("localhost", 2000)
        client.set_timeout(60.0)
        world = None

        try:
            world = client.load_world("{default_town}")
            world.tick()
            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(2.5)
            traffic_manager.global_percentage_speed_difference(-10.0)

            original_settings = world.get_settings()
            settings = world.get_settings()
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.05
            world.apply_settings(settings)

            apply_environment(world, args)

            spawn_points = choose_spawn_points(world)
            ego_spawn = spawn_points[0]
            ego_bp = pick_ego_blueprint(world.get_blueprint_library())
            ego_vehicle = world.try_spawn_actor(ego_bp, ego_spawn)
            if ego_vehicle is None:
                for spawn in spawn_points[1:]:
                    ego_vehicle = world.try_spawn_actor(ego_bp, spawn)
                    if ego_vehicle is not None:
                        ego_spawn = spawn
                        break
            if ego_vehicle is None:
                raise RuntimeError("Failed to spawn ego vehicle")
            actors.append(ego_vehicle)
            ego_vehicle.set_autopilot(True, traffic_manager.get_port())

            camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
            camera_bp.set_attribute("image_size_x", "1280")
            camera_bp.set_attribute("image_size_y", "720")
            camera_bp.set_attribute("fov", "110")
            camera_transform = carla.Transform(
                carla.Location(x=0.8, y=0.0, z=1.4),
                carla.Rotation(pitch=8.0),
            )
            camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
            sensors.append(camera)
            camera.listen(make_camera_callback("front"))

            actors.extend(spawn_background_traffic(world, traffic_manager, ego_spawn, log_handle))

            for _ in range(20):
                world.tick()
            time.sleep(0.25)

            max_frames = max(1, int(float(args.duration) * 20))
            for frame_index in range(1, max_frames + 1):
                world.tick()
                if frame_ready.wait(timeout=1.0):
                    with images_lock:
                        image = images_received["front"]
                        images_received["front"] = None
                    frame_ready.clear()
                    save_image_to_disk(image, front_dir / f"front_frame_{{frame_index:08d}}.png")
                else:
                    log(f"[WARNING] Camera timeout at frame {{frame_index}}", log_handle)

            log("[INFO] Completed fallback scaffold run", log_handle)
        finally:
            for sensor in sensors:
                try:
                    sensor.stop()
                except Exception:
                    pass
            for actor in reversed(actors):
                try:
                    actor.destroy()
                except Exception:
                    pass
            try:
                if original_settings is not None and world is not None:
                    world.apply_settings(original_settings)
            except Exception:
                pass


if __name__ == "__main__":
    main()
"""

    def _generate_shot(self, scene: Dict[str, Any]) -> Tuple[Dict[str, Any], int, str]:
        scenario_keyword = str(scene["scenario_keyword"])
        paths = self._shot_paths(scenario_keyword, shot_index=0)
        llm_generation_count = 0

        original_prompt = str(scene["prompt"])
        scene_specifications = normalize_scene_specifications_for_generation(
            str(scene.get("scene_specifications", "") or ""),
            scene_prompt=original_prompt,
        )
        enhanced_prompt = original_prompt
        full_prompt = self.build_full_prompt(
            template=self.prompt_template,
            scenario_keyword=scenario_keyword,
            scene_prompt=enhanced_prompt,
            scene_specifications=scene_specifications,
        )

        code_seed_source = ""
        status = "generated"
        generated_code = ""

        if not self.force_regenerate:
            latest = self._latest_seed_code(scenario_keyword)
            if latest and latest.exists():
                shutil.copy2(latest, paths["code_file"])
                generated_code = paths["code_file"].read_text(encoding="utf-8", errors="ignore")
                code_seed_source = str(latest.resolve())
                status = "prepared_from_existing"

        if not generated_code:
            if self.backend is not None:
                full_prompt = self.build_full_prompt(
                    template=self.prompt_template,
                    scenario_keyword=scenario_keyword,
                    scene_prompt=enhanced_prompt,
                    scene_specifications=scene_specifications,
                )

                try:
                    generated_code = self.backend.generate_code(
                        full_prompt=full_prompt,
                        base_max_tokens=8000,
                        retry_max_tokens=12000,
                        context_label=f"{scenario_keyword} shot_0",
                    )
                    llm_generation_count += 1
                except AgentBackendError:
                    latest = self._latest_seed_code(scenario_keyword)
                    if latest and latest.exists():
                        shutil.copy2(latest, paths["code_file"])
                        generated_code = paths["code_file"].read_text(encoding="utf-8", errors="ignore")
                        code_seed_source = str(latest.resolve())
                        status = "prepared_from_existing"

            if not generated_code:
                generated_code = self._build_fallback_code(scenario_keyword)
                status = "pending_agent_code"

            paths["code_file"].write_text(generated_code, encoding="utf-8")

        paths["prompt_file"].write_text(
            (
                f"KEYWORD: {scene['keyword']}\n"
                f"SCENARIO_KEYWORD: {scenario_keyword}\n"
                "SHOT_INDEX: 0\n\n"
                f"ORIGINAL PROMPT:\n{original_prompt}\n\n"
                f"SCENE SPECIFICATIONS:\n{scene_specifications}\n\n"
                f"ENHANCED PROMPT:\n{enhanced_prompt}\n"
            ),
            encoding="utf-8",
        )
        paths["full_prompt_file"].write_text(full_prompt, encoding="utf-8")

        shot = ShotArtifacts(
            shot_index=0,
            prompt_text=enhanced_prompt,
            user_adjustment_prompt="",
            prompt_file=str(paths["prompt_file"].resolve()),
            full_prompt_file=str(paths["full_prompt_file"].resolve()),
            code_file=str(paths["code_file"].resolve()),
            scene_output_dir=str(paths["output_dir"].resolve()),
            sim_success="",
            frames_ok="",
            accepted_by_user="",
            status=status,
            timestamp=datetime.now().isoformat(),
        )

        return shot.__dict__, llm_generation_count, code_seed_source

    def _write_handoff(self, scene: Dict[str, Any], shot: Dict[str, Any]) -> Path:
        from carla_wine_bridge import build_wine_runtime_config, posix_to_windows_path

        runtime = build_wine_runtime_config()
        code_path = Path(shot["code_file"]).resolve()
        output_path = Path(shot["scene_output_dir"]).resolve()

        handoff_run_dir = self.handoff_dir / self.run_id / str(scene["scenario_keyword"])
        handoff_run_dir.mkdir(parents=True, exist_ok=True)

        manifest_path = handoff_run_dir / "manifest.json"
        manifest = {
            "run_id": self.run_id,
            "scene_keyword": scene["keyword"],
            "scenario_keyword": scene["scenario_keyword"],
            "scene_serial": scene.get("normalized_serial", scene.get("serial")),
            "shot_index": int(shot.get("shot_index", 0)),
            "code_file_posix": str(code_path),
            "code_file_windows": posix_to_windows_path(code_path, runtime.wineprefix),
            "output_dir_posix": str(output_path),
            "output_dir_windows": posix_to_windows_path(output_path, runtime.wineprefix),
            "duration_seconds": int(self.handoff_duration),
            "prompt_file": shot.get("prompt_file"),
            "full_prompt_file": shot.get("full_prompt_file"),
            "generated_at": datetime.now().isoformat(),
            "generation_mode": "agentic_scene_generator",
            "generation_status": shot.get("status"),
            "scene_prompt_original": scene.get("prompt"),
            "scene_specifications": str(scene.get("scene_specifications", "") or ""),
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        next_steps = handoff_run_dir / "NEXT_STEPS.md"
        next_steps.write_text(
            "\n".join(
                [
                    f"# Next Steps: {scene['scenario_keyword']}",
                    "",
                    "1. (Wine CMD) Run stage 2:",
                    '   cd "C:\\Program Files\\WindowsNoEditor\\VLM-AV"',
                    "   02_run_latest_scene.cmd",
                    "",
                    "2. (macOS) Run stage 3 evaluation:",
                    "   ./03_evaluate_latest.sh",
                    "",
                ]
            ),
            encoding="utf-8",
        )

        latest_posix = self.handoff_dir / "LATEST_MANIFEST_POSIX.txt"
        latest_windows = self.handoff_dir / "LATEST_MANIFEST_WINDOWS.txt"
        latest_posix.parent.mkdir(parents=True, exist_ok=True)
        latest_posix.write_text(str(manifest_path.resolve()), encoding="utf-8")
        latest_windows.write_text(
            posix_to_windows_path(manifest_path.resolve(), runtime.wineprefix),
            encoding="utf-8",
        )

        return manifest_path

    def _append_summary_row(self, record: Dict[str, Any]) -> None:
        self.intervention_summary_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = self.intervention_summary_csv.exists()
        with self.intervention_summary_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "timestamp",
                    "run_id",
                    "scenario_keyword",
                    "scene_keyword",
                    "status",
                    "intervention_count",
                    "llm_generation_count",
                ],
            )
            if not exists:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": record["timestamp"],
                    "run_id": record["run_id"],
                    "scenario_keyword": record["scenario_keyword"],
                    "scene_keyword": record["scene_keyword"],
                    "status": record["final_status"],
                    "intervention_count": record["intervention_count"],
                    "llm_generation_count": record["llm_generation_count"],
                }
            )

    def _append_scenario_row(self, record: Dict[str, Any]) -> None:
        self.scenario_intervention_csv.parent.mkdir(parents=True, exist_ok=True)
        exists = self.scenario_intervention_csv.exists()
        with self.scenario_intervention_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "run_id",
                    "scenario_keyword",
                    "scene_keyword",
                    "tested",
                    "final_success",
                    "intervention_count",
                    "llm_generation_count",
                    "shot_0_prompt_text",
                    "shot_0_code_file",
                    "shot_0_scene_output_dir",
                    "shot_0_status",
                ],
            )
            if not exists:
                writer.writeheader()
            shot0 = (record.get("shot_artifacts") or [{}])[0]
            writer.writerow(
                {
                    "run_id": record.get("run_id"),
                    "scenario_keyword": record.get("scenario_keyword"),
                    "scene_keyword": record.get("scene_keyword"),
                    "tested": record.get("tested"),
                    "final_success": record.get("final_success"),
                    "intervention_count": record.get("intervention_count"),
                    "llm_generation_count": record.get("llm_generation_count"),
                    "shot_0_prompt_text": shot0.get("prompt_text"),
                    "shot_0_code_file": shot0.get("code_file"),
                    "shot_0_scene_output_dir": shot0.get("scene_output_dir"),
                    "shot_0_status": shot0.get("status"),
                }
            )

    def run_simulation(
        self,
        code_file: Path,
        output_dir: Path,
        duration_seconds: int,
    ) -> Tuple[bool, Optional[str]]:
        """Run one generated script through selected runtime mode."""
        output_flag = "--output"
        try:
            code_text = code_file.read_text(encoding="utf-8", errors="ignore")
            if "--output-dir" in code_text:
                output_flag = "--output-dir"
        except Exception:
            output_flag = "--output"

        timeout = int(duration_seconds) + 120

        try:
            result = run_python_script(
                script_path=code_file,
                script_args=["--duration", str(int(duration_seconds)), output_flag, str(output_dir)],
                runtime_mode=self.runtime_mode,
                timeout=timeout,
                capture_output=True,
                text=True,
                cwd=self.base_dir,
            )
        except subprocess.TimeoutExpired:
            return False, f"Timeout after {timeout}s"
        except Exception as err:
            return False, f"Execution error: {type(err).__name__}: {err}"

        if int(result.returncode) == 0:
            return True, None

        message = f"Exit {result.returncode}"
        if result.stderr:
            message += f"\nSTDERR:\n{result.stderr}"
        if result.stdout:
            message += f"\nSTDOUT:\n{result.stdout}"
        return False, message

    def verify_frames(self, scene_output_dir: Path, duration_seconds: int, min_success_ratio: float = 0.5) -> bool:
        expected_frames = max(1, int(duration_seconds) * 20)
        camera_views = ["front", "front_left", "front_right", "rear"]

        counts: Dict[str, int] = {}
        for view in camera_views:
            view_dir = scene_output_dir / view
            if view_dir.exists():
                counts[view] = len(list(view_dir.glob(f"{view}_frame_*.png")))

        if counts:
            best_stream = max(counts.values())
            return (best_stream / expected_frames) >= min_success_ratio

        frame_files = [*scene_output_dir.glob("*.png"), *scene_output_dir.rglob("*.png")]
        return (len(frame_files) / expected_frames) >= min_success_ratio

    def process_scene(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        shot, llm_count, code_seed_source = self._generate_shot(scene)
        scenario_keyword = str(scene["scenario_keyword"])

        final_success = True
        final_status = "READY"
        runtime_error: Optional[str] = None
        frames_ok: Optional[bool] = None

        if not self.generate_only:
            sim_success, runtime_error = self.run_simulation(
                code_file=Path(shot["code_file"]),
                output_dir=Path(shot["scene_output_dir"]),
                duration_seconds=self.handoff_duration,
            )
            frames_ok = self.verify_frames(Path(shot["scene_output_dir"]), self.handoff_duration)
            final_success = bool(sim_success or frames_ok)
            final_status = "SUCCESS" if final_success else "FAILED"
            shot["sim_success"] = sim_success
            shot["frames_ok"] = frames_ok
            shot["status"] = "passed" if final_success else "failed_runtime_or_frames"

        record = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "execution_mode": self.execution_mode,
            "scene_keyword": scene["keyword"],
            "scenario_keyword": scenario_keyword,
            "scene_serial": scene.get("normalized_serial", scene.get("serial")),
            "tested": not self.generate_only,
            "excluded_from_tested": False,
            "intervention_count": 0,
            "fix_rounds": 0,
            "llm_generation_count": llm_count,
            "sim_runs": 0 if self.generate_only else 1,
            "final_success": final_success,
            "final_status": final_status,
            "runtime_error": runtime_error,
            "frames_ok": frames_ok,
            "code_seed_source": code_seed_source,
            "shot_artifacts": [shot],
        }

        if self.prepare_handoff:
            manifest_path = self._write_handoff(scene, shot)
            record["handoff_manifest"] = str(manifest_path)

        self._append_summary_row(record)
        self._append_scenario_row(record)
        return record

    def run(self) -> Dict[str, Any]:
        scenes = self._select_scenes(self.read_unique_scenes_from_excel())
        records = [self.process_scene(scene) for scene in scenes]

        payload = {
            "run_id": self.run_id,
            "timestamp": datetime.now().isoformat(),
            "mode": "generate-only" if self.generate_only else ("test" if self.test_mode else "full"),
            "execution_mode": self.execution_mode,
            "runtime_requested": self.runtime_requested,
            "runtime_mode": self.runtime_mode,
            "records": records,
            "totals": {
                "scenes": len(records),
                "success": sum(1 for row in records if row.get("final_success")),
                "fail": sum(1 for row in records if not row.get("final_success")),
            },
        }
        self.intervention_report_file.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Agentic CARLA scene generator")
    parser.add_argument("--mode", choices=MODE_CHOICES, default="generate-only")
    parser.add_argument("--execution-mode", choices=EXECUTION_CHOICES, default="interactive")
    parser.add_argument("--scenario-limit", default=None, help="Integer or 'all'")
    parser.add_argument("--runtime", choices=RUNTIME_CHOICES, default="auto")
    parser.add_argument("--max-attempts", type=int, default=3)

    parser.add_argument("--scene-keyword", help="Process exactly one scene by keyword")
    parser.add_argument("--scene-serial", type=int, help="Process one scene by serial index")
    parser.add_argument("--force-regenerate", action="store_true")

    parser.add_argument("--prepare-handoff", action="store_true")
    parser.add_argument("--handoff-dir", default=str(Path(__file__).parent / "handoffs"))
    parser.add_argument("--handoff-duration", type=int, default=20)

    parser.add_argument("--backend", choices=BACKEND_CHOICES, default="none")
    parser.add_argument("--model", default=None)
    parser.add_argument("--yes", action="store_true", help="Run non-interactively")
    return parser.parse_args()


def _parse_scenario_limit(raw: Optional[str]) -> Optional[int]:
    if raw is None:
        return None
    if str(raw).strip().lower() == "all":
        return None
    value = int(raw)
    if value <= 0:
        raise ValueError("--scenario-limit must be > 0 or 'all'")
    return value


def _build_backend(args: argparse.Namespace, workspace_dir: Path) -> Optional[AgentBackend]:
    if args.backend == "none":
        return None
    return CodexCliBackend(workspace_dir=workspace_dir, model=args.model)


def main() -> int:
    args = parse_args()

    scenario_limit = _parse_scenario_limit(args.scenario_limit)
    mode = args.mode
    test_mode = mode == "test"
    generate_only = mode == "generate-only"

    if mode == "test" and scenario_limit is None:
        scenario_limit = 2

    backend = _build_backend(args, Path(__file__).parent.resolve())

    generator = AgenticSceneGenerator(
        test_mode=test_mode,
        generate_only=generate_only,
        scenario_limit=scenario_limit,
        execution_mode=args.execution_mode,
        max_attempts=args.max_attempts,
        runtime=args.runtime,
        backend=backend,
        scene_keyword=args.scene_keyword,
        scene_serial=args.scene_serial,
        prepare_handoff=bool(args.prepare_handoff),
        handoff_dir=Path(args.handoff_dir),
        handoff_duration=int(args.handoff_duration),
        force_regenerate=bool(args.force_regenerate),
    )

    result = generator.run()
    print(json.dumps({
        "run_id": result["run_id"],
        "scenes": result["totals"]["scenes"],
        "success": result["totals"]["success"],
        "fail": result["totals"]["fail"],
        "intervention_report": str(generator.intervention_report_file),
    }, indent=2))
    return 0 if result["totals"]["fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
