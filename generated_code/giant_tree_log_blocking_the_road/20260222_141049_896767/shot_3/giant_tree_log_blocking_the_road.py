#!/usr/bin/env python3
"""Standalone CARLA scenario script (V3 mesh-compliance sync).

Version: V3
Scene: Giant Tree Log Blocking the Road
Scenario Keyword: giant_tree_log_blocking_the_road
Prompt: Generate Carla PythonAPI code for a suburban road where a massive fallen tree log completely blocks one lane. Vehicles must brake, slow, or swerve into the opposite lane (if safe) to continue forward. Include roadside vegetation, traffic flow, and hazard placement with proper collision behavior.

Scene Specifications:
Road Context: Signalized urban 4-way intersection baseline, two lanes per direction, with active cross-traffic in all approaches.
Traffic Density: Minimum traffic density: >=20 moving vehicles per approach direction at event onset, with >=80 active vehicles in broader scene. CRITICAL: Spawn majority of vehicles NEAR ego vehicle location (within 100m radius), not randomly across map. Vehicles must be visible behind, beside, and in oncoming lanes near ego.
Camera Contract: Save synchronized front, front_left, front_right, rear, and drone_follow streams using matched frame ids.
Event Contract: Use static.prop.beech_tree blueprint to create a REAL fallen tree across the lane. Spawn the tree rotated 90 degrees (roll=90) so it lies horizontally across the road like a fallen trunk. Add multiple beech_tree props end-to-end for a massive log appearance. Surround with debris (dirtdebris, plantpot, bush props). Add warning cones on approach side.
Success Criteria: Fallen tree is clearly recognizable as a tree (not barrels/boxes). Blockage spans full lane width. Ego and nearby traffic visibly brake and adjust path. Dense traffic visible around ego from all directions.

V3 Mesh Compliance Notes:
- Uses `static.prop.mesh` / `mesh_path` for primary custom-object staging where relevant.
- Includes reference breadcrumb to SCENE_GENERATION_REFERENCE.md §9.
- Source manifest (seed lineage): not included in public repo
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image


def _ensure_vlmav_project_root_on_syspath() -> None:
    script_path = Path(__file__).resolve()
    for parent in script_path.parents:
        if (parent / "scene_utils" / "time_weather_matrix.py").exists():
            parent_str = str(parent)
            if parent_str not in sys.path:
                sys.path.insert(0, parent_str)
            return


_ensure_vlmav_project_root_on_syspath()

from scene_utils.time_weather_matrix import (  # noqa: E402
    SCENE_DEFAULT_KEY,
    TimeWeatherSpecError,
    VALID_STREETLIGHT_MODES,
    canonical_time_key,
    canonical_weather_key,
    get_time_preset,
    get_weather_preset,
    load_time_weather_spec,
)

try:
    import carla
except Exception as exc:
    raise RuntimeError("CARLA Python API is required") from exc


FRAME_DT = 0.05
FPS = 20
# Default to front-camera-only capture unless multi-angle capture is explicitly requested.
CAMERA_KEYS = ["front"]
SCENARIO_KEYWORD = "giant_tree_log_blocking_the_road"
SCENE_KEYWORD = "Giant Tree Log Blocking the Road"
SCENE_PROMPT = "Generate Carla PythonAPI code for a suburban road where a massive fallen tree log completely blocks one lane. Vehicles must brake, slow, or swerve into the opposite lane (if safe) to continue forward. Include roadside vegetation, traffic flow, and hazard placement with proper collision behavior."
SCENE_SPECIFICATIONS = "Road Context: Tree-lined multilane bidirectional road corridor on a straight segment just after a 4-way intersection (ego and fallen tree are NOT inside the intersection). Prefer a corridor with visible opposite-direction traffic flow.\nTraffic Density: Minimum traffic density target remains >=20 moving vehicles per required direction when map capacity allows, with dense traffic concentrated near ego. Spawn vehicles relative to the ego after ego spawn, fill adjacent lanes beside ego, and keep visible oncoming traffic beyond the fallen tree on the opposite side.\nCamera Contract: Front camera only by default (`front`). Only generate 5-angle capture (`front`, `front_left`, `front_right`, `rear`, `drone_follow`) if explicitly requested.\nEvent Contract: Use `static.prop.mesh` with beech-tree meshes to create a REAL fallen tree/log that appears to fall from the sidewalk/curb sideways into the ego-side lanes. Place the fallen tree within 100m of ego (target ~90m). Do NOT place warning cones or barriers before the fallen tree.\nSuccess Criteria: Front camera clearly shows ego approaching a sideways fallen tree blocking ego-side lanes, dense adjacent traffic around ego, and visible oncoming traffic flowing on the opposite side beyond the tree."
VERSION_TAG = "V3"
COMPLIANCE_UPDATE_PROFILE = "mesh-staticmeshfactory-sync"
SOURCE_MANIFEST_POSIX = "not included in public repo"
SOURCE_RUN_ID = "20260220_123040_646317"
SOURCE_SHOT_INDEX = 2
EVENT_MODE = "fallen_tree_log"
ROAD_MODE = "local"
WEATHER_MODE = "clear"
SHOT_INDEX = 3
MESH_EVENT_HINT = "Fallen tree/log meshes spawned via static.prop.mesh with physics-enabled mass."
MIN_TOTAL_ACTIVE = 80
MIN_PER_DIRECTION = 20
SCENE_DEFAULT_ENV: Dict[str, float] = {}
MIN_SIDE_DENSITY = MIN_PER_DIRECTION * 3
TRAFFIC_SPEED_DIFF_RANGE = (-22.0, -6.0)
OPPOSITE_TRAFFIC_SPEED_DIFF_RANGE = (-55.0, -28.0)
TRAFFIC_FOLLOW_DISTANCE_RANGE = (1.4, 3.0)
EGO_SPEED_DIFF = 45.0
OPPOSITE_START_SPAWN_DISTANCE_M = 100.0
OPPOSITE_START_SPAWN_BAND_M = 18.0
OPPOSITE_START_SPAWN_TARGET = 14
ADJACENT_LANE_FILL_TARGET_PER_LANE = 10
ADJACENT_LANE_FILL_MAX_AHEAD_M = 140.0
ADJACENT_LANE_FILL_MAX_BEHIND_M = 35.0
EGO_RELATIVE_TRAFFIC_TARGET_PER_LANE = 14
EGO_RELATIVE_TRAFFIC_MAX_AHEAD_M = 180.0
EGO_RELATIVE_TRAFFIC_MAX_BEHIND_M = 55.0
POST_INTERSECTION_MIN_M = 8.0
POST_INTERSECTION_MAX_M = 70.0
FOUR_WAY_MIN_UNIQUE_ROADS = 4
OPPOSITE_TREE_SAME_ROAD_TARGET = 18
OPPOSITE_TREE_SAME_ROAD_AHEAD_MIN_M = 5.0
OPPOSITE_TREE_SAME_ROAD_AHEAD_MAX_M = 80.0
# Keep the fallen tree within the 100m requirement while giving a long approach.
TREE_EVENT_AHEAD_M = 90.0
TREE_MAX_VISIBLE_DISTANCE_M = 100.0
TREE_MESH_FALLEN_ORDER = ["SM_Beech", "SM_Oak_L_v1", "SM_Maple_L_v1", "SM_Pine_L_v1"]

images_received: Dict[str, Optional[carla.Image]] = {k: None for k in CAMERA_KEYS}
images_lock = threading.Lock()
frame_ready = threading.Event()


def log(msg: str, log_file) -> None:
    line = msg.rstrip()
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def save_image_to_disk(image: carla.Image, output_path: Path) -> bool:
    try:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]  # BGR to RGB
        Image.fromarray(array).save(output_path, "PNG")
        return True
    except Exception as exc:
        print(f"[ERROR] save_image_to_disk failed for {output_path}: {exc}")
        return False


def make_camera_callback(name: str):
    def _callback(image: carla.Image) -> None:
        with images_lock:
            images_received[name] = image
            if all(images_received[k] is not None for k in CAMERA_KEYS):
                frame_ready.set()
    return _callback


def reset_camera_buffers() -> None:
    with images_lock:
        for k in CAMERA_KEYS:
            images_received[k] = None
    frame_ready.clear()


def map_token(name: str) -> str:
    raw = str(name)
    return raw.rsplit('/', 1)[-1] if '/' in raw else raw


def is_retryable_carla_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(t in msg for t in ("time-out", "timeout", "failed to connect", "connection closed", "rpc", "socket"))


def safe_tick(world: carla.World, stage: str, log_file, retries: int = 3, fatal: bool = True) -> bool:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            world.tick()
            return True
        except RuntimeError as exc:
            last_error = exc
            if not is_retryable_carla_error(exc):
                raise
            log(f"[WARN] world.tick retry stage={stage} attempt={attempt}/{retries}: {exc}", log_file)
            time.sleep(min(0.5 * attempt, 1.5))
    if fatal and last_error is not None:
        raise RuntimeError(f"Persistent world.tick failure at stage={stage}") from last_error
    log(f"[WARN] Continuing after repeated tick failure stage={stage}", log_file)
    return False


def distance_2d(a: carla.Location, b: carla.Location) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def get_vegetation_bbox_centers(world: carla.World, log_file) -> List[carla.Location]:
    try:
        label = getattr(getattr(carla, "CityObjectLabel", None), "Vegetation", None)
        if label is None or not hasattr(world, "get_level_bbs"):
            return []
        boxes = list(world.get_level_bbs(label))
    except Exception as exc:
        log(f"[WARN] Vegetation bbox query unavailable: {exc}", log_file)
        return []
    centers: List[carla.Location] = []
    for bb in boxes:
        try:
            centers.append(bb.location)
        except Exception:
            pass
    return centers


def roadside_vegetation_score(spawn: carla.Transform, vegetation_centers: List[carla.Location]) -> float:
    if not vegetation_centers:
        return 0.0
    tf = spawn
    loc = tf.location
    fwd = tf.get_forward_vector()
    right = tf.get_right_vector()
    score = 0.0
    for vc in vegetation_centers:
        dx = vc.x - loc.x
        dy = vc.y - loc.y
        along = dx * fwd.x + dy * fwd.y
        side = abs(dx * right.x + dy * right.y)
        dist = math.sqrt(dx * dx + dy * dy)
        if dist > 95.0:
            continue
        if -25.0 <= along <= 110.0 and 6.0 <= side <= 34.0:
            # Reward visible roadside vegetation corridor on either side.
            score += 1.0 if along >= 0.0 else 0.35
    return score


def classify_direction(reference_loc: carla.Location, candidate_loc: carla.Location) -> str:
    dx = candidate_loc.x - reference_loc.x
    dy = candidate_loc.y - reference_loc.y
    if abs(dx) > abs(dy):
        return "east" if dx > 0.0 else "west"
    return "north" if dy > 0.0 else "south"


def gather_direction_counts(
    spawn_points: List[carla.Transform],
    center_location: carla.Location,
    min_dist: float = 18.0,
    max_dist: float = 140.0,
) -> Dict[str, int]:
    counts = {"north": 0, "south": 0, "east": 0, "west": 0}
    for spawn in spawn_points:
        dist = distance_2d(spawn.location, center_location)
        if min_dist <= dist <= max_dist:
            counts[classify_direction(center_location, spawn.location)] += 1
    return counts


def pick_best_intersection_center(
    spawn_points: List[carla.Transform],
    traffic_lights,
) -> Optional[carla.TrafficLight]:
    best_tl = None
    best_score = -1.0
    for tl in traffic_lights:
        try:
            loc = tl.get_location()
        except Exception:
            continue
        counts = gather_direction_counts(spawn_points, loc, min_dist=18.0, max_dist=140.0)
        score = (min(counts.values()) * 100.0) + sum(counts.values())
        if score > best_score:
            best_score = score
            best_tl = tl
    return best_tl


def pick_ego_spawn_facing_light(
    spawn_points: List[carla.Transform],
    target_tl: carla.TrafficLight,
) -> Optional[carla.Transform]:
    best_spawn = None
    best_score = -1.0
    tl_loc = target_tl.get_location()
    for spawn in spawn_points:
        dist = distance_2d(spawn.location, tl_loc)
        if dist < 35.0 or dist > 85.0:
            continue
        forward = spawn.rotation.get_forward_vector()
        to_light = tl_loc - spawn.location
        planar = math.sqrt((to_light.x * to_light.x) + (to_light.y * to_light.y))
        if planar < 1e-3:
            continue
        to_light_norm = carla.Vector3D(to_light.x / planar, to_light.y / planar, 0.0)
        alignment = (forward.x * to_light_norm.x) + (forward.y * to_light_norm.y)
        if alignment < 0.78:
            continue
        score = (alignment * 2.0) - (abs(dist - 55.0) / 90.0)
        if score > best_score:
            best_score = score
            best_spawn = spawn
    return best_spawn


def choose_map(client: carla.Client, log_file) -> str:
    preferred = ["Town04", "Town05", "Town06"] if ROAD_MODE == "highway" else ["Town03", "Town05", "Town10HD"]
    try:
        available = [map_token(v) for v in client.get_available_maps()]
    except Exception as exc:
        log(f"[WARN] get_available_maps failed: {exc}; using {preferred[0]}", log_file)
        return preferred[0]
    lower = {m.lower(): m for m in available}
    for p in preferred:
        if p.lower() in lower:
            return lower[p.lower()]
        for a in available:
            if a.lower().startswith(p.lower()):
                return a
    return available[0] if available else preferred[0]


def load_world_with_retry(client: carla.Client, target_map: str, log_file) -> carla.World:
    last_error: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            client.set_timeout(120.0)
            log(f"[INFO] Loading map {target_map} attempt={attempt}/4", log_file)
            return client.load_world(target_map)
        except RuntimeError as exc:
            last_error = exc
            if not is_retryable_carla_error(exc):
                raise
            log(f"[WARN] load_world retryable error attempt={attempt}/4: {exc}", log_file)
            time.sleep(min(1.0 * attempt, 4.0))
            try:
                probe = client.get_world()
                if map_token(probe.get_map().name).lower() == target_map.lower():
                    log(f"[INFO] Active map already switched to {target_map}", log_file)
                    return probe
            except Exception:
                pass
    if last_error is not None:
        raise RuntimeError(f"Failed to load map {target_map}") from last_error
    raise RuntimeError(f"Failed to load map {target_map}")


def apply_sync_settings(world: carla.World, client: carla.Client, log_file) -> carla.WorldSettings:
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FRAME_DT
    try:
        settings.max_substep_delta_time = 0.01
        settings.max_substeps = 10
    except Exception:
        pass
    for attempt in range(1, 5):
        try:
            client.set_timeout(60.0)
            world.apply_settings(settings)
            return original
        except RuntimeError as exc:
            if not is_retryable_carla_error(exc):
                raise
            log(f"[WARN] apply_settings retry attempt={attempt}/4: {exc}", log_file)
            time.sleep(min(0.8 * attempt, 3.0))
    raise RuntimeError("Failed to enable synchronous mode")


def find_blueprint(bp_lib: carla.BlueprintLibrary, preferred_ids: List[str], fallback_pattern: str) -> carla.ActorBlueprint:
    for bp_id in preferred_ids:
        try:
            return bp_lib.find(bp_id)
        except Exception:
            continue
    choices = list(bp_lib.filter(fallback_pattern))
    if not choices:
        raise RuntimeError(f"No blueprint for pattern {fallback_pattern}")
    return random.choice(choices)


def set_vehicle_attributes(bp: carla.ActorBlueprint, role_name: str = "autopilot") -> None:
    if bp.has_attribute('role_name'):
        bp.set_attribute('role_name', role_name)
    if bp.has_attribute('color'):
        colors = bp.get_attribute('color').recommended_values
        if colors:
            bp.set_attribute('color', random.choice(colors))
    if bp.has_attribute('driver_id'):
        ids = bp.get_attribute('driver_id').recommended_values
        if ids:
            bp.set_attribute('driver_id', random.choice(ids))


# Mesh helpers per SCENE_GENERATION_REFERENCE.md §9 (StaticMeshFactory / static.prop.mesh)
def mesh_content_path(category: str, subcategory: str, mesh_name: str) -> str:
    return f"/Game/Carla/Static/{category}/{subcategory}/{mesh_name}.{mesh_name}"


def spawn_static_mesh(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    mesh_path: str,
    transform: carla.Transform,
    actors_to_cleanup: List[carla.Actor],
    log_file,
    mass: Optional[float] = None,
    label: str = "mesh",
) -> Optional[carla.Actor]:
    try:
        bp = bp_lib.find('static.prop.mesh')
    except Exception as exc:
        log(f"[MESH][FAIL] static.prop.mesh unavailable label={label}: {exc}", log_file)
        return None

    try:
        bp.set_attribute('mesh_path', mesh_path)
        if mass is not None:
            bp.set_attribute('mass', str(float(mass)))
        actor = world.try_spawn_actor(bp, transform)
        if actor is None:
            log(f"[MESH][FAIL] spawn returned None label={label} mesh_path={mesh_path}", log_file)
            return None
        actors_to_cleanup.append(actor)
        log(f"[MESH][OK] label={label} mesh_path={mesh_path} actor_id={actor.id}", log_file)
        return actor
    except Exception as exc:
        log(f"[MESH][FAIL] exception label={label} mesh_path={mesh_path}: {exc}", log_file)
        return None


def spawn_static_mesh_with_fallback(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    mesh_path: str,
    fallback_patterns: List[str],
    transform: carla.Transform,
    actors_to_cleanup: List[carla.Actor],
    log_file,
    mass: Optional[float] = None,
    label: str = 'mesh_fallback',
) -> Optional[carla.Actor]:
    actor = spawn_static_mesh(world, bp_lib, mesh_path, transform, actors_to_cleanup, log_file, mass=mass, label=label)
    if actor is not None:
        return actor

    for pattern in fallback_patterns:
        try:
            choices = list(bp_lib.filter(pattern))
        except Exception:
            choices = []
        if not choices:
            continue
        try:
            fb = random.choice(choices)
            actor = world.try_spawn_actor(fb, transform)
            if actor is not None:
                actors_to_cleanup.append(actor)
                log(f"[MESH][FALLBACK] label={label} pattern={pattern} actor_id={actor.id}", log_file)
                return actor
        except Exception as exc:
            log(f"[MESH][FALLBACK-FAIL] label={label} pattern={pattern}: {exc}", log_file)
    return None


def collect_density(world: carla.World, ego: carla.Vehicle, radius_m: float = 260.0) -> Dict[str, int]:
    counts = {'ahead': 0, 'behind': 0, 'left': 0, 'right': 0, 'same': 0, 'opposite': 0, 'total': 0, 'distinct': 0}
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    distinct = set()
    for actor in world.get_actors().filter('vehicle.*'):
        if actor.id == ego.id:
            continue
        loc = actor.get_location()
        if distance_2d(loc, ego_loc) > radius_m:
            continue
        dx = loc.x - ego_loc.x
        dy = loc.y - ego_loc.y
        local_fwd = dx * fwd.x + dy * fwd.y
        local_side = -dx * fwd.y + dy * fwd.x
        if abs(local_fwd) >= abs(local_side):
            counts['ahead' if local_fwd >= 0 else 'behind'] += 1
        else:
            counts['right' if local_side >= 0 else 'left'] += 1
        of = actor.get_transform().get_forward_vector()
        dot = fwd.x * of.x + fwd.y * of.y + fwd.z * of.z
        counts['same' if dot >= 0 else 'opposite'] += 1
        counts['total'] += 1
        distinct.add(actor.type_id)
    counts['distinct'] = len(distinct)
    return counts


def spawn_batch_vehicles(
    client,
    world,
    traffic_manager,
    blueprints,
    transforms,
    actors_to_cleanup,
    ego_forward: Optional[carla.Vector3D] = None,
) -> int:
    if not transforms:
        return 0
    batch = []
    port = traffic_manager.get_port()
    for tf in transforms:
        bp = random.choice(blueprints)
        set_vehicle_attributes(bp, 'autopilot')
        batch.append(carla.command.SpawnActor(bp, tf).then(carla.command.SetAutopilot(carla.command.FutureActor, True, port)))
    responses = client.apply_batch_sync(batch, True)
    actor_ids = [r.actor_id for r in responses if not r.error]
    if not actor_ids:
        return 0
    for actor in world.get_actors(actor_ids):
        actors_to_cleanup.append(actor)
        try:
            traffic_manager.auto_lane_change(actor, True)
            speed_range = TRAFFIC_SPEED_DIFF_RANGE
            if ego_forward is not None:
                try:
                    af = actor.get_transform().get_forward_vector()
                    dot = (af.x * ego_forward.x) + (af.y * ego_forward.y) + (af.z * ego_forward.z)
                    if dot <= -0.35:
                        speed_range = OPPOSITE_TRAFFIC_SPEED_DIFF_RANGE
                except Exception:
                    pass
            traffic_manager.vehicle_percentage_speed_difference(actor, random.uniform(*speed_range))
            traffic_manager.distance_to_leading_vehicle(actor, random.uniform(*TRAFFIC_FOLLOW_DISTANCE_RANGE))
            traffic_manager.ignore_walkers_percentage(actor, 0.0)
        except Exception:
            pass
    return len(actor_ids)


def spawn_dense_traffic(world, client, traffic_manager, ego, spawn_points, vehicle_blueprints, actors_to_cleanup, log_file) -> Dict[str, int]:
    ego_loc = ego.get_location()
    ego_tf = ego.get_transform()
    fwd = ego_tf.get_forward_vector()
    candidates = [sp for sp in spawn_points if 16.0 <= distance_2d(sp.location, ego_loc) <= (240.0 if ROAD_MODE != 'highway' else 360.0)]
    side_priority = []
    opposite_front_priority = []
    opposite_priority = []
    other_candidates = []
    for sp in candidates:
        dx = sp.location.x - ego_loc.x
        dy = sp.location.y - ego_loc.y
        local_fwd = dx * fwd.x + dy * fwd.y
        local_side = -dx * fwd.y + dy * fwd.x
        sp_fwd = sp.rotation.get_forward_vector()
        heading_dot = (sp_fwd.x * fwd.x) + (sp_fwd.y * fwd.y) + (sp_fwd.z * fwd.z)
        if ROAD_MODE != 'highway' and 5.0 <= abs(local_side) <= 75.0 and -120.0 <= local_fwd <= 170.0:
            if heading_dot <= -0.35:
                if 0.0 <= local_fwd <= 210.0:
                    opposite_front_priority.append(sp)
                else:
                    opposite_priority.append(sp)
            else:
                side_priority.append(sp)
        else:
            other_candidates.append(sp)
    random.shuffle(side_priority)
    random.shuffle(opposite_front_priority)
    random.shuffle(opposite_priority)
    random.shuffle(other_candidates)
    candidates = opposite_front_priority + opposite_priority + side_priority + other_candidates
    log(
        "[TRAFFIC] candidate buckets "
        f"opp_front={len(opposite_front_priority)} opp_other={len(opposite_priority)} "
        f"side={len(side_priority)} other={len(other_candidates)}",
        log_file,
    )
    target_total = 190 if ROAD_MODE != 'highway' else 95
    spawned = 0
    for idx in range(0, min(len(candidates), target_total + 90), 28):
        chunk = candidates[idx: idx + 28]
        if not chunk:
            break
        spawned += spawn_batch_vehicles(
            client,
            world,
            traffic_manager,
            vehicle_blueprints,
            chunk,
            actors_to_cleanup,
            ego_forward=fwd,
        )
        for _ in range(8):
            safe_tick(world, 'traffic_settle', log_file, fatal=False)
        density = collect_density(world, ego)
        log(f"[TRAFFIC] spawned={spawned} density={density}", log_file)
        if ROAD_MODE == 'highway':
            if density['same'] >= MIN_PER_DIRECTION and density['opposite'] >= MIN_PER_DIRECTION and density['total'] >= MIN_TOTAL_ACTIVE and density['distinct'] >= 8:
                return density
        else:
            if (
                density['ahead'] >= MIN_PER_DIRECTION
                and density['behind'] >= MIN_PER_DIRECTION
                and density['left'] >= MIN_SIDE_DENSITY
                and density['right'] >= MIN_SIDE_DENSITY
                and density['opposite'] >= MIN_PER_DIRECTION
                and density['same'] >= MIN_PER_DIRECTION
                and density['total'] >= max(MIN_TOTAL_ACTIVE, 120)
                and density['distinct'] >= 8
            ):
                return density
    density = collect_density(world, ego)
    log(f"[WARN] Density target not fully met; achieved={density}", log_file)
    return density


def seed_ego_relative_traffic(
    world,
    client,
    traffic_manager,
    ego,
    vehicle_blueprints,
    actors_to_cleanup,
    log_file,
) -> int:
    """Spawn traffic from lane waypoints around the ego after ego placement (same road corridor first)."""
    map_obj = world.get_map()
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_fwd = ego_tf.get_forward_vector()
    try:
        ego_wp = map_obj.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    except Exception:
        ego_wp = None
    if ego_wp is None:
        log("[TRAFFIC][EGO_REL] skipped (ego waypoint unavailable)", log_file)
        return 0

    base_road_id = getattr(ego_wp, 'road_id', None)
    lanes: List[Any] = [ego_wp]
    seen = {(int(getattr(ego_wp, "road_id", 0)), int(getattr(ego_wp, "section_id", 0)), int(getattr(ego_wp, "lane_id", 0)))}
    for getter_name in ("get_left_lane", "get_right_lane"):
        lane = ego_wp
        for _ in range(10):
            try:
                lane = getattr(lane, getter_name)()
            except Exception:
                lane = None
            if lane is None or lane.lane_type != carla.LaneType.Driving:
                break
            key = (int(getattr(lane, "road_id", 0)), int(getattr(lane, "section_id", 0)), int(getattr(lane, "lane_id", 0)))
            if key in seen:
                break
            seen.add(key)
            if base_road_id is not None and getattr(lane, 'road_id', None) != base_road_id:
                continue
            if getattr(lane, 'is_junction', False):
                continue
            lanes.append(lane)

    if not lanes:
        log("[TRAFFIC][EGO_REL] no lanes available", log_file)
        return 0

    offsets = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 72.0, 84.0, 96.0, 110.0, 124.0, 138.0, 152.0, 168.0, -12.0, -24.0, -36.0, -48.0]
    ranked: List[tuple] = []
    lane_meta = []
    for lane in lanes:
        lf = lane.transform.get_forward_vector()
        heading_dot = (lf.x * ego_fwd.x) + (lf.y * ego_fwd.y) + (lf.z * ego_fwd.z)
        lane_meta.append((int(getattr(lane, 'lane_id', 0)), round(float(heading_dot), 2)))
        for offset_m in offsets:
            probe = lane
            moved = 0.0
            step = 6.0
            advance_forward = offset_m >= 0.0
            target = abs(offset_m)
            while moved < target:
                try:
                    nxt = probe.next(step) if advance_forward else probe.previous(step)
                except Exception:
                    nxt = []
                if not nxt:
                    probe = None
                    break
                chosen = None
                for cand in nxt:
                    if cand is None or cand.lane_type != carla.LaneType.Driving:
                        continue
                    if base_road_id is not None and getattr(cand, 'road_id', None) != base_road_id:
                        continue
                    chosen = cand
                    break
                if chosen is None:
                    probe = None
                    break
                probe = chosen
                moved += step
                if getattr(probe, 'is_junction', False):
                    probe = None
                    break
            if probe is None:
                continue

            loc = probe.transform.location
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            local_fwd = dx * ego_fwd.x + dy * ego_fwd.y
            local_side = -dx * ego_fwd.y + dy * ego_fwd.x
            if local_fwd > EGO_RELATIVE_TRAFFIC_MAX_AHEAD_M or local_fwd < -EGO_RELATIVE_TRAFFIC_MAX_BEHIND_M:
                continue
            # allow ego lane and adjacent lanes, but skip exact ego overlap region
            if abs(local_fwd) < 8.0 and abs(local_side) < 4.0:
                continue

            tf = carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.35),
                probe.transform.rotation,
            )
            # Prefer visible traffic around ego; oncoming/ahead lanes rank highest.
            score = (
                abs(local_fwd - 38.0) * (0.65 if heading_dot <= -0.35 else 0.95)
                + abs(abs(local_side) - 7.5) * 0.45
                + (0.0 if local_fwd >= 0.0 else 14.0)
            )
            ranked.append((score, random.random(), tf))

    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: List[carla.Transform] = []
    max_total = max(12, len(lanes) * EGO_RELATIVE_TRAFFIC_TARGET_PER_LANE)
    for _, _, tf in ranked:
        if any(distance_2d(tf.location, picked.location) < 8.0 for picked in selected):
            continue
        selected.append(tf)
        if len(selected) >= max_total:
            break

    seeded = 0
    if selected:
        for idx in range(0, len(selected), 24):
            seeded += spawn_batch_vehicles(
                client,
                world,
                traffic_manager,
                vehicle_blueprints,
                selected[idx: idx + 24],
                actors_to_cleanup,
                ego_forward=ego_fwd,
            )
            for _ in range(4):
                safe_tick(world, 'ego_relative_traffic_settle', log_file, fatal=False)

    density = collect_density(world, ego)
    log(
        "[TRAFFIC][EGO_REL] "
        f"lanes={len(lanes)} lane_meta={lane_meta} "
        f"candidates={len(ranked)} selected={len(selected)} seeded={seeded} density={density}",
        log_file,
    )
    return seeded


def seed_adjacent_lanes_near_ego(
    world,
    client,
    traffic_manager,
    ego,
    vehicle_blueprints,
    actors_to_cleanup,
    log_file,
) -> int:
    if ROAD_MODE == 'highway':
        return 0
    map_obj = world.get_map()
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_fwd = ego_tf.get_forward_vector()
    try:
        ego_wp = map_obj.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    except Exception:
        ego_wp = None
    if ego_wp is None:
        log("[TRAFFIC][ADJ_FILL] skipped (ego waypoint unavailable)", log_file)
        return 0

    base_road_id = getattr(ego_wp, 'road_id', None)
    lane_groups: List[Any] = []
    seen = {(int(getattr(ego_wp, "road_id", 0)), int(getattr(ego_wp, "section_id", 0)), int(getattr(ego_wp, "lane_id", 0)))}
    for getter_name in ("get_left_lane", "get_right_lane"):
        lane = ego_wp
        for _ in range(8):
            try:
                lane = getattr(lane, getter_name)()
            except Exception:
                lane = None
            if lane is None or lane.lane_type != carla.LaneType.Driving:
                break
            lane_key = (int(getattr(lane, "road_id", 0)), int(getattr(lane, "section_id", 0)), int(getattr(lane, "lane_id", 0)))
            if lane_key in seen:
                break
            seen.add(lane_key)
            if getattr(lane, 'is_junction', False):
                continue
            if base_road_id is not None and getattr(lane, 'road_id', None) != base_road_id:
                continue
            lane_groups.append(lane)

    if not lane_groups:
        log("[TRAFFIC][ADJ_FILL] no adjacent driving lanes found", log_file)
        return 0

    offsets = [12.0, 24.0, 36.0, 48.0, 60.0, 72.0, 84.0, 98.0, 112.0, 126.0, -14.0, -28.0]
    ranked: List[tuple] = []
    lane_meta = []
    for lane in lane_groups:
        lf = lane.transform.get_forward_vector()
        heading_dot = (lf.x * ego_fwd.x) + (lf.y * ego_fwd.y) + (lf.z * ego_fwd.z)
        lane_meta.append((int(getattr(lane, 'lane_id', 0)), heading_dot))
        for offset_m in offsets:
            probe = lane
            moved = 0.0
            step = 6.0
            advance_forward = offset_m >= 0.0
            target = abs(offset_m)
            while moved < target:
                try:
                    nxt = probe.next(step) if advance_forward else probe.previous(step)
                except Exception:
                    nxt = []
                if not nxt:
                    probe = None
                    break
                chosen = None
                for cand in nxt:
                    if cand is None or cand.lane_type != carla.LaneType.Driving:
                        continue
                    if getattr(cand, 'road_id', None) != base_road_id:
                        continue
                    chosen = cand
                    break
                if chosen is None:
                    probe = None
                    break
                probe = chosen
                moved += step
                if getattr(probe, 'is_junction', False):
                    probe = None
                    break
            if probe is None:
                continue

            loc = probe.transform.location
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            local_fwd = dx * ego_fwd.x + dy * ego_fwd.y
            local_side = -dx * ego_fwd.y + dy * ego_fwd.x
            if local_fwd > ADJACENT_LANE_FILL_MAX_AHEAD_M or local_fwd < -ADJACENT_LANE_FILL_MAX_BEHIND_M:
                continue
            if abs(local_side) < 3.5:
                continue

            tf = carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.35),
                probe.transform.rotation,
            )
            # Prefer visibly adjacent, forward placements; oncoming lanes ahead are highest priority.
            score = (
                abs(local_fwd - 45.0) * (0.7 if heading_dot <= -0.35 else 1.0)
                + abs(abs(local_side) - 8.0) * 0.5
                + (0.0 if local_fwd >= 0.0 else 18.0)
            )
            ranked.append((score, random.random(), tf))

    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: List[carla.Transform] = []
    max_total = max(8, len(lane_groups) * ADJACENT_LANE_FILL_TARGET_PER_LANE)
    for _, _, tf in ranked:
        if any(distance_2d(tf.location, picked.location) < 8.0 for picked in selected):
            continue
        selected.append(tf)
        if len(selected) >= max_total:
            break

    seeded = 0
    if selected:
        for idx in range(0, len(selected), 24):
            seeded += spawn_batch_vehicles(
                client,
                world,
                traffic_manager,
                vehicle_blueprints,
                selected[idx: idx + 24],
                actors_to_cleanup,
                ego_forward=ego_fwd,
            )
            for _ in range(4):
                safe_tick(world, 'adjacent_lane_fill_settle', log_file, fatal=False)

    density = collect_density(world, ego)
    log(
        "[TRAFFIC][ADJ_FILL] "
        f"adjacent_lanes={len(lane_groups)} lane_meta={lane_meta} "
        f"candidates={len(ranked)} selected={len(selected)} seeded={seeded} density={density}",
        log_file,
    )
    return seeded


def _lane_direction_neighbor_counts(wp) -> tuple:
    base_fwd = wp.transform.get_forward_vector()
    same_count = 0
    opposite_count = 0
    seen = {(int(getattr(wp, "road_id", 0)), int(getattr(wp, "section_id", 0)), int(getattr(wp, "lane_id", 0)))}
    for getter_name in ("get_left_lane", "get_right_lane"):
        lane = wp
        for _ in range(6):
            try:
                lane = getattr(lane, getter_name)()
            except Exception:
                lane = None
            if lane is None or lane.lane_type != carla.LaneType.Driving:
                break
            key = (int(getattr(lane, "road_id", 0)), int(getattr(lane, "section_id", 0)), int(getattr(lane, "lane_id", 0)))
            if key in seen:
                break
            seen.add(key)
            if getattr(lane, "is_junction", False):
                continue
            if getattr(lane, "road_id", None) != getattr(wp, "road_id", None):
                continue
            lf = lane.transform.get_forward_vector()
            dot = (base_fwd.x * lf.x) + (base_fwd.y * lf.y) + (base_fwd.z * lf.z)
            if dot > 0.3:
                same_count += 1
            elif dot < -0.3:
                opposite_count += 1
    return same_count, opposite_count


def _advance_waypoint_along_lane(start_wp, distance_m: float, *, forward: bool) -> tuple:
    wp = start_wp
    moved = 0.0
    step = 5.0
    while moved < distance_m:
        try:
            nxt = wp.next(step) if forward else wp.previous(step)
        except Exception:
            nxt = []
        if not nxt:
            return None, moved
        chosen = None
        for cand in nxt:
            if cand is None or cand.lane_type != carla.LaneType.Driving:
                continue
            if getattr(cand, "road_id", None) != getattr(start_wp, "road_id", None):
                continue
            chosen = cand
            break
        if chosen is None:
            chosen = nxt[0]
        wp = chosen
        moved += step
        if getattr(wp, "is_junction", False):
            return None, moved
    return wp, moved


def _junction_unique_driving_roads(junction_obj) -> int:
    if junction_obj is None:
        return 0
    road_ids = set()
    try:
        pairs = junction_obj.get_waypoints(carla.LaneType.Driving)
    except Exception:
        pairs = []
    for pair in pairs or []:
        for wp in pair:
            if wp is None:
                continue
            rid = getattr(wp, "road_id", None)
            if rid is not None:
                road_ids.add(int(rid))
    return len(road_ids)


def _find_four_way_junction_behind(wp, max_distance_m: float = POST_INTERSECTION_MAX_M) -> tuple:
    if wp is None:
        return None, None, 0.0
    probe = wp
    moved = 0.0
    step = 5.0
    while moved < max_distance_m:
        try:
            prevs = probe.previous(step)
        except Exception:
            prevs = []
        if not prevs:
            break
        chosen = None
        for cand in prevs:
            if cand is None or cand.lane_type != carla.LaneType.Driving:
                continue
            if getattr(cand, "road_id", None) != getattr(wp, "road_id", None):
                continue
            chosen = cand
            break
        if chosen is None:
            chosen = prevs[0]
        probe = chosen
        moved += step
        if not getattr(probe, "is_junction", False):
            continue
        junction_obj = None
        try:
            junction_obj = probe.get_junction()
        except Exception:
            junction_obj = None
        unique_roads = _junction_unique_driving_roads(junction_obj)
        if unique_roads >= FOUR_WAY_MIN_UNIQUE_ROADS and moved >= POST_INTERSECTION_MIN_M:
            return probe, junction_obj, moved
    return None, None, moved


def pick_tree_anchor_and_ego_spawn(world_map, ordered_spawns, spawn_points, log_file):
    best = None
    checked = 0
    for anchor_tf in ordered_spawns[:220]:
        try:
            anchor_wp = world_map.get_waypoint(anchor_tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            anchor_wp = None
        if anchor_wp is None or getattr(anchor_wp, "is_junction", False):
            continue

        same_neighbors, opposite_neighbors = _lane_direction_neighbor_counts(anchor_wp)
        if opposite_neighbors < 1:
            continue

        ego_wp, prev_moved = _advance_waypoint_along_lane(anchor_wp, TREE_EVENT_AHEAD_M, forward=False)
        if ego_wp is None or prev_moved < max(25.0, TREE_EVENT_AHEAD_M - 5.0):
            continue
        j_wp, j_obj, dist_junction_behind_ego = _find_four_way_junction_behind(ego_wp, POST_INTERSECTION_MAX_M)
        if j_wp is None or j_obj is None:
            continue
        if dist_junction_behind_ego < POST_INTERSECTION_MIN_M or dist_junction_behind_ego > POST_INTERSECTION_MAX_M:
            continue
        junction_unique_roads = _junction_unique_driving_roads(j_obj)
        if junction_unique_roads < FOUR_WAY_MIN_UNIQUE_ROADS:
            continue
        ahead_wp, next_moved = _advance_waypoint_along_lane(anchor_wp, 60.0, forward=True)
        if ahead_wp is None or next_moved < 35.0:
            continue

        anchor_loc = anchor_wp.transform.location
        anchor_fwd = anchor_wp.transform.get_forward_vector()
        road_total = 0
        road_opposite = 0
        road_same = 0
        for sp in spawn_points:
            try:
                sp_wp = world_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            except Exception:
                sp_wp = None
            if sp_wp is None or getattr(sp_wp, "is_junction", False):
                continue
            if getattr(sp_wp, "road_id", None) != getattr(anchor_wp, "road_id", None):
                continue
            d = distance_2d(sp.location, anchor_loc)
            if d > 220.0:
                continue
            road_total += 1
            sf = sp_wp.transform.get_forward_vector()
            dot = (sf.x * anchor_fwd.x) + (sf.y * anchor_fwd.y) + (sf.z * anchor_fwd.z)
            if dot < -0.35:
                road_opposite += 1
            else:
                road_same += 1
        if road_opposite < 4:
            continue

        score = (
            road_total * 1.0
            + road_opposite * 3.0
            + road_same * 1.0
            + same_neighbors * 6.0
            + opposite_neighbors * 9.0
            + junction_unique_roads * 20.0
            + max(0.0, 30.0 - abs(dist_junction_behind_ego - 25.0)) * 2.0
        )
        checked += 1
        if best is None or score > best[0]:
            best = (
                score,
                anchor_wp,
                ego_wp,
                road_total,
                road_same,
                road_opposite,
                same_neighbors,
                opposite_neighbors,
                junction_unique_roads,
                dist_junction_behind_ego,
            )

    if best is None:
        log("[INFO] No tree-anchor corridor preselection match found; falling back to ego-first spawn", log_file)
        return None, None

    (
        _,
        anchor_wp,
        ego_wp,
        road_total,
        road_same,
        road_opposite,
        same_neighbors,
        opposite_neighbors,
        junction_unique_roads,
        dist_junction_behind_ego,
    ) = best
    ego_tf = carla.Transform(
        carla.Location(
            x=ego_wp.transform.location.x,
            y=ego_wp.transform.location.y,
            z=ego_wp.transform.location.z + 0.35,
        ),
        ego_wp.transform.rotation,
    )
    log(
        "[INFO] Preselected tree anchor corridor "
        f"road_id={getattr(anchor_wp, 'road_id', None)} lane_id={getattr(anchor_wp, 'lane_id', None)} "
        f"same_neighbors={same_neighbors} opposite_neighbors={opposite_neighbors} "
        f"road_spawn_counts=(total={road_total}, same={road_same}, opposite={road_opposite}) "
        f"post_intersection=(junction_roads={junction_unique_roads}, dist_behind_ego={dist_junction_behind_ego:.1f}m) "
        f"ego_to_tree_target={TREE_EVENT_AHEAD_M:.1f}m checked={checked}",
        log_file,
    )
    return ego_tf, anchor_wp


def seed_opposite_flow_at_spawn(
    world,
    client,
    traffic_manager,
    ego,
    spawn_points,
    vehicle_blueprints,
    actors_to_cleanup,
    log_file,
) -> int:
    map_obj = world.get_map()
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    fwd = ego_tf.get_forward_vector()
    target_d = OPPOSITE_START_SPAWN_DISTANCE_M
    band = OPPOSITE_START_SPAWN_BAND_M
    try:
        ego_wp = map_obj.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    except Exception:
        ego_wp = None
    ego_road_id = getattr(ego_wp, 'road_id', None)

    ranked: List[tuple] = []
    same_road_candidates = 0
    for sp in spawn_points:
        try:
            sp_wp = map_obj.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            sp_wp = None
        if sp_wp is None:
            continue
        if getattr(sp_wp, 'is_junction', False):
            continue
        if ego_road_id is not None and getattr(sp_wp, 'road_id', None) != ego_road_id:
            continue
        same_road_candidates += 1
        d = distance_2d(sp.location, ego_loc)
        if d < (target_d - band) or d > (target_d + band):
            continue
        dx = sp.location.x - ego_loc.x
        dy = sp.location.y - ego_loc.y
        local_fwd = dx * fwd.x + dy * fwd.y
        local_side = -dx * fwd.y + dy * fwd.x
        if local_fwd < 55.0 or local_fwd > 150.0:
            continue
        if abs(local_side) < 4.0 or abs(local_side) > 95.0:
            continue
        sp_fwd = sp.rotation.get_forward_vector()
        heading_dot = (sp_fwd.x * fwd.x) + (sp_fwd.y * fwd.y) + (sp_fwd.z * fwd.z)
        if heading_dot > -0.45:
            continue
        score = (
            abs(d - target_d) * 2.0
            + abs(local_fwd - target_d) * 1.25
            + max(0.0, abs(local_side) - 50.0) * 0.25
        )
        ranked.append((score, random.random(), sp))

    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: List[carla.Transform] = []
    for _, _, sp in ranked:
        if any(distance_2d(sp.location, picked.location) < 9.0 for picked in selected):
            continue
        selected.append(sp)
        if len(selected) >= OPPOSITE_START_SPAWN_TARGET:
            break

    seeded = 0
    if selected:
        seeded = spawn_batch_vehicles(
            client,
            world,
            traffic_manager,
            vehicle_blueprints,
            selected,
            actors_to_cleanup,
            ego_forward=fwd,
        )
        for _ in range(8):
            safe_tick(world, 'opposite_start_seed_settle', log_file, fatal=False)

    density = collect_density(world, ego)
    log(
        "[TRAFFIC][OPPOSITE_START] "
        f"target_dist={target_d:.0f}m band={band:.0f}m "
        f"same_road_candidates={same_road_candidates} "
        f"candidates={len(ranked)} selected={len(selected)} seeded={seeded} density={density}",
        log_file,
    )
    return seeded


def seed_opposite_flow_beyond_tree_same_road(
    world,
    client,
    traffic_manager,
    ego,
    event_center,
    spawn_points,
    vehicle_blueprints,
    actors_to_cleanup,
    log_file,
) -> int:
    if event_center is None:
        log("[TRAFFIC][OPPOSITE_TREE_SAME_ROAD] skipped (missing event center)", log_file)
        return 0
    map_obj = world.get_map()
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    ego_fwd = ego_tf.get_forward_vector()
    try:
        ego_wp = map_obj.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        center_wp = map_obj.get_waypoint(event_center, project_to_road=True, lane_type=carla.LaneType.Driving)
    except Exception:
        ego_wp = None
        center_wp = None
    ego_road_id = getattr(ego_wp, 'road_id', None)
    center_road_id = getattr(center_wp, 'road_id', None)

    dx_tree = event_center.x - ego_loc.x
    dy_tree = event_center.y - ego_loc.y
    tree_local_fwd = dx_tree * ego_fwd.x + dy_tree * ego_fwd.y

    ranked: List[tuple] = []
    same_road_candidates = 0
    lane_based_candidates = 0
    opposite_lanes_found = 0
    generated_waypoint_candidates = 0

    # First: directly build oncoming spawn transforms from the opposite-direction
    # lane(s) at the tree location to guarantee same-road, same-corridor traffic.
    if center_wp is not None:
        lane_scan = [center_wp]
        seen_lane_keys = {(int(getattr(center_wp, "road_id", 0)), int(getattr(center_wp, "lane_id", 0)))}
        for getter_name in ("get_left_lane", "get_right_lane"):
            lane = center_wp
            while True:
                try:
                    lane = getattr(lane, getter_name)()
                except Exception:
                    lane = None
                if lane is None or lane.lane_type != carla.LaneType.Driving:
                    break
                lane_key = (int(getattr(lane, "road_id", 0)), int(getattr(lane, "lane_id", 0)))
                if lane_key in seen_lane_keys:
                    break
                seen_lane_keys.add(lane_key)
                if getattr(lane, "is_junction", False):
                    break
                lane_scan.append(lane)

        opposite_lanes = []
        for lane in lane_scan:
            lf = lane.transform.get_forward_vector()
            heading_dot = (lf.x * ego_fwd.x) + (lf.y * ego_fwd.y) + (lf.z * ego_fwd.z)
            if heading_dot <= -0.35:
                opposite_lanes.append(lane)
        opposite_lanes_found = len(opposite_lanes)

        for lane in opposite_lanes:
            probe = lane
            # For opposite lanes, `previous()` usually moves away from ego (beyond tree).
            for _ in range(24):
                prevs = []
                try:
                    prevs = probe.previous(6.0)
                except Exception:
                    prevs = []
                if not prevs:
                    break
                probe = prevs[0]
                if probe is None or probe.lane_type != carla.LaneType.Driving:
                    break
                if getattr(probe, 'is_junction', False):
                    continue
                if ego_road_id is not None and getattr(probe, 'road_id', None) != ego_road_id:
                    continue
                if center_road_id is not None and getattr(probe, 'road_id', None) != center_road_id:
                    continue

                loc = probe.transform.location
                dx = loc.x - ego_loc.x
                dy = loc.y - ego_loc.y
                local_fwd = dx * ego_fwd.x + dy * ego_fwd.y
                local_side = -dx * ego_fwd.y + dy * ego_fwd.x
                ahead_of_tree = local_fwd - tree_local_fwd
                if ahead_of_tree < OPPOSITE_TREE_SAME_ROAD_AHEAD_MIN_M or ahead_of_tree > OPPOSITE_TREE_SAME_ROAD_AHEAD_MAX_M:
                    continue
                if abs(local_side) < 4.0 or abs(local_side) > 95.0:
                    continue
                dist_ego = distance_2d(loc, ego_loc)
                if dist_ego > 260.0:
                    continue

                lane_based_candidates += 1
                tf = carla.Transform(
                    carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.35),
                    probe.transform.rotation,
                )
                score = (
                    abs(ahead_of_tree - 24.0) * 1.2
                    + abs(abs(local_side) - 10.0) * 0.5
                    + abs(dist_ego - 120.0) * 0.35
                )
                ranked.append((score, random.random(), tf))

    # Fallback 2: sample map waypoints near the tree corridor directly. This works
    # even when opposite carriageways do not share the same road_id / spawn points.
    if lane_based_candidates < max(4, OPPOSITE_TREE_SAME_ROAD_TARGET // 3):
        try:
            generated_wps = map_obj.generate_waypoints(6.0)
        except Exception:
            generated_wps = []
        for wp in generated_wps:
            if wp is None or wp.lane_type != carla.LaneType.Driving:
                continue
            if getattr(wp, 'is_junction', False):
                continue
            loc = wp.transform.location
            if distance_2d(loc, event_center) > 95.0:
                continue
            dx = loc.x - ego_loc.x
            dy = loc.y - ego_loc.y
            local_fwd = dx * ego_fwd.x + dy * ego_fwd.y
            local_side = -dx * ego_fwd.y + dy * ego_fwd.x
            ahead_of_tree = local_fwd - tree_local_fwd
            if ahead_of_tree < OPPOSITE_TREE_SAME_ROAD_AHEAD_MIN_M or ahead_of_tree > OPPOSITE_TREE_SAME_ROAD_AHEAD_MAX_M:
                continue
            if abs(local_side) < 4.0 or abs(local_side) > 80.0:
                continue
            dist_ego = distance_2d(loc, ego_loc)
            if dist_ego > 260.0:
                continue
            wf = wp.transform.get_forward_vector()
            heading_dot = (wf.x * ego_fwd.x) + (wf.y * ego_fwd.y) + (wf.z * ego_fwd.z)
            if heading_dot > -0.45:
                continue
            generated_waypoint_candidates += 1
            tf = carla.Transform(
                carla.Location(x=loc.x, y=loc.y, z=loc.z + 0.35),
                wp.transform.rotation,
            )
            score = (
                abs(ahead_of_tree - 22.0) * 1.1
                + abs(abs(local_side) - 10.0) * 0.45
                + abs(dist_ego - 120.0) * 0.30
            ) + 2.5  # Prefer direct adjacent-lane scan slightly when available.
            ranked.append((score, random.random(), tf))

    # Fallback: spawn-point based same-road candidates if lane-based extraction is sparse.
    for sp in spawn_points:
        try:
            sp_wp = map_obj.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        except Exception:
            sp_wp = None
        if sp_wp is None:
            continue
        if getattr(sp_wp, 'is_junction', False):
            continue
        sp_road_id = getattr(sp_wp, 'road_id', None)
        if ego_road_id is not None and sp_road_id != ego_road_id:
            continue
        if center_road_id is not None and sp_road_id != center_road_id:
            continue
        same_road_candidates += 1

        dx = sp.location.x - ego_loc.x
        dy = sp.location.y - ego_loc.y
        local_fwd = dx * ego_fwd.x + dy * ego_fwd.y
        local_side = -dx * ego_fwd.y + dy * ego_fwd.x
        ahead_of_tree = local_fwd - tree_local_fwd
        if ahead_of_tree < OPPOSITE_TREE_SAME_ROAD_AHEAD_MIN_M or ahead_of_tree > OPPOSITE_TREE_SAME_ROAD_AHEAD_MAX_M:
            continue
        if abs(local_side) < 4.0 or abs(local_side) > 95.0:
            continue
        if distance_2d(sp.location, ego_loc) > 260.0:
            continue

        sp_fwd = sp.rotation.get_forward_vector()
        heading_dot = (sp_fwd.x * ego_fwd.x) + (sp_fwd.y * ego_fwd.y) + (sp_fwd.z * ego_fwd.z)
        if heading_dot > -0.45:
            continue

        tf = carla.Transform(
            carla.Location(x=sp.location.x, y=sp.location.y, z=sp.location.z + 0.35),
            sp.rotation,
        )
        score = (
            abs(ahead_of_tree - 25.0) * 1.5
            + abs(abs(local_side) - 10.0) * 0.6
            + abs(distance_2d(sp.location, ego_loc) - 120.0) * 0.4
        ) + 8.0  # Slightly prefer lane-derived candidates over raw spawn points.
        ranked.append((score, random.random(), tf))

    ranked.sort(key=lambda item: (item[0], item[1]))
    selected: List[carla.Transform] = []
    for _, _, tf in ranked:
        if any(distance_2d(tf.location, picked.location) < 9.0 for picked in selected):
            continue
        selected.append(tf)
        if len(selected) >= OPPOSITE_TREE_SAME_ROAD_TARGET:
            break

    seeded = 0
    if selected:
        seeded = spawn_batch_vehicles(
            client,
            world,
            traffic_manager,
            vehicle_blueprints,
            selected,
            actors_to_cleanup,
            ego_forward=ego_fwd,
        )
        for _ in range(10):
            safe_tick(world, 'opposite_tree_same_road_settle', log_file, fatal=False)

    density = collect_density(world, ego)
    log(
        "[TRAFFIC][OPPOSITE_TREE_SAME_ROAD] "
        f"ego_road={ego_road_id} center_road={center_road_id} "
        f"tree_local_fwd={tree_local_fwd:.1f}m "
        f"opposite_lanes={opposite_lanes_found} lane_based_candidates={lane_based_candidates} "
        f"generated_waypoint_candidates={generated_waypoint_candidates} "
        f"same_road_candidates={same_road_candidates} "
        f"candidates={len(ranked)} selected={len(selected)} seeded={seeded} density={density}",
        log_file,
    )
    return seeded


def spawn_context_props(world, bp_lib, ego, spawn_points, actors_to_cleanup, log_file) -> int:
    if EVENT_MODE == 'fallen_tree_log':
        log("[INFO] Context warning props disabled for fallen_tree_log scene", log_file)
        return 0
    patterns = ['static.prop.trafficcone*', 'static.prop.warning*', 'static.prop.streetbarrier*', 'static.prop.barrel*']
    ego_loc = ego.get_location()
    nearby = [sp for sp in spawn_points if 12.0 <= distance_2d(sp.location, ego_loc) <= 120.0]
    random.shuffle(nearby)
    map_obj = world.get_map()
    spawned = 0
    for sp in nearby:
        if spawned >= 28:
            break
        wp = map_obj.get_waypoint(sp.location)
        if wp is None:
            continue
        right = wp.transform.get_right_vector()
        lateral = random.choice([-1.0, 1.0]) * random.uniform(6.0, 11.0)
        loc = sp.location + carla.Location(x=right.x * lateral, y=right.y * lateral, z=0.15)
        try:
            lane_wp = map_obj.get_waypoint(loc, project_to_road=False, lane_type=carla.LaneType.Any)
        except Exception:
            lane_wp = None
        if lane_wp is not None and getattr(lane_wp, 'lane_type', None) == carla.LaneType.Driving:
            continue
        choices = []
        for p in patterns:
            try:
                choices.extend(list(bp_lib.filter(p)))
            except Exception:
                pass
        if not choices:
            continue
        actor = world.try_spawn_actor(random.choice(choices), carla.Transform(loc, carla.Rotation(yaw=random.uniform(-180, 180))))
        if actor is not None:
            actors_to_cleanup.append(actor)
            spawned += 1
    log(f"[INFO] Context props spawned={spawned}", log_file)
    return spawned


def setup_cameras(world, ego, actors_to_cleanup, log_file):
    cam_bp = world.get_blueprint_library().find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', '1280')
    cam_bp.set_attribute('image_size_y', '720')
    cam_bp.set_attribute('fov', '110')
    configs = {
        'front': carla.Transform(carla.Location(x=0.8, y=0.0, z=1.4), carla.Rotation(pitch=8)),
        'front_left': carla.Transform(carla.Location(x=-0.1, y=-0.4, z=1.2), carla.Rotation(yaw=-60)),
        'front_right': carla.Transform(carla.Location(x=-0.1, y=0.4, z=1.2), carla.Rotation(yaw=60)),
        'rear': carla.Transform(carla.Location(x=-0.2, y=0.3, z=1.25), carla.Rotation(yaw=180, pitch=5)),
        'drone_follow': carla.Transform(carla.Location(x=-12.0, y=0.0, z=12.0), carla.Rotation(pitch=-25, yaw=0, roll=0)),
    }
    cams = {}
    for key in CAMERA_KEYS:
        tf = configs[key]
        cam = world.spawn_actor(cam_bp, tf, attach_to=ego)
        cam.listen(make_camera_callback(key))
        actors_to_cleanup.append(cam)
        cams[key] = cam
    log(f"[INFO] Cameras attached keys={list(cams.keys())}", log_file)
    return cams


def update_spectator_follow(world, ego, log_file=None) -> bool:
    """Keep the CARLA spectator near the ego so the on-screen viewport matches the scene area."""
    try:
        spectator = world.get_spectator()
        ego_tf = ego.get_transform()
        fwd = ego_tf.get_forward_vector()
        right = ego_tf.get_right_vector()
        loc = ego_tf.location + carla.Location(
            x=-fwd.x * 10.5 + right.x * 0.8,
            y=-fwd.y * 10.5 + right.y * 0.8,
            z=6.5,
        )
        rot = carla.Rotation(
            pitch=-18.0,
            yaw=ego_tf.rotation.yaw,
            roll=0.0,
        )
        spectator.set_transform(carla.Transform(loc, rot))
        return True
    except Exception as exc:
        if log_file is not None and not getattr(update_spectator_follow, "_warned", False):
            log(f"[WARN] Failed to update spectator follow camera: {exc}", log_file)
            update_spectator_follow._warned = True
        return False


def _set_streetlights_mode(world: carla.World, mode: str, log_file) -> None:
    mode_key = str(mode or "auto").strip().lower()
    if mode_key == "auto":
        return
    if mode_key not in VALID_STREETLIGHT_MODES:
        log(f"[WARN] Invalid streetlights mode '{mode}'; using auto", log_file)
        return
    try:
        light_manager = world.get_lightmanager()
    except Exception as exc:
        log(f"[WARN] Light manager unavailable ({exc}); skipping streetlights override", log_file)
        return
    try:
        street_group = getattr(getattr(carla, "LightGroup", None), "Street", None)
        lights = None
        if street_group is not None:
            try:
                lights = light_manager.get_all_lights(street_group)
            except TypeError:
                lights = None
        if lights is None:
            lights = light_manager.get_all_lights()
        if mode_key == "on":
            light_manager.turn_on(lights)
        elif mode_key == "off":
            light_manager.turn_off(lights)
        log(f"[INFO] Streetlights forced {mode_key}", log_file)
    except Exception as exc:
        log(f"[WARN] Failed to apply streetlights={mode_key}: {exc}", log_file)


def _env_overrides_requested(args: argparse.Namespace) -> bool:
    if canonical_time_key(getattr(args, "time_preset", SCENE_DEFAULT_KEY)) != SCENE_DEFAULT_KEY:
        return True
    if canonical_weather_key(getattr(args, "weather_preset", SCENE_DEFAULT_KEY)) != SCENE_DEFAULT_KEY:
        return True
    for attr in (
        "sun_altitude",
        "sun_azimuth",
        "cloudiness",
        "precipitation",
        "precipitation_deposits",
        "wind_intensity",
        "fog_density",
        "fog_distance",
        "wetness",
    ):
        if getattr(args, attr, None) is not None:
            return True
    return str(getattr(args, "streetlights", "auto")).strip().lower() != "auto"


def apply_scene_weather(
    world: carla.World,
    time_preset: str = SCENE_DEFAULT_KEY,
    weather_preset: str = SCENE_DEFAULT_KEY,
    sun_altitude: Optional[float] = None,
    sun_azimuth: Optional[float] = None,
    streetlights: str = "auto",
    cloudiness: Optional[float] = None,
    precipitation: Optional[float] = None,
    precipitation_deposits: Optional[float] = None,
    wind_intensity: Optional[float] = None,
    fog_density: Optional[float] = None,
    fog_distance: Optional[float] = None,
    wetness: Optional[float] = None,
    log_file=None,
) -> None:
    current = world.get_weather()
    merged: Dict[str, float] = {
        "cloudiness": float(current.cloudiness),
        "precipitation": float(current.precipitation),
        "precipitation_deposits": float(current.precipitation_deposits),
        "wind_intensity": float(current.wind_intensity),
        "fog_density": float(current.fog_density),
        "fog_distance": float(current.fog_distance),
        "wetness": float(current.wetness),
        "sun_altitude_angle": float(current.sun_altitude_angle),
        "sun_azimuth_angle": float(current.sun_azimuth_angle),
    }
    merged.update(SCENE_DEFAULT_ENV)

    time_key = canonical_time_key(time_preset)
    weather_key = canonical_weather_key(weather_preset)
    time_cfg: Optional[Dict[str, Any]] = None
    weather_cfg: Optional[Dict[str, Any]] = None

    if (time_key and time_key != SCENE_DEFAULT_KEY) or (weather_key and weather_key != SCENE_DEFAULT_KEY):
        try:
            spec = load_time_weather_spec()
            time_cfg = get_time_preset(spec, time_key)
            weather_cfg = get_weather_preset(spec, weather_key)
            if time_key != SCENE_DEFAULT_KEY and time_cfg is None:
                log(f"[WARN] Unknown time preset '{time_preset}'; keeping scene default time", log_file)
            if weather_key != SCENE_DEFAULT_KEY and weather_cfg is None:
                log(f"[WARN] Unknown weather preset '{weather_preset}'; keeping scene default weather", log_file)
        except TimeWeatherSpecError as exc:
            log(f"[WARN] Failed loading shared time/weather spec ({exc}); using scene defaults/CLI overrides", log_file)

    if time_cfg is not None:
        merged["sun_altitude_angle"] = float(time_cfg["sun_altitude_angle"])
        merged["sun_azimuth_angle"] = float(time_cfg["sun_azimuth_angle"])
    if weather_cfg is not None:
        for key in (
            "cloudiness",
            "precipitation",
            "precipitation_deposits",
            "wind_intensity",
            "fog_density",
            "fog_distance",
            "wetness",
        ):
            merged[key] = float(weather_cfg[key])

    explicit = {
        "sun_altitude_angle": sun_altitude,
        "sun_azimuth_angle": sun_azimuth,
        "cloudiness": cloudiness,
        "precipitation": precipitation,
        "precipitation_deposits": precipitation_deposits,
        "wind_intensity": wind_intensity,
        "fog_density": fog_density,
        "fog_distance": fog_distance,
        "wetness": wetness,
    }
    for key, value in explicit.items():
        if value is not None:
            merged[key] = float(value)

    weather = world.get_weather()
    weather.sun_altitude_angle = float(merged["sun_altitude_angle"])
    weather.sun_azimuth_angle = float(merged["sun_azimuth_angle"])
    weather.cloudiness = float(merged["cloudiness"])
    weather.wind_intensity = float(merged["wind_intensity"])
    weather.precipitation = float(merged["precipitation"])
    weather.precipitation_deposits = float(merged["precipitation_deposits"])
    weather.wetness = float(merged["wetness"])
    weather.fog_density = float(merged["fog_density"])
    weather.fog_distance = float(merged["fog_distance"])
    try:
        weather.fog_falloff = getattr(weather, "fog_falloff", 0.35)
    except Exception:
        pass
    world.set_weather(weather)

    light_mode = str(streetlights or "auto").strip().lower()
    if light_mode not in VALID_STREETLIGHT_MODES:
        log(f"[WARN] Invalid streetlights arg '{streetlights}'; using auto", log_file)
        light_mode = "auto"
    if light_mode == "auto" and time_cfg is not None:
        light_mode = str(time_cfg.get("streetlights", "auto")).strip().lower()
    _set_streetlights_mode(world, light_mode, log_file)

    log(
        "[INFO] Applied environment "
        f"time_preset={time_key or SCENE_DEFAULT_KEY} "
        f"weather_preset={weather_key or SCENE_DEFAULT_KEY} "
        f"sun_alt={merged['sun_altitude_angle']:.2f} "
        f"sun_azm={merged['sun_azimuth_angle']:.2f} "
        f"cloud={merged['cloudiness']:.1f} rain={merged['precipitation']:.1f} "
        f"puddles={merged['precipitation_deposits']:.1f} wind={merged['wind_intensity']:.1f} "
        f"fog_density={merged['fog_density']:.1f} fog_distance={merged['fog_distance']:.1f} "
        f"wetness={merged['wetness']:.1f} streetlights={light_mode}",
        log_file,
    )


def apply_weather(world, phase: float) -> None:
    weather = carla.WeatherParameters()
    envelope = 0.0 if phase < 0.2 else min(1.0, (phase - 0.2) / 0.5)
    envelope = math.sin(math.pi * envelope)
    weather.sun_altitude_angle = 55.0
    weather.cloudiness = 15.0 + 50.0 * envelope
    weather.wind_intensity = 10.0 + 25.0 * envelope
    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0
    weather.wetness = 8.0
    weather.fog_density = 2.0 + 8.0 * envelope
    weather.fog_distance = 80.0
    weather.fog_falloff = 0.35
    try:
        weather.scattering_intensity = 0.7
        weather.mie_scattering_scale = 0.05
    except Exception:
        pass

    if WEATHER_MODE == 'storm':
        weather.cloudiness = 95.0
        weather.wind_intensity = 75.0
        weather.precipitation = 70.0
        weather.precipitation_deposits = 40.0
        weather.fog_density = 18.0 + 20.0 * envelope
    elif WEATHER_MODE == 'fog':
        weather.cloudiness = 85.0
        weather.fog_density = 32.0 + 20.0 * envelope
        weather.wind_intensity = 20.0
    elif WEATHER_MODE == 'night_anomaly':
        weather.cloudiness = 95.0
        weather.sun_altitude_angle = -12.0 + 12.0 * (1.0 - envelope)
        weather.wind_intensity = 35.0
        weather.fog_density = 15.0 + 18.0 * envelope
        try:
            weather.scattering_intensity = 1.3
            weather.mie_scattering_scale = 0.3
        except Exception:
            pass
    world.set_weather(weather)


def _try_spawn_vehicle(world, bp_lib, preferred_ids, tf, actors_to_cleanup, log_file, label='event_vehicle'):
    try:
        bp = find_blueprint(bp_lib, preferred_ids, 'vehicle.*')
        set_vehicle_attributes(bp, role_name=label)
        actor = world.try_spawn_actor(bp, tf)
        if actor is not None:
            actors_to_cleanup.append(actor)
            log(f"[EVENT] vehicle spawned label={label} type={actor.type_id}", log_file)
        return actor
    except Exception as exc:
        log(f"[EVENT] vehicle spawn failed label={label}: {exc}", log_file)
        return None


def stage_mesh_event(world, bp_lib, ego, actors_to_cleanup, log_file, center_wp_override=None) -> Dict[str, Any]:
    state: Dict[str, Any] = {'mesh_actors': [], 'event_actor': None, 'center': None, 'event_props': []}
    map_obj = world.get_map()
    ego_wp = map_obj.get_waypoint(ego.get_location())
    if ego_wp is None and center_wp_override is None:
        log('[EVENT] No ego waypoint; skipping event staging', log_file)
        return state
    if center_wp_override is not None:
        center_wp = center_wp_override
        log('[EVENT] Using preselected traffic-corridor anchor waypoint for fallen-tree placement', log_file)
    else:
        center_wp = ego_wp
        moved = 0.0
        while moved < TREE_EVENT_AHEAD_M:
            nxt = center_wp.next(5.0)
            if not nxt:
                break
            center_wp = nxt[0]
            moved += 5.0
    center = center_wp.transform.location
    state['center'] = center
    fwd = center_wp.transform.get_forward_vector()
    right = center_wp.transform.get_right_vector()
    yaw = center_wp.transform.rotation.yaw
    event_dist = distance_2d(center, ego.get_location())
    log(f"[EVENT] center selected dist_from_ego={event_dist:.1f}m (target<={TREE_MAX_VISIBLE_DISTANCE_M:.1f}m)", log_file)

    def mesh(category, subcat, name, xoff, yoff, zoff=0.3, *, mass=None, roll=0.0, pitch=0.0, yaw_offset=0.0, fallbacks=None, label='mesh'):
        tf = carla.Transform(
            center + carla.Location(x=fwd.x * xoff + right.x * yoff, y=fwd.y * xoff + right.y * yoff, z=zoff),
            carla.Rotation(pitch=pitch, yaw=yaw + yaw_offset, roll=roll),
        )
        actor = spawn_static_mesh_with_fallback(
            world,
            bp_lib,
            mesh_content_path(category, subcat, name),
            fallbacks or ['static.prop.*'],
            tf,
            actors_to_cleanup,
            log_file,
            mass=mass,
            label=label,
        )
        if actor is not None:
            state['mesh_actors'].append(actor)
        return actor

    def prop(pattern, xoff, yoff, zoff=0.25, label='prop'):
        tf = carla.Transform(center + carla.Location(x=fwd.x * xoff + right.x * yoff, y=fwd.y * xoff + right.y * yoff, z=zoff), carla.Rotation(yaw=yaw))
        for p in ([pattern] if isinstance(pattern, str) else list(pattern)):
            try:
                choices = list(bp_lib.filter(p))
            except Exception:
                choices = []
            if not choices:
                continue
            actor = world.try_spawn_actor(random.choice(choices), tf)
            if actor is not None:
                actors_to_cleanup.append(actor)
                state['event_props'].append(actor)
                log(f"[EVENT] fallback prop spawned label={label} pattern={p} actor_id={actor.id}", log_file)
                return actor
        return None

    def beech_tree(xoff, yoff, zoff=0.32, *, roll=-90.0, pitch=0.0, yaw_offset=90.0, label='beech_tree_fallen'):
        base_loc = center + carla.Location(
            x=fwd.x * xoff + right.x * yoff,
            y=fwd.y * xoff + right.y * yoff,
            z=zoff,
        )
        base_rot = carla.Rotation(pitch=pitch, yaw=yaw + yaw_offset, roll=roll)
        actor = None
        z_attempts = [0.0, 0.15, 0.30, -0.05]
        lateral_attempts = [0.0, 0.5, -0.5, 1.0, -1.0]
        yaw_attempts = [0.0, -10.0, 10.0]
        for sm_name in TREE_MESH_FALLEN_ORDER:
            if actor is not None:
                break
            for z_delta in z_attempts:
                if actor is not None:
                    break
                for lat_delta in lateral_attempts:
                    if actor is not None:
                        break
                    for yaw_delta in yaw_attempts:
                        tf = carla.Transform(
                            carla.Location(
                                x=base_loc.x + right.x * lat_delta,
                                y=base_loc.y + right.y * lat_delta,
                                z=base_loc.z + z_delta,
                            ),
                            carla.Rotation(
                                pitch=base_rot.pitch,
                                yaw=base_rot.yaw + yaw_delta,
                                roll=base_rot.roll,
                            ),
                        )
                        actor = spawn_static_mesh_with_fallback(
                            world,
                            bp_lib,
                            mesh_content_path('Vegetation', 'Trees', sm_name),
                            ['static.prop.beech_tree*', 'static.prop.tree*', 'static.prop.*'],
                            tf,
                            actors_to_cleanup,
                            log_file,
                            mass=800.0,
                            label=f"{label}_{sm_name}",
                        )
                        if actor is not None:
                            log(
                                f"[EVENT] fallen tree mesh placed label={label} mesh={sm_name} "
                                f"z_delta={z_delta:.2f} lat_delta={lat_delta:.2f} yaw_delta={yaw_delta:.1f}",
                                log_file,
                            )
                            break
        if actor is None:
            log(f"[EVENT][WARN] All fallen tree mesh attempts failed for label={label}", log_file)
        if actor is not None:
            state['mesh_actors'].append(actor)
        return actor

    # Always log the compliance hint for maintainers.
    log(f"[V3][MESH] {MESH_EVENT_HINT}", log_file)
    log('[V3][MESH] Reference: SCENE_GENERATION_REFERENCE.md §9', log_file)

    if EVENT_MODE == 'fallen_tree_log':
        lane_w = max(3.4, float(getattr(center_wp, 'lane_width', 3.5)))
        base_fwd = center_wp.transform.get_forward_vector()
        same_dir_sign = 0.0
        opposite_dir_sign = 0.0
        for sign, getter_name in [(-1.0, 'get_left_lane'), (1.0, 'get_right_lane')]:
            try:
                nb = getattr(center_wp, getter_name)()
            except Exception:
                nb = None
            if nb is None or nb.lane_type != carla.LaneType.Driving:
                continue
            nb_fwd = nb.transform.get_forward_vector()
            dot = (base_fwd.x * nb_fwd.x) + (base_fwd.y * nb_fwd.y) + (base_fwd.z * nb_fwd.z)
            if dot > 0.3 and same_dir_sign == 0.0:
                same_dir_sign = sign
            elif dot < -0.3 and opposite_dir_sign == 0.0:
                opposite_dir_sign = sign
        if same_dir_sign == 0.0:
            # Fallback: assume adjacent same-direction lane on left to keep blockage localized.
            same_dir_sign = -1.0
        log(
            f"[EVENT] lane-side inference same_dir_sign={same_dir_sign:+.0f} "
            f"opposite_dir_sign={opposite_dir_sign:+.0f}",
            log_file,
        )

        # Make the tree look like it fell from the sidewalk/curb into the ego lanes.
        curb_side_sign = -same_dir_sign if same_dir_sign != 0.0 else 1.0
        log(f"[EVENT] curb-side inference curb_side_sign={curb_side_sign:+.0f}", log_file)
        # Rotate the trunk from the curb toward the road center (not parallel to lane direction).
        curb_fall_yaw = -90.0 * curb_side_sign
        beech_tree(-0.8, curb_side_sign * lane_w * 1.45, zoff=0.30, roll=-90.0, yaw_offset=curb_fall_yaw + 0.0, label='fallen_beech_primary')
        beech_tree(0.8, curb_side_sign * lane_w * 0.95, zoff=0.28, roll=-88.0, yaw_offset=curb_fall_yaw + 6.0, label='fallen_beech_into_lane_1')
        beech_tree(2.2, curb_side_sign * lane_w * 0.35, zoff=0.26, roll=-86.0, yaw_offset=curb_fall_yaw - 6.0, label='fallen_beech_into_lane_2')
        # Add extra trunk/debris mass to make the blockage read clearly.
        mesh('Vegetation', 'Trees', 'SM_OakTree_S_v3', 1.2, curb_side_sign * lane_w * 0.55, zoff=0.26, mass=320.0, roll=-86.0, yaw_offset=curb_fall_yaw + 3.0, label='fallen_tree_secondary_mesh')
        # No warning cones/barriers for this scene variant; keep the focus on the fallen tree.

    elif EVENT_MODE == 'floating_rocks':
        for i, (xoff, yoff) in enumerate([(12, 0), (16, 2.2), (18, -2.0), (22, 0.8)]):
            mesh('Vegetation', 'Rocks', random.choice(['SM_Stone_1', 'SM_Stone_2', 'SM_Stone_3', 'SM_Stone_4']), xoff, yoff, zoff=1.3 + 0.5 * (i % 2), mass=120.0, label=f'floating_rock_{i}')

    elif EVENT_MODE == 'rolling_sphere':
        actor = mesh('Vegetation', 'Rocks', 'SM_Stone_4', 20.0, -1.0, zoff=0.9, mass=650.0, label='rolling_boulder')
        state['event_actor'] = actor

    elif EVENT_MODE == 'landslide_debris':
        for i in range(8):
            mesh('Vegetation', 'Rocks', random.choice(['SM_Stone_1', 'SM_Stone_2', 'SM_Stone_3']), 14.0 + i * 1.3, random.uniform(-3.5, 3.5), zoff=0.45, mass=180.0, label=f'landslide_rock_{i}')
        for i in range(4):
            mesh('Dynamic', 'Trash', random.choice(['SM_Box01', 'SM_box02', 'SM_box03', 'SM_Barrel']), 12.0 + i * 1.6, random.uniform(-2.0, 2.0), zoff=0.35, mass=25.0, label=f'landslide_debris_{i}')

    elif EVENT_MODE == 'overhead_sign_collapse':
        mesh('Pole', 'Pole', 'SM_RoadSigns01', 18.0, 0.0, zoff=1.0, mass=200.0, roll=70.0, yaw_offset=20.0, label='sign_pole_collapse')
        mesh('TrafficSign', 'PostSigns', 'SM_RoundSign', 19.0, 0.6, zoff=0.8, mass=60.0, roll=85.0, label='sign_plate_collapse')
        for i in range(-2, 3):
            mesh('Dynamic', 'Construction', 'SM_TrafficCones_01', 14.0, i * 1.0, zoff=0.2, mass=2.0, label=f'sign_cone_{i+2}')

    elif EVENT_MODE == 'drawbridge_lift':
        mesh('Bridge', 'Bridge', 'SM_Bridge_Rail', 18.0, -1.5, zoff=0.8, mass=120.0, pitch=18.0, label='drawbridge_rail_left')
        mesh('Bridge', 'Bridge', 'SM_Bridge_Rail', 18.0, 1.5, zoff=0.8, mass=120.0, pitch=18.0, label='drawbridge_rail_right')
        mesh('Bridge', 'Bridge', 'SM_Beam01', 19.0, 0.0, zoff=1.2, mass=240.0, pitch=22.0, label='drawbridge_beam')
        for i in range(-2, 3):
            prop('static.prop.streetbarrier*', 12.5, i * 1.2, label=f'drawbridge_barrier_{i}')

    elif EVENT_MODE == 'collapsing_bridge':
        for i in range(3):
            mesh('Bridge', 'Bridge', random.choice(['SM_Beam01', 'SM_Bridge_Rail']), 17.0 + i * 2.0, random.uniform(-2.5, 2.5), zoff=0.7 + i * 0.2, mass=220.0, roll=random.uniform(25.0, 80.0), yaw_offset=random.uniform(-30.0, 30.0), label=f'bridge_debris_{i}')
        for i in range(4):
            mesh('Dynamic', 'Construction', random.choice(['SM_StreetBarrier', 'SM_ConstructionCone', 'SM_TrafficCones_02']), 12.0 + i, random.uniform(-3.0, 3.0), zoff=0.25, mass=4.0, label=f'bridge_warning_{i}')

    elif EVENT_MODE == 'drone_crash':
        actor = mesh('Static', 'Static', 'SM_LightBox1', 16.0, 0.0, zoff=7.0, mass=35.0, label='drone_proxy_mesh')
        state['event_actor'] = actor
        for i in range(3):
            mesh('Dynamic', 'Trash', random.choice(['SM_Box01', 'SM_Can01', 'SM_BottlePlastic']), 18.0 + i, random.uniform(-1.5, 1.5), zoff=2.0 + i * 0.4, mass=5.0, label=f'drone_debris_{i}')

    elif EVENT_MODE == 'obstacle_drop':
        actor = mesh('Dynamic', 'Trash', 'SM_Box01', 14.0, 0.2, zoff=5.0, mass=30.0, label='dropped_box_mesh')
        state['event_actor'] = actor
        prop(['static.prop.trafficcone*', 'static.prop.warning*'], 11.0, 2.0, label='drop_warning')

    elif EVENT_MODE == 'fallen_power_lines':
        mesh('Pole', 'PoweLine', 'SM_ElectricPole01', 18.0, 2.0, zoff=1.0, mass=350.0, roll=80.0, yaw_offset=12.0, label='power_pole_fallen')
        mesh('Pole', 'Pole', 'SM_wire', 19.0, 0.0, zoff=0.7, mass=40.0, roll=20.0, label='power_wire_proxy')
        for i in range(-2, 3):
            prop('static.prop.warning*', 12.0, i * 1.1, label=f'power_warning_{i}')

    elif EVENT_MODE == 'sinkhole_opening':
        for i in range(8):
            ang = 2.0 * math.pi * i / 8.0
            mesh('Vegetation', 'Rocks', random.choice(['SM_Stone_1', 'SM_Stone_2']), 16.0 + math.cos(ang) * 2.6, math.sin(ang) * 2.2, zoff=0.25, mass=30.0, label=f'sinkhole_ring_{i}')
        for i in range(-2, 3):
            prop(['static.prop.streetbarrier*', 'static.prop.trafficcone*'], 12.0, i * 1.3, label=f'sinkhole_warning_{i}')

    elif EVENT_MODE == 'meteor_shower':
        for i in range(5):
            mesh('Vegetation', 'Rocks', random.choice(['SM_Stone_2', 'SM_Stone_3', 'SM_Stone_4']), 16.0 + i * 1.8, random.uniform(-2.8, 2.8), zoff=5.0 + i * 1.4, mass=80.0, label=f'meteor_proxy_{i}')
        for i in range(4):
            mesh('Other', 'Other', random.choice(['SM_DirtDebris01', 'SM_DirtDebris02', 'SM_DirtDebris03']), 18.0 + i, random.uniform(-2.0, 2.0), zoff=0.4, mass=15.0, label=f'impact_debris_{i}')

    elif EVENT_MODE == 'ufo_hover':
        actor = mesh('Other', 'Other', 'SM_Plane', 18.0, 0.0, zoff=8.5, mass=100.0, label='ufo_plane_proxy')
        state['event_actor'] = actor
        for i in range(6):
            mesh('Static', 'TrafficSign', 'SM_RoundSign', 18.0 + math.cos(i) * 4.0, math.sin(i) * 3.0, zoff=0.5, mass=10.0, label=f'ufo_perimeter_{i}')

    elif EVENT_MODE == 'tentacle_roadblock':
        for i in range(6):
            mesh('Pole', 'Pole', random.choice(['SM_PoleCylinder', 'SM_InterchangeSign_03_Pole']), 16.0 + random.uniform(-2.5, 2.5), (i - 2.5) * 1.2, zoff=1.0, mass=140.0, roll=random.uniform(-25.0, 25.0), pitch=random.uniform(-15.0, 15.0), label=f'tentacle_proxy_{i}')
        for i in range(-2, 3):
            mesh('Dynamic', 'Construction', 'SM_StreetBarrier', 13.0, i * 1.3, zoff=0.2, mass=8.0, label=f'tentacle_barrier_{i}')

    elif EVENT_MODE == 'portal_disruption':
        for i in range(10):
            ang = 2.0 * math.pi * i / 10.0
            mesh('Pole', 'Pole', 'SM_PoleCylinder', 18.0 + math.cos(ang) * 3.5, math.sin(ang) * 3.0, zoff=1.4, mass=50.0, pitch=90.0, yaw_offset=math.degrees(ang), label=f'portal_ring_{i}')
        for i in range(6):
            mesh('Vegetation', 'Rocks', random.choice(['SM_Stone_1', 'SM_Stone_3']), 18.0 + random.uniform(-2.2, 2.2), random.uniform(-2.2, 2.2), zoff=0.8 + random.uniform(0.0, 1.2), mass=60.0, label=f'portal_debris_{i}')

    elif EVENT_MODE == 'illusionary_split':
        for i in range(6):
            mesh('TrafficSign', 'TrafficSigns_A1', random.choice(['SM_A01yieldPed', 'SM_A01onlyCrossw', 'SM_A01tow']), 14.0 + i * 1.7, random.choice([-2.4, 2.4]), zoff=0.9, mass=12.0, label=f'illusion_sign_{i}')
        for i in range(4):
            mesh('Dynamic', 'Construction', random.choice(['SM_TrafficCones_01', 'SM_TrafficCones_02']), 16.0 + i, (i - 1.5) * 1.6, zoff=0.2, mass=2.0, label=f'illusion_marker_{i}')

    elif EVENT_MODE == 'zero_gravity':
        for i in range(7):
            mesh('Static', 'Static', random.choice(['SM_Bench01', 'SM_TrashCan1', 'SM_Plantpot01', 'SM_Basket1' if False else 'SM_Bench02']), 14.0 + random.uniform(-3.0, 3.0), random.uniform(-3.5, 3.5), zoff=1.6 + random.uniform(0.0, 2.5), mass=20.0, label=f'zero_g_street_{i}')
        for i in range(3):
            mesh('Vegetation', 'Rocks', random.choice(['SM_Stone_1', 'SM_Stone_2']), 18.0 + i * 1.5, random.uniform(-2.0, 2.0), zoff=2.5 + i * 0.7, mass=40.0, label=f'zero_g_rock_{i}')

    else:
        # Generic mesh compliance fallback for custom-object scenes.
        mesh('Vegetation', 'Rocks', 'SM_Stone_2', 16.0, 0.0, zoff=0.5, mass=80.0, label='generic_mesh_hazard')
        mesh('Dynamic', 'Construction', 'SM_StreetBarrier', 13.0, 2.0, zoff=0.2, mass=5.0, label='generic_barrier')

    return state


def apply_event_dynamics(world, ego, event_state: Dict[str, Any], elapsed: float, total_duration: float, log_file) -> None:
    phase = 0.0 if total_duration <= 0.0 else max(0.0, min(1.0, elapsed / total_duration))
    mesh_actors: List[carla.Actor] = list(event_state.get('mesh_actors') or [])
    primary = event_state.get('event_actor')

    if EVENT_MODE in {'floating_rocks', 'zero_gravity', 'portal_disruption', 'ufo_hover', 'tentacle_roadblock'}:
        for idx, actor in enumerate(mesh_actors):
            if actor is None:
                continue
            try:
                actor.add_angular_impulse(carla.Vector3D(
                    x=math.sin(elapsed * 1.7 + idx) * 40.0,
                    y=math.cos(elapsed * 1.3 + idx) * 40.0,
                    z=math.sin(elapsed * 1.9 + idx) * 55.0,
                ))
                if 0.2 <= phase <= 0.85 and idx % 2 == 0:
                    actor.add_impulse(carla.Vector3D(
                        x=math.sin(elapsed + idx) * 35.0,
                        y=math.cos(elapsed + idx) * 35.0,
                        z=30.0,
                    ))
            except Exception:
                pass

    if EVENT_MODE in {'rolling_sphere', 'drone_crash', 'obstacle_drop'} and primary is not None:
        try:
            if not event_state.get('_primary_impulse_sent') and phase >= 0.28:
                if EVENT_MODE == 'rolling_sphere':
                    primary.add_impulse(carla.Vector3D(x=500.0, y=0.0, z=20.0))
                elif EVENT_MODE == 'drone_crash':
                    primary.add_impulse(carla.Vector3D(x=140.0, y=0.0, z=-1200.0))
                else:
                    primary.add_impulse(carla.Vector3D(x=80.0, y=0.0, z=-900.0))
                event_state['_primary_impulse_sent'] = True
                log(f"[EVENT] Primary impulse triggered mode={EVENT_MODE}", log_file)
        except Exception:
            pass

    if EVENT_MODE in {'meteor_shower', 'landslide_debris'} and 0.22 <= phase <= 0.75:
        for idx, actor in enumerate(mesh_actors[:8]):
            try:
                actor.add_impulse(carla.Vector3D(
                    x=random.uniform(-30.0, 60.0),
                    y=random.uniform(-25.0, 25.0),
                    z=-120.0 if EVENT_MODE == 'meteor_shower' else -40.0,
                ))
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Standalone CARLA scenario: {SCENARIO_KEYWORD} ({VERSION_TAG})")
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--duration', type=float, default=20.0)
    parser.add_argument('--output-dir', default=f'./scenes/{SCENARIO_KEYWORD}')
    parser.add_argument('--output', dest='output_alias', default=None)
    parser.add_argument('--time-preset', default=SCENE_DEFAULT_KEY)
    parser.add_argument('--weather-preset', default=SCENE_DEFAULT_KEY)
    parser.add_argument('--sun-altitude', type=float, default=None)
    parser.add_argument('--sun-azimuth', type=float, default=None)
    parser.add_argument('--streetlights', default='auto')
    parser.add_argument('--cloudiness', type=float, default=None)
    parser.add_argument('--precipitation', type=float, default=None)
    parser.add_argument('--precipitation-deposits', type=float, default=None)
    parser.add_argument('--wind-intensity', type=float, default=None)
    parser.add_argument('--fog-density', type=float, default=None)
    parser.add_argument('--fog-distance', type=float, default=None)
    parser.add_argument('--wetness', type=float, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_alias or args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    output_dirs = {k: output_root / k for k in CAMERA_KEYS}
    for d in output_dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    cleared_old_frames = 0
    for old_frame in output_root.rglob("*_frame_*.png"):
        try:
            old_frame.unlink()
            cleared_old_frames += 1
        except Exception:
            pass

    log_path = output_root / f'{SCENARIO_KEYWORD}_simulation.log'

    client: Optional[carla.Client] = None
    world: Optional[carla.World] = None
    original_settings: Optional[carla.WorldSettings] = None
    traffic_manager = None
    actors_to_cleanup: List[carla.Actor] = []

    with log_path.open('w', encoding='utf-8') as log_file:
        log(f'[INFO][{VERSION_TAG}] Scenario start: {SCENARIO_KEYWORD} shot={SHOT_INDEX}', log_file)
        log(f'[INFO] Cleared old frame files: {cleared_old_frames}', log_file)
        log(f'[INFO][{VERSION_TAG}] Compliance profile: {COMPLIANCE_UPDATE_PROFILE}', log_file)
        log(f'[INFO][{VERSION_TAG}] Source manifest: {SOURCE_MANIFEST_POSIX} (run={SOURCE_RUN_ID}, shot={SOURCE_SHOT_INDEX})', log_file)
        log(f'[INFO][{VERSION_TAG}] Prompt preserved: {SCENE_PROMPT}', log_file)
        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(60.0)

            selected_map = choose_map(client, log_file)
            log(f'[INFO] Map selected: {selected_map}', log_file)
            world = load_world_with_retry(client, selected_map, log_file)
            original_settings = apply_sync_settings(world, client, log_file)
            use_fixed_env = _env_overrides_requested(args)
            if use_fixed_env:
                apply_scene_weather(
                    world,
                    time_preset=args.time_preset,
                    weather_preset=args.weather_preset,
                    sun_altitude=args.sun_altitude,
                    sun_azimuth=args.sun_azimuth,
                    streetlights=args.streetlights,
                    cloudiness=args.cloudiness,
                    precipitation=args.precipitation,
                    precipitation_deposits=args.precipitation_deposits,
                    wind_intensity=args.wind_intensity,
                    fog_density=args.fog_density,
                    fog_distance=args.fog_distance,
                    wetness=args.wetness,
                    log_file=log_file,
                )
            else:
                log('[INFO] Using scene-default dynamic weather progression', log_file)

            log('[INFO] Clearing existing vehicles from map', log_file)
            for actor in world.get_actors().filter('vehicle.*'):
                try:
                    actor.destroy()
                except Exception:
                    pass

            bp_lib = world.get_blueprint_library()
            world_map = world.get_map()
            spawn_points = world_map.get_spawn_points()
            if not spawn_points:
                raise RuntimeError('No spawn points available')
            vegetation_centers = get_vegetation_bbox_centers(world, log_file)
            log(f"[INFO] Vegetation bbox centers available for corridor scoring: {len(vegetation_centers)}", log_file)

            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(4.0)
            try:
                traffic_manager.global_percentage_speed_difference(-8.0)
            except Exception:
                pass
            try:
                traffic_manager.set_hybrid_physics_mode(False)
            except Exception:
                pass

            ego_bp = find_blueprint(bp_lib, ['vehicle.tesla.model3', 'vehicle.audi.a2'], 'vehicle.*')
            set_vehicle_attributes(ego_bp, 'hero')
            ego = None
            local_multilane_spawns: List[tuple] = []
            local_singlelane_spawns: List[tuple] = []
            fallback_spawns: List[carla.Transform] = []
            for sp in spawn_points:
                try:
                    wp = world_map.get_waypoint(
                        sp.location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                except Exception:
                    wp = None
                if wp is None:
                    fallback_spawns.append(sp)
                    continue
                if getattr(wp, "is_junction", False):
                    continue
                # Avoid approaches that enter a junction/intersection soon; this scene
                # should focus on the fallen tree roadblock on a corridor segment.
                probe_wp = wp
                clear_non_junction_m = 0.0
                valid_runway = True
                while clear_non_junction_m < 110.0:
                    nxt = probe_wp.next(5.0)
                    if not nxt:
                        valid_runway = False
                        break
                    probe_wp = nxt[0]
                    clear_non_junction_m += 5.0
                    if getattr(probe_wp, "is_junction", False):
                        valid_runway = False
                        break
                if not valid_runway:
                    continue
                has_same_dir_neighbor = False
                for nb in (wp.get_left_lane(), wp.get_right_lane()):
                    if nb is None or nb.lane_type != carla.LaneType.Driving:
                        continue
                    f0 = wp.transform.get_forward_vector()
                    f1 = nb.transform.get_forward_vector()
                    dot = (f0.x * f1.x) + (f0.y * f1.y) + (f0.z * f1.z)
                    if dot > 0.3:
                        has_same_dir_neighbor = True
                        break
                if has_same_dir_neighbor:
                    local_multilane_spawns.append((roadside_vegetation_score(sp, vegetation_centers), sp))
                else:
                    local_singlelane_spawns.append((roadside_vegetation_score(sp, vegetation_centers), sp))

            random.shuffle(local_multilane_spawns)
            random.shuffle(local_singlelane_spawns)
            local_multilane_spawns.sort(key=lambda item: item[0], reverse=True)
            local_singlelane_spawns.sort(key=lambda item: item[0], reverse=True)
            random.shuffle(fallback_spawns)
            ordered_spawns = [sp for _, sp in local_multilane_spawns] + [sp for _, sp in local_singlelane_spawns] + fallback_spawns
            best_multilane_score = local_multilane_spawns[0][0] if local_multilane_spawns else 0.0
            best_singlelane_score = local_singlelane_spawns[0][0] if local_singlelane_spawns else 0.0
            log(
                f"[INFO] Local-road spawn candidates: multilane={len(local_multilane_spawns)} "
                f"singlelane={len(local_singlelane_spawns)} fallback={len(fallback_spawns)} "
                f"best_tree_corridor_scores=(multilane={best_multilane_score:.1f}, singlelane={best_singlelane_score:.1f}) "
                f"(no intersection/traffic-light requirement)",
                log_file,
            )
            tree_anchor_wp = None
            if ROAD_MODE != 'highway':
                anchored_ego_tf, tree_anchor_wp = pick_tree_anchor_and_ego_spawn(world_map, ordered_spawns, spawn_points, log_file)
                if anchored_ego_tf is not None:
                    ego = world.try_spawn_actor(ego_bp, anchored_ego_tf)
                    if ego is not None:
                        log('[INFO] Ego spawned using preselected tree-anchor corridor', log_file)
            for tf in ordered_spawns:
                if ego is not None:
                    break
                ego = world.try_spawn_actor(ego_bp, tf)
                if ego is not None:
                    break
            if ego is None:
                raise RuntimeError('Failed to spawn ego vehicle')
            actors_to_cleanup.append(ego)
            ego.set_autopilot(True, traffic_manager.get_port())
            try:
                traffic_manager.vehicle_percentage_speed_difference(ego, EGO_SPEED_DIFF)
                traffic_manager.auto_lane_change(ego, True)
            except Exception:
                pass
            update_spectator_follow(world, ego, log_file)
            log('[INFO] Spectator camera set to follow ego vehicle', log_file)

            vehicle_bps = []
            for bp in bp_lib.filter('vehicle.*'):
                try:
                    if bp.has_attribute('number_of_wheels') and int(bp.get_attribute('number_of_wheels')) < 4:
                        continue
                except Exception:
                    pass
                vehicle_bps.append(bp)
            if not vehicle_bps:
                vehicle_bps = list(bp_lib.filter('vehicle.*'))

            ego_relative_seed = seed_ego_relative_traffic(
                world,
                client,
                traffic_manager,
                ego,
                vehicle_bps,
                actors_to_cleanup,
                log_file,
            )
            adjacent_lane_fill_seed = seed_adjacent_lanes_near_ego(
                world,
                client,
                traffic_manager,
                ego,
                vehicle_bps,
                actors_to_cleanup,
                log_file,
            )
            update_spectator_follow(world, ego, log_file)
            opposite_seed_start = seed_opposite_flow_at_spawn(
                world,
                client,
                traffic_manager,
                ego,
                spawn_points,
                vehicle_bps,
                actors_to_cleanup,
                log_file,
            )
            update_spectator_follow(world, ego, log_file)
            density_initial = spawn_dense_traffic(world, client, traffic_manager, ego, spawn_points, vehicle_bps, actors_to_cleanup, log_file)
            props_count = spawn_context_props(world, bp_lib, ego, spawn_points, actors_to_cleanup, log_file)
            event_state = stage_mesh_event(world, bp_lib, ego, actors_to_cleanup, log_file, center_wp_override=tree_anchor_wp)
            opposite_seed_tree_same_road = seed_opposite_flow_beyond_tree_same_road(
                world,
                client,
                traffic_manager,
                ego,
                event_state.get('center'),
                spawn_points,
                vehicle_bps,
                actors_to_cleanup,
                log_file,
            )
            update_spectator_follow(world, ego, log_file)
            setup_cameras(world, ego, actors_to_cleanup, log_file)

            reset_camera_buffers()
            for _ in range(20):
                safe_tick(world, 'camera_warmup', log_file, fatal=False)
                update_spectator_follow(world, ego, log_file)
            reset_camera_buffers()
            log('[INFO] Cameras warmed and buffers reset', log_file)

            max_frames = max(1, int(args.duration * FPS))
            frame_number = 1

            while frame_number <= max_frames:
                phase = (frame_number - 1) / float(max_frames)
                if not use_fixed_env:
                    apply_weather(world, phase)
                apply_event_dynamics(world, ego, event_state, (frame_number - 1) * FRAME_DT, args.duration, log_file)
                safe_tick(world, 'main_loop', log_file, fatal=False)
                update_spectator_follow(world, ego, log_file)

                if frame_ready.wait(timeout=1.2):
                    with images_lock:
                        for key in CAMERA_KEYS:
                            img = images_received.get(key)
                            if img is None:
                                continue
                            out = output_dirs[key] / f'{key}_frame_{frame_number:08d}.png'
                            save_image_to_disk(img, out)
                else:
                    log(f'[WARN] camera timeout frame={frame_number}', log_file)

                if frame_number % 80 == 0:
                    v = ego.get_velocity()
                    speed_kmh = 3.6 * math.sqrt(v.x * v.x + v.y * v.y + v.z * v.z)
                    density = collect_density(world, ego)
                    log(f'[PROGRESS] frame={frame_number}/{max_frames} speed_kmh={speed_kmh:.1f} density={density}', log_file)

                reset_camera_buffers()
                frame_number += 1

            final_density = collect_density(world, ego)
            log(
                f'[SUCCESS][{VERSION_TAG}] Completed scenario. '
                f'ego_relative_seed={ego_relative_seed} '
                f'adjacent_lane_fill_seed={adjacent_lane_fill_seed} '
                f'opposite_seed_start={opposite_seed_start} '
                f'opposite_seed_tree_same_road={opposite_seed_tree_same_road} '
                f'density_initial={density_initial} final_density={final_density} context_props={props_count}',
                log_file,
            )

        except Exception as exc:
            log(f'[ERROR] Scenario failure: {exc}', log_file)
            raise

        finally:
            log('[INFO] Cleanup start', log_file)
            for actor in actors_to_cleanup:
                try:
                    if 'sensor.camera' in actor.type_id:
                        actor.stop()
                except Exception:
                    pass
            for actor in reversed(actors_to_cleanup):
                try:
                    actor.destroy()
                except Exception:
                    pass
            if world is not None and original_settings is not None:
                try:
                    world.apply_settings(original_settings)
                except Exception:
                    pass
            if traffic_manager is not None:
                try:
                    traffic_manager.set_synchronous_mode(False)
                except Exception:
                    pass
            log('[INFO] Cleanup complete', log_file)


if __name__ == '__main__':
    main()
