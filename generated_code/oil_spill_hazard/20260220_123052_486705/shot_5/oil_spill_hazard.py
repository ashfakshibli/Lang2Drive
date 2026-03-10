#!/usr/bin/env python3
"""Standalone CARLA simulation: Oil Spill Hazard on Highway.

An overturned tanker truck has spilled oil across a highway lane. Emergency
vehicles (ambulance, police) are parked nearby. Construction cones, warning
signs, and debris mark the hazard zone. Slippery conditions with high wetness.

Shot 5  |  Seed 3406  |  Duration 20 s  |  Map preference: Town04 (highway)
Enhancements: More emergency vehicles, wider spill zone, enhanced debris.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import carla
except ImportError:
    print("[ERROR] CARLA Python API not found.")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
FRAME_DT = 0.05
FPS = int(round(1.0 / FRAME_DT))
# Default to front-camera-only capture unless a scene explicitly requests all views.
CAMERA_KEYS = ["front"]
MAP_PREFERENCE = ["Town04", "Town03", "Town10HD"]
MIN_FLOW_PER_DIRECTION = 20
MIN_TOTAL_VEHICLES = 80
UNIFORM_TRAFFIC_SPEED_DIFF = 8.0
EGO_APPROACH_SPEED_DIFF = 8.0

# ---------------------------------------------------------------------------
# Thread-safe image buffer
# ---------------------------------------------------------------------------
images_received: Dict[str, Optional[carla.Image]] = {k: None for k in CAMERA_KEYS}
images_lock = threading.Lock()
frame_ready = threading.Event()


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------
def log(msg: str, log_file) -> None:
    line = msg.rstrip()
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def map_token(name: str) -> str:
    raw = str(name).strip()
    if "/" in raw:
        return raw.rsplit("/", 1)[-1]
    return raw


def is_retryable_carla_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        tok in msg
        for tok in (
            "time-out",
            "failed to connect to newly created map",
            "failed to connect",
            "connection closed",
            "rpc",
            "socket",
        )
    )


def distance_2d(a: carla.Location, b: carla.Location) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def get_speed_kmh(vehicle: carla.Actor) -> float:
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2) * 3.6


def next_waypoint(start_wp: carla.Waypoint, distance_m: float) -> carla.Waypoint:
    current = start_wp
    moved = 0.0
    while moved < distance_m:
        nxt = current.next(5.0)
        if not nxt:
            break
        current = nxt[0]
        moved += 5.0
    return current


def is_highway_candidate(wp: carla.Waypoint) -> bool:
    """True if the waypoint sits on a likely highway segment (>= 2 lanes same dir)."""
    if wp.is_junction:
        return False
    left = wp.get_left_lane()
    right = wp.get_right_lane()
    return (
        (left is not None and left.lane_type == carla.LaneType.Driving) or
        (right is not None and right.lane_type == carla.LaneType.Driving)
    )


def has_long_drivable_segment(wp: carla.Waypoint, min_len_m: float = 160.0) -> bool:
    cur = wp
    moved = 0.0
    while moved < min_len_m:
        nxt = cur.next(6.0)
        if not nxt:
            return False
        cur = nxt[0]
        moved += 6.0
        if cur.is_junction:
            return False
    return True


def classify_flow_alignment(base_tf: carla.Transform, other_tf: carla.Transform) -> str:
    base_fwd = base_tf.get_forward_vector()
    other_fwd = other_tf.get_forward_vector()
    dot = base_fwd.x * other_fwd.x + base_fwd.y * other_fwd.y + base_fwd.z * other_fwd.z
    return "same" if dot >= 0.0 else "opposite"


def rightmost_driving_lane(wp: carla.Waypoint) -> carla.Waypoint:
    cur = wp
    while True:
        nxt = cur.get_right_lane()
        if nxt is None or nxt.lane_type != carla.LaneType.Driving:
            break
        if classify_flow_alignment(cur.transform, nxt.transform) != "same":
            break
        cur = nxt
    return cur


def lane_offset_location(
    wp: carla.Waypoint,
    forward_m: float = 0.0,
    lateral_m: float = 0.0,
    z: float = 0.0,
) -> carla.Location:
    tf = wp.transform
    fwd = tf.get_forward_vector()
    right = tf.get_right_vector()
    return carla.Location(
        x=tf.location.x + fwd.x * forward_m + right.x * lateral_m,
        y=tf.location.y + fwd.y * forward_m + right.y * lateral_m,
        z=tf.location.z + z,
    )


def relative_forward_lateral(base_tf: carla.Transform, location: carla.Location) -> Tuple[float, float]:
    dv = location - base_tf.location
    fwd = base_tf.get_forward_vector()
    right = base_tf.get_right_vector()
    rel_fwd = dv.x * fwd.x + dv.y * fwd.y + dv.z * fwd.z
    rel_lat = dv.x * right.x + dv.y * right.y + dv.z * right.z
    return rel_fwd, rel_lat


# ---------------------------------------------------------------------------
# safe_tick / map / sync
# ---------------------------------------------------------------------------
def safe_tick(world: carla.World, stage: str, log_file,
              retries: int = 3, fatal: bool = True) -> bool:
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            world.tick()
            return True
        except RuntimeError as exc:
            last_error = exc
            if not is_retryable_carla_error(exc):
                raise
            log(f"[WARN] world.tick retryable failure stage={stage} "
                f"attempt={attempt}/{retries}: {exc}", log_file)
            if attempt < retries:
                time.sleep(min(0.6 * attempt, 1.5))
    if fatal and last_error is not None:
        raise RuntimeError(f"Persistent tick failure at stage={stage}") from last_error
    log(f"[WARN] Continuing despite repeated tick failure at stage={stage}", log_file)
    return False


def choose_preferred_map(client: carla.Client, requested: str, log_file) -> str:
    candidates = [requested, *MAP_PREFERENCE]
    ordered: List[str] = []
    for c in candidates:
        if c and c not in ordered:
            ordered.append(c)

    try:
        available = [map_token(n) for n in client.get_available_maps()]
    except Exception as exc:
        log(f"[WARN] Could not query available maps; using requested={requested}. "
            f"details={exc}", log_file)
        return requested

    available_by_lower = {n.lower(): n for n in available}
    for candidate in ordered:
        c = candidate.lower()
        if c in available_by_lower:
            chosen = available_by_lower[c]
            log(f"[INFO] Map selector picked: {chosen} (candidate={candidate})", log_file)
            return chosen
        for token in available:
            if token.lower().startswith(c):
                log(f"[INFO] Map selector picked: {token} "
                    f"(candidate={candidate}, prefix-match)", log_file)
                return token

    log(f"[WARN] No preferred map found; using requested={requested}", log_file)
    return requested


def load_world_with_retry(client: carla.Client, selected_map: str,
                          log_file, retries: int = 4) -> carla.World:
    chosen = map_token(selected_map)
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        try:
            client.set_timeout(30.0)
            current_world = client.get_world()
            active = map_token(current_world.get_map().name)
            if active.lower() == chosen.lower():
                log(f"[INFO] Reusing active map without reload: {active}", log_file)
                return current_world
        except Exception:
            pass

        timeout_seconds = 120.0 + (attempt - 1) * 45.0
        try:
            client.set_timeout(timeout_seconds)
            log(f"[INFO] Loading map attempt={attempt}/{retries} "
                f"map={chosen} timeout={timeout_seconds:.0f}s", log_file)
            return client.load_world(chosen)
        except RuntimeError as exc:
            last_error = exc
            if not is_retryable_carla_error(exc):
                raise
            log(f"[WARN] load_world retryable failure map={chosen} "
                f"attempt={attempt}/{retries}: {exc}", log_file)
            try:
                client.set_timeout(45.0)
                probe = client.get_world()
                active = map_token(probe.get_map().name)
                if active.lower() == chosen.lower():
                    log(f"[INFO] Active map switched to target after retry "
                        f"failure: {active}", log_file)
                    return probe
            except Exception as reconnect_exc:
                log(f"[WARN] reconnect probe failed after load_world error: "
                    f"{reconnect_exc}", log_file)
            time.sleep(min(2.0 * attempt, 6.0))

    if last_error is not None:
        raise RuntimeError(
            f"Failed to load map '{chosen}' after {retries} attempts"
        ) from last_error
    raise RuntimeError(f"Failed to load map '{chosen}'")


def apply_sync_settings(world: carla.World, client: carla.Client,
                        log_file) -> carla.WorldSettings:
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FRAME_DT
    settings.max_substep_delta_time = 0.01
    settings.max_substeps = 10

    last_error: Optional[Exception] = None
    for attempt in range(1, 5):
        try:
            client.set_timeout(90.0 + (attempt - 1) * 15.0)
            world.apply_settings(settings)
            client.set_timeout(45.0)
            return original
        except RuntimeError as exc:
            last_error = exc
            if not is_retryable_carla_error(exc):
                raise
            log(f"[WARN] apply_settings retryable failure "
                f"attempt={attempt}/4: {exc}", log_file)
            time.sleep(min(1.5 * attempt, 6.0))
    raise RuntimeError("Failed to apply synchronous world settings") from last_error


# ---------------------------------------------------------------------------
# Blueprint helpers
# ---------------------------------------------------------------------------
def find_blueprint(
    bp_lib: carla.BlueprintLibrary,
    preferred_ids: List[str],
    fallback_pattern: str,
    must_contain: Optional[str] = None,
) -> carla.ActorBlueprint:
    for bp_id in preferred_ids:
        try:
            return bp_lib.find(bp_id)
        except Exception:
            continue
    candidates = list(bp_lib.filter(fallback_pattern))
    if must_contain:
        narrowed = [bp for bp in candidates if must_contain in bp.id.lower()]
        if narrowed:
            candidates = narrowed
    if not candidates:
        raise RuntimeError(f"No blueprint matched {fallback_pattern}")
    return random.choice(candidates)


def set_vehicle_attributes(bp: carla.ActorBlueprint,
                           role_name: str = "autopilot") -> None:
    if bp.has_attribute("role_name"):
        bp.set_attribute("role_name", role_name)
    if bp.has_attribute("color"):
        colors = bp.get_attribute("color").recommended_values
        if colors:
            bp.set_attribute("color", random.choice(colors))
    if bp.has_attribute("driver_id"):
        ids = bp.get_attribute("driver_id").recommended_values
        if ids:
            bp.set_attribute("driver_id", random.choice(ids))


# ---------------------------------------------------------------------------
# Traffic spawning
# ---------------------------------------------------------------------------
def collect_flow_counts(
    world: carla.World, ego: carla.Vehicle, radius_m: float = 240.0,
) -> Tuple[Dict[str, int], int, int]:
    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    counts = {"same": 0, "opposite": 0}
    total = 0
    distinct = set()
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego.id:
            continue
        if distance_2d(actor.get_location(), ego_loc) > radius_m:
            continue
        flow = classify_flow_alignment(ego_tf, actor.get_transform())
        counts[flow] += 1
        total += 1
        distinct.add(actor.type_id)
    return counts, total, len(distinct)


def spawn_batch(
    client: carla.Client,
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    blueprints: List[carla.ActorBlueprint],
    transforms: List[carla.Transform],
    actors_to_cleanup: List[carla.Actor],
) -> int:
    if not transforms:
        return 0
    actor_ids: List[int] = []
    chunk_size = 12

    for start in range(0, len(transforms), chunk_size):
        batch = []
        for tf in transforms[start:start + chunk_size]:
            bp = random.choice(blueprints)
            set_vehicle_attributes(bp, role_name="autopilot")
            spawn = carla.command.SpawnActor(bp, tf)
            autopilot = carla.command.SetAutopilot(
                carla.command.FutureActor, True, traffic_manager.get_port())
            batch.append(spawn.then(autopilot))

        try:
            # Avoid server-side tick in apply_batch_sync for large batches; tick explicitly.
            responses = client.apply_batch_sync(batch, False)
        except RuntimeError as exc:
            if not is_retryable_carla_error(exc):
                raise
            responses = []
            for cmd in batch:
                try:
                    responses.extend(client.apply_batch_sync([cmd], False))
                except RuntimeError:
                    continue

        actor_ids.extend(r.actor_id for r in responses if not r.error)

        try:
            world.tick()
        except Exception:
            pass

    if not actor_ids:
        return 0

    for actor in world.get_actors(actor_ids):
        actors_to_cleanup.append(actor)
        try:
            traffic_manager.auto_lane_change(actor, False)
            traffic_manager.ignore_walkers_percentage(actor, 0.0)
            traffic_manager.vehicle_percentage_speed_difference(actor, UNIFORM_TRAFFIC_SPEED_DIFF)
        except Exception:
            pass
    return len(actor_ids)


def select_spawn_candidates_for_flow(
    available: List[carla.Transform],
    used_indices: set,
    ego_tf: carla.Transform,
    need_same: int,
    need_opposite: int,
    max_take: int,
) -> List[carla.Transform]:
    picks: List[carla.Transform] = []

    for idx, sp in enumerate(available):
        if idx in used_indices:
            continue
        flow = classify_flow_alignment(ego_tf, sp)
        if flow == "same" and need_same > 0:
            picks.append(sp)
            used_indices.add(idx)
            need_same -= 1
        elif flow == "opposite" and need_opposite > 0:
            picks.append(sp)
            used_indices.add(idx)
            need_opposite -= 1
        if len(picks) >= max_take:
            return picks

    for idx, sp in enumerate(available):
        if idx in used_indices:
            continue
        picks.append(sp)
        used_indices.add(idx)
        if len(picks) >= max_take:
            break
    return picks


def spawn_dense_highway_traffic(
    world: carla.World,
    client: carla.Client,
    traffic_manager: carla.TrafficManager,
    ego: carla.Vehicle,
    spawn_points: List[carla.Transform],
    vehicle_blueprints: List[carla.ActorBlueprint],
    actors_to_cleanup: List[carla.Actor],
    log_file,
) -> Dict[str, object]:
    """Highway-tuned traffic spawner: wider radius, preference for highway segments."""
    ego_loc = ego.get_location()
    world_map = world.get_map()
    close_available: List[carla.Transform] = []
    far_available: List[carla.Transform] = []
    for sp in spawn_points:
        d = distance_2d(sp.location, ego_loc)
        if 8.0 <= d <= 120.0:
            close_available.append(sp)
        elif 120.0 < d <= 250.0:
            far_available.append(sp)

    # Prefer highway spawns
    highway_close = []
    nonhighway_close = []
    for sp in close_available:
        wp = world_map.get_waypoint(sp.location, project_to_road=True,
                                    lane_type=carla.LaneType.Driving)
        if wp is not None and is_highway_candidate(wp):
            highway_close.append(sp)
        else:
            nonhighway_close.append(sp)
    available = highway_close + nonhighway_close + far_available

    random.shuffle(available)
    used_indices: set = set()
    ego_tf = ego.get_transform()
    ego_wp = world_map.get_waypoint(
        ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    ego_lane_w = max(3.4, float(getattr(ego_wp, "lane_width", 3.5))) if ego_wp is not None else 3.5

    def _spawn_priority(sp: carla.Transform) -> Tuple[int, float]:
        rel_fwd, rel_lat = relative_forward_lateral(ego_tf, sp.location)
        same_dir = classify_flow_alignment(ego_tf, sp) == "same"
        score = 0
        if same_dir:
            score += 1000
            if 8.0 <= rel_fwd <= 180.0:
                score += 500
            elif rel_fwd > 180.0:
                score += 250
            if abs(rel_lat) >= ego_lane_w * 0.75:
                score += 220
        else:
            if 20.0 <= rel_fwd <= 220.0:
                score += 80
        return (score, random.random())

    available = sorted(available, key=_spawn_priority, reverse=True)

    initial_pick = select_spawn_candidates_for_flow(
        available, used_indices, ego_tf,
        need_same=max(40, MIN_FLOW_PER_DIRECTION * 2),
        need_opposite=MIN_FLOW_PER_DIRECTION,
        max_take=min(len(available), 140),
    )
    spawned = spawn_batch(
        client, world, traffic_manager, vehicle_blueprints,
        initial_pick, actors_to_cleanup)
    for _ in range(35):
        safe_tick(world, "traffic_settle_initial", log_file, fatal=False)

    for attempt in range(8):
        flow_counts, total, distinct = collect_flow_counts(
            world, ego, radius_m=280.0)
        log(f"[TRAFFIC] attempt={attempt} flow={flow_counts} total={total} "
            f"distinct_types={distinct} spawned={spawned}", log_file)
        same_direction_target = 40
        need_same = max(0, same_direction_target - flow_counts["same"])
        need_opposite = max(0, MIN_FLOW_PER_DIRECTION - flow_counts["opposite"])
        total_deficit = max(0, MIN_TOTAL_VEHICLES - total)

        if need_same <= 0 and need_opposite <= 0 and total >= MIN_TOTAL_VEHICLES and distinct >= 8:
            return {
                "flow_counts": flow_counts,
                "total": total,
                "distinct_types": distinct,
                "spawned": spawned,
            }

        extra = select_spawn_candidates_for_flow(
            available, used_indices, ego_tf,
            need_same=max(need_same * 3, total_deficit),
            need_opposite=max(need_opposite * 2, total_deficit // 2),
            max_take=max(28, (need_same + need_opposite + total_deficit) * 2),
        )
        if not extra:
            break
        spawned += spawn_batch(
            client, world, traffic_manager, vehicle_blueprints,
            extra, actors_to_cleanup)
        for _ in range(22):
            safe_tick(world, "traffic_settle_extra", log_file, fatal=False)

    flow_counts, total, distinct = collect_flow_counts(
        world, ego, radius_m=280.0)
    if total < MIN_TOTAL_VEHICLES:
        log(f"[WARN] Traffic density target not fully met: "
            f"flow={flow_counts} total={total} distinct_types={distinct} "
            f"spawned={spawned} (continuing anyway)", log_file)
    return {
        "flow_counts": flow_counts,
        "total": total,
        "distinct_types": distinct,
        "spawned": spawned,
    }


# ===========================================================================
# Scenario-specific: Oil Spill Hazard (Shot 5)
# ===========================================================================

def apply_scene_weather(world: carla.World) -> None:
    """Grey overcast, wet road with high precipitation deposits for oil-sheen."""
    weather = carla.WeatherParameters(
        cloudiness=65.0,
        precipitation=5.0,
        precipitation_deposits=80.0,
        wind_intensity=18.0,
        fog_density=8.0,
        fog_distance=90.0,
        wetness=90.0,
        sun_altitude_angle=35.0,
    )
    world.set_weather(weather)


def _try_spawn_prop(
    world: carla.World,
    bp: carla.ActorBlueprint,
    location: carla.Location,
    rotation: carla.Rotation,
    actors_to_cleanup: List[carla.Actor],
) -> Optional[carla.Actor]:
    tf = carla.Transform(location, rotation)
    actor = world.try_spawn_actor(bp, tf)
    if actor is not None:
        actors_to_cleanup.append(actor)
    return actor


def _resolve_prop_bp(
    bp_lib: carla.BlueprintLibrary,
    preferred: List[str],
    fallback_pattern: str,
) -> Optional[carla.ActorBlueprint]:
    for bp_id in preferred:
        try:
            return bp_lib.find(bp_id)
        except Exception:
            continue
    candidates = list(bp_lib.filter(fallback_pattern))
    if candidates:
        return random.choice(candidates)
    return None


def _try_spawn_vehicle_prop(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    preferred_ids: List[str],
    fallback_pattern: str,
    location: carla.Location,
    rotation: carla.Rotation,
    actors_to_cleanup: List[carla.Actor],
    autopilot: bool = False,
    traffic_manager: Optional[carla.TrafficManager] = None,
) -> Optional[carla.Actor]:
    """Spawn a vehicle (e.g. emergency) as a scene prop."""
    bp = None
    for bp_id in preferred_ids:
        try:
            bp = bp_lib.find(bp_id)
            break
        except Exception:
            continue
    if bp is None:
        candidates = list(bp_lib.filter(fallback_pattern))
        if not candidates:
            return None
        bp = random.choice(candidates)

    set_vehicle_attributes(bp, role_name="scene_prop")
    tf = carla.Transform(location, rotation)
    actor = world.try_spawn_actor(bp, tf)
    if actor is not None:
        actors_to_cleanup.append(actor)
        if autopilot and traffic_manager is not None:
            try:
                actor.set_autopilot(True, traffic_manager.get_port())
                traffic_manager.vehicle_percentage_speed_difference(actor, 100.0)
            except Exception:
                pass
    return actor


def _freeze_scene_vehicle(actor: carla.Actor) -> None:
    try:
        actor.set_autopilot(False)
    except Exception:
        pass
    try:
        actor.set_simulate_physics(False)
    except Exception:
        pass
    try:
        actor.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
    except Exception:
        pass


def _set_vehicle_lights(actor: carla.Actor, emergency: bool = False) -> None:
    try:
        lights = carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        if emergency:
            lights = lights | carla.VehicleLightState.Special1 | carla.VehicleLightState.Special2
        actor.set_light_state(carla.VehicleLightState(lights))
    except Exception:
        pass


def stage_oil_spill_assets(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    hazard_wp: carla.Waypoint,
    actors_to_cleanup: List[carla.Actor],
    traffic_manager: carla.TrafficManager,
    log_file,
) -> Tuple[carla.Location, Dict[str, int]]:
    """Stage oil spill hazard zone on highway.

    Components: overturned truck (roll=170), oil-debris, cones, barriers,
    emergency vehicles (ambulance, police), additional debris.
    """
    hazard_wp = rightmost_driving_lane(hazard_wp)
    center = hazard_wp.transform.location
    forward = hazard_wp.transform.get_forward_vector()
    right = hazard_wp.transform.get_right_vector()
    base_yaw = hazard_wp.transform.rotation.yaw
    lane_w = max(3.4, float(getattr(hazard_wp, "lane_width", 3.5)))
    shoulder_lat = lane_w * 0.95
    stats: Dict[str, int] = {
        "overturned_truck": 0, "wrecked_cars": 0, "debris": 0, "cones": 0,
        "warnings": 0, "emergency_vehicles": 0, "barrels": 0,
    }

    # -- 1. Overturned truck centered on the rightmost lane oil-spill area ---
    truck = None
    truck_spawn_rot = carla.Rotation(yaw=base_yaw + 88.0)
    truck_lane_center_lat = -lane_w * 0.55  # center across rightmost + adjacent lane
    for spawn_z in (2.8, 3.3, 3.8):
        truck = _try_spawn_vehicle_prop(
            world,
            bp_lib,
            ["vehicle.carlamotors.european_hgv"],
            "vehicle.*hgv*",
            lane_offset_location(hazard_wp, forward_m=0.0, lateral_m=truck_lane_center_lat, z=spawn_z),
            truck_spawn_rot,
            actors_to_cleanup,
        )
        if truck is not None:
            break
    if truck is None:
        for spawn_z in (2.6, 3.0, 3.4):
            truck = _try_spawn_vehicle_prop(
                world,
                bp_lib,
                ["vehicle.carlamotors.firetruck"],
                "vehicle.*truck*",
                lane_offset_location(hazard_wp, forward_m=0.0, lateral_m=truck_lane_center_lat, z=spawn_z),
                truck_spawn_rot,
                actors_to_cleanup,
            )
            if truck is not None:
                break
    if truck is not None:
        placed = False
        for final_z in (3.0, 3.3, 3.6):
            try:
                truck.set_transform(
                    carla.Transform(
                        lane_offset_location(
                            hazard_wp,
                            forward_m=0.0,
                            lateral_m=truck_lane_center_lat,
                            z=final_z,
                        ),
                        carla.Rotation(yaw=base_yaw + 88.0, pitch=2.0, roll=108.0),
                    )
                )
                placed = True
                break
            except Exception:
                continue
        if not placed:
            try:
                truck.set_transform(
                    carla.Transform(
                        lane_offset_location(
                            hazard_wp,
                            forward_m=0.0,
                            lateral_m=truck_lane_center_lat,
                            z=4.0,
                        ),
                        carla.Rotation(yaw=base_yaw + 88.0, pitch=2.0, roll=108.0),
                    )
                )
            except Exception:
                pass
        _freeze_scene_vehicle(truck)
        stats["overturned_truck"] = 1

    # Wrecked car near the semi (looks like impact/collision aftermath)
    wreck = None
    for z_spawn in (0.25, 0.45, 0.7):
        wreck = _try_spawn_vehicle_prop(
            world,
            bp_lib,
            ["vehicle.audi.tt", "vehicle.dodge.charger_2020", "vehicle.nissan.patrol_2021"],
            "vehicle.*",
            lane_offset_location(
                hazard_wp,
                forward_m=-2.8,
                lateral_m=-lane_w * 1.1,
                z=z_spawn,
            ),
            carla.Rotation(yaw=base_yaw + 120.0, pitch=0.0, roll=8.0),
            actors_to_cleanup,
        )
        if wreck is not None:
            break
    if wreck is not None:
        _freeze_scene_vehicle(wreck)
        stats["wrecked_cars"] += 1

    # -- 2. Oil-spill/debris patch mostly on rightmost lane + shoulder -------
    debris_bp = _resolve_prop_bp(
        bp_lib,
        ["static.prop.dirtdebris01", "static.prop.dirtdebris02", "static.prop.dirtdebris03"],
        "static.prop.dirtdebris*",
    )
    if debris_bp is not None:
        for _ in range(34):
            loc = lane_offset_location(
                hazard_wp,
                forward_m=random.uniform(-8.0, 7.0),
                lateral_m=random.uniform(-lane_w * 0.35, lane_w * 0.85),
                z=0.05,
            )
            rot = carla.Rotation(yaw=random.uniform(0.0, 360.0))
            actor = _try_spawn_prop(world, debris_bp, loc, rot, actors_to_cleanup)
            if actor is not None:
                stats["debris"] += 1

    # -- 3. Barrels around the truck / shoulder ------------------------------
    barrel_bp = _resolve_prop_bp(bp_lib, ["static.prop.barrel"], "static.prop.barrel*")
    if barrel_bp is not None:
        barrel_offsets = [
            (-5.5, 0.6), (-4.0, 1.2), (-2.8, 0.2), (-1.4, 0.9),
            (1.0, 1.1), (2.3, 0.4), (3.8, 1.4), (5.0, 0.8),
        ]
        for fwd_off, lat_off in barrel_offsets:
            loc = lane_offset_location(
                hazard_wp,
                forward_m=fwd_off + random.uniform(-0.5, 0.5),
                lateral_m=lat_off + random.uniform(-0.25, 0.25),
                z=0.08,
            )
            rot = carla.Rotation(
                yaw=random.uniform(0.0, 360.0),
                pitch=random.uniform(-10.0, 15.0),
                roll=random.uniform(-15.0, 15.0),
            )
            actor = _try_spawn_prop(world, barrel_bp, loc, rot, actors_to_cleanup)
            if actor is not None:
                stats["barrels"] += 1

    # -- 4. Lane-closure cones: taper + perimeter around the truck ----------
    cone_bp = _resolve_prop_bp(
        bp_lib,
        ["static.prop.constructioncone", "static.prop.trafficcone01", "static.prop.trafficcone02"],
        "static.prop.*cone*",
    )
    if cone_bp is not None:
        # Taper to close the rightmost lane while leaving adjacent lanes open.
        taper = [
            (70.0, lane_w * 0.95), (62.0, lane_w * 0.85), (54.0, lane_w * 0.72),
            (46.0, lane_w * 0.58), (38.0, lane_w * 0.46), (32.0, lane_w * 0.35),
            (26.0, lane_w * 0.25), (20.0, lane_w * 0.15), (14.0, lane_w * 0.10),
        ]
        for dist_back, lat in taper:
            for lat_delta in (0.0, -0.9):
                actor = _try_spawn_prop(
                    world,
                    cone_bp,
                    lane_offset_location(
                        hazard_wp,
                        forward_m=-dist_back,
                        lateral_m=lat + lat_delta + random.uniform(-0.12, 0.12),
                        z=0.05,
                    ),
                    carla.Rotation(yaw=random.uniform(0.0, 360.0)),
                    actors_to_cleanup,
                )
                if actor is not None:
                    stats["cones"] += 1

        perimeter_offsets = [
            (-6.5, -0.7), (-5.5, 0.7), (-3.5, -0.9), (-2.5, 1.0),
            (-0.5, -0.8), (0.8, 1.0), (2.5, -0.6), (3.5, 1.1),
            (5.2, -0.4), (6.0, 1.0),
        ]
        for fwd_off, lat_off in perimeter_offsets:
            actor = _try_spawn_prop(
                world,
                cone_bp,
                lane_offset_location(
                    hazard_wp,
                    forward_m=fwd_off + random.uniform(-0.15, 0.15),
                    lateral_m=lat_off + random.uniform(-0.15, 0.15),
                    z=0.05,
                ),
                carla.Rotation(yaw=random.uniform(0.0, 360.0)),
                actors_to_cleanup,
            )
            if actor is not None:
                stats["cones"] += 1

    # -- 5. Lane-closed signs/barriers on right shoulder ---------------------
    warning_bp = _resolve_prop_bp(
        bp_lib,
        ["static.prop.warningconstruction", "static.prop.warningaccident",
         "static.prop.trafficwarning", "static.prop.streetbarrier"],
        "static.prop.*warning*",
    )
    if warning_bp is not None:
        sign_layout = [
            (-72.0, lane_w * 1.20, 0.20, 0.0),
            (-50.0, lane_w * 1.05, 0.18, 4.0),
            (-28.0, lane_w * 0.95, 0.18, 6.0),
            (-10.0, lane_w * 0.75, 0.15, 8.0),
        ]
        for fwd_off, lat_off, z_off, yaw_delta in sign_layout:
            actor = _try_spawn_prop(
                world,
                warning_bp,
                lane_offset_location(hazard_wp, forward_m=fwd_off, lateral_m=lat_off, z=z_off),
                carla.Rotation(yaw=base_yaw + yaw_delta),
                actors_to_cleanup,
            )
            if actor is not None:
                stats["warnings"] += 1

    # -- 6. Emergency response vehicles around the overturned truck ----------
    emergency_layout = [
        {
            "name": "ambulance",
            "preferred": ["vehicle.ford.ambulance", "vehicle.mercedes.sprinter"],
            "fallback": "vehicle.*ambulance*",
            # Place ambulance after (past) the overturned truck.
            "fwd_off": -22.0,
            "lat_off": lane_w * 0.35,
            "yaw_delta": 5.0,
            "emergency_lights": True,
        },
        {
            "name": "police",
            "preferred": ["vehicle.dodge.charger_police_2020", "vehicle.dodge.charger_police"],
            "fallback": "vehicle.*police*",
            "fwd_off": -18.0,
            "lat_off": -lane_w * 0.35,
            "yaw_delta": 7.0,
            "emergency_lights": True,
        },
        {
            "name": "police_beside_truck",
            "preferred": ["vehicle.dodge.charger_police_2020", "vehicle.dodge.charger_police"],
            "fallback": "vehicle.*police*",
            "fwd_off": 1.0,
            "lat_off": lane_w * 0.32,
            "yaw_delta": 6.0,
            "emergency_lights": True,
        },
    ]
    for cfg in emergency_layout:
        actor = None
        for z_spawn in (0.2, 0.4, 0.7):
            actor = _try_spawn_vehicle_prop(
                world,
                bp_lib,
                cfg["preferred"],
                cfg["fallback"],
                lane_offset_location(
                    hazard_wp,
                    forward_m=cfg["fwd_off"],
                    lateral_m=cfg["lat_off"],
                    z=z_spawn,
                ),
                carla.Rotation(yaw=base_yaw + cfg["yaw_delta"]),
                actors_to_cleanup,
            )
            if actor is not None:
                break
        if actor is not None:
            _freeze_scene_vehicle(actor)
            _set_vehicle_lights(actor, emergency=bool(cfg.get("emergency_lights")))
            stats["emergency_vehicles"] += 1

    log(f"[INFO] Oil spill assets staged: {stats}", log_file)
    return center, stats


# ---------------------------------------------------------------------------
# Camera rig
# ---------------------------------------------------------------------------
def clear_frame_outputs(output_dir: Path) -> Dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    camera_dirs: Dict[str, Path] = {}
    for key in CAMERA_KEYS:
        path = output_dir / key
        path.mkdir(parents=True, exist_ok=True)
        camera_dirs[key] = path
        for ext in ("*.png", "*.jpg", "*.jpeg"):
            for img in path.glob(ext):
                try:
                    img.unlink()
                except Exception:
                    pass
    return camera_dirs


def make_camera_callback(camera_key: str):
    def callback(image: carla.Image) -> None:
        with images_lock:
            images_received[camera_key] = image
            if all(images_received[k] is not None for k in CAMERA_KEYS):
                frame_ready.set()
    return callback


def clear_camera_buffers() -> None:
    with images_lock:
        for key in CAMERA_KEYS:
            images_received[key] = None
    frame_ready.clear()


def attach_cameras(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    ego: carla.Vehicle,
    actors_to_cleanup: List[carla.Actor],
    log_file,
) -> None:
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "110")
    cam_bp.set_attribute("sensor_tick", str(FRAME_DT))

    transforms = {
        "front": carla.Transform(
            carla.Location(x=0.8, y=0.0, z=1.4),
            carla.Rotation(pitch=8.0)),
        "front_left": carla.Transform(
            carla.Location(x=-0.1, y=-0.4, z=1.2),
            carla.Rotation(yaw=-60.0)),
        "front_right": carla.Transform(
            carla.Location(x=-0.1, y=0.4, z=1.2),
            carla.Rotation(yaw=60.0)),
        "rear": carla.Transform(
            carla.Location(x=-0.2, y=0.3, z=1.25),
            carla.Rotation(yaw=180.0, pitch=5.0)),
        "drone_follow": carla.Transform(
            carla.Location(x=-12.0, y=0.0, z=12.0),
            carla.Rotation(pitch=-25.0)),
    }

    for key in CAMERA_KEYS:
        tf = transforms[key]
        cam = world.spawn_actor(cam_bp, tf, attach_to=ego)
        actors_to_cleanup.append(cam)
        cam.listen(make_camera_callback(key))

    for _ in range(20):
        safe_tick(world, "camera_warmup", log_file, retries=3, fatal=False)
    time.sleep(0.5)
    clear_camera_buffers()
    log(f"[INFO] Attached and warmed up {len(CAMERA_KEYS)} camera(s): "
        f"{', '.join(CAMERA_KEYS)}", log_file)


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------
def run_capture_loop(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    ego: carla.Vehicle,
    spill_center: carla.Location,
    camera_dirs: Dict[str, Path],
    duration_s: float,
    log_file,
) -> Dict[str, int]:
    max_frames = min(int(duration_s * 20), 500)
    frame_counts = {key: 0 for key in CAMERA_KEYS}
    log(f"[INFO] Starting simulation loop: target_frames={max_frames}", log_file)

    for frame_idx in range(1, max_frames + 1):
        if not safe_tick(world, f"run_frame_{frame_idx}", log_file,
                         retries=2, fatal=False):
            continue

        dist = distance_2d(ego.get_location(), spill_center)
        try:
            traffic_manager.vehicle_percentage_speed_difference(ego, EGO_APPROACH_SPEED_DIFF)
        except Exception:
            pass

        if frame_ready.wait(timeout=1.2):
            with images_lock:
                for key in CAMERA_KEYS:
                    image = images_received[key]
                    if image is None:
                        continue
                    out = camera_dirs[key] / f"{key}_frame_{frame_idx:08d}.png"
                    image.save_to_disk(str(out))
                    frame_counts[key] += 1
                    images_received[key] = None
            frame_ready.clear()
        else:
            clear_camera_buffers()
            if frame_idx % 20 == 0:
                log(f"[WARN] Camera frame timeout at frame={frame_idx}", log_file)

        if frame_idx % 20 == 0:
            flow_counts, total, distinct = collect_flow_counts(
                world, ego, radius_m=280.0)
            speed = get_speed_kmh(ego)
            log(f"[PROGRESS] frame={frame_idx}/{max_frames} "
                f"ego_speed={speed:.1f}km/h spill_dist={dist:.1f}m "
                f"flow={flow_counts} total={total} distinct={distinct} "
                f"captures={frame_counts}", log_file)

    for _ in range(8):
        safe_tick(world, "run_drain", log_file, retries=2, fatal=False)

    return frame_counts


# ===========================================================================
# Main entry point
# ===========================================================================
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Oil Spill Hazard on Highway -- Shot 5")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output-dir",
                        default="./scenes/oil_spill_hazard")
    parser.add_argument("--output", dest="output_alias", default=None)
    parser.add_argument("--seed", type=int, default=3406)
    parser.add_argument("--map", default="Town04")
    args = parser.parse_args()

    random.seed(int(args.seed))

    output_root = Path(args.output_alias or args.output_dir).resolve()
    camera_dirs = clear_frame_outputs(output_root)
    log_path = output_root / "oil_spill_hazard_simulation.log"
    output_root.mkdir(parents=True, exist_ok=True)

    client: Optional[carla.Client] = None
    world: Optional[carla.World] = None
    traffic_manager: Optional[carla.TrafficManager] = None
    original_settings: Optional[carla.WorldSettings] = None
    actors_to_cleanup: List[carla.Actor] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            log("[INFO] Scenario: Oil Spill Hazard on Highway (Shot 5)", log_file)
            log(f"[INFO] Seed: {args.seed}", log_file)
            log(f"[INFO] Connecting to CARLA at {args.host}:{args.port}", log_file)

            client = carla.Client(args.host, int(args.port))
            client.set_timeout(60.0)

            selected_map = choose_preferred_map(client, args.map, log_file)
            world = load_world_with_retry(client, selected_map, log_file, retries=4)
            client.set_timeout(45.0)

            world_map = world.get_map()
            active_map = map_token(world_map.name)
            log(f"[INFO] Active map: {active_map}", log_file)

            original_settings = apply_sync_settings(world, client, log_file)

            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(3.0)
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(90.0)
            traffic_manager.global_percentage_speed_difference(UNIFORM_TRAFFIC_SPEED_DIFF)

            apply_scene_weather(world)
            for _ in range(10):
                safe_tick(world, "setup_warmup", log_file, fatal=False)

            removed_existing = 0
            for actor in world.get_actors().filter("vehicle.*"):
                try:
                    actor.destroy()
                    removed_existing += 1
                except Exception:
                    pass
            log(f"[INFO] Cleared pre-existing vehicles: {removed_existing}", log_file)

            spawn_points = list(world_map.get_spawn_points())
            if not spawn_points:
                raise RuntimeError("No spawn points available on map")
            random.shuffle(spawn_points)

            bp_lib = world.get_blueprint_library()

            ego_bp = find_blueprint(
                bp_lib,
                ["vehicle.tesla.model3", "vehicle.lincoln.mkz_2020",
                 "vehicle.audi.tt"],
                "vehicle.*",
            )
            set_vehicle_attributes(ego_bp, role_name="ego")

            # Prefer spawning on a highway segment
            ego_vehicle: Optional[carla.Vehicle] = None
            for sp in spawn_points:
                wp = world_map.get_waypoint(
                    sp.location, project_to_road=True,
                    lane_type=carla.LaneType.Driving)
                if wp is None or wp.is_junction:
                    continue
                if not is_highway_candidate(wp):
                    continue
                if not has_long_drivable_segment(wp, min_len_m=160.0):
                    continue
                ego_spawn_wp = rightmost_driving_lane(wp)
                ego_tf = ego_spawn_wp.transform
                ego_vehicle = world.try_spawn_actor(
                    ego_bp,
                    carla.Transform(
                        carla.Location(
                            x=ego_tf.location.x,
                            y=ego_tf.location.y,
                            z=ego_tf.location.z + 0.15,
                        ),
                        ego_tf.rotation,
                    ),
                )
                if ego_vehicle is not None:
                    break

            # Fallback: any long segment
            if ego_vehicle is None:
                for sp in spawn_points:
                    wp = world_map.get_waypoint(
                        sp.location, project_to_road=True,
                        lane_type=carla.LaneType.Driving)
                    if wp is None or wp.is_junction:
                        continue
                    if not has_long_drivable_segment(wp, min_len_m=120.0):
                        continue
                    ego_spawn_wp = rightmost_driving_lane(wp)
                    ego_tf = ego_spawn_wp.transform
                    ego_vehicle = world.try_spawn_actor(
                        ego_bp,
                        carla.Transform(
                            carla.Location(
                                x=ego_tf.location.x,
                                y=ego_tf.location.y,
                                z=ego_tf.location.z + 0.15,
                            ),
                            ego_tf.rotation,
                        ),
                    )
                    if ego_vehicle is not None:
                        break

            if ego_vehicle is None:
                for sp in spawn_points:
                    wp = world_map.get_waypoint(
                        sp.location, project_to_road=True,
                        lane_type=carla.LaneType.Driving)
                    if wp is None:
                        continue
                    ego_spawn_wp = rightmost_driving_lane(wp)
                    ego_tf = ego_spawn_wp.transform
                    ego_vehicle = world.try_spawn_actor(
                        ego_bp,
                        carla.Transform(
                            carla.Location(
                                x=ego_tf.location.x,
                                y=ego_tf.location.y,
                                z=ego_tf.location.z + 0.15,
                            ),
                            ego_tf.rotation,
                        ),
                    )
                    if ego_vehicle is not None:
                        break
            if ego_vehicle is None:
                raise RuntimeError("Failed to spawn ego vehicle")

            actors_to_cleanup.append(ego_vehicle)
            try:
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=1.0, hand_brake=True))
            except Exception:
                pass
            log(f"[INFO] Ego spawned: {ego_vehicle.type_id} id={ego_vehicle.id}",
                log_file)

            attach_cameras(world, bp_lib, ego_vehicle, actors_to_cleanup, log_file)

            ego_wp = world_map.get_waypoint(
                ego_vehicle.get_location(), project_to_road=True,
                lane_type=carla.LaneType.Driving)
            if ego_wp is None:
                raise RuntimeError("Failed to resolve ego waypoint for hazard staging")
            ego_wp = rightmost_driving_lane(ego_wp)
            hazard_wp = rightmost_driving_lane(next_waypoint(ego_wp, 110.0))
            spill_center, asset_counts = stage_oil_spill_assets(
                world, bp_lib, hazard_wp, actors_to_cleanup,
                traffic_manager, log_file)
            log(f"[INFO] Oil spill centre at "
                f"({spill_center.x:.1f}, {spill_center.y:.1f}, "
                f"{spill_center.z:.1f})", log_file)

            for _ in range(15):
                safe_tick(world, "spill_settle", log_file, fatal=False)

            vehicle_blueprints = list(bp_lib.filter("vehicle.*"))
            traffic_diag = spawn_dense_highway_traffic(
                world=world,
                client=client,
                traffic_manager=traffic_manager,
                ego=ego_vehicle,
                spawn_points=spawn_points,
                vehicle_blueprints=vehicle_blueprints,
                actors_to_cleanup=actors_to_cleanup,
                log_file=log_file,
            )
            log(f"[INFO] Traffic density result: {traffic_diag}", log_file)

            # Start ego after scene staging so the front camera captures the full approach.
            try:
                ego_vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0, hand_brake=False))
            except Exception:
                pass
            ego_vehicle.set_autopilot(True, traffic_manager.get_port())
            traffic_manager.auto_lane_change(ego_vehicle, False)
            traffic_manager.ignore_lights_percentage(ego_vehicle, 0.0)
            traffic_manager.ignore_walkers_percentage(ego_vehicle, 0.0)
            traffic_manager.vehicle_percentage_speed_difference(ego_vehicle, EGO_APPROACH_SPEED_DIFF)
            for _ in range(4):
                safe_tick(world, "ego_rollout_start", log_file, fatal=False)

            frame_counts = run_capture_loop(
                world=world,
                traffic_manager=traffic_manager,
                ego=ego_vehicle,
                spill_center=spill_center,
                camera_dirs=camera_dirs,
                duration_s=float(args.duration),
                log_file=log_file,
            )
            log(f"[INFO] Capture complete: {frame_counts}", log_file)
            log("[SUCCESS] Oil Spill Hazard on Highway (Shot 5) scenario completed",
                log_file)

        except Exception as exc:
            log(f"[ERROR] {type(exc).__name__}: {exc}", log_file)
            raise

        finally:
            clear_camera_buffers()
            log("[INFO] Cleanup started", log_file)

            for actor in actors_to_cleanup:
                if actor is None:
                    continue
                if actor.type_id.startswith("sensor."):
                    try:
                        actor.stop()
                    except Exception:
                        pass

            if client is not None and actors_to_cleanup:
                destroy_cmds = [
                    carla.command.DestroyActor(actor.id)
                    for actor in actors_to_cleanup if actor is not None
                ]
                if destroy_cmds:
                    try:
                        client.apply_batch(destroy_cmds)
                    except Exception:
                        pass

            if traffic_manager is not None:
                try:
                    traffic_manager.set_synchronous_mode(False)
                except Exception:
                    pass

            if world is not None and original_settings is not None:
                try:
                    world.apply_settings(original_settings)
                except Exception:
                    pass

            log("[INFO] Cleanup complete", log_file)


if __name__ == "__main__":
    main()
