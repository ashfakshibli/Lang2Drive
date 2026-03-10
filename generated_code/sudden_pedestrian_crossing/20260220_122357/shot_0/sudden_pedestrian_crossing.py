#!/usr/bin/env python3
"""Standalone CARLA scenario script.

Scene: Sudden Pedestrian Crossing
Prompt: Generate Carla PythonAPI code for a four-lane city road where a pedestrian crowd suddenly runs from the sidewalk into the middle of the road when the ego vehicle approaches.

Scene Specifications:
Road Context: Signalized urban 4-way intersection baseline, two lanes per direction, with active cross-traffic in all approaches.
Traffic Density: Minimum traffic density: >=20 moving vehicles per approach direction (north/south/east/west equivalent) at event onset, with >=80 active vehicles in broader scene; fail run if thresholds are not met after retries.
Camera Contract: Save synchronized front, front_left, front_right, rear, and drone_follow streams using matched frame ids.
Event Contract: 20-30 pedestrians (mixed adults/children, male/female where supported) wait on a sidewalk and suddenly run into the road as ego approaches.
Success Criteria: Sudden crowd crossing onset is clearly visible on a city road with sidewalks and nearby traffic; ego/traffic reaction is observable.
"""

from __future__ import annotations

import argparse
import math
import random
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import carla
except Exception as exc:
    raise RuntimeError("CARLA Python API is required") from exc


FRAME_DT = 0.05
FPS = 20
CAMERA_KEYS = ["front"]
SCENARIO_KEYWORD = "sudden_pedestrian_crossing"
SCENE_KEYWORD = "Sudden Pedestrian Crossing"
SCENE_PROMPT = "Generate Carla PythonAPI code for a four-lane city road where a pedestrian crowd suddenly runs across the street outside of a crosswalk. Vehicles are approaching at normal urban speed and must react to avoid hitting the pedestrians."
SCENE_SPECIFICATIONS = "Road Context: Signalized urban 4-way intersection baseline, two lanes per direction, with active cross-traffic in all approaches.\nTraffic Density: Minimum traffic density: >=20 moving vehicles per approach direction (north/south/east/west equivalent) at event onset, with >=80 active vehicles in broader scene; fail run if thresholds are not met after retries.\nCamera Contract: Save synchronized front, front_left, front_right, rear, and drone_follow streams using matched frame ids.\nEvent Contract: Spawn 20-30 pedestrians on a sidewalk and trigger a sudden run into the road when ego approaches; use a city road with sidewalks.\nSuccess Criteria: Crossing event is visible with clear temporal onset and braking/yield response."
EVENT_MODE = "sudden_pedestrian_crowd_crossing"
ROAD_MODE = "urban"
WEATHER_MODE = "clear"
SHOT_INDEX = 0
MIN_TOTAL_ACTIVE = 80
MIN_PER_DIRECTION = 20
SCENE_DEFAULT_KEY = "scene_default"
CROWD_TRIGGER_DISTANCE_M = 60.0
CROWD_CROSSING_LOOKAHEAD_M = 10.0
CROWD_STAGE_LOOKAHEAD_M = 100.0
EGO_TARGET_SPEED_KMH = 35.0
EGO_BRAKE_DISTANCE_M = 30.0
EGO_STOP_DISTANCE_M = 15.0
MIN_CROWD_SIZE = 5
MAX_CROWD_SIZE = 8
SIDEWALK_SEGMENT_SPACING_M = 10.0
SIDEWALK_SEGMENT_ALONG_OFFSETS_M = (-10.0, 0.0, 10.0)
SIDEWALK_LANE = getattr(carla.LaneType, "Sidewalk", carla.LaneType.Any)
SCENE_LAYOUT_SEED = 20260220

images_received: Dict[str, Optional[carla.Image]] = {k: None for k in CAMERA_KEYS}
images_lock = threading.Lock()
frame_ready = threading.Event()
ENV_CONFIG: Dict[str, Optional[str]] = {
    'time_preset': 'scene_default',
    'weather_preset': 'scene_default',
    'sun_altitude': None,
    'sun_azimuth': None,
    'streetlights': 'auto',
    'cloudiness': None,
    'precipitation': None,
    'precipitation_deposits': None,
    'wind_intensity': None,
    'fog_density': None,
    'fog_distance': None,
    'wetness': None,
}
NIGHTLIKE_TIME_PRESETS = {'night', 'almost_night_no_streetlights'}


def log(msg: str, log_file) -> None:
    line = msg.rstrip()
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def configure_environment_from_args(args: argparse.Namespace) -> None:
    ENV_CONFIG['time_preset'] = str(getattr(args, 'time_preset', 'scene_default') or 'scene_default')
    ENV_CONFIG['weather_preset'] = str(getattr(args, 'weather_preset', 'scene_default') or 'scene_default')
    ENV_CONFIG['sun_altitude'] = getattr(args, 'sun_altitude', None)
    ENV_CONFIG['sun_azimuth'] = getattr(args, 'sun_azimuth', None)
    ENV_CONFIG['streetlights'] = str(getattr(args, 'streetlights', 'auto') or 'auto').lower()
    ENV_CONFIG['cloudiness'] = getattr(args, 'cloudiness', None)
    ENV_CONFIG['precipitation'] = getattr(args, 'precipitation', None)
    ENV_CONFIG['precipitation_deposits'] = getattr(args, 'precipitation_deposits', None)
    ENV_CONFIG['wind_intensity'] = getattr(args, 'wind_intensity', None)
    ENV_CONFIG['fog_density'] = getattr(args, 'fog_density', None)
    ENV_CONFIG['fog_distance'] = getattr(args, 'fog_distance', None)
    ENV_CONFIG['wetness'] = getattr(args, 'wetness', None)


def _matrix_night_like() -> bool:
    return str(ENV_CONFIG.get('time_preset', '')).strip().lower() in NIGHTLIKE_TIME_PRESETS


def apply_vehicle_headlights_for_matrix(world: carla.World, log_file) -> None:
    if not _matrix_night_like():
        return
    try:
        state = carla.VehicleLightState(
            carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        )
    except Exception:
        return
    updated = 0
    for actor in world.get_actors().filter('vehicle.*'):
        try:
            actor.set_light_state(state)
            updated += 1
        except Exception:
            pass
    if updated > 0:
        log(f"[ENV] Applied night headlight matrix rule to {updated} vehicles", log_file)


def save_image_to_disk(image: carla.Image, output_path: Path) -> bool:
    try:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        array = array[:, :, :3][:, :, ::-1]
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
    return raw.rsplit("/", 1)[-1] if "/" in raw else raw


def is_retryable_carla_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "time-out",
            "failed to connect",
            "connection closed",
            "rpc",
            "socket",
            "timeout",
        )
    )


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


def choose_map(client: carla.Client, log_file) -> str:
    preferred = ["Town04", "Town05", "Town06"] if ROAD_MODE == "highway" else ["Town03", "Town05", "Town10HD"]
    try:
        available = [map_token(v) for v in client.get_available_maps()]
    except Exception as exc:
        log(f"[WARN] Failed to query maps: {exc}; defaulting to {preferred[0]}", log_file)
        return preferred[0]

    by_lower = {m.lower(): m for m in available}
    for candidate in preferred:
        if candidate.lower() in by_lower:
            log(f"[INFO] Map selected: {by_lower[candidate.lower()]}", log_file)
            return by_lower[candidate.lower()]
        for avail in available:
            if avail.lower().startswith(candidate.lower()):
                log(f"[INFO] Map selected by prefix: {avail}", log_file)
                return avail

    fallback = available[0] if available else preferred[0]
    log(f"[WARN] Preferred maps unavailable, fallback={fallback}", log_file)
    return fallback


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
            try:
                probe_world = client.get_world()
                active = map_token(probe_world.get_map().name)
                if active.lower() == target_map.lower():
                    log(f"[INFO] Reusing already-active target map {active}", log_file)
                    return probe_world
            except Exception:
                pass
            time.sleep(min(1.0 * attempt, 4.0))

    if last_error is not None:
        raise RuntimeError(f"Failed to load map {target_map}") from last_error
    raise RuntimeError(f"Failed to load map {target_map}")


def apply_sync_settings(world: carla.World, client: carla.Client, log_file) -> carla.WorldSettings:
    original = world.get_settings()
    settings = world.get_settings()
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = FRAME_DT
    settings.max_substeps = 10
    settings.max_substep_delta_time = 0.01
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
    raise RuntimeError("Failed to enable synchronous settings")


def find_blueprint(bp_lib: carla.BlueprintLibrary, preferred_ids: List[str], fallback_pattern: str) -> carla.ActorBlueprint:
    for bp_id in preferred_ids:
        try:
            return bp_lib.find(bp_id)
        except Exception:
            continue
    candidates = list(bp_lib.filter(fallback_pattern))
    if not candidates:
        raise RuntimeError(f"No blueprint available for pattern={fallback_pattern}")
    return random.choice(candidates)


def set_vehicle_attributes(bp: carla.ActorBlueprint, role_name: str = "autopilot") -> None:
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


def estimate_distance_to_junction(wp: carla.Waypoint, step_m: float = 5.0, max_checks: int = 18) -> float:
    probe = wp
    traversed = 0.0
    for _ in range(max_checks):
        if probe.is_junction:
            return traversed
        nxt = probe.next(step_m)
        if not nxt:
            break
        probe = nxt[0]
        traversed += step_m
    return 999.0


def drive_ego_straight(
    ego: carla.Vehicle,
    world_map: carla.Map,
    target_speed_kmh: float = EGO_TARGET_SPEED_KMH,
    lookahead_m: float = 8.0,
) -> None:
    """Apply throttle/steer/brake to keep ego driving straight along waypoints."""
    vel = ego.get_velocity()
    speed_kmh = 3.6 * math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)

    tf = ego.get_transform()
    wp = world_map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
    if wp is None:
        ego.apply_control(carla.VehicleControl(throttle=0.4, steer=0.0))
        return

    # Look ahead along the road for a steering target.
    target_wp = wp
    for nxt in (wp.next(lookahead_m) or []):
        target_wp = nxt
        break

    # Compute steering toward the target waypoint.
    target_loc = target_wp.transform.location
    ego_fwd = tf.get_forward_vector()
    dx = target_loc.x - tf.location.x
    dy = target_loc.y - tf.location.y
    dist = math.sqrt(dx * dx + dy * dy)
    if dist > 0.1:
        # Cross product gives signed lateral error: positive = target is to the right.
        cross = ego_fwd.x * dy - ego_fwd.y * dx
        steer = max(-0.3, min(0.3, cross / dist * 2.0))
    else:
        steer = 0.0

    # Simple proportional speed control.
    if target_speed_kmh < 1.0:
        # Emergency stop requested.
        throttle = 0.0
        brake = 1.0 if speed_kmh > 2.0 else 0.5
    else:
        speed_error = target_speed_kmh - speed_kmh
        if speed_error > 2.0:
            throttle = min(0.7, 0.15 + speed_error * 0.02)
            brake = 0.0
        elif speed_error < -5.0:
            throttle = 0.0
            brake = min(0.8, abs(speed_error) * 0.04)
        else:
            throttle = 0.2
            brake = 0.0

    ego.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=brake))


def find_sidewalk_anchor(
    world_map: carla.Map,
    center_wp: carla.Waypoint,
    side_sign: float,
) -> Optional[carla.Waypoint]:
    forward = center_wp.transform.get_forward_vector()
    right = center_wp.transform.get_right_vector()
    samples_along = [-10.0, -6.0, -2.0, 2.0, 6.0, 10.0]
    lateral_offsets = [6.0, 7.5, 9.0, 10.5, 12.0, 14.0]

    for along in samples_along:
        for lateral in lateral_offsets:
            loc = center_wp.transform.location + carla.Location(
                x=forward.x * along + right.x * lateral * side_sign,
                y=forward.y * along + right.y * lateral * side_sign,
                z=0.3,
            )
            wp = world_map.get_waypoint(loc, project_to_road=True, lane_type=SIDEWALK_LANE)
            if wp is not None and wp.lane_type == SIDEWALK_LANE:
                return wp
    return None


def stable_transform_sort_key(tf: carla.Transform) -> Tuple[float, float, float, float]:
    loc = tf.location
    yaw = tf.rotation.yaw % 360.0
    return (round(loc.x, 2), round(loc.y, 2), round(loc.z, 2), round(yaw, 1))


def pick_city_ego_spawn(world_map: carla.Map, spawn_points: List[carla.Transform]) -> Optional[carla.Transform]:
    candidates = sorted(spawn_points, key=stable_transform_sort_key)
    best_tf: Optional[carla.Transform] = None
    best_score: Optional[Tuple[int, float, float, float, float, float]] = None

    for tf in candidates:
        wp = world_map.get_waypoint(tf.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None or wp.is_junction:
            continue

        dist_to_junction = estimate_distance_to_junction(wp)
        if not (25.0 <= dist_to_junction <= 85.0):
            continue

        sidewalk_hits = 0
        for side_sign in (-1.0, 1.0):
            if find_sidewalk_anchor(world_map, wp, side_sign) is not None:
                sidewalk_hits += 1
        if sidewalk_hits >= 1:
            x, y, z, yaw = stable_transform_sort_key(tf)
            score = (
                0 if sidewalk_hits >= 2 else 1,
                abs(dist_to_junction - 55.0),
                x,
                y,
                z,
                yaw,
            )
            if best_score is None or score < best_score:
                best_score = score
                best_tf = tf

    return best_tf


def distance_2d(a: carla.Location, b: carla.Location) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def clear_ego_to_event_corridor(
    world: carla.World,
    ego: carla.Vehicle,
    target_loc: Optional[carla.Location],
    log_file,
    lateral_half_width_m: float = 8.5,
    rear_buffer_m: float = 1.5,
    front_buffer_m: float = 10.0,
) -> int:
    if target_loc is None:
        return 0

    ego_loc = ego.get_location()
    vx = target_loc.x - ego_loc.x
    vy = target_loc.y - ego_loc.y
    seg_len = math.sqrt(vx * vx + vy * vy)
    if seg_len < 3.0:
        return 0

    ux = vx / seg_len
    uy = vy / seg_len
    removed = 0

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego.id:
            continue
        try:
            loc = actor.get_location()
        except Exception:
            continue

        rx = loc.x - ego_loc.x
        ry = loc.y - ego_loc.y
        proj = rx * ux + ry * uy
        lateral = abs((-uy * rx) + (ux * ry))

        if -rear_buffer_m <= proj <= (seg_len + front_buffer_m) and lateral <= lateral_half_width_m:
            try:
                actor.destroy()
                removed += 1
            except Exception:
                pass

    if removed > 0:
        log(
            f"[TRAFFIC] Cleared ego->event corridor vehicles removed={removed} "
            f"segment_len={seg_len:.1f}m width={lateral_half_width_m:.1f}m",
            log_file,
        )
    return removed


def collect_density(world: carla.World, ego: carla.Vehicle, radius_m: float = 260.0) -> Dict[str, int]:
    counts = {
        "ahead": 0,
        "behind": 0,
        "left": 0,
        "right": 0,
        "same": 0,
        "opposite": 0,
        "total": 0,
        "distinct": 0,
    }

    ego_tf = ego.get_transform()
    ego_loc = ego_tf.location
    forward = ego_tf.get_forward_vector()
    distinct_types = set()

    for actor in world.get_actors().filter("vehicle.*"):
        if actor.id == ego.id:
            continue
        loc = actor.get_location()
        if distance_2d(loc, ego_loc) > radius_m:
            continue

        rel_x = loc.x - ego_loc.x
        rel_y = loc.y - ego_loc.y
        local_fwd = rel_x * forward.x + rel_y * forward.y
        local_side = -rel_x * forward.y + rel_y * forward.x

        if abs(local_fwd) >= abs(local_side):
            if local_fwd >= 0:
                counts["ahead"] += 1
            else:
                counts["behind"] += 1
        else:
            if local_side >= 0:
                counts["right"] += 1
            else:
                counts["left"] += 1

        other_fwd = actor.get_transform().get_forward_vector()
        dot = forward.x * other_fwd.x + forward.y * other_fwd.y + forward.z * other_fwd.z
        if dot >= 0:
            counts["same"] += 1
        else:
            counts["opposite"] += 1

        counts["total"] += 1
        distinct_types.add(actor.type_id)

    counts["distinct"] = len(distinct_types)
    return counts


def spawn_batch_vehicles(
    client: carla.Client,
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    blueprints: List[carla.ActorBlueprint],
    transforms: List[carla.Transform],
    actors_to_cleanup: List[carla.Actor],
) -> int:
    if not transforms:
        return 0

    batch = []
    tm_port = traffic_manager.get_port()
    for tf in transforms:
        bp = random.choice(blueprints)
        set_vehicle_attributes(bp, role_name="autopilot")
        cmd = carla.command.SpawnActor(bp, tf).then(
            carla.command.SetAutopilot(carla.command.FutureActor, True, tm_port)
        )
        batch.append(cmd)

    responses = client.apply_batch_sync(batch, True)
    actor_ids = [r.actor_id for r in responses if not r.error]
    if not actor_ids:
        return 0

    for actor in world.get_actors(actor_ids):
        actors_to_cleanup.append(actor)
        try:
            traffic_manager.auto_lane_change(actor, True)
            traffic_manager.vehicle_percentage_speed_difference(actor, random.uniform(8.0, 35.0))
            traffic_manager.distance_to_leading_vehicle(actor, random.uniform(3.0, 6.0))
            traffic_manager.ignore_walkers_percentage(actor, 0.0)
        except Exception:
            pass
    return len(actor_ids)


def spawn_dense_traffic(
    world: carla.World,
    client: carla.Client,
    traffic_manager: carla.TrafficManager,
    ego: carla.Vehicle,
    spawn_points: List[carla.Transform],
    vehicle_blueprints: List[carla.ActorBlueprint],
    actors_to_cleanup: List[carla.Actor],
    log_file,
) -> Dict[str, int]:
    ego_loc = ego.get_location()
    candidates = [sp for sp in spawn_points if 20.0 <= distance_2d(sp.location, ego_loc) <= 360.0]
    random.shuffle(candidates)

    target_total = 110 if ROAD_MODE != "highway" else 95
    chunk_size = 28
    spawned = 0

    for idx in range(0, min(len(candidates), target_total + 80), chunk_size):
        chunk = candidates[idx : idx + chunk_size]
        if not chunk:
            break
        spawned += spawn_batch_vehicles(
            client,
            world,
            traffic_manager,
            vehicle_blueprints,
            chunk,
            actors_to_cleanup,
        )
        for _ in range(8):
            safe_tick(world, "traffic_settle", log_file, fatal=False)

        density = collect_density(world, ego)
        log(
            f"[TRAFFIC] spawned={spawned} density={density}",
            log_file,
        )

        if ROAD_MODE == "highway":
            if (
                density["same"] >= MIN_PER_DIRECTION
                and density["opposite"] >= MIN_PER_DIRECTION
                and density["total"] >= MIN_TOTAL_ACTIVE
                and density["distinct"] >= 8
            ):
                return density
        else:
            if (
                density["ahead"] >= MIN_PER_DIRECTION
                and density["behind"] >= MIN_PER_DIRECTION
                and density["left"] >= MIN_PER_DIRECTION
                and density["right"] >= MIN_PER_DIRECTION
                and density["total"] >= MIN_TOTAL_ACTIVE
                and density["distinct"] >= 8
            ):
                return density

    density = collect_density(world, ego)
    log(
        f"[WARN] Density target not fully met; achieved density={density}",
        log_file,
    )
    return density


def spawn_context_props(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    ego: carla.Vehicle,
    spawn_points: List[carla.Transform],
    actors_to_cleanup: List[carla.Actor],
    log_file,
) -> int:
    patterns = [
        "static.prop.trafficcone*",
        "static.prop.warning*",
        "static.prop.streetbarrier*",
        "static.prop.barrel*",
        "static.prop.construction*",
    ]

    ego_loc = ego.get_location()
    nearby = [sp for sp in spawn_points if 12.0 <= distance_2d(sp.location, ego_loc) <= 120.0]
    random.shuffle(nearby)

    spawned = 0
    map_obj = world.get_map()

    for base_tf in nearby:
        if spawned >= 34:
            break
        base_wp = map_obj.get_waypoint(base_tf.location)
        if base_wp is None:
            continue

        right = base_wp.transform.get_right_vector()
        lateral = random.choice([-1.0, 1.0]) * random.uniform(6.0, 11.0)
        target_loc = base_tf.location + carla.Location(x=right.x * lateral, y=right.y * lateral, z=0.15)

        lane_wp = map_obj.get_waypoint(target_loc, project_to_road=False, lane_type=carla.LaneType.Any)
        if lane_wp is not None and lane_wp.lane_type == carla.LaneType.Driving:
            continue

        pattern = random.choice(patterns)
        choices = list(bp_lib.filter(pattern))
        if not choices:
            continue
        bp = random.choice(choices)
        tf = carla.Transform(target_loc, carla.Rotation(yaw=random.uniform(-180.0, 180.0)))
        actor = world.try_spawn_actor(bp, tf)
        if actor is not None:
            actors_to_cleanup.append(actor)
            spawned += 1

    log(f"[INFO] Context props spawned={spawned}", log_file)
    return spawned


def apply_weather(world: carla.World, phase: float) -> None:
    weather = carla.WeatherParameters()

    envelope = max(0.0, min(1.0, (phase - 0.2) / 0.5))
    envelope = math.sin(math.pi * envelope)

    weather.sun_altitude_angle = 50.0
    weather.cloudiness = 20.0 + 60.0 * envelope
    weather.wind_intensity = 12.0 + 50.0 * envelope
    weather.precipitation = 0.0
    weather.precipitation_deposits = 0.0
    weather.wetness = 10.0
    weather.fog_density = 2.0 + 20.0 * envelope
    weather.fog_distance = 80.0
    weather.fog_falloff = 0.35
    weather.scattering_intensity = 0.7
    weather.mie_scattering_scale = 0.05

    if WEATHER_MODE == "sandstorm":
        weather.fog_density = 25.0 + 45.0 * envelope
        weather.cloudiness = 60.0 + 35.0 * envelope
        weather.wind_intensity = 45.0 + 45.0 * envelope
        if hasattr(weather, "dust_storm"):
            weather.dust_storm = 35.0 + 65.0 * envelope
    elif WEATHER_MODE == "snow":
        weather.precipitation = 75.0 + 25.0 * envelope
        weather.precipitation_deposits = 80.0 + 20.0 * envelope
        weather.wetness = 35.0
        weather.fog_density = 8.0 + 14.0 * envelope
        weather.cloudiness = 85.0
        if hasattr(weather, "snow"):
            weather.snow = 55.0 + 25.0 * envelope
    elif WEATHER_MODE == "ash":
        weather.cloudiness = 90.0
        weather.fog_density = 35.0 + 45.0 * envelope
        weather.wind_intensity = 35.0 + 40.0 * envelope
        if hasattr(weather, "dust_storm"):
            weather.dust_storm = 25.0 + 45.0 * envelope
    elif WEATHER_MODE == "smoke":
        weather.cloudiness = 80.0
        weather.fog_density = 18.0 + 50.0 * envelope
        weather.wind_intensity = 28.0 + 30.0 * envelope
        weather.precipitation = 5.0
    elif WEATHER_MODE == "anomaly":
        weather.cloudiness = 95.0
        weather.sun_altitude_angle = -8.0 + 20.0 * (1.0 - envelope)
        weather.fog_density = 22.0 + 30.0 * envelope
        weather.scattering_intensity = 1.2 + 1.0 * envelope
        weather.mie_scattering_scale = 0.2 + 0.5 * envelope

    if ENV_CONFIG.get('sun_altitude') is not None:
        weather.sun_altitude_angle = float(ENV_CONFIG['sun_altitude'])
    if ENV_CONFIG.get('sun_azimuth') is not None:
        try:
            weather.sun_azimuth_angle = float(ENV_CONFIG['sun_azimuth'])
        except Exception:
            pass
    if ENV_CONFIG.get('cloudiness') is not None:
        weather.cloudiness = float(ENV_CONFIG['cloudiness'])
    if ENV_CONFIG.get('precipitation') is not None:
        weather.precipitation = float(ENV_CONFIG['precipitation'])
    if ENV_CONFIG.get('precipitation_deposits') is not None:
        weather.precipitation_deposits = float(ENV_CONFIG['precipitation_deposits'])
    if ENV_CONFIG.get('wind_intensity') is not None:
        weather.wind_intensity = float(ENV_CONFIG['wind_intensity'])
    if ENV_CONFIG.get('fog_density') is not None:
        weather.fog_density = float(ENV_CONFIG['fog_density'])
    if ENV_CONFIG.get('fog_distance') is not None:
        weather.fog_distance = float(ENV_CONFIG['fog_distance'])
    if ENV_CONFIG.get('wetness') is not None:
        weather.wetness = float(ENV_CONFIG['wetness'])

    world.set_weather(weather)


def setup_cameras(world: carla.World, ego: carla.Vehicle, actors_to_cleanup: List[carla.Actor]):
    cam_bp = world.get_blueprint_library().find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "1280")
    cam_bp.set_attribute("image_size_y", "720")
    cam_bp.set_attribute("fov", "110")

    configs = {
        "front": carla.Transform(carla.Location(x=0.8, y=0.0, z=1.4), carla.Rotation(pitch=8)),
    }

    cameras = {}
    for name, tf in configs.items():
        cam = world.spawn_actor(cam_bp, tf, attach_to=ego)
        cam.listen(make_camera_callback(name))
        cameras[name] = cam
        actors_to_cleanup.append(cam)
    return cameras


def stage_primary_event(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    ego: carla.Vehicle,
    actors_to_cleanup: List[carla.Actor],
    log_file,
) -> Dict[str, object]:
    state: Dict[str, object] = {
        "event_vehicle": None,
        "event_props": [],
        "traffic_lights": [],
        "drone_actor": None,
        "dropped": False,
        "hood_actor": None,
        "center": None,
        "crowd_center": None,
        "crowd_plans": [],
        "crowd_triggered": False,
        "crowd_trigger_time": None,
        "crowd_trigger_distance_m": CROWD_TRIGGER_DISTANCE_M,
        "crowd_started_count": 0,
        "ego_crossing_anchor": None,
        "ego_cross_forward": None,
        "ego_cross_right": None,
        "ego_signal_lock_started": False,
        "ego_signal_locked_light_id": None,
    }

    map_obj = world.get_map()
    ego_wp = map_obj.get_waypoint(ego.get_location())
    if ego_wp is None:
        return state

    center_wp = ego_wp
    center_step_m = 5.0
    center_advance_steps = 8
    if EVENT_MODE == "sudden_pedestrian_crowd_crossing":
        center_advance_steps = max(1, int(round(CROWD_STAGE_LOOKAHEAD_M / center_step_m)))
    for _ in range(center_advance_steps):
        nxt = center_wp.next(center_step_m)
        if not nxt:
            break
        center_wp = nxt[0]
    center_loc = center_wp.transform.location
    state["center"] = center_loc

    forward = center_wp.transform.get_forward_vector()
    right = center_wp.transform.get_right_vector()

    def spawn_prop(pattern: str, loc: carla.Location, yaw: float = 0.0) -> Optional[carla.Actor]:
        choices = list(bp_lib.filter(pattern))
        if not choices:
            return None
        actor = world.try_spawn_actor(
            random.choice(choices),
            carla.Transform(loc, carla.Rotation(yaw=yaw)),
        )
        if actor is not None:
            actors_to_cleanup.append(actor)
            state["event_props"].append(actor)
        return actor

    def spawn_vehicle(preferred_ids: List[str], tf: carla.Transform, role: str) -> Optional[carla.Vehicle]:
        bp = find_blueprint(bp_lib, preferred_ids, "vehicle.*")
        set_vehicle_attributes(bp, role_name=role)
        actor = world.try_spawn_actor(bp, tf)
        if actor is not None:
            actors_to_cleanup.append(actor)
        return actor

    if EVENT_MODE == "scooter_swerving":
        tf = carla.Transform(
            center_loc + carla.Location(x=forward.x * 10.0, y=forward.y * 10.0, z=0.3),
            center_wp.transform.rotation,
        )
        veh = spawn_vehicle(["vehicle.vespa.zx125", "vehicle.yamaha.yzf", "vehicle.harley-davidson.low_rider"], tf, "event_scooter")
        if veh is not None:
            try:
                veh.set_autopilot(False)
            except Exception:
                pass
            state["event_vehicle"] = veh

    elif EVENT_MODE == "motorcycle_wheelie":
        tf = carla.Transform(
            center_loc + carla.Location(x=forward.x * 14.0, y=forward.y * 14.0, z=0.45),
            carla.Rotation(
                yaw=center_wp.transform.rotation.yaw,
                pitch=6.0,
            ),
        )
        veh = spawn_vehicle(["vehicle.kawasaki.ninja", "vehicle.yamaha.yzf", "vehicle.harley-davidson.low_rider"], tf, "event_motorcycle")
        if veh is not None:
            try:
                veh.set_autopilot(False)
            except Exception:
                pass
            state["event_vehicle"] = veh

    elif EVENT_MODE == "truck_jackknifing":
        tf = carla.Transform(
            center_loc + carla.Location(x=forward.x * 6.0, y=forward.y * 6.0, z=0.35),
            carla.Rotation(yaw=center_wp.transform.rotation.yaw + 58.0),
        )
        veh = spawn_vehicle(["vehicle.carlamotors.firetruck", "vehicle.mercedes.sprinter", "vehicle.tesla.cybertruck"], tf, "event_jackknife")
        if veh is not None:
            try:
                veh.set_autopilot(False)
                veh.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
            except Exception:
                pass
            state["event_vehicle"] = veh

    elif EVENT_MODE == "pedestrian_drop":
        walker_bp = find_blueprint(bp_lib, [], "walker.pedestrian.*")
        walker_tf = carla.Transform(
            center_loc + carla.Location(x=forward.x * 4.0 + right.x * 2.8, y=forward.y * 4.0 + right.y * 2.8, z=0.2),
            carla.Rotation(yaw=center_wp.transform.rotation.yaw - 90.0),
        )
        walker = world.try_spawn_actor(walker_bp, walker_tf)
        if walker is not None:
            actors_to_cleanup.append(walker)
            state["event_props"].append(walker)
        bag_loc = center_loc + carla.Location(x=forward.x * 7.5, y=forward.y * 7.5, z=0.4)
        spawn_prop("static.prop.briefcase*", bag_loc, yaw=center_wp.transform.rotation.yaw)

    elif EVENT_MODE == "sudden_pedestrian_crowd_crossing":
        try:
            walker_controller_bp = bp_lib.find("controller.ai.walker")
        except Exception:
            walker_controller_bp = None
        walker_blueprints = sorted(list(bp_lib.filter("walker.pedestrian.*")), key=lambda bp: bp.id)
        if not walker_blueprints:
            raise RuntimeError("No walker blueprints available for crowd crossing event")

        crowd_layout_seed = (
            SCENE_LAYOUT_SEED
            + int(round(center_loc.x * 10.0)) * 10007
            + int(round(center_loc.y * 10.0)) * 10009
        )
        crowd_rng = random.Random(crowd_layout_seed)
        state["crowd_layout_seed"] = crowd_layout_seed

        segment_offsets = list(SIDEWALK_SEGMENT_ALONG_OFFSETS_M)
        segment_targets = {
            segment_offset: crowd_rng.randint(MIN_CROWD_SIZE, MAX_CROWD_SIZE)
            for segment_offset in segment_offsets
        }
        crowd_count_target = sum(segment_targets.values())
        # Prefer ego-left sidewalk so pedestrians cross left -> right relative to ego heading.
        spawn_side = -1.0
        sidewalk_wp = find_sidewalk_anchor(map_obj, center_wp, spawn_side)
        if sidewalk_wp is None:
            spawn_side = 1.0
            sidewalk_wp = find_sidewalk_anchor(map_obj, center_wp, spawn_side)
        opposite_sidewalk_wp = find_sidewalk_anchor(map_obj, center_wp, -spawn_side) if sidewalk_wp is not None else None

        if sidewalk_wp is not None:
            base_sidewalk_loc = sidewalk_wp.transform.location
        else:
            # Fallback: place crowd at a large lateral offset near the event center.
            base_sidewalk_loc = center_loc + carla.Location(x=right.x * 10.0 * spawn_side, y=right.y * 10.0 * spawn_side, z=0.2)

        if opposite_sidewalk_wp is not None:
            target_sidewalk_loc = opposite_sidewalk_wp.transform.location
        else:
            target_sidewalk_loc = center_loc + carla.Location(x=right.x * -10.0 * spawn_side, y=right.y * -10.0 * spawn_side, z=0.2)

        state["crowd_center"] = base_sidewalk_loc
        state["crowd_spawn_side"] = spawn_side
        try:
            crowd_spawn_dist = distance_2d(ego.get_location(), base_sidewalk_loc)
            state["crowd_spawn_distance_from_ego_m"] = crowd_spawn_dist
        except Exception:
            crowd_spawn_dist = None
        demographics = {"male": 0, "female": 0, "boy": 0, "girl": 0, "other": 0}
        desired_mix = ["male", "female", "boy", "girl"]

        def apply_walker_demographics(bp: carla.ActorBlueprint, label: str) -> None:
            applied = False
            if bp.has_attribute("gender"):
                values = [v.lower() for v in bp.get_attribute("gender").recommended_values]
                target = "male" if label in {"male", "boy"} else "female"
                if target in values:
                    bp.set_attribute("gender", target)
                    applied = True
            if bp.has_attribute("age"):
                values = [v.lower() for v in bp.get_attribute("age").recommended_values]
                target = "child" if label in {"boy", "girl"} else "adult"
                if target in values:
                    bp.set_attribute("age", target)
                    applied = True
            if not applied and bp.has_attribute("speed"):
                # Keep a deterministic-ish spread when explicit demographic attributes are unavailable.
                values = bp.get_attribute("speed").recommended_values
                if values:
                    try:
                        bp.set_attribute("speed", crowd_rng.choice(list(values)))
                    except Exception:
                        pass

        crowd_plans = []
        segment_spawned_counts: Dict[float, int] = {}
        walker_serial = 0

        for segment_offset in segment_offsets:
            segment_target = segment_targets[segment_offset]
            segment_spawned = 0
            cols = 3

            for local_idx in range(segment_target):
                walker_bp = crowd_rng.choice(walker_blueprints)
                if walker_bp.has_attribute("is_invincible"):
                    walker_bp.set_attribute("is_invincible", "false")

                label = desired_mix[walker_serial % len(desired_mix)]
                walker_serial += 1
                apply_walker_demographics(walker_bp, label)

                row = local_idx // cols
                col = local_idx % cols
                local_along = (col - (cols - 1) * 0.5) * 2.0 + crowd_rng.uniform(-0.25, 0.25) + row * 0.4
                along_offset = segment_offset + local_along
                sidewalk_depth = 0.8 + (row * 1.2) + crowd_rng.uniform(0.0, 0.4)
                side_jitter = crowd_rng.uniform(-0.2, 0.2)
                spawn_loc = base_sidewalk_loc + carla.Location(
                    x=forward.x * along_offset + right.x * spawn_side * (sidewalk_depth + side_jitter),
                    y=forward.y * along_offset + right.y * spawn_side * (sidewalk_depth + side_jitter),
                    z=0.25,
                )

                spawn_yaw = center_wp.transform.rotation.yaw - (90.0 * spawn_side)
                walker = None
                for z_bump in (0.0, 0.15, 0.35):
                    spawn_tf = carla.Transform(
                        carla.Location(x=spawn_loc.x, y=spawn_loc.y, z=spawn_loc.z + z_bump),
                        carla.Rotation(yaw=spawn_yaw),
                    )
                    walker = world.try_spawn_actor(walker_bp, spawn_tf)
                    if walker is not None:
                        break

                if walker is None:
                    continue

                actors_to_cleanup.append(walker)
                state["event_props"].append(walker)
                demographics[label] = demographics.get(label, 0) + 1
                segment_spawned += 1

                # We intentionally avoid walker AI routing for this scene; direct WalkerControl will force the crossing.
                controller = None
                if walker_controller_bp is not None:
                    try:
                        controller = world.spawn_actor(walker_controller_bp, carla.Transform(), attach_to=walker)
                        actors_to_cleanup.append(controller)
                    except Exception:
                        controller = None

                target_along = segment_offset + (local_along * 0.85) + crowd_rng.uniform(-1.5, 1.5)
                target_sidewalk_depth = crowd_rng.uniform(0.6, 1.6)
                target_loc = target_sidewalk_loc + carla.Location(
                    x=forward.x * target_along + right.x * (-spawn_side) * target_sidewalk_depth,
                    y=forward.y * target_along + right.y * (-spawn_side) * target_sidewalk_depth,
                    z=0.1,
                )
                mid_target_loc = center_loc + carla.Location(
                    x=forward.x * target_along + right.x * crowd_rng.uniform(-0.7, 0.7),
                    y=forward.y * target_along + right.y * crowd_rng.uniform(-0.7, 0.7),
                    z=0.1,
                )

                crowd_plans.append(
                    {
                        "walker": walker,
                        "controller": controller,
                        "mid_target": mid_target_loc,
                        "target": target_loc,
                        "speed": crowd_rng.uniform(3.8, 5.2),
                        "crossing_fwd_jitter": crowd_rng.uniform(-0.75, 0.75),
                        "crossing_center_jitter": crowd_rng.uniform(-0.4, 0.4),
                        "cross_sign": 1.0,
                        "started": False,
                        "entry_reached": False,
                        "mid_reached": False,
                        "last_loc": None,
                        "stuck_ticks": 0,
                        "nudge_count": 0,
                    }
                )

            segment_spawned_counts[segment_offset] = segment_spawned

        state["crowd_plans"] = crowd_plans
        state["crowd_mix"] = demographics
        state["crowd_segment_targets"] = segment_targets
        state["crowd_segment_spawned"] = segment_spawned_counts
        try:
            state["traffic_lights"] = [
                tl for tl in world.get_actors().filter("traffic.traffic_light*")
                if distance_2d(tl.get_location(), center_loc) <= 90.0
            ]
        except Exception:
            state["traffic_lights"] = []
        segment_summary = ", ".join(
            f"{int(seg):+d}m:{segment_spawned_counts.get(seg, 0)}/{segment_targets.get(seg, 0)}"
            for seg in segment_offsets
        )
        spawn_dist_suffix = (
            f" spawn_dist_from_ego={crowd_spawn_dist:.1f}m"
            if crowd_spawn_dist is not None
            else ""
        )
        log(
            f"[INFO] Crowd staged target={crowd_count_target} spawned={len(crowd_plans)} "
            f"segments=[{segment_summary}] mix={demographics} sidewalk_side={'right' if spawn_side > 0 else 'left'} "
            f"layout_seed={crowd_layout_seed}{spawn_dist_suffix}",
            log_file,
        )

    elif EVENT_MODE == "traffic_light_malfunction":
        lights = []
        for actor in world.get_actors().filter("traffic.traffic_light*"):
            if distance_2d(actor.get_location(), center_loc) <= 90.0:
                lights.append(actor)
        state["traffic_lights"] = lights

    elif EVENT_MODE == "drawbridge_lift":
        for idx in range(-2, 3):
            loc = center_loc + carla.Location(x=right.x * idx * 1.8, y=right.y * idx * 1.8, z=0.3)
            spawn_prop("static.prop.streetbarrier*", loc, yaw=center_wp.transform.rotation.yaw)

    elif EVENT_MODE == "overhead_sign_collapse":
        for idx in range(-3, 4):
            loc = center_loc + carla.Location(
                x=forward.x * 5.0 + right.x * idx * 1.3,
                y=forward.y * 5.0 + right.y * idx * 1.3,
                z=0.35,
            )
            spawn_prop("static.prop.constructionbarrier*", loc, yaw=center_wp.transform.rotation.yaw)

    elif EVENT_MODE == "tunnel_blockage":
        for idx in range(5):
            tf = carla.Transform(
                center_loc + carla.Location(x=forward.x * (6.0 + idx * 6.0), y=forward.y * (6.0 + idx * 6.0), z=0.3),
                center_wp.transform.rotation,
            )
            veh = spawn_vehicle(["vehicle.audi.a2", "vehicle.citroen.c3", "vehicle.dodge.charger_2020"], tf, "event_stalled")
            if veh is not None:
                try:
                    veh.set_autopilot(False)
                    veh.apply_control(carla.VehicleControl(hand_brake=True, brake=1.0))
                except Exception:
                    pass

    elif EVENT_MODE == "hood_popup":
        prop_loc = carla.Location(x=1.0, y=0.0, z=1.5)
        prop_tf = carla.Transform(prop_loc, carla.Rotation(pitch=72.0))
        try:
            prop_bp = find_blueprint(bp_lib, [], "static.prop.*")
            hood = world.try_spawn_actor(prop_bp, prop_tf, attach_to=ego)
            if hood is not None:
                actors_to_cleanup.append(hood)
                state["hood_actor"] = hood
        except Exception:
            pass

    elif EVENT_MODE == "drone_crash":
        loc = center_loc + carla.Location(x=0.0, y=0.0, z=12.0)
        drone = spawn_prop("static.prop.*drone*", loc, yaw=0.0)
        if drone is None:
            drone = spawn_prop("static.prop.box*", loc, yaw=0.0)
        if drone is not None:
            try:
                drone.set_simulate_physics(False)
            except Exception:
                pass
            state["drone_actor"] = drone

    elif EVENT_MODE in {"magnetic_anomaly", "illusionary_split", "portal_disruption", "bridge_icing", "fire_smoke", "sandstorm", "snow_ice", "volcanic_ash", "earthquake"}:
        radius = 6.0 if EVENT_MODE != "portal_disruption" else 8.5
        num_props = 10 if EVENT_MODE != "portal_disruption" else 14
        for idx in range(num_props):
            ang = (2.0 * math.pi * idx) / float(num_props)
            loc = center_loc + carla.Location(
                x=math.cos(ang) * radius,
                y=math.sin(ang) * radius,
                z=0.35 if EVENT_MODE != "portal_disruption" else 1.0 + 0.4 * math.sin(ang * 2.0),
            )
            spawn_prop("static.prop.trafficcone*", loc, yaw=math.degrees(ang))

    log(f"[INFO] Event staged mode={EVENT_MODE} props={len(state['event_props'])}", log_file)
    return state


def apply_event_dynamics(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    ego: carla.Vehicle,
    event_state: Dict[str, object],
    elapsed: float,
    total_duration: float,
    log_file=None,
) -> None:
    phase = 0.0 if total_duration <= 0.0 else max(0.0, min(1.0, elapsed / total_duration))

    if EVENT_MODE == "sudden_pedestrian_crowd_crossing":
        # Force all nearby traffic lights green so the ego (driven by direct
        # control, not TM) is never blocked by a red signal.
        traffic_lights = event_state.get("traffic_lights") or []
        for tl in traffic_lights:
            try:
                tl.set_state(carla.TrafficLightState.Green)
                tl.freeze(True)
            except Exception:
                pass
        if traffic_lights and not bool(event_state.get("ego_signal_lock_started")):
            event_state["ego_signal_lock_started"] = True
            if log_file is not None:
                log(f"[EVENT] All {len(traffic_lights)} nearby traffic lights frozen GREEN", log_file)

    if EVENT_MODE == "sudden_pedestrian_crowd_crossing":
        crowd_center = event_state.get("crowd_center")
        crowd_plans = event_state.get("crowd_plans") or []
        if crowd_center is not None and crowd_plans:
            ego_loc = ego.get_location()
            ego_tf = ego.get_transform()
            ego_forward = ego_tf.get_forward_vector()
            ego_right = ego_tf.get_right_vector()
            crowd_dist = distance_2d(ego_loc, crowd_center)
            event_state["crowd_distance_to_ego"] = crowd_dist

            # Track forward-signed distance to the crossing anchor (where
            # pedestrians actually cross).  Positive = anchor is ahead.
            crossing_anchor = event_state.get("dynamic_crossing_anchor") or event_state.get("ego_crossing_anchor")
            if isinstance(crossing_anchor, carla.Location):
                dx_a = crossing_anchor.x - ego_loc.x
                dy_a = crossing_anchor.y - ego_loc.y
                forward_dist = dx_a * ego_forward.x + dy_a * ego_forward.y
                event_state["crossing_forward_dist"] = forward_dist
            else:
                event_state["crossing_forward_dist"] = crowd_dist

            trigger_distance = float(event_state.get("crowd_trigger_distance_m") or CROWD_TRIGGER_DISTANCE_M)
            triggered = bool(event_state.get("crowd_triggered"))
            if (not triggered) and crowd_dist <= trigger_distance:
                event_state["crowd_triggered"] = True
                event_state["crowd_trigger_time"] = elapsed
                anchor_at_trigger = carla.Location(
                    x=ego_loc.x + ego_forward.x * CROWD_CROSSING_LOOKAHEAD_M,
                    y=ego_loc.y + ego_forward.y * CROWD_CROSSING_LOOKAHEAD_M,
                    z=ego_loc.z,
                )
                try:
                    anchor_wp_at_trigger = world.get_map().get_waypoint(
                        anchor_at_trigger,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                except Exception:
                    anchor_wp_at_trigger = None
                if anchor_wp_at_trigger is not None:
                    anchor_at_trigger = anchor_wp_at_trigger.transform.location
                event_state["ego_crossing_anchor"] = anchor_at_trigger
                event_state["ego_cross_forward"] = carla.Vector3D(x=ego_forward.x, y=ego_forward.y, z=ego_forward.z)
                event_state["ego_cross_right"] = carla.Vector3D(x=ego_right.x, y=ego_right.y, z=ego_right.z)
                event_state["dynamic_crossing_anchor"] = anchor_at_trigger
                if log_file is not None:
                    log(
                        f"[EVENT] Crowd trigger activated at t={elapsed:.2f}s ego_distance={crowd_dist:.1f}m "
                        f"walkers={len(crowd_plans)} anchor=({anchor_at_trigger.x:.1f},{anchor_at_trigger.y:.1f}) "
                        f"cross_dir=left_to_right_relative_to_ego",
                        log_file,
                    )

            if bool(event_state.get("crowd_triggered")):
                started_now = 0
                started_total = int(event_state.get("crowd_started_count") or 0)
                crossing_anchor = event_state.get("ego_crossing_anchor")
                if not isinstance(crossing_anchor, carla.Location):
                    crossing_anchor = carla.Location(
                        x=ego_loc.x + ego_forward.x * CROWD_CROSSING_LOOKAHEAD_M,
                        y=ego_loc.y + ego_forward.y * CROWD_CROSSING_LOOKAHEAD_M,
                        z=ego_loc.z,
                    )
                    event_state["ego_crossing_anchor"] = crossing_anchor
                cross_forward = event_state.get("ego_cross_forward")
                cross_right = event_state.get("ego_cross_right")
                if not isinstance(cross_forward, carla.Vector3D):
                    cross_forward = carla.Vector3D(x=ego_forward.x, y=ego_forward.y, z=ego_forward.z)
                    event_state["ego_cross_forward"] = cross_forward
                if not isinstance(cross_right, carla.Vector3D):
                    cross_right = carla.Vector3D(x=ego_right.x, y=ego_right.y, z=ego_right.z)
                    event_state["ego_cross_right"] = cross_right
                try:
                    anchor_wp = world.get_map().get_waypoint(
                        crossing_anchor,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                except Exception:
                    anchor_wp = None
                if anchor_wp is not None:
                    crossing_anchor = anchor_wp.transform.location
                    lane_width = float(getattr(anchor_wp, "lane_width", 3.5) or 3.5)
                else:
                    lane_width = 3.5
                road_half_span = max(6.5, lane_width * 2.2)
                far_side_span = road_half_span + 2.0
                event_state["dynamic_crossing_anchor"] = crossing_anchor

                for plan in crowd_plans:
                    if bool(plan.get("started")):
                        continue

                    walker = plan.get("walker")
                    controller = plan.get("controller")
                    run_speed = float(plan.get("speed", 4.2))

                    if walker is None or not getattr(walker, "is_alive", False):
                        plan["started"] = True
                        continue

                    # Force manual crossing behavior; do not let walker AI traffic rules block movement.
                    if controller is not None and getattr(controller, "is_alive", False):
                        try:
                            controller.stop()
                        except Exception:
                            pass
                    started = True

                    plan["started"] = True
                    if started:
                        started_now += 1
                        started_total += 1

                event_state["crowd_started_count"] = started_total
                if started_now and log_file is not None:
                    log(
                        f"[EVENT] Crowd runners released +{started_now} (total={started_total}/{len(crowd_plans)})",
                        log_file,
                    )

                # Keep pedestrians moving across the road centerline and then to the opposite sidewalk.
                for plan in crowd_plans:
                    if not bool(plan.get("started")):
                        continue
                    walker = plan.get("walker")
                    if walker is None or not getattr(walker, "is_alive", False):
                        continue

                    current_loc = walker.get_location()
                    rel_x = current_loc.x - crossing_anchor.x
                    rel_y = current_loc.y - crossing_anchor.y
                    local_side = (rel_x * cross_right.x) + (rel_y * cross_right.y)
                    cross_sign = 1.0  # force left -> right relative to ego
                    plan["cross_sign"] = cross_sign
                    fwd_jitter = float(plan.get("crossing_fwd_jitter") or 0.0)
                    center_jitter = float(plan.get("crossing_center_jitter") or 0.0)
                    entry_target = carla.Location(
                        x=crossing_anchor.x + cross_forward.x * fwd_jitter + cross_right.x * (-far_side_span),
                        y=crossing_anchor.y + cross_forward.y * fwd_jitter + cross_right.y * (-far_side_span),
                        z=crossing_anchor.z + 0.1,
                    )
                    mid_target = carla.Location(
                        x=crossing_anchor.x + cross_forward.x * fwd_jitter + cross_right.x * center_jitter,
                        y=crossing_anchor.y + cross_forward.y * fwd_jitter + cross_right.y * center_jitter,
                        z=crossing_anchor.z + 0.1,
                    )
                    final_target = carla.Location(
                        x=crossing_anchor.x + cross_forward.x * fwd_jitter + cross_right.x * (float(cross_sign) * far_side_span),
                        y=crossing_anchor.y + cross_forward.y * fwd_jitter + cross_right.y * (float(cross_sign) * far_side_span),
                        z=crossing_anchor.z + 0.1,
                    )
                    plan["entry_target"] = entry_target
                    plan["mid_target"] = mid_target
                    plan["target"] = final_target

                    # If a walker somehow starts on the ego-right side, force them back to the left entry side.
                    if local_side > 0.4 and not bool(plan.get("entry_reached")):
                        try:
                            walker.set_location(
                                carla.Location(
                                    x=entry_target.x + cross_right.x * random.uniform(-0.3, 0.3),
                                    y=entry_target.y + cross_right.y * random.uniform(-0.3, 0.3),
                                    z=current_loc.z,
                                )
                            )
                            current_loc = walker.get_location()
                        except Exception:
                            pass

                    if not bool(plan.get("entry_reached")) and distance_2d(current_loc, entry_target) <= 1.8:
                        plan["entry_reached"] = True
                    if bool(plan.get("entry_reached")) and (not bool(plan.get("mid_reached"))):
                        if distance_2d(current_loc, mid_target) <= 1.8:
                            plan["mid_reached"] = True
                    if not bool(plan.get("entry_reached")):
                        active_target = entry_target
                    elif not bool(plan.get("mid_reached")):
                        active_target = mid_target
                    else:
                        active_target = final_target
                    run_speed = float(plan.get("speed", 4.2))

                    controller = plan.get("controller")
                    if controller is not None and getattr(controller, "is_alive", False):
                        try:
                            controller.stop()
                        except Exception:
                            pass

                    # Manual steering (primary path) keeps the crowd visibly crossing toward the far sidewalk.
                    try:
                        dx = active_target.x - current_loc.x
                        dy = active_target.y - current_loc.y
                        dist = math.sqrt(dx * dx + dy * dy)
                        if dist > 0.5:
                            prev_loc = plan.get("last_loc")
                            if isinstance(prev_loc, carla.Location):
                                moved = distance_2d(current_loc, prev_loc)
                                if moved < 0.04:
                                    plan["stuck_ticks"] = int(plan.get("stuck_ticks") or 0) + 1
                                else:
                                    plan["stuck_ticks"] = 0
                            else:
                                plan["stuck_ticks"] = 0
                            plan["last_loc"] = carla.Location(x=current_loc.x, y=current_loc.y, z=current_loc.z)

                            direction = carla.Vector3D(x=dx / dist, y=dy / dist, z=0.0)
                            stuck_ticks = int(plan.get("stuck_ticks") or 0)
                            forced_speed = run_speed + (1.5 if stuck_ticks >= 4 else 0.6)
                            try:
                                # Mirrors 3CSIM-style direct WalkerControl triggering, but updated every tick.
                                walker.apply_control(carla.WalkerControl(direction=direction, speed=forced_speed, jump=False))
                            except Exception:
                                walker.set_target_velocity(
                                    carla.Vector3D(x=direction.x * forced_speed, y=direction.y * forced_speed, z=0.0)
                                )

                            # If a pedestrian remains stuck, nudge them programmatically toward the crossing target.
                            if stuck_ticks >= 8:
                                nudge_step = min(0.35, max(0.10, dist * 0.15))
                                try:
                                    walker.set_location(
                                        carla.Location(
                                            x=current_loc.x + direction.x * nudge_step,
                                            y=current_loc.y + direction.y * nudge_step,
                                            z=current_loc.z,
                                        )
                                    )
                                    plan["nudge_count"] = int(plan.get("nudge_count") or 0) + 1
                                    plan["stuck_ticks"] = 0
                                except Exception:
                                    pass
                        else:
                            plan["last_loc"] = carla.Location(x=current_loc.x, y=current_loc.y, z=current_loc.z)
                            plan["stuck_ticks"] = 0
                    except Exception:
                        pass

    event_vehicle = event_state.get("event_vehicle")
    if EVENT_MODE in {"scooter_swerving", "motorcycle_wheelie"} and event_vehicle is not None:
        try:
            steer = 0.35 * math.sin(elapsed * 1.8)
            throttle = 0.42 if EVENT_MODE == "scooter_swerving" else 0.5
            event_vehicle.apply_control(carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0))
        except Exception:
            pass

    if EVENT_MODE == "traffic_light_malfunction":
        lights = event_state.get("traffic_lights") or []
        try:
            if 0.30 <= phase < 0.55:
                for light in lights:
                    light.set_state(carla.TrafficLightState.Green)
                    light.freeze(True)
            elif 0.55 <= phase < 0.78:
                for light in lights:
                    light.set_state(carla.TrafficLightState.Red)
                    light.freeze(True)
            else:
                for light in lights:
                    light.freeze(False)
        except Exception:
            pass

    if EVENT_MODE == "hood_popup":
        try:
            if 0.25 <= phase <= 0.75:
                traffic_manager.vehicle_percentage_speed_difference(ego, 60.0)
            else:
                traffic_manager.vehicle_percentage_speed_difference(ego, 25.0)
        except Exception:
            pass

    if EVENT_MODE == "earthquake":
        if 0.32 <= phase <= 0.62 and int(elapsed * 10.0) % 4 == 0:
            try:
                ego.add_impulse(
                    carla.Vector3D(
                        x=random.uniform(-1200.0, 1200.0),
                        y=random.uniform(-1200.0, 1200.0),
                        z=random.uniform(60.0, 180.0),
                    )
                )
            except Exception:
                pass

    if EVENT_MODE == "drone_crash":
        drone_actor = event_state.get("drone_actor")
        if drone_actor is not None and not event_state.get("dropped") and phase >= 0.35:
            try:
                drone_actor.set_simulate_physics(True)
                drone_actor.add_impulse(carla.Vector3D(x=80.0, y=0.0, z=-4200.0))
                event_state["dropped"] = True
            except Exception:
                pass

    if EVENT_MODE in {"magnetic_anomaly", "illusionary_split", "portal_disruption"}:
        for actor in event_state.get("event_props") or []:
            try:
                actor.add_angular_impulse(
                    carla.Vector3D(
                        x=random.uniform(-150.0, 150.0),
                        y=random.uniform(-150.0, 150.0),
                        z=random.uniform(-220.0, 220.0),
                    )
                )
            except Exception:
                pass


def main() -> None:
    parser = argparse.ArgumentParser(description=f"Standalone CARLA scenario: {SCENE_KEYWORD}")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=20.0)
    parser.add_argument("--output-dir", default=f"./scenes/{SCENARIO_KEYWORD}")
    parser.add_argument("--output", dest="output_alias", default=None)
    parser.add_argument("--time-preset", default=SCENE_DEFAULT_KEY)
    parser.add_argument("--weather-preset", default=SCENE_DEFAULT_KEY)
    parser.add_argument("--sun-altitude", type=float, default=None)
    parser.add_argument("--sun-azimuth", type=float, default=None)
    parser.add_argument("--streetlights", default="auto")
    parser.add_argument("--cloudiness", type=float, default=None)
    parser.add_argument("--precipitation", type=float, default=None)
    parser.add_argument("--precipitation-deposits", type=float, default=None)
    parser.add_argument("--wind-intensity", type=float, default=None)
    parser.add_argument("--fog-density", type=float, default=None)
    parser.add_argument("--fog-distance", type=float, default=None)
    parser.add_argument("--wetness", type=float, default=None)
    args = parser.parse_args()
    configure_environment_from_args(args)

    output_root = Path(args.output_alias or args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    cleared_old_frames = 0
    for old_frame in output_root.rglob("*_frame_*.png"):
        try:
            old_frame.unlink()
            cleared_old_frames += 1
        except Exception:
            pass
    output_dirs = {k: output_root / k for k in CAMERA_KEYS}
    for directory in output_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    log_path = output_root / f"{SCENARIO_KEYWORD}_simulation.log"

    client: Optional[carla.Client] = None
    world: Optional[carla.World] = None
    original_settings: Optional[carla.WorldSettings] = None
    actors_to_cleanup: List[carla.Actor] = []
    traffic_manager: Optional[carla.TrafficManager] = None

    with log_path.open("w", encoding="utf-8") as log_file:
        log(f"[INFO] Stage1 scene script start scenario={SCENARIO_KEYWORD} shot={SHOT_INDEX}", log_file)
        log(f"[INFO] Cleared old frame files: {cleared_old_frames}", log_file)
        log(f"[INFO] Scene keyword: {SCENE_KEYWORD}", log_file)
        log(f"[INFO] Prompt: {SCENE_PROMPT}", log_file)
        log(f"[INFO] Specifications: {SCENE_SPECIFICATIONS}", log_file)
        log(
            f"[ENV] time_preset={ENV_CONFIG['time_preset']} weather_preset={ENV_CONFIG['weather_preset']} "
            f"streetlights={ENV_CONFIG['streetlights']}",
            log_file,
        )

        try:
            client = carla.Client(args.host, args.port)
            client.set_timeout(60.0)

            selected_map = choose_map(client, log_file)
            world = load_world_with_retry(client, selected_map, log_file)
            original_settings = apply_sync_settings(world, client, log_file)

            log("[INFO] Clearing existing vehicles from map", log_file)
            for actor in world.get_actors().filter("vehicle.*"):
                try:
                    actor.destroy()
                except Exception:
                    pass

            bp_lib = world.get_blueprint_library()
            world_map = world.get_map()
            spawn_points = world_map.get_spawn_points()
            if not spawn_points:
                raise RuntimeError("No spawn points available")

            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(4.0)
            traffic_manager.set_hybrid_physics_mode(False)

            ego_bp = find_blueprint(
                bp_lib,
                ["vehicle.tesla.model3", "vehicle.audi.a2", "vehicle.mercedes.coupe"],
                "vehicle.*",
            )
            set_vehicle_attributes(ego_bp, role_name="hero")

            ego_vehicle: Optional[carla.Vehicle] = None
            shuffled_spawns = sorted(spawn_points, key=stable_transform_sort_key)
            preferred_city_spawn = pick_city_ego_spawn(world_map, spawn_points)
            if preferred_city_spawn is not None:
                shuffled_spawns = [preferred_city_spawn] + [tf for tf in shuffled_spawns if tf is not preferred_city_spawn]
                log("[INFO] City-road ego spawn candidate prioritized (sidewalk/junction nearby)", log_file)
            else:
                log("[WARN] No city-road preferred spawn found; using deterministic fallback order", log_file)
            # Pre-filter spawn transforms using the map waypoint data (the
            # transform location from get_spawn_points() is reliable, whereas
            # actor.get_location() may return (0,0,0) under Wine/cross-platform).
            validated_spawns: List[carla.Transform] = []
            rejected_spawns = 0
            for tf in shuffled_spawns:
                sp_loc = tf.location
                sp_wp = world_map.get_waypoint(
                    sp_loc, project_to_road=True, lane_type=carla.LaneType.Driving,
                )
                wp_ok = (
                    sp_wp is not None
                    and sp_wp.lane_type == carla.LaneType.Driving
                    and not sp_wp.is_junction
                    and bool(sp_wp.next(5.0))
                )
                road_dist = distance_2d(sp_loc, sp_wp.transform.location) if sp_wp is not None else 999.0
                if wp_ok and road_dist <= 3.0:
                    validated_spawns.append(tf)
                else:
                    rejected_spawns += 1
            log(
                f"[INFO] Spawn validation: {len(validated_spawns)} valid, {rejected_spawns} rejected "
                f"out of {len(shuffled_spawns)} total",
                log_file,
            )
            if not validated_spawns:
                raise RuntimeError("No spawn points passed driving-lane validation")

            for tf in validated_spawns:
                ego_vehicle = world.try_spawn_actor(ego_bp, tf)
                if ego_vehicle is not None:
                    break
            if ego_vehicle is None:
                raise RuntimeError("Failed to spawn ego vehicle at a valid driving-lane position")
            # Use the known-good transform location for logging (actor.get_location()
            # can return origin under Wine).
            ego_spawn_tf = tf
            log(
                f"[INFO] Ego spawned (deterministic area) at "
                f"({ego_spawn_tf.location.x:.1f}, {ego_spawn_tf.location.y:.1f}, {ego_spawn_tf.location.z:.1f})",
                log_file,
            )

            actors_to_cleanup.append(ego_vehicle)
            # Drive ego directly via waypoint-following (TM autopilot is
            # unreliable under Wine).  No set_autopilot() call.
            log(f"[INFO] Ego using direct waypoint-following at {EGO_TARGET_SPEED_KMH:.0f} km/h", log_file)

            vehicle_blueprints: List[carla.ActorBlueprint] = []
            for bp in bp_lib.filter("vehicle.*"):
                try:
                    if bp.has_attribute("number_of_wheels") and int(bp.get_attribute("number_of_wheels")) < 4:
                        continue
                except Exception:
                    pass
                vehicle_blueprints.append(bp)
            if not vehicle_blueprints:
                vehicle_blueprints = list(bp_lib.filter("vehicle.*"))

            density = spawn_dense_traffic(
                world,
                client,
                traffic_manager,
                ego_vehicle,
                spawn_points,
                vehicle_blueprints,
                actors_to_cleanup,
                log_file,
            )

            props_count = spawn_context_props(
                world,
                bp_lib,
                ego_vehicle,
                spawn_points,
                actors_to_cleanup,
                log_file,
            )

            event_state = stage_primary_event(
                world,
                bp_lib,
                ego_vehicle,
                actors_to_cleanup,
                log_file,
            )
            corridor_target_loc = event_state.get("center") or event_state.get("crowd_center")
            event_state["corridor_target_loc"] = corridor_target_loc
            corridor_target_current = event_state.get("dynamic_crossing_anchor") or corridor_target_loc
            corridor_removed_total = clear_ego_to_event_corridor(
                world,
                ego_vehicle,
                corridor_target_current if isinstance(corridor_target_current, carla.Location) else None,
                log_file,
            )
            event_state["corridor_removed_total"] = corridor_removed_total
            apply_vehicle_headlights_for_matrix(world, log_file)

            setup_cameras(world, ego_vehicle, actors_to_cleanup)
            reset_camera_buffers()

            for _ in range(20):
                drive_ego_straight(ego_vehicle, world_map)
                safe_tick(world, "camera_warmup", log_file, fatal=False)
                corridor_target_current = event_state.get("dynamic_crossing_anchor") or corridor_target_loc
                corridor_removed_total += clear_ego_to_event_corridor(
                    world,
                    ego_vehicle,
                    corridor_target_current if isinstance(corridor_target_current, carla.Location) else None,
                    log_file,
                )
                event_state["corridor_removed_total"] = corridor_removed_total

            log(f"[INFO] Cameras ready with keys={CAMERA_KEYS}", log_file)
            max_frames = max(1, int(args.duration * FPS))
            frame_number = 1

            while frame_number <= max_frames:
                phase = (frame_number - 1) / float(max_frames)
                corridor_target_current = event_state.get("dynamic_crossing_anchor") or corridor_target_loc
                corridor_removed_total += clear_ego_to_event_corridor(
                    world,
                    ego_vehicle,
                    corridor_target_current if isinstance(corridor_target_current, carla.Location) else None,
                    log_file,
                )
                event_state["corridor_removed_total"] = corridor_removed_total
                apply_weather(world, phase)
                apply_event_dynamics(
                    world,
                    traffic_manager,
                    ego_vehicle,
                    event_state,
                    (frame_number - 1) * FRAME_DT,
                    args.duration,
                    log_file=log_file,
                )

                # Drive ego forward; brake when approaching the crossing crowd.
                ego_speed_target = EGO_TARGET_SPEED_KMH
                if bool(event_state.get("crowd_triggered")):
                    fwd_dist = event_state.get("crossing_forward_dist")
                    if fwd_dist is not None:
                        if fwd_dist <= EGO_STOP_DISTANCE_M:
                            # At or past the crossing — full stop.
                            ego_speed_target = 0.0
                        elif fwd_dist <= EGO_BRAKE_DISTANCE_M:
                            ratio = (fwd_dist - EGO_STOP_DISTANCE_M) / (EGO_BRAKE_DISTANCE_M - EGO_STOP_DISTANCE_M)
                            ego_speed_target = EGO_TARGET_SPEED_KMH * ratio
                drive_ego_straight(ego_vehicle, world_map, target_speed_kmh=ego_speed_target)

                safe_tick(world, "main_loop", log_file, fatal=False)

                if frame_ready.wait(timeout=1.2):
                    with images_lock:
                        for key in CAMERA_KEYS:
                            image = images_received[key]
                            if image is None:
                                continue
                            out_path = output_dirs[key] / f"{key}_frame_{frame_number:08d}.png"
                            save_image_to_disk(image, out_path)
                else:
                    log(f"[WARN] Frame timeout at frame={frame_number}", log_file)

                if frame_number % 80 == 0:
                    apply_vehicle_headlights_for_matrix(world, log_file)
                    speed = 3.6 * math.sqrt(
                        ego_vehicle.get_velocity().x ** 2
                        + ego_vehicle.get_velocity().y ** 2
                        + ego_vehicle.get_velocity().z ** 2
                    )
                    crowd_started = int(event_state.get("crowd_started_count") or 0)
                    crowd_triggered = bool(event_state.get("crowd_triggered"))
                    crowd_dist = event_state.get("crowd_distance_to_ego") or 0.0
                    crossing_fwd = event_state.get("crossing_forward_dist")
                    log(
                        f"[PROGRESS] frame={frame_number}/{max_frames} speed_kmh={speed:.1f} "
                        f"crowd_triggered={crowd_triggered} crowd_started={crowd_started} "
                        f"crowd_dist={crowd_dist:.1f} crossing_fwd={crossing_fwd} "
                        f"ego_target_kmh={ego_speed_target:.1f}",
                        log_file,
                    )

                reset_camera_buffers()
                frame_number += 1

            final_density = collect_density(world, ego_vehicle)
            log(
                f"[SUCCESS] Completed scenario={SCENARIO_KEYWORD} shot={SHOT_INDEX} "
                f"final_density={final_density} context_props={props_count} initial_density={density}",
                log_file,
            )

        except Exception as exc:
            log(f"[ERROR] Scenario failure: {exc}", log_file)
            raise

        finally:
            log("[INFO] Cleanup start", log_file)

            for actor in actors_to_cleanup:
                try:
                    if "sensor.camera" in actor.type_id:
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

            log("[INFO] Cleanup complete", log_file)


if __name__ == "__main__":
    main()
