#!/usr/bin/env python3
"""Standalone CARLA bicycle-in-highway-lane simulation (no shared runtime dependency).

Scenario: A bicycle occupies an active highway lane on Town04, forcing the ego
vehicle and surrounding traffic to slow down or merge around it.  The bicycle
follows lane geometry via Traffic-Manager autopilot at a much lower speed than
surrounding highway traffic.
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
CAMERA_KEYS = ["front"]
MAP_PREFERENCE = ["Town04"]
MIN_FLOW_PER_DIRECTION = 20
VEHICLE_SPAWN_Z_OFFSET = 1.0
BICYCLE_SPEED_RATIO_TO_EGO = 0.5

# Bicycle blueprint candidates (tried in order; first success wins)
BICYCLE_BLUEPRINTS = [
    "vehicle.bh.crossbike",
    "vehicle.diamondback.century",
    "vehicle.gazelle.omafiets",
]


# ---------------------------------------------------------------------------
# Thread-safe image buffer
# ---------------------------------------------------------------------------

images_received: Dict[str, Optional[carla.Image]] = {key: None for key in CAMERA_KEYS}
images_lock = threading.Lock()
frame_ready = threading.Event()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str, log_file) -> None:
    """Print and persist a log line."""
    line = msg.rstrip()
    print(line)
    log_file.write(line + "\n")
    log_file.flush()


def map_token(name: str) -> str:
    """Extract bare map name from a CARLA map path."""
    raw = str(name).strip()
    if "/" in raw:
        return raw.rsplit("/", 1)[-1]
    return raw


def is_retryable_carla_error(exc: Exception) -> bool:
    """Return True if the exception looks like a transient CARLA/RPC error."""
    msg = str(exc).lower()
    return any(
        token in msg
        for token in (
            "time-out",
            "failed to connect to newly created map",
            "failed to connect",
            "connection closed",
            "rpc",
            "socket",
        )
    )


def distance_2d(a: carla.Location, b: carla.Location) -> float:
    """Euclidean distance ignoring Z."""
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def get_speed_kmh(vehicle: carla.Actor) -> float:
    """Current speed in km/h."""
    vel = vehicle.get_velocity()
    return math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z) * 3.6


def next_waypoint(start_wp: carla.Waypoint, distance_m: float) -> carla.Waypoint:
    """Walk forward along a lane by *distance_m* metres."""
    current = start_wp
    moved = 0.0
    while moved < distance_m:
        nxt = current.next(5.0)
        if not nxt:
            break
        current = nxt[0]
        moved += 5.0
    return current


def lifted_transform(tf: carla.Transform, z_offset: float = VEHICLE_SPAWN_Z_OFFSET) -> carla.Transform:
    """Return a copy of *tf* lifted slightly to avoid tire clipping into asphalt."""
    return carla.Transform(
        carla.Location(
            x=float(tf.location.x),
            y=float(tf.location.y),
            z=float(tf.location.z) + float(z_offset),
        ),
        carla.Rotation(
            pitch=float(tf.rotation.pitch),
            yaw=float(tf.rotation.yaw),
            roll=float(tf.rotation.roll),
        ),
    )


def snap_and_lift_to_driving_lane(
    world_map: carla.Map, tf: carla.Transform, z_offset: float = VEHICLE_SPAWN_Z_OFFSET
) -> carla.Transform:
    """Project a spawn transform to a driving lane waypoint, then lift it."""
    try:
        wp = world_map.get_waypoint(
            tf.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
    except Exception:
        wp = None
    base_tf = wp.transform if wp is not None else tf
    return lifted_transform(base_tf, z_offset=z_offset)


def speed_fraction_from_tm_diff(speed_diff_pct: float) -> float:
    """Convert Traffic Manager speed difference % to target speed fraction of speed limit."""
    return max(0.0, 1.0 - float(speed_diff_pct) / 100.0)


def tm_diff_for_speed_fraction(speed_fraction: float) -> float:
    """Convert target speed fraction of speed limit to Traffic Manager speed difference %."""
    return max(-50.0, min(95.0, 100.0 * (1.0 - float(speed_fraction))))


def bicycle_tm_diff_from_ego_tm_diff(ego_speed_diff_pct: float) -> float:
    """Set bicycle target speed to ~half of the ego target speed."""
    ego_fraction = speed_fraction_from_tm_diff(ego_speed_diff_pct)
    bicycle_fraction = max(0.05, min(1.0, ego_fraction * BICYCLE_SPEED_RATIO_TO_EGO))
    return tm_diff_for_speed_fraction(bicycle_fraction)


def has_long_drivable_segment(wp: carla.Waypoint, min_len_m: float = 140.0) -> bool:
    """Return True if *wp* sits on a straight, non-junction segment >= min_len_m."""
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


def is_highway_candidate(wp: carla.Waypoint) -> bool:
    """Return True if *wp* looks like a highway lane suitable for this scenario."""
    if wp.is_junction or wp.lane_type != carla.LaneType.Driving:
        return False
    if float(wp.lane_width) < 3.2:
        return False
    return has_long_drivable_segment(wp, min_len_m=140.0)


def classify_flow_alignment(
    base_tf: carla.Transform, other_tf: carla.Transform
) -> str:
    """Classify whether *other_tf* travels in the same or opposite direction."""
    base_fwd = base_tf.get_forward_vector()
    other_fwd = other_tf.get_forward_vector()
    dot = (
        base_fwd.x * other_fwd.x
        + base_fwd.y * other_fwd.y
        + base_fwd.z * other_fwd.z
    )
    return "same" if dot >= 0.0 else "opposite"


# ---------------------------------------------------------------------------
# safe_tick / map loading / sync settings
# ---------------------------------------------------------------------------

def safe_tick(
    world: carla.World,
    stage: str,
    log_file,
    retries: int = 3,
    fatal: bool = True,
) -> bool:
    """Tick the world with retry logic for transient failures."""
    last_error: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            world.tick()
            return True
        except RuntimeError as exc:
            last_error = exc
            if not is_retryable_carla_error(exc):
                raise
            log(
                f"[WARN] world.tick retryable failure stage={stage} "
                f"attempt={attempt}/{retries}: {exc}",
                log_file,
            )
            if attempt < retries:
                time.sleep(min(0.6 * attempt, 1.5))
    if fatal and last_error is not None:
        raise RuntimeError(
            f"Persistent tick failure at stage={stage}"
        ) from last_error
    log(
        f"[WARN] Continuing despite repeated tick failure at stage={stage}",
        log_file,
    )
    return False


def choose_preferred_map(
    client: carla.Client, requested: str, log_file
) -> str:
    """Pick the best available map from the preference list."""
    candidates = [requested, *MAP_PREFERENCE]
    ordered: List[str] = []
    for c in candidates:
        if c and c not in ordered:
            ordered.append(c)

    try:
        available = [map_token(name) for name in client.get_available_maps()]
    except Exception as exc:
        log(
            f"[WARN] Could not query available maps; using requested={requested}. "
            f"details={exc}",
            log_file,
        )
        return requested

    available_by_lower = {name.lower(): name for name in available}
    for candidate in ordered:
        c = candidate.lower()
        if c in available_by_lower:
            chosen = available_by_lower[c]
            log(
                f"[INFO] Map selector picked: {chosen} (candidate={candidate})",
                log_file,
            )
            return chosen
        for token in available:
            if token.lower().startswith(c):
                log(
                    f"[INFO] Map selector picked: {token} "
                    f"(candidate={candidate}, prefix-match)",
                    log_file,
                )
                return token

    log(
        f"[WARN] No preferred map found; using requested={requested}",
        log_file,
    )
    return requested


def load_world_with_retry(
    client: carla.Client,
    selected_map: str,
    log_file,
    retries: int = 4,
) -> carla.World:
    """Load a map with reconnect retries on transient errors."""
    chosen = map_token(selected_map)
    last_error: Optional[Exception] = None

    for attempt in range(1, retries + 1):
        # Check if target map is already active
        try:
            client.set_timeout(30.0)
            current_world = client.get_world()
            active = map_token(current_world.get_map().name)
            if active.lower() == chosen.lower():
                log(
                    f"[INFO] Reusing active map without reload: {active}",
                    log_file,
                )
                return current_world
        except Exception:
            pass

        timeout_seconds = 120.0 + (attempt - 1) * 45.0
        try:
            client.set_timeout(timeout_seconds)
            log(
                f"[INFO] Loading map attempt={attempt}/{retries} "
                f"map={chosen} timeout={timeout_seconds:.0f}s",
                log_file,
            )
            return client.load_world(chosen)
        except RuntimeError as exc:
            last_error = exc
            if not is_retryable_carla_error(exc):
                raise
            log(
                f"[WARN] load_world retryable failure map={chosen} "
                f"attempt={attempt}/{retries}: {exc}",
                log_file,
            )
            # Probe whether the map actually switched
            try:
                client.set_timeout(45.0)
                probe = client.get_world()
                active = map_token(probe.get_map().name)
                if active.lower() == chosen.lower():
                    log(
                        f"[INFO] Active map switched to target after "
                        f"retry failure: {active}",
                        log_file,
                    )
                    return probe
            except Exception as reconnect_exc:
                log(
                    f"[WARN] reconnect probe failed after load_world "
                    f"error: {reconnect_exc}",
                    log_file,
                )
            time.sleep(min(2.0 * attempt, 6.0))

    if last_error is not None:
        raise RuntimeError(
            f"Failed to load map '{chosen}' after {retries} attempts"
        ) from last_error
    raise RuntimeError(f"Failed to load map '{chosen}'")


def apply_sync_settings(
    world: carla.World, client: carla.Client, log_file
) -> carla.WorldSettings:
    """Enable synchronous mode and return the original settings for later restore."""
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
            log(
                f"[WARN] apply_settings retryable failure "
                f"attempt={attempt}/4: {exc}",
                log_file,
            )
            time.sleep(min(1.5 * attempt, 6.0))

    raise RuntimeError(
        "Failed to apply synchronous world settings"
    ) from last_error


# ---------------------------------------------------------------------------
# Blueprint helpers
# ---------------------------------------------------------------------------

def find_blueprint(
    bp_lib: carla.BlueprintLibrary,
    preferred_ids: List[str],
    fallback_pattern: str,
    must_contain: Optional[str] = None,
) -> carla.ActorBlueprint:
    """Try preferred IDs first, then fall back to a pattern search."""
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


def set_vehicle_attributes(
    bp: carla.ActorBlueprint, role_name: str = "autopilot"
) -> None:
    """Set common vehicle attributes (role_name, color, driver_id)."""
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
# Traffic flow helpers
# ---------------------------------------------------------------------------

def collect_flow_counts(
    world: carla.World, ego: carla.Vehicle, radius_m: float = 240.0
) -> Tuple[Dict[str, int], int, int]:
    """Count nearby vehicles classified as same/opposite direction flow."""
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
    """Batch-spawn vehicles with autopilot and return the count spawned."""
    if not transforms:
        return 0
    world_map = world.get_map()
    batch = []
    for tf in transforms:
        bp = random.choice(blueprints)
        set_vehicle_attributes(bp, role_name="autopilot")
        spawn_tf = snap_and_lift_to_driving_lane(world_map, tf)
        spawn = carla.command.SpawnActor(bp, spawn_tf)
        autopilot = carla.command.SetAutopilot(
            carla.command.FutureActor, True, traffic_manager.get_port()
        )
        batch.append(spawn.then(autopilot))
    responses = client.apply_batch_sync(batch, True)
    actor_ids = [r.actor_id for r in responses if not r.error]
    if not actor_ids:
        return 0

    for actor in world.get_actors(actor_ids):
        actors_to_cleanup.append(actor)
        try:
            traffic_manager.auto_lane_change(actor, True)
            traffic_manager.ignore_walkers_percentage(actor, 0.0)
            # shot_1: make traffic faster so they actively pass bicycles/ego
            traffic_manager.vehicle_percentage_speed_difference(
                actor, random.uniform(-15.0, 5.0)  # mostly faster than speed limit
            )
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
    """Select spawn transforms biased toward the directions that still need vehicles."""
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

    # Fill remaining quota from any direction
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
    """Spawn dense highway traffic concentrated behind and beside ego.

    shot_1: Prioritise spawns within 80m of ego so vehicles are visible
    in rear and side cameras. Left-lane vehicles set faster to pass.
    """
    ego_loc = ego.get_location()
    # shot_1: prioritise close spawns (behind/beside ego)
    close_available: List[carla.Transform] = []
    far_available: List[carla.Transform] = []
    for sp in spawn_points:
        d = distance_2d(sp.location, ego_loc)
        if 10.0 <= d <= 80.0:
            close_available.append(sp)
        elif 80.0 < d <= 240.0:
            far_available.append(sp)
    available = close_available + far_available

    random.shuffle(available)
    used_indices: set = set()
    ego_tf = ego.get_transform()

    # Initial large batch
    initial_pick = select_spawn_candidates_for_flow(
        available,
        used_indices,
        ego_tf,
        need_same=MIN_FLOW_PER_DIRECTION * 2,
        need_opposite=MIN_FLOW_PER_DIRECTION * 2,
        max_take=min(len(available), 120),
    )
    spawned = spawn_batch(
        client, world, traffic_manager, vehicle_blueprints,
        initial_pick, actors_to_cleanup,
    )
    for _ in range(35):
        safe_tick(world, "traffic_settle_initial", log_file, fatal=False)

    # Iterative fill-up
    for attempt in range(8):
        flow_counts, total, distinct = collect_flow_counts(
            world, ego, radius_m=240.0
        )
        log(
            f"[TRAFFIC] attempt={attempt} flow={flow_counts} total={total} "
            f"distinct_types={distinct} spawned={spawned}",
            log_file,
        )
        need_same = max(0, MIN_FLOW_PER_DIRECTION - flow_counts["same"])
        need_opposite = max(
            0, MIN_FLOW_PER_DIRECTION - flow_counts["opposite"]
        )
        if need_same <= 0 and need_opposite <= 0 and distinct >= 10:
            return {
                "flow_counts": flow_counts,
                "total": total,
                "distinct_types": distinct,
                "spawned": spawned,
            }

        extra = select_spawn_candidates_for_flow(
            available,
            used_indices,
            ego_tf,
            need_same=need_same * 3,
            need_opposite=need_opposite * 3,
            max_take=max(24, (need_same + need_opposite) * 3),
        )
        if not extra:
            break
        spawned += spawn_batch(
            client, world, traffic_manager, vehicle_blueprints,
            extra, actors_to_cleanup,
        )
        for _ in range(22):
            safe_tick(world, "traffic_settle_extra", log_file, fatal=False)

    flow_counts, total, distinct = collect_flow_counts(
        world, ego, radius_m=240.0
    )
    raise RuntimeError(
        "Traffic density contract not met after retries: "
        f"flow={flow_counts} total={total} distinct_types={distinct} "
        f"spawned={spawned}"
    )


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def apply_streetlights_mode(world: carla.World, mode: str, log_file) -> None:
    """Best-effort streetlight override used by matrix runs."""
    requested = str(mode or "auto").strip().lower()
    if requested == "auto":
        log("[INFO] Streetlights mode: auto (no explicit override)", log_file)
        return
    if requested not in {"on", "off"}:
        log(
            f"[WARN] Unknown streetlights mode '{mode}', expected auto/on/off",
            log_file,
        )
        return

    try:
        light_manager = world.get_lightmanager()
        if hasattr(carla, "LightGroup"):
            try:
                street_lights = light_manager.get_all_lights(carla.LightGroup.Street)
            except Exception:
                street_lights = light_manager.get_all_lights()
        else:
            street_lights = light_manager.get_all_lights()
        if requested == "on":
            light_manager.turn_on(street_lights)
        else:
            light_manager.turn_off(street_lights)
        log(
            f"[INFO] Streetlights override applied: mode={requested} count={len(street_lights)}",
            log_file,
        )
    except Exception as exc:
        log(
            f"[WARN] Streetlights override '{requested}' not applied: {exc}",
            log_file,
        )


def apply_highway_weather(
    world: carla.World, args: argparse.Namespace, log_file
) -> None:
    """Set default highway weather and apply matrix/CLI overrides when provided."""
    weather = carla.WeatherParameters(
        cloudiness=15.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=8.0,
        fog_density=0.0,
        wetness=0.0,
        sun_altitude_angle=45.0,
        sun_azimuth_angle=220.0,
    )

    if args.cloudiness is not None:
        weather.cloudiness = float(args.cloudiness)
    if args.precipitation is not None:
        weather.precipitation = float(args.precipitation)
    if args.precipitation_deposits is not None:
        weather.precipitation_deposits = float(args.precipitation_deposits)
    if args.wind_intensity is not None:
        weather.wind_intensity = float(args.wind_intensity)
    if args.fog_density is not None:
        weather.fog_density = float(args.fog_density)
    if args.fog_distance is not None:
        weather.fog_distance = float(args.fog_distance)
    if args.wetness is not None:
        weather.wetness = float(args.wetness)
    if args.sun_altitude is not None:
        weather.sun_altitude_angle = float(args.sun_altitude)
    if args.sun_azimuth is not None:
        weather.sun_azimuth_angle = float(args.sun_azimuth)

    world.set_weather(weather)
    log(
        "[INFO] Weather applied "
        f"(time_preset={args.time_preset}, weather_preset={args.weather_preset}, "
        f"sun_alt={weather.sun_altitude_angle:.1f}, sun_az={weather.sun_azimuth_angle:.1f}, "
        f"cloud={weather.cloudiness:.1f}, rain={weather.precipitation:.1f}, "
        f"rain_deposits={weather.precipitation_deposits:.1f}, wind={weather.wind_intensity:.1f}, "
        f"fog_density={weather.fog_density:.1f}, fog_distance={weather.fog_distance:.1f}, "
        f"wetness={weather.wetness:.1f})",
        log_file,
    )
    apply_streetlights_mode(world, args.streetlights, log_file)


def is_night_like_time_variant(args: argparse.Namespace) -> bool:
    """Return True when matrix/default args imply a night-like lighting condition."""
    time_key = str(getattr(args, "time_preset", "") or "").strip().lower()
    if "night" in time_key:
        return True
    sun_alt = getattr(args, "sun_altitude", None)
    try:
        if sun_alt is not None and float(sun_alt) <= 0.0:
            return True
    except Exception:
        pass
    return False


def apply_vehicle_headlights_for_time_variant(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    args: argparse.Namespace,
    log_file,
) -> None:
    """Force headlights on for night matrix variants (4-wheel vehicles primarily)."""
    enable_headlights = is_night_like_time_variant(args)
    if not enable_headlights:
        log("[INFO] Vehicle headlights: daytime/default variant (no force-on applied)", log_file)
        return

    if not hasattr(carla, "VehicleLightState"):
        log("[WARN] CARLA VehicleLightState unavailable; cannot force headlights", log_file)
        return

    try:
        headlight_state = (
            carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam
        )
    except Exception:
        headlight_state = carla.VehicleLightState.Position

    updated = 0
    failed = 0
    for actor in world.get_actors().filter("vehicle.*"):
        try:
            if hasattr(traffic_manager, "update_vehicle_lights"):
                traffic_manager.update_vehicle_lights(actor, True)
        except Exception:
            pass
        try:
            actor.set_light_state(headlight_state)
            updated += 1
        except Exception:
            failed += 1

    log(
        "[INFO] Vehicle headlights forced ON for night-like variant "
        f"(time_preset={getattr(args, 'time_preset', None)}, "
        f"sun_altitude={getattr(args, 'sun_altitude', None)}, "
        f"updated={updated}, failed={failed})",
        log_file,
    )


# ---------------------------------------------------------------------------
# Bicycle spawning
# ---------------------------------------------------------------------------

def find_rightmost_driving_lane(wp: carla.Waypoint) -> carla.Waypoint:
    """Walk to the rightmost driving lane from the given waypoint."""
    current = wp
    while True:
        right = current.get_right_lane()
        if right is None:
            break
        if right.lane_type != carla.LaneType.Driving:
            break
        # Make sure direction is the same (positive dot product on forward)
        fwd_cur = current.transform.get_forward_vector()
        fwd_right = right.transform.get_forward_vector()
        dot = fwd_cur.x * fwd_right.x + fwd_cur.y * fwd_right.y
        if dot < 0.5:
            break
        current = right
    return current


def spawn_bicycle_ahead(
    world: carla.World,
    bp_lib: carla.BlueprintLibrary,
    traffic_manager: carla.TrafficManager,
    ego: carla.Vehicle,
    world_map: carla.Map,
    actors_to_cleanup: List[carla.Actor],
    log_file,
) -> Tuple[List[carla.Actor], carla.Location]:
    """Spawn 3-5 bicycles in a cluster ahead of ego in the rightmost lane.

    shot_1: Multiple bicycles instead of single, spaced 8-12m apart.
    Returns (list_of_bicycle_actors, cluster_center_location).
    """
    ego_wp = world_map.get_waypoint(
        ego.get_location(),
        project_to_road=True,
        lane_type=carla.LaneType.Driving,
    )
    if ego_wp is None:
        raise RuntimeError("Cannot resolve ego waypoint for bicycle spawn")

    # Move to rightmost driving lane so bicycles are on the slow lane
    rightmost_wp = find_rightmost_driving_lane(ego_wp)

    # Place first bicycle 35m ahead, then stagger 8-12m apart
    base_ahead = 35.0
    bicycle_spacings = [0.0, 10.0, 20.0, 28.0, 38.0]  # 5 positions
    target_count = random.randint(3, 5)

    bicycles: List[carla.Actor] = []
    cluster_center: Optional[carla.Location] = None

    for i in range(target_count):
        ahead_distance = base_ahead + bicycle_spacings[i]
        bicycle_wp = next_waypoint(rightmost_wp, ahead_distance)

        # Skip junctions
        if bicycle_wp.is_junction:
            bicycle_wp = next_waypoint(rightmost_wp, ahead_distance + 15.0)

        bicycle_tf = lifted_transform(bicycle_wp.transform)

        # Try each bicycle blueprint
        bicycle_actor: Optional[carla.Actor] = None
        used_bp_id = ""
        # Cycle through blueprints so bikes look different
        bp_rotation = BICYCLE_BLUEPRINTS[i % len(BICYCLE_BLUEPRINTS):]
        bp_rotation += BICYCLE_BLUEPRINTS[:i % len(BICYCLE_BLUEPRINTS)]
        for bp_id in bp_rotation:
            try:
                bike_bp = bp_lib.find(bp_id)
            except Exception:
                continue
            set_vehicle_attributes(bike_bp, role_name="bicycle")
            bicycle_actor = world.try_spawn_actor(bike_bp, bicycle_tf)
            if bicycle_actor is not None:
                used_bp_id = bp_id
                break

        if bicycle_actor is None:
            # Fallback: any 2-wheel vehicle
            try:
                bike_bp = find_blueprint(bp_lib, [], "vehicle.*", must_contain="bike")
                set_vehicle_attributes(bike_bp, role_name="bicycle")
                bicycle_actor = world.try_spawn_actor(bike_bp, bicycle_tf)
                if bicycle_actor is not None:
                    used_bp_id = bike_bp.id
            except Exception:
                pass

        if bicycle_actor is None:
            log(f"[WARN] Failed to spawn bicycle #{i} at distance={ahead_distance:.0f}m", log_file)
            continue

        actors_to_cleanup.append(bicycle_actor)
        bicycles.append(bicycle_actor)
        spawn_loc = bicycle_actor.get_location()

        if cluster_center is None:
            cluster_center = spawn_loc

        # Configure TM: bicycles ride at about half the ego target speed, in-lane.
        bicycle_actor.set_autopilot(True, traffic_manager.get_port())
        traffic_manager.vehicle_percentage_speed_difference(
            bicycle_actor, bicycle_tm_diff_from_ego_tm_diff(65.0)
        )
        traffic_manager.auto_lane_change(bicycle_actor, False)
        traffic_manager.ignore_lights_percentage(bicycle_actor, 100.0)
        traffic_manager.ignore_signs_percentage(bicycle_actor, 100.0)
        traffic_manager.distance_to_leading_vehicle(bicycle_actor, 1.5)

        log(
            f"[INFO] Bicycle #{i} spawned: {used_bp_id} id={bicycle_actor.id} "
            f"at ({spawn_loc.x:.1f}, {spawn_loc.y:.1f}), "
            f"ahead_distance={ahead_distance:.1f}m",
            log_file,
        )

    if not bicycles:
        raise RuntimeError("Failed to spawn any bicycles at highway waypoint")

    if cluster_center is None:
        cluster_center = bicycles[0].get_location()

    log(f"[INFO] Total bicycles spawned: {len(bicycles)} (target was {target_count})", log_file)

    # shot_1: Set ego to follow closely behind bicycle group
    traffic_manager.vehicle_percentage_speed_difference(ego, 65.0)
    traffic_manager.auto_lane_change(ego, False)  # stay behind bikes
    traffic_manager.distance_to_leading_vehicle(ego, 3.0)

    return bicycles, cluster_center


# ---------------------------------------------------------------------------
# Camera rig
# ---------------------------------------------------------------------------

def clear_frame_outputs(output_dir: Path) -> Dict[str, Path]:
    """Create (and wipe) per-camera output directories."""
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
    """Return a callback that stores camera images in the thread-safe buffer."""
    def callback(image: carla.Image) -> None:
        with images_lock:
            images_received[camera_key] = image
            if all(images_received[k] is not None for k in CAMERA_KEYS):
                frame_ready.set()
    return callback


def clear_camera_buffers() -> None:
    """Reset all camera image slots and clear the ready event."""
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
    """Attach the configured synchronised RGB camera rig to the ego vehicle."""
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x", "800")
    cam_bp.set_attribute("image_size_y", "450")
    cam_bp.set_attribute("fov", "110")
    cam_bp.set_attribute("sensor_tick", str(FRAME_DT))

    transforms = {
        "front": carla.Transform(
            carla.Location(x=1.5, y=0.0, z=1.6),
            carla.Rotation(pitch=-2.0),
        ),
        "front_left": carla.Transform(
            carla.Location(x=1.2, y=-0.6, z=1.6),
            carla.Rotation(yaw=-55, pitch=-3.0),
        ),
        "front_right": carla.Transform(
            carla.Location(x=1.2, y=0.6, z=1.6),
            carla.Rotation(yaw=55, pitch=-3.0),
        ),
        "rear": carla.Transform(
            carla.Location(x=-1.8, y=0.0, z=1.6),
            carla.Rotation(yaw=180, pitch=-2.0),
        ),
        "drone_follow": carla.Transform(
            carla.Location(x=-10.0, y=0.0, z=14.0),
            carla.Rotation(pitch=-36.0),
        ),
    }

    for key in CAMERA_KEYS:
        tf = transforms[key]
        cam = world.spawn_actor(cam_bp, tf, attach_to=ego)
        actors_to_cleanup.append(cam)
        cam.listen(make_camera_callback(key))

    for _ in range(12):
        safe_tick(world, "camera_warmup", log_file, retries=3, fatal=False)
    clear_camera_buffers()
    log(f"[INFO] Attached and warmed up {len(CAMERA_KEYS)} camera(s)", log_file)


# ---------------------------------------------------------------------------
# Capture loop
# ---------------------------------------------------------------------------

def run_capture_loop(
    world: carla.World,
    traffic_manager: carla.TrafficManager,
    ego: carla.Vehicle,
    bicycle: carla.Actor,
    bicycle_group: List[carla.Actor],
    bicycle_spawn_loc: carla.Location,
    camera_dirs: Dict[str, Path],
    duration_s: float,
    log_file,
) -> Dict[str, int]:
    """Run the main simulation loop, modulating ego speed based on bicycle proximity."""
    max_frames = max(1, int(round(duration_s * FPS)))
    frame_counts = {key: 0 for key in CAMERA_KEYS}
    log(
        f"[INFO] Starting simulation loop: target_frames={max_frames} "
        f"duration={duration_s}s",
        log_file,
    )

    for frame_idx in range(1, max_frames + 1):
        if not safe_tick(
            world, f"run_frame_{frame_idx}", log_file,
            retries=2, fatal=False,
        ):
            continue

        # --- Ego speed modulation based on distance to bicycle ---
        ego_loc = ego.get_location()
        bicycle_loc = bicycle.get_location()
        dist_to_bicycle = distance_2d(ego_loc, bicycle_loc)

        if dist_to_bicycle < 15.0:
            # Very close -- slow way down to avoid collision
            ego_speed_diff_target = 65.0
        elif dist_to_bicycle < 30.0:
            # Approaching -- moderate slowdown
            ego_speed_diff_target = 45.0
        elif dist_to_bicycle < 60.0:
            # Closing in -- begin reducing speed
            ego_speed_diff_target = 28.0
        else:
            # Far away -- cruise at near highway speed
            ego_speed_diff_target = 20.0

        traffic_manager.vehicle_percentage_speed_difference(ego, ego_speed_diff_target)
        bicycle_speed_diff_target = bicycle_tm_diff_from_ego_tm_diff(ego_speed_diff_target)
        for bike in bicycle_group:
            try:
                traffic_manager.vehicle_percentage_speed_difference(
                    bike, bicycle_speed_diff_target
                )
            except Exception:
                pass

        # --- Capture camera frames ---
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
                log(
                    f"[WARN] Camera frame timeout at frame={frame_idx}",
                    log_file,
                )

        # --- Progress logging ---
        if frame_idx % 20 == 0 or frame_idx == 1:
            flow_counts, total, distinct = collect_flow_counts(
                world, ego, radius_m=240.0
            )
            ego_speed = get_speed_kmh(ego)
            bicycle_speed = get_speed_kmh(bicycle)
            log(
                f"[PROGRESS] frame={frame_idx}/{max_frames} "
                f"ego_speed={ego_speed:.1f}km/h "
                f"bicycle_speed={bicycle_speed:.1f}km/h "
                f"bicycle_target_ratio={BICYCLE_SPEED_RATIO_TO_EGO:.2f} "
                f"bicycle_dist={dist_to_bicycle:.1f}m "
                f"flow={flow_counts} total={total} distinct={distinct} "
                f"captures={frame_counts}",
                log_file,
            )

    # Drain remaining frames
    for _ in range(8):
        safe_tick(world, "run_drain", log_file, retries=2, fatal=False)

    return frame_counts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone Bicycle in Highway Lane Shot-0"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--duration", type=float, default=16.0)
    parser.add_argument(
        "--output-dir", default="./scenes/bicycle_in_highway_lane"
    )
    parser.add_argument("--output", dest="output_alias", default=None)
    parser.add_argument("--seed", type=int, default=402)
    parser.add_argument("--map", default="Town04")
    # Accept matrix environment overrides even if this scene uses its own defaults.
    parser.add_argument("--time-preset", default="scene_default")
    parser.add_argument("--weather-preset", default="scene_default")
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

    random.seed(int(args.seed))

    output_root = Path(args.output_alias or args.output_dir).resolve()
    camera_dirs = clear_frame_outputs(output_root)
    log_path = output_root / "bicycle_in_highway_lane_simulation.log"
    output_root.mkdir(parents=True, exist_ok=True)

    client: Optional[carla.Client] = None
    world: Optional[carla.World] = None
    traffic_manager: Optional[carla.TrafficManager] = None
    original_settings: Optional[carla.WorldSettings] = None
    actors_to_cleanup: List[carla.Actor] = []

    with log_path.open("w", encoding="utf-8") as log_file:
        try:
            # ----------------------------------------------------------
            # 1. Connect
            # ----------------------------------------------------------
            log("[INFO] Scenario: Bicycle in Highway Lane", log_file)
            log(f"[INFO] Seed: {args.seed}", log_file)
            log(
                f"[INFO] Connecting to CARLA at {args.host}:{args.port}",
                log_file,
            )

            client = carla.Client(args.host, int(args.port))
            client.set_timeout(60.0)

            # ----------------------------------------------------------
            # 2. Load Town04
            # ----------------------------------------------------------
            selected_map = choose_preferred_map(client, args.map, log_file)
            world = load_world_with_retry(
                client, selected_map, log_file, retries=4
            )
            client.set_timeout(45.0)

            world_map = world.get_map()
            active_map = map_token(world_map.name)
            log(f"[INFO] Active map: {active_map}", log_file)
            if active_map.lower() != "town04":
                raise RuntimeError(
                    f"Scene specification requires Town04 highway segment only; got {active_map}"
                )

            # ----------------------------------------------------------
            # 3. Synchronous mode
            # ----------------------------------------------------------
            original_settings = apply_sync_settings(world, client, log_file)

            # ----------------------------------------------------------
            # 4. Traffic Manager
            # ----------------------------------------------------------
            traffic_manager = client.get_trafficmanager(8000)
            traffic_manager.set_synchronous_mode(True)
            traffic_manager.set_global_distance_to_leading_vehicle(2.4)
            traffic_manager.set_hybrid_physics_mode(True)
            traffic_manager.set_hybrid_physics_radius(180.0)
            traffic_manager.global_percentage_speed_difference(2.0)

            # ----------------------------------------------------------
            # 5. Weather -- clear daytime, good visibility
            # ----------------------------------------------------------
            apply_highway_weather(world, args, log_file)
            for _ in range(10):
                safe_tick(world, "setup_warmup", log_file, fatal=False)

            # ----------------------------------------------------------
            # 6. Clear pre-existing vehicles
            # ----------------------------------------------------------
            removed_existing = 0
            for actor in world.get_actors().filter("vehicle.*"):
                try:
                    actor.destroy()
                    removed_existing += 1
                except Exception:
                    pass
            log(
                f"[INFO] Cleared pre-existing vehicles: {removed_existing}",
                log_file,
            )

            # ----------------------------------------------------------
            # 7. Spawn ego on highway
            # ----------------------------------------------------------
            spawn_points = list(world_map.get_spawn_points())
            if not spawn_points:
                raise RuntimeError("No spawn points available on map")
            random.shuffle(spawn_points)

            bp_lib = world.get_blueprint_library()
            ego_bp = find_blueprint(
                bp_lib,
                [
                    "vehicle.tesla.model3",
                    "vehicle.lincoln.mkz_2020",
                    "vehicle.audi.tt",
                ],
                "vehicle.*",
            )
            # TM hybrid physics uses hero-tagged vehicles as the full-physics anchor.
            set_vehicle_attributes(ego_bp, role_name="hero")

            ego_vehicle: Optional[carla.Vehicle] = None
            ego_highway_wp: Optional[carla.Waypoint] = None
            for sp in spawn_points:
                wp = world_map.get_waypoint(
                    sp.location,
                    project_to_road=True,
                    lane_type=carla.LaneType.Driving,
                )
                if wp is None:
                    continue
                rightmost_wp = find_rightmost_driving_lane(wp)
                if rightmost_wp is None or not is_highway_candidate(rightmost_wp):
                    continue
                ego_tf = lifted_transform(rightmost_wp.transform)
                ego_vehicle = world.try_spawn_actor(ego_bp, ego_tf)
                if ego_vehicle is not None:
                    ego_highway_wp = rightmost_wp
                    break

            # Fallback: try any spawn point
            if ego_vehicle is None:
                log(
                    "[WARN] No highway-candidate spawn found; "
                    "falling back to any spawn",
                    log_file,
                )
                for sp in spawn_points:
                    fallback_tf = lifted_transform(sp)
                    fallback_wp = world_map.get_waypoint(
                        sp.location,
                        project_to_road=True,
                        lane_type=carla.LaneType.Driving,
                    )
                    if fallback_wp is not None:
                        rightmost_wp = find_rightmost_driving_lane(fallback_wp)
                        if rightmost_wp is not None:
                            fallback_tf = lifted_transform(rightmost_wp.transform)
                    ego_vehicle = world.try_spawn_actor(ego_bp, fallback_tf)
                    if ego_vehicle is not None:
                        ego_highway_wp = fallback_wp
                        break
            if ego_vehicle is None:
                raise RuntimeError("Failed to spawn ego vehicle")

            actors_to_cleanup.append(ego_vehicle)
            ego_vehicle.set_autopilot(True, traffic_manager.get_port())
            # Keep ego in the right lane so it follows directly behind the bicycles.
            traffic_manager.auto_lane_change(ego_vehicle, False)
            traffic_manager.ignore_lights_percentage(ego_vehicle, 0.0)
            traffic_manager.ignore_walkers_percentage(ego_vehicle, 0.0)
            traffic_manager.vehicle_percentage_speed_difference(
                ego_vehicle, 20.0
            )
            log(
                f"[INFO] Ego spawned: {ego_vehicle.type_id} "
                f"id={ego_vehicle.id}",
                log_file,
            )
            ego_wp_check = world_map.get_waypoint(
                ego_vehicle.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if ego_wp_check is not None:
                rightmost_check = find_rightmost_driving_lane(ego_wp_check)
                on_rightmost = (
                    rightmost_check is not None
                    and ego_wp_check.road_id == rightmost_check.road_id
                    and ego_wp_check.section_id == rightmost_check.section_id
                    and ego_wp_check.lane_id == rightmost_check.lane_id
                )
                log(
                    "[INFO] Ego lane check: "
                    f"lane_id={ego_wp_check.lane_id} "
                    f"rightmost_lane_id={rightmost_check.lane_id if rightmost_check else 'n/a'} "
                    f"on_rightmost={on_rightmost}",
                    log_file,
                )

            # Let ego settle for a few ticks before attaching cameras
            for _ in range(8):
                safe_tick(world, "ego_settle", log_file, fatal=False)

            # ----------------------------------------------------------
            # 8. Attach cameras
            # ----------------------------------------------------------
            attach_cameras(
                world, bp_lib, ego_vehicle, actors_to_cleanup, log_file
            )

            # ----------------------------------------------------------
            # 9. Spawn bicycle ahead of ego
            # ----------------------------------------------------------
            bicycle_actors, bicycle_spawn_loc = spawn_bicycle_ahead(
                world=world,
                bp_lib=bp_lib,
                traffic_manager=traffic_manager,
                ego=ego_vehicle,
                world_map=world_map,
                actors_to_cleanup=actors_to_cleanup,
                log_file=log_file,
            )
            # shot_1: bicycle_actor is now the first bicycle for distance tracking
            bicycle_actor = bicycle_actors[0]

            # Let bicycles settle on autopilot for a few ticks
            for _ in range(15):
                safe_tick(world, "bicycle_settle", log_file, fatal=False)

            # Verify lead bicycle is still alive and on-road
            bicycle_wp_check = world_map.get_waypoint(
                bicycle_actor.get_location(),
                project_to_road=True,
                lane_type=carla.LaneType.Driving,
            )
            if bicycle_wp_check is not None:
                off_road_dist = distance_2d(
                    bicycle_actor.get_location(),
                    bicycle_wp_check.transform.location,
                )
                log(
                    f"[INFO] Bicycle on-road check: "
                    f"offset={off_road_dist:.2f}m "
                    f"lane_width={bicycle_wp_check.lane_width:.2f}m",
                    log_file,
                )

            # ----------------------------------------------------------
            # 10. Spawn dense highway traffic
            # ----------------------------------------------------------
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
            log(
                f"[INFO] Traffic density satisfied: {traffic_diag}",
                log_file,
            )
            apply_vehicle_headlights_for_time_variant(
                world=world,
                traffic_manager=traffic_manager,
                args=args,
                log_file=log_file,
            )

            # ----------------------------------------------------------
            # 11. Run capture loop
            # ----------------------------------------------------------
            frame_counts = run_capture_loop(
                world=world,
                traffic_manager=traffic_manager,
                ego=ego_vehicle,
                bicycle=bicycle_actor,
                bicycle_group=bicycle_actors,
                bicycle_spawn_loc=bicycle_spawn_loc,
                camera_dirs=camera_dirs,
                duration_s=float(args.duration),
                log_file=log_file,
            )
            log(f"[INFO] Capture complete: {frame_counts}", log_file)
            log(
                "[SUCCESS] Bicycle in Highway Lane scenario completed",
                log_file,
            )

        except Exception as exc:
            log(f"[ERROR] {type(exc).__name__}: {exc}", log_file)
            raise

        finally:
            # ----------------------------------------------------------
            # Deterministic cleanup and world settings restore
            # ----------------------------------------------------------
            clear_camera_buffers()
            log("[INFO] Cleanup started", log_file)

            # Stop sensors first
            for actor in actors_to_cleanup:
                if actor is None:
                    continue
                if actor.type_id.startswith("sensor."):
                    try:
                        actor.stop()
                    except Exception:
                        pass

            # Batch-destroy all spawned actors
            if client is not None and actors_to_cleanup:
                destroy_cmds = [
                    carla.command.DestroyActor(actor.id)
                    for actor in actors_to_cleanup
                    if actor is not None
                ]
                if destroy_cmds:
                    try:
                        client.apply_batch(destroy_cmds)
                    except Exception:
                        pass

            # Restore TM sync mode
            if traffic_manager is not None:
                try:
                    traffic_manager.set_synchronous_mode(False)
                except Exception:
                    pass

            # Restore original world settings
            if world is not None and original_settings is not None:
                try:
                    world.apply_settings(original_settings)
                except Exception:
                    pass

            log("[INFO] Cleanup complete", log_file)


if __name__ == "__main__":
    main()
