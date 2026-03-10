#!/usr/bin/env python3

import sys
import os
import time
import argparse
import random
import math
import threading
import numpy as np
from PIL import Image

try:
    import carla
except ImportError:
    print("[ERROR] CARLA Python API not found. Please install it.")
    sys.exit(1)

# ========================================================================
# SHARED STATE FOR 4-CAMERA SYNCHRONIZATION
# ========================================================================
images_received = {'front': None, 'front_left': None, 'front_right': None, 'rear': None}
images_lock = threading.Lock()
frame_ready = threading.Event()

# ========================================================================
# IMAGE SAVING WITH BGR TO RGB CONVERSION
# ========================================================================
def save_image_to_disk(image, output_path):
    """Save CARLA image as RGB PNG (CARLA uses BGRA format)"""
    try:
        array = np.frombuffer(image.raw_data, dtype=np.uint8)
        array = array.reshape((image.height, image.width, 4))
        # CRITICAL: CARLA uses BGRA - convert BGR to RGB
        array = array[:, :, :3][:, :, ::-1]  # BGR to RGB conversion
        img = Image.fromarray(array)
        img.save(output_path, 'PNG')
        return True
    except Exception as e:
        print(f"[ERROR] Failed to save image: {e}")
        return False

def make_camera_callback(camera_name):
    """Create callback for camera synchronization"""
    def callback(image):
        with images_lock:
            images_received[camera_name] = image
            if all(img is not None for img in images_received.values()):
                frame_ready.set()
    return callback

# ========================================================================
# COLLISION DETECTION
# ========================================================================
collision_detected = False

def on_collision(event):
    global collision_detected
    collision_detected = True
    actor_type = 'Unknown'
    if event.other_actor:
        actor_type = event.other_actor.type_id
    print(f"[WARNING] Collision detected with {actor_type} at frame {event.frame}")

# ========================================================================
# MAIN SIMULATION
# ========================================================================
def main():
    global collision_detected
    
    parser = argparse.ArgumentParser(description='CARLA Red Light Violation Scenario')
    parser.add_argument('--host', default='127.0.0.1', help='CARLA server host')
    parser.add_argument('--port', type=int, default=2000, help='CARLA server port')
    parser.add_argument('--duration', type=float, default=25.0, help='Simulation duration in seconds')
    parser.add_argument('--output-dir', default='./scenes/red_light_violation', help='Output directory')
    args = parser.parse_args()
    
    # Create output directories
    output_dirs = {
        'front': os.path.join(args.output_dir, 'front'),
        'front_left': os.path.join(args.output_dir, 'front_left'),
        'front_right': os.path.join(args.output_dir, 'front_right'),
        'rear': os.path.join(args.output_dir, 'rear')
    }
    
    for dir_path in output_dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    
    log_file_path = os.path.join(args.output_dir, 'red_light_violation_simulation.log')
    
    # Connect to CARLA
    client = None
    world = None
    original_settings = None
    actors_to_cleanup = []
    
    try:
        print("[INFO] Connecting to CARLA server...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(60.0)
        
        # Test connection
        version = client.get_server_version()
        print(f"[INFO] Connected to CARLA server version {version}")
        
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()
        spawn_points = world.get_map().get_spawn_points()
        
        # Save original settings
        original_settings = world.get_settings()
        
        # Configure synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05  # 20 FPS
        world.apply_settings(settings)
        print("[INFO] Synchronous mode enabled at 20 FPS")
        
        # ========================================================================
        # FIND SPAWN POINT FACING A TRAFFIC LIGHT (40-55m BEFORE)
        # ========================================================================
        print("[INFO] Finding suitable spawn point near traffic light...")
        traffic_lights = world.get_actors().filter('traffic.traffic_light')
        
        best_spawn = None
        target_tl = None
        min_angle_diff = float('inf')
        
        for sp in spawn_points:
            sp_forward = sp.rotation.get_forward_vector()
            for tl in traffic_lights:
                dist = sp.location.distance(tl.get_location())
                if 40.0 < dist < 55.0:
                    # Check if spawn point faces the traffic light
                    to_tl = tl.get_location() - sp.location
                    to_tl_len = math.sqrt(to_tl.x**2 + to_tl.y**2)
                    if to_tl_len > 0:
                        to_tl_norm = carla.Vector3D(to_tl.x/to_tl_len, to_tl.y/to_tl_len, 0)
                        dot = sp_forward.x * to_tl_norm.x + sp_forward.y * to_tl_norm.y
                        if dot > 0.8:  # Well-aligned facing light
                            angle_diff = math.acos(max(-1.0, min(1.0, dot)))
                            if angle_diff < min_angle_diff:
                                best_spawn = sp
                                target_tl = tl
                                min_angle_diff = angle_diff
        
        if not best_spawn or not target_tl:
            print("[ERROR] Could not find suitable spawn point facing traffic light")
            return
        
        print(f"[INFO] Selected spawn point {best_spawn.location} at {best_spawn.location.distance(target_tl.get_location()):.1f}m from traffic light")
        
        # ========================================================================
        # SET TARGET TRAFFIC LIGHT TO RED AND FREEZE IT
        # ========================================================================
        target_tl.set_state(carla.TrafficLightState.Red)
        target_tl.set_red_time(9999.0)
        target_tl.freeze(True)
        world.tick()
        print(f"[INFO] Target traffic light at {target_tl.get_location()} set to RED and frozen")
        
        # ========================================================================
        # SPAWN VIOLATING VEHICLE
        # ========================================================================
        print("[INFO] Spawning violating vehicle...")
        vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
        violator_vehicle = world.try_spawn_actor(vehicle_bp, best_spawn)
        
        if violator_vehicle is None:
            # Try alternative spawn
            for offset_dist in [2.0, -2.0, 4.0, -4.0]:
                alt_spawn = carla.Transform(
                    carla.Location(
                        x=best_spawn.location.x + offset_dist * best_spawn.rotation.get_forward_vector().x,
                        y=best_spawn.location.y + offset_dist * best_spawn.rotation.get_forward_vector().y,
                        z=best_spawn.location.z
                    ),
                    best_spawn.rotation
                )
                violator_vehicle = world.try_spawn_actor(vehicle_bp, alt_spawn)
                if violator_vehicle:
                    break
        
        if violator_vehicle is None:
            print("[ERROR] Failed to spawn violating vehicle")
            return
        
        actors_to_cleanup.append(violator_vehicle)
        print(f"[INFO] Violating vehicle spawned at {violator_vehicle.get_location()}")
        
        # ========================================================================
        # ATTACH COLLISION SENSOR
        # ========================================================================
        collision_bp = blueprint_library.find('sensor.other.collision')
        collision_sensor = world.spawn_actor(collision_bp, carla.Transform(), attach_to=violator_vehicle)
        actors_to_cleanup.append(collision_sensor)
        collision_sensor.listen(on_collision)
        
        # ========================================================================
        # SPAWN COMPLIANT VEHICLES (NOT IN VIOLATOR'S PATH)
        # ========================================================================
        print("[INFO] Spawning compliant vehicles...")
        compliant_count = 0
        violator_forward = best_spawn.rotation.get_forward_vector()
        
        for sp in spawn_points[:20]:  # Try first 20 spawn points
            if compliant_count >= 6:
                break
            
            # Skip if too close to violator spawn
            if sp.location.distance(best_spawn.location) < 20.0:
                continue
            
            # Check if this spawn point is in violator's forward path
            to_sp = sp.location - best_spawn.location
            to_sp_len = math.sqrt(to_sp.x**2 + to_sp.y**2)
            if to_sp_len > 0:
                to_sp_norm = carla.Vector3D(to_sp.x/to_sp_len, to_sp.y/to_sp_len, 0)
                dot = violator_forward.x * to_sp_norm.x + violator_forward.y * to_sp_norm.y
                
                # Skip if in front of violator (same direction)
                if dot > 0.7:
                    continue
            
            # Spawn compliant vehicle
            vehicle_bp = random.choice(blueprint_library.filter('vehicle.*'))
            if 'bike' in vehicle_bp.id or 'motor' in vehicle_bp.id:
                continue
            
            compliant_vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if compliant_vehicle:
                actors_to_cleanup.append(compliant_vehicle)
                compliant_vehicle.set_autopilot(True)
                compliant_count += 1
        
        print(f"[INFO] Spawned {compliant_count} compliant vehicles")
        
        # ========================================================================
        # SETUP 4 DRIVER-POV CAMERAS
        # ========================================================================
        print("[INFO] Setting up 4 driver-POV cameras...")
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '1280')
        camera_bp.set_attribute('image_size_y', '720')
        camera_bp.set_attribute('fov', '110')
        
        # Front camera (hood-cam view)
        front_transform = carla.Transform(
            carla.Location(x=0.8, y=0.0, z=1.4),
            carla.Rotation(pitch=8)
        )
        front_camera = world.spawn_actor(camera_bp, front_transform, attach_to=violator_vehicle)
        actors_to_cleanup.append(front_camera)
        front_camera.listen(make_camera_callback('front'))
        
        # Front-left camera (driver's left view)
        front_left_transform = carla.Transform(
            carla.Location(x=-0.1, y=-0.4, z=1.2),
            carla.Rotation(yaw=-60)
        )
        front_left_camera = world.spawn_actor(camera_bp, front_left_transform, attach_to=violator_vehicle)
        actors_to_cleanup.append(front_left_camera)
        front_left_camera.listen(make_camera_callback('front_left'))
        
        # Front-right camera (driver's right view)
        front_right_transform = carla.Transform(
            carla.Location(x=-0.1, y=0.4, z=1.2),
            carla.Rotation(yaw=60)
        )
        front_right_camera = world.spawn_actor(camera_bp, front_right_transform, attach_to=violator_vehicle)
        actors_to_cleanup.append(front_right_camera)
        front_right_camera.listen(make_camera_callback('front_right'))
        
        # Rear camera (rearview mirror)
        rear_transform = carla.Transform(
            carla.Location(x=-0.2, y=0.3, z=1.25),
            carla.Rotation(yaw=180, pitch=5)
        )
        rear_camera = world.spawn_actor(camera_bp, rear_transform, attach_to=violator_vehicle)
        actors_to_cleanup.append(rear_camera)
        rear_camera.listen(make_camera_callback('rear'))
        
        print("[INFO] All 4 cameras attached")
        
        # ========================================================================
        # CAMERA WARM-UP
        # ========================================================================
        print("[INFO] Warming up cameras...")
        for _ in range(20):
            world.tick()
        time.sleep(0.5)
        print("[INFO] Cameras ready")
        
        # ========================================================================
        # MAIN SIMULATION LOOP
        # ========================================================================
        print("[INFO] Starting red light violation simulation...")
        
        max_frames = min(int(args.duration * 20), 500)
        frame_number = 1
        violation_logged = False
        
        TARGET_SPEED = 40.0 / 3.6  # 40 km/h in m/s
        
        log_entries = []
        log_entries.append("frame,timestamp,vehicle_x,vehicle_y,vehicle_z,speed_kmh,traffic_light_state,distance_to_light\n")
        
        # Get initial distance to traffic light for reference
        tl_location = target_tl.get_location()
        
        while frame_number <= max_frames:
            world.tick()
            
            # Get vehicle state
            vehicle_location = violator_vehicle.get_location()
            vehicle_velocity = violator_vehicle.get_velocity()
            current_speed = math.sqrt(vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)
            current_speed_kmh = current_speed * 3.6
            
            # Calculate distance to traffic light
            distance_to_light = vehicle_location.distance(tl_location)
            
            # Manual control to drive through red light
            if current_speed < TARGET_SPEED:
                throttle = 0.7
            else:
                throttle = 0.3
            
            # Simple steering to go straight (follow road)
            vehicle_transform = violator_vehicle.get_transform()
            forward = vehicle_transform.rotation.get_forward_vector()
            waypoint = world.get_map().get_waypoint(vehicle_location, project_to_road=True)
            
            steer = 0.0
            if waypoint:
                wp_forward = waypoint.transform.rotation.get_forward_vector()
                # Calculate angle difference
                cross = forward.x * wp_forward.y - forward.y * wp_forward.x
                steer = max(-1.0, min(1.0, cross * 2.0))
            
            control = carla.VehicleControl(throttle=throttle, steer=steer, brake=0.0)
            violator_vehicle.apply_control(control)
            
            # Check if violation occurred (crossed stop line with red light)
            if distance_to_light < 5.0 and not violation_logged:
                tl_state = target_tl.get_state()
                if tl_state == carla.TrafficLightState.Red:
                    print(f"[VIOLATION] Frame {frame_number}: Vehicle crossed stop line during RED light!")
                    violation_logged = True
            
            # Wait for all 4 cameras to capture
            if frame_ready.wait(timeout=1.0):
                with images_lock:
                    # Save all 4 camera images
                    front_path = os.path.join(output_dirs['front'], f'front_frame_{frame_number:08d}.png')
                    front_left_path = os.path.join(output_dirs['front_left'], f'front_left_frame_{frame_number:08d}.png')
                    front_right_path = os.path.join(output_dirs['front_right'], f'front_right_frame_{frame_number:08d}.png')
                    rear_path = os.path.join(output_dirs['rear'], f'rear_frame_{frame_number:08d}.png')
                    
                    save_image_to_disk(images_received['front'], front_path)
                    save_image_to_disk(images_received['front_left'], front_left_path)
                    save_image_to_disk(images_received['front_right'], front_right_path)
                    save_image_to_disk(images_received['rear'], rear_path)
                
                # Log vehicle state
                tl_state_str = str(target_tl.get_state()).split('.')[-1]
                log_entries.append(f"{frame_number},{frame_number*0.05:.2f},{vehicle_location.x:.2f},{vehicle_location.y:.2f},{vehicle_location.z:.2f},{current_speed_kmh:.2f},{tl_state_str},{distance_to_light:.2f}\n")
                
                # Progress logging every 200 frames
                if frame_number % 200 == 0:
                    print(f"[PROGRESS] Frame {frame_number}/{max_frames} | Speed: {current_speed_kmh:.1f} km/h | Distance to light: {distance_to_light:.1f}m")
                
                frame_number += 1
            else:
                print(f"[WARNING] Frame {frame_number} camera timeout - skipping frame")
            
            # Clear frame synchronization
            frame_ready.clear()
            with images_lock:
                images_received['front'] = None
                images_received['front_left'] = None
                images_received['front_right'] = None
                images_received['rear'] = None
            
            # Update spectator to follow vehicle
            spectator = world.get_spectator()
            spectator_transform = carla.Transform(
                carla.Location(
                    x=vehicle_location.x - 20,
                    y=vehicle_location.y - 20,
                    z=vehicle_location.z + 15
                ),
                carla.Rotation(pitch=-30, yaw=45)
            )
            spectator.set_transform(spectator_transform)
        
        # Save log file
        print("[INFO] Saving simulation log...")
        with open(log_file_path, 'w') as log_file:
            log_file.writelines(log_entries)
        
        print(f"[SUCCESS] Simulation complete! {frame_number-1} frames captured")
        print(f"[INFO] Output saved to {args.output_dir}")
        if collision_detected:
            print("[WARNING] Collision was detected during simulation")
    
    except Exception as e:
        print(f"[ERROR] Simulation failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        print("[INFO] Cleaning up actors...")
        
        if world is not None:
            # Stop all sensors
            for actor in actors_to_cleanup:
                if actor.type_id.startswith('sensor.'):
                    actor.stop()
            
            # Destroy all actors
            for actor in actors_to_cleanup:
                if actor.is_alive:
                    actor.destroy()
        
        # Restore original settings
        if world is not None and original_settings is not None:
            world.apply_settings(original_settings)
            print("[INFO] Original world settings restored")
        
        print("[INFO] Cleanup complete")

if __name__ == '__main__':
    main()