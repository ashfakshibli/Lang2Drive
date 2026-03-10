#!/usr/bin/env python

"""
CARLA Editable Simulation with Latency Measurement
==================================================

Combines the latency measurement from previous code with 
interactive scenario editing capabilities similar to Talk2Traffic.

Features:
- Real-time scenario editing via natural language
- Comprehensive latency tracking
- Dynamic weather/lighting changes
- Vehicle behavior modifications
- Performance impact analysis of edits
"""

import glob
import os
import sys
import time
import threading
import json
from collections import deque
import numpy as np
import random


# CARLA setup
try:
    carla_egg_path = r'C:\Carla\WindowsNoEditor\PythonAPI\carla\dist\carla-*.egg'
    egg_file = glob.glob(carla_egg_path)[0]
    sys.path.append(egg_file)
    
    # Also add the carla directory itself
    carla_dir = r'C:\Carla\WindowsNoEditor\PythonAPI\carla'
    if carla_dir not in sys.path:
        sys.path.append(carla_dir)
        
    import carla
    print("✅ Successfully imported carla module!")
except (IndexError, ImportError) as e:
    print(f"CARLA import failed: {e}")
    print("Trying pip-installed carla module...")
    try:
        import carla
        print("✅ Successfully imported carla module via pip!")
    except ImportError:
        print(f"❌ Failed to import carla module: {e}")
        print("\n🔧 Troubleshooting:")
        print("1. Make sure CARLA server is running")
        print("2. Install CARLA: pip install carla")
        sys.exit(1)


class EditableLatencyTracker:
    """Enhanced latency tracker that also tracks performance impact of edits"""
    
    def __init__(self, max_samples=1000):
        self.frame_timestamps = deque(maxlen=max_samples)
        self.processing_times = deque(maxlen=max_samples)
        self.capture_to_process_times = deque(maxlen=max_samples)
        self.frame_intervals = deque(maxlen=max_samples)
        self.edit_events = deque(maxlen=max_samples)  # Track when edits happen
        self.last_frame_time = None
        self.lock = threading.Lock()
        
    def record_edit_event(self, edit_type: str, timestamp: float):
        """Record when an edit was made to analyze performance impact"""
        with self.lock:
            self.edit_events.append({
                'edit_type': edit_type,
                'timestamp': timestamp,
                'frame_count': len(self.frame_timestamps)
            })
    
    def record_frame_start(self, carla_timestamp, system_timestamp):
        """Record when frame capture started"""
        with self.lock:
            if self.last_frame_time is not None:
                interval = system_timestamp - self.last_frame_time
                self.frame_intervals.append(interval * 1000)
            
            self.last_frame_time = system_timestamp
            
            # Handle different timestamp formats (pip vs egg file)
            if hasattr(carla_timestamp, 'frame'):
                frame_id = carla_timestamp.frame
            else:
                # For pip-installed CARLA, timestamp is just a float
                frame_id = int(carla_timestamp * 1000)  # Use timestamp as frame ID
            
            return {
                'carla_timestamp': carla_timestamp,
                'system_start': system_timestamp,
                'frame_id': frame_id
            }
    
    def record_frame_end(self, frame_data, processing_start_time):
        """Record when frame processing completed"""
        processing_end_time = time.time()
        processing_duration = (processing_end_time - processing_start_time) * 1000
        
        # Handle different timestamp formats
        carla_timestamp = frame_data['carla_timestamp']
        if hasattr(carla_timestamp, 'elapsed_seconds'):
            carla_sim_time = carla_timestamp.elapsed_seconds
        else:
            # For pip-installed CARLA, timestamp is just a float (seconds)
            carla_sim_time = carla_timestamp
            
        system_time_when_captured = frame_data['system_start']
        capture_to_process_latency = (processing_start_time - system_time_when_captured) * 1000
        
        with self.lock:
            self.processing_times.append(processing_duration)
            self.capture_to_process_times.append(capture_to_process_latency)
            self.frame_timestamps.append({
                'frame_id': frame_data['frame_id'],
                'carla_time': carla_sim_time,
                'system_capture_time': system_time_when_captured,
                'processing_start': processing_start_time,
                'processing_end': processing_end_time,
                'processing_duration_ms': processing_duration,
                'capture_to_process_ms': capture_to_process_latency
            })
    
    def get_stats_around_edits(self, edit_window_seconds=5.0):
        """Get performance stats before/after edits to measure impact"""
        with self.lock:
            if not self.edit_events or not self.frame_timestamps:
                return None
            
            results = []
            for edit in self.edit_events:
                edit_time = edit['timestamp']
                
                # Find frames before/after edit
                before_frames = []
                after_frames = []
                
                for frame in self.frame_timestamps:
                    time_diff = frame['system_capture_time'] - edit_time
                    if -edit_window_seconds <= time_diff <= 0:
                        before_frames.append(frame['processing_duration_ms'])
                    elif 0 < time_diff <= edit_window_seconds:
                        after_frames.append(frame['processing_duration_ms'])
                
                if before_frames and after_frames:
                    results.append({
                        'edit_type': edit['edit_type'],
                        'before_avg_ms': np.mean(before_frames),
                        'after_avg_ms': np.mean(after_frames),
                        'performance_impact': np.mean(after_frames) - np.mean(before_frames)
                    })
            
            return results
    
    def get_stats(self):
        """Get current latency statistics"""
        with self.lock:
            if not self.processing_times:
                return None
                
            return {
                'processing_latency': {
                    'avg_ms': np.mean(list(self.processing_times)),
                    'min_ms': np.min(list(self.processing_times)),
                    'max_ms': np.max(list(self.processing_times)),
                    'std_ms': np.std(list(self.processing_times)),
                    'p95_ms': np.percentile(list(self.processing_times), 95)
                },
                'capture_to_process_latency': {
                    'avg_ms': np.mean(list(self.capture_to_process_times)),
                    'min_ms': np.min(list(self.capture_to_process_times)),
                    'max_ms': np.max(list(self.capture_to_process_times)),
                    'std_ms': np.std(list(self.capture_to_process_times)),
                    'p95_ms': np.percentile(list(self.capture_to_process_times), 95)
                },
                'frame_rate': {
                    'avg_fps': 1000 / np.mean(list(self.frame_intervals)) if self.frame_intervals else 0,
                    'avg_interval_ms': np.mean(list(self.frame_intervals)) if self.frame_intervals else 0,
                    'fps_std': np.std([1000/interval for interval in self.frame_intervals]) if self.frame_intervals else 0
                },
                'total_frames': len(self.frame_timestamps),
                'total_edits': len(self.edit_events)
            }


class EditableCameraSimulation:
    """Main simulation class combining editing and latency measurement"""
    
    def __init__(self, host='localhost', port=2000, output_dir='output_images'):
        self.client = carla.Client(host, port)
        self.client.set_timeout(10.0)
        self.world = self.client.get_world()
        self.blueprint_library = self.world.get_blueprint_library()
        
        self.output_dir = output_dir
        self.frame_count = 0
        self.latency_tracker = EditableLatencyTracker()
        
        # Editing state
        self.vehicles = []
        self.camera = None
        self.editing_enabled = True
        self.command_queue = []
        
        # Camera and recording state
        self.camera_mode = "intersection"  # "intersection" or "follow_car"
        self.followed_vehicle = None
        self.recording_enabled = False  # Start with recording disabled
        
        # Real-time metrics logging
        self.metrics_log_file = os.path.join(self.output_dir, 'real_time_metrics.txt')
        self.metrics_file_handle = None
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Output directory: {os.path.abspath(self.output_dir)}")
        print(f"✅ Connected to CARLA server at {host}:{port}")
        
        # Initialize metrics log file
        self._init_metrics_log()
    
    def _init_metrics_log(self):
        """Initialize real-time metrics logging"""
        try:
            self.metrics_file_handle = open(self.metrics_log_file, 'w')
            self.metrics_file_handle.write("CARLA SIMULATION REAL-TIME METRICS LOG\n")
            self.metrics_file_handle.write("="*50 + "\n")
            self.metrics_file_handle.write(f"Started at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            self.metrics_file_handle.write("Format: Timestamp | Frame | Processing(ms) | FPS | Latency(ms) | Total Edits\n")
            self.metrics_file_handle.write("-"*80 + "\n")
            self.metrics_file_handle.flush()
            print(f"📊 Real-time metrics logging to: {self.metrics_log_file}")
        except Exception as e:
            print(f"⚠️  Could not initialize metrics log: {e}")
            self.metrics_file_handle = None
    
    def _log_real_time_metrics(self, frame_count, stats):
        """Log real-time metrics to file"""
        if self.metrics_file_handle and stats:
            try:
                timestamp = time.strftime('%H:%M:%S')
                processing_ms = stats['processing_latency']['avg_ms']
                fps = stats['frame_rate']['avg_fps']
                latency_ms = stats['capture_to_process_latency']['avg_ms']
                total_edits = stats['total_edits']
                
                log_line = f"{timestamp} | {frame_count:>5} | {processing_ms:>10.2f} | {fps:>6.1f} | {latency_ms:>9.2f} | {total_edits:>10}\n"
                self.metrics_file_handle.write(log_line)
                self.metrics_file_handle.flush()
            except Exception as e:
                print(f"⚠️  Error writing metrics: {e}")
    
    def spawn_scenario(self):
        """Spawn initial scenario with vehicles and camera"""
        try:
            # Get spawn points and find a good intersection location
            spawn_points = self.world.get_map().get_spawn_points()
            
            # Setup traffic manager with basic settings
            tm = self.client.get_trafficmanager()
            tm.set_global_distance_to_leading_vehicle(1.5)  # Closer following
            tm.set_synchronous_mode(True)
            tm.set_random_device_seed(42)  # Consistent behavior
            
            # Spawn multiple vehicles around the intersection area
            vehicle_blueprints = self.blueprint_library.filter('vehicle.*')
            vehicles_to_spawn = min(12, len(spawn_points))  # More vehicles for better selection
            
            # Filter for reliable car types that move well
            preferred_vehicles = [
                'vehicle.tesla.model3',
                'vehicle.audi.tt',
                'vehicle.bmw.grandtourer', 
                'vehicle.mercedes.coupe',
                'vehicle.nissan.patrol',
                'vehicle.ford.mustang',
                'vehicle.chevrolet.impala',
                'vehicle.toyota.prius',
                'vehicle.lincoln.mkz_2020'
            ]
            
            # Try to use preferred vehicles first
            car_blueprints = []
            for preferred in preferred_vehicles:
                for bp in vehicle_blueprints:
                    if preferred in bp.id:
                        car_blueprints.append(bp)
            
            # Fall back to any car if preferred not available
            if not car_blueprints:
                car_blueprints = [bp for bp in vehicle_blueprints if 'bike' not in bp.id.lower() and 'motorcycle' not in bp.id.lower()]
            
            if not car_blueprints:
                car_blueprints = vehicle_blueprints
            
            successful_spawns = 0
            for i in range(vehicles_to_spawn):
                if successful_spawns >= 8:  # Limit to 8 vehicles for performance
                    break
                    
                try:
                    # Use preferred car blueprints
                    vehicle_bp = random.choice(car_blueprints)
                    
                    # Try multiple spawn points if one fails
                    for attempt in range(3):
                        try:
                            spawn_point = spawn_points[(i + attempt) % len(spawn_points)]
                            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                            break
                        except:
                            continue
                    else:
                        continue  # Skip if all spawn attempts failed
                    
                    # Configure vehicle for continuous movement
                    vehicle.set_autopilot(True, tm.get_port())
                    
                    # Set traffic manager parameters for this vehicle
                    tm.vehicle_percentage_speed_difference(vehicle, -30.0)  # 30% faster than speed limit
                    tm.distance_to_leading_vehicle(vehicle, 1.0)  # Close following
                    tm.ignore_vehicles_percentage(vehicle, 70.0)  # Aggressive overtaking
                    tm.ignore_lights_percentage(vehicle, 100.0)  # Ignore ALL traffic lights
                    tm.ignore_signs_percentage(vehicle, 100.0)   # Ignore ALL stop signs
                    tm.auto_lane_change(vehicle, True)  # Enable lane changes
                    
                    # Give initial momentum
                    forward = spawn_point.get_forward_vector()
                    impulse = carla.Vector3D(forward.x * 15.0, forward.y * 15.0, 0.0)
                    vehicle.add_impulse(impulse)
                    
                    self.vehicles.append(vehicle)
                    successful_spawns += 1
                    
                    print(f"🚗 Spawned vehicle {successful_spawns}: {vehicle_bp.id}")
                    
                except Exception as e:
                    print(f"⚠️  Could not spawn vehicle {i+1}: {e}")
                    continue
            
            if not self.vehicles:
                raise Exception("Failed to spawn any vehicles")
            
            # Wait for traffic manager to initialize
            print("⏳ Initializing traffic management...")
            time.sleep(1.0)
            
            # Tick the world to get physics and AI going
            for _ in range(20):
                self.world.tick()
                time.sleep(0.05)
            
            # Setup initial camera (intersection view by default)
            self._setup_camera_view()
            
            # Print vehicle status
            print(f"✅ Scenario spawned:")
            print(f"   🚗 {len(self.vehicles)} vehicles with enhanced autopilot")
            print(f"   🚦 Traffic lights: IGNORED for continuous movement")
            print(f"   📹 Camera mode: {self.camera_mode}")
            if self.camera_mode == "follow_car" and self.followed_vehicle:
                print(f"   🎯 Following vehicle: {self.followed_vehicle.type_id}")
            
            # Check initial vehicle speeds
            print("🏃 Initial vehicle speeds:")
            moving_vehicles = 0
            for i, vehicle in enumerate(self.vehicles):
                if vehicle.is_alive:
                    velocity = vehicle.get_velocity()
                    speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                    if speed > 0.5:
                        moving_vehicles += 1
                    print(f"   Vehicle {i+1}: {speed:.2f} m/s")
            
            print(f"✅ {moving_vehicles}/{len(self.vehicles)} vehicles moving initially")
            
        except Exception as e:
            print(f"❌ Failed to spawn scenario: {e}")
            sys.exit(1)
    
    def _setup_camera_view(self):
        """Setup camera based on current mode"""
        if self.camera and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()
        
        camera_bp = self.blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', '800')
        camera_bp.set_attribute('image_size_y', '600')
        camera_bp.set_attribute('fov', '110')
        
        if self.camera_mode == "intersection":
            # Static intersection camera (elevated view)
            spawn_points = self.world.get_map().get_spawn_points()
            intersection_point = spawn_points[len(spawn_points)//2]
            
            camera_transform = carla.Transform(
                carla.Location(
                    x=intersection_point.location.x,
                    y=intersection_point.location.y, 
                    z=intersection_point.location.z + 15.0
                ),
                carla.Rotation(pitch=-30.0, yaw=intersection_point.rotation.yaw)
            )
            
            self.camera = self.world.spawn_actor(camera_bp, camera_transform)
            print(f"📹 Intersection camera positioned at: {camera_transform.location}")
            
        elif self.camera_mode == "follow_car" and self.vehicles:
            # Select and monitor the best vehicle to follow
            self._select_and_monitor_follow_vehicle()
            
            if self.followed_vehicle and self.followed_vehicle.is_alive:
                # Verify vehicle is not stuck at spawn
                location = self.followed_vehicle.get_location()
                velocity = self.followed_vehicle.get_velocity()
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                
                print(f"🚗 Selected vehicle: {self.followed_vehicle.type_id}")
                print(f"📍 Location: {location}")
                print(f"🏃 Speed: {speed:.2f} m/s")
                
                # Ensure vehicle has proper autopilot settings
                tm = self.client.get_trafficmanager()
                self.followed_vehicle.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(self.followed_vehicle, -40.0)  # 40% faster
                tm.distance_to_leading_vehicle(self.followed_vehicle, 0.5)  # Very close following
                tm.ignore_vehicles_percentage(self.followed_vehicle, 90.0)  # Very aggressive
                tm.ignore_lights_percentage(self.followed_vehicle, 100.0)  # Ignore ALL traffic lights
                tm.ignore_signs_percentage(self.followed_vehicle, 100.0)   # Ignore ALL stop signs
                tm.auto_lane_change(self.followed_vehicle, True)  # Enable lane changes
                
                # Give it a boost if it's slow
                if speed < 2.0:
                    waypoint = self.world.get_map().get_waypoint(location)
                    if waypoint:
                        forward = waypoint.transform.get_forward_vector()
                        impulse = carla.Vector3D(forward.x * 20.0, forward.y * 20.0, 0.0)
                        self.followed_vehicle.add_impulse(impulse)
                        print("🚀 Boosted followed vehicle")
                
                # Attach camera with optimized settings for better following
                camera_transform = carla.Transform(
                    carla.Location(x=-6.0, z=2.5),   # Closer and lower for better view
                    carla.Rotation(pitch=-5.0)        # Less steep angle
                )
                
                self.camera = self.world.spawn_actor(camera_bp, camera_transform, attach_to=self.followed_vehicle)
                print(f"📹 Camera attached to vehicle: {self.followed_vehicle.type_id}")
                
            else:
                print("❌ No suitable vehicle found to follow")
        
        if self.camera:
            self.camera.listen(self.process_image)
    
    def _select_and_monitor_follow_vehicle(self):
        """Select the best vehicle to follow and set up monitoring"""
        best_vehicle = None
        best_score = -1
        
        print("🔍 Selecting best vehicle to follow...")
        
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_alive:
                location = vehicle.get_location()
                velocity = vehicle.get_velocity()
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                
                # Score based on speed and position (prefer moving vehicles)
                score = speed * 10  # Base score from speed
                
                # Bonus for being away from spawn area (likely to keep moving)
                spawn_points = self.world.get_map().get_spawn_points()
                min_spawn_distance = min([location.distance(sp.location) for sp in spawn_points])
                if min_spawn_distance > 20:  # 20 meters from spawn
                    score += 50
                
                print(f"  Vehicle {i+1}: speed={speed:.1f} m/s, score={score:.1f}")
                
                if score > best_score:
                    best_score = score
                    best_vehicle = vehicle
        
        # If no good vehicle found, force movement on the first vehicle
        if best_vehicle is None and self.vehicles:
            best_vehicle = self.vehicles[0]
            print("🔧 No moving vehicle found, forcing movement on first vehicle")
            
            # Force movement
            location = best_vehicle.get_location()
            waypoint = self.world.get_map().get_waypoint(location)
            if waypoint:
                forward = waypoint.transform.get_forward_vector()
                impulse = carla.Vector3D(forward.x * 25.0, forward.y * 25.0, 0.0)
                best_vehicle.add_impulse(impulse)
        
        self.followed_vehicle = best_vehicle
        if best_vehicle:
            print(f"✅ Selected vehicle: {best_vehicle.type_id} (score: {best_score:.1f})")
    
    def _print_current_weather(self):
        """Print current weather status for debugging"""
        weather = self.world.get_weather()
        print("🌤️  Current Weather Status:")
        print(f"   ☔ Precipitation: {weather.precipitation:.1f}%")
        print(f"   ☁️  Cloudiness: {weather.cloudiness:.1f}%")
        print(f"   💧 Wetness: {weather.wetness:.1f}%")
        print(f"   🌫️  Fog: {weather.fog_density:.1f}%")
        print(f"   💨 Wind: {weather.wind_intensity:.1f}%")
        print(f"   ☀️  Sun altitude: {weather.sun_altitude_angle:.1f}°")
        print(f"   🧭 Sun azimuth: {weather.sun_azimuth_angle:.1f}°")
        
        # Determine conditions
        if weather.precipitation > 50:
            print("   Weather: HEAVY RAIN 🌧️")
        elif weather.precipitation > 0:
            print("   Weather: LIGHT RAIN 🌦️")
        else:
            print("   Weather: CLEAR ☀️")
            
        if weather.sun_altitude_angle > 30:
            print("   Time: BRIGHT DAY ☀️")
        elif weather.sun_altitude_angle > 0:
            print("   Time: DAWN/DUSK 🌅")
        else:
            print("   Time: NIGHT 🌙")
    
    def _ensure_vehicle_movement(self):
        """Ensure vehicles are actually moving by checking and fixing autopilot"""
        print("🔧 Ensuring vehicle movement...")
        
        # Get traffic manager and set it up properly
        tm = self.client.get_trafficmanager()
        tm.set_global_distance_to_leading_vehicle(2.0)
        tm.set_synchronous_mode(True)
        
        for i, vehicle in enumerate(self.vehicles):
            if vehicle.is_alive:
                # Re-enable autopilot
                vehicle.set_autopilot(True)
                
                # Check current speed
                velocity = vehicle.get_velocity()
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                location = vehicle.get_location()
                
                print(f"  🚗 Vehicle {i+1} ({vehicle.type_id}): {speed:.1f} m/s at {location}")
                
                # If vehicle is stuck, give it a small nudge
                if speed < 0.1:
                    # Get a nearby waypoint and set as destination
                    waypoint = self.world.get_map().get_waypoint(location)
                    if waypoint:
                        next_waypoints = waypoint.next(10.0)  # 10 meters ahead
                        if next_waypoints:
                            target_location = next_waypoints[0].transform.location
                            # Give vehicle a small impulse towards the target
                            impulse_direction = carla.Vector3D(
                                target_location.x - location.x,
                                target_location.y - location.y,
                                0.0
                            )
                            # Normalize and scale
                            length = (impulse_direction.x**2 + impulse_direction.y**2)**0.5
                            if length > 0:
                                impulse_direction.x = (impulse_direction.x / length) * 5.0
                                impulse_direction.y = (impulse_direction.y / length) * 5.0
                                vehicle.add_impulse(impulse_direction)
                                print(f"  🚀 Gave impulse to vehicle {i+1}")
    
    def _select_best_vehicle_to_follow(self):
        """Select the best vehicle to follow (preferably one that's moving)"""
        best_vehicle = None
        best_speed = 0.0
        
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                velocity = vehicle.get_velocity()
                speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                
                # Prefer vehicles that are already moving
                if speed > best_speed:
                    best_speed = speed
                    best_vehicle = vehicle
        
        # If no vehicle is moving, just pick the first one
        if best_vehicle is None and self.vehicles:
            best_vehicle = self.vehicles[0]
            
        return best_vehicle
    
    def process_image(self, image):
        """Process camera image with latency measurement - optimized for better FPS"""
        # Only process and save if recording is enabled
        if not self.recording_enabled:
            return
            
        system_timestamp = time.time()
        frame_data = self.latency_tracker.record_frame_start(image.timestamp, system_timestamp)
        processing_start_time = time.time()
        
        try:
            # Optimized image processing for better FPS
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            array = array[:, :, ::-1]  # BGR to RGB
            
            filename = f"frame_{self.frame_count:06d}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Use optimized saving
            try:
                from PIL import Image
                img = Image.fromarray(array)
                img.save(filepath, optimize=True, compress_level=1)  # Fast compression
            except ImportError:
                np.save(filepath.replace('.png', '.npy'), array)
            
            self.latency_tracker.record_frame_end(frame_data, processing_start_time)
            self.frame_count += 1
            
            # Print stats less frequently for better performance (every 60 frames)
            if self.frame_count % 60 == 0:
                stats = self.latency_tracker.get_stats()
                if stats:
                    # Check if followed vehicle is still alive and moving
                    if (self.camera_mode == "follow_car" and 
                        self.followed_vehicle and 
                        self.followed_vehicle.is_alive):
                        
                        velocity = self.followed_vehicle.get_velocity()
                        speed = (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
                        status = f"Following: {speed:.1f} m/s"
                    else:
                        status = "Static view"
                    
                    # Print to console
                    print(f"📸 Frame {self.frame_count} | "
                          f"FPS: {stats['frame_rate']['avg_fps']:.1f} | "
                          f"Proc: {stats['processing_latency']['avg_ms']:.0f}ms | "
                          f"{status}")
                    
                    # Log to file
                    self._log_real_time_metrics(self.frame_count, stats)
                    
        except Exception as e:
            print(f"❌ Error processing image: {e}")
    
    def process_edit_command(self, command: str):
        """Process editing command and track performance impact"""
        edit_time = time.time()
        command = command.lower().strip()
        
        print(f"🎯 Processing: '{command}'")
        
        try:
            # Recording control commands
            if 'start recording' in command or 'start capture' in command:
                self.recording_enabled = True
                print("🔴 Recording started - frames will be saved")
                return True
                
            elif 'stop recording' in command or 'stop capture' in command:
                self.recording_enabled = False
                print("⏹️  Recording stopped - frames will not be saved")
                return True
            
            # Camera view commands
            elif 'intersection view' in command or 'switch to intersection' in command:
                self.camera_mode = "intersection"
                self._setup_camera_view()
                self.latency_tracker.record_edit_event('camera_intersection', edit_time)
                print("📹 Switched to intersection view")
                return True
                
            elif 'follow car' in command or 'third person' in command or 'car view' in command:
                self.camera_mode = "follow_car"
                self._setup_camera_view()
                self.latency_tracker.record_edit_event('camera_follow_car', edit_time)
                print("📹 Switched to third-person car following view")
                return True
                
            elif 'switch car' in command and self.camera_mode == "follow_car":
                # Switch to next vehicle
                if self.vehicles:
                    current_index = 0
                    if self.followed_vehicle in self.vehicles:
                        current_index = self.vehicles.index(self.followed_vehicle)
                    next_index = (current_index + 1) % len(self.vehicles)
                    self.followed_vehicle = self.vehicles[next_index]
                    self._setup_camera_view()
                    self.latency_tracker.record_edit_event('camera_switch_car', edit_time)
                    print(f"🔄 Now following: {self.followed_vehicle.type_id}")
                return True
            
            # Weather commands with enhanced effects
            elif 'rain' in command and 'stop' not in command:
                # Heavy rain with maximum visibility
                self._modify_weather({
                    'precipitation': 90.0,
                    'cloudiness': 90.0
                })
                self.latency_tracker.record_edit_event('start_rain', edit_time)
                
            elif 'stop' in command and 'rain' in command:
                # Clear weather
                self._modify_weather({
                    'precipitation': 0.0,
                    'cloudiness': 10.0
                })
                self.latency_tracker.record_edit_event('stop_rain', edit_time)
                
            elif 'night' in command:
                # Strong nighttime effect
                self._modify_lighting({
                    'sun_altitude_angle': -60.0  # Deep night
                })
                self.latency_tracker.record_edit_event('night_time', edit_time)
                
            elif 'day' in command or 'morning' in command:
                # Bright daytime effect
                self._modify_lighting({
                    'sun_altitude_angle': 60.0   # High sun
                })
                self.latency_tracker.record_edit_event('day_time', edit_time)
                
            elif 'weather status' in command or 'check weather' in command:
                # Check current weather without changing it
                self._print_current_weather()
                return True
                
            elif 'heavy rain' in command:
                # Extreme rain effect
                self._modify_weather({
                    'precipitation': 100.0,
                    'cloudiness': 100.0
                })
                self.latency_tracker.record_edit_event('heavy_rain', edit_time)
                
            elif 'clear weather' in command:
                # Perfect clear day
                self._modify_weather({
                    'precipitation': 0.0,
                    'cloudiness': 0.0
                })
                self._modify_lighting({
                    'sun_altitude_angle': 45.0
                })
                self.latency_tracker.record_edit_event('clear_weather', edit_time)
                
            elif 'faster' in command or 'speed up' in command:
                self._modify_vehicle_speeds(1.5)
                self.latency_tracker.record_edit_event('speed_increase', edit_time)
                
            elif 'slower' in command or 'slow down' in command:
                self._modify_vehicle_speeds(0.7)
                self.latency_tracker.record_edit_event('speed_decrease', edit_time)
                
            elif 'add' in command and ('car' in command or 'vehicle' in command):
                self._add_vehicle()
                self.latency_tracker.record_edit_event('add_vehicle', edit_time)
                
            elif 'remove' in command and ('car' in command or 'vehicle' in command):
                self._remove_vehicle()
                self.latency_tracker.record_edit_event('remove_vehicle', edit_time)
                
            else:
                print("❓ Command not recognized")
                return False
            
            print("✅ Edit applied successfully")
            return True
            
        except Exception as e:
            print(f"❌ Edit failed: {e}")
            return False
    
    def _modify_weather(self, params):
        """Modify weather settings with enhanced visibility"""
        weather = self.world.get_weather()
        
        # Apply the requested changes
        for key, value in params.items():
            if hasattr(weather, key):
                setattr(weather, key, value)
        
        # Ensure other weather parameters are set for better visibility
        if 'precipitation' in params:
            if params['precipitation'] > 0:
                # Rain setup
                weather.precipitation_deposits = min(100.0, params['precipitation'])
                weather.wind_intensity = 30.0
                weather.wetness = min(100.0, params['precipitation'])
                weather.fog_density = 20.0
                print(f"🌧️  Rain applied: {params['precipitation']}% precipitation")
            else:
                # Clear weather
                weather.precipitation_deposits = 0.0
                weather.wind_intensity = 5.0
                weather.wetness = 0.0
                weather.fog_density = 0.0
                print(f"☀️  Clear weather applied")
        
        # Apply weather to world
        self.world.set_weather(weather)
        
        # Force weather update by ticking the world
        for _ in range(5):
            self.world.tick()
        
        # Print current weather state for verification
        current_weather = self.world.get_weather()
        print(f"🌦️  Weather updated:")
        print(f"   Precipitation: {current_weather.precipitation:.1f}%")
        print(f"   Cloudiness: {current_weather.cloudiness:.1f}%")
        print(f"   Wetness: {current_weather.wetness:.1f}%")
        print(f"   Wind: {current_weather.wind_intensity:.1f}%")
    
    def _modify_lighting(self, params):
        """Modify lighting/time settings with enhanced visibility"""
        weather = self.world.get_weather()
        
        # Apply the requested changes
        for key, value in params.items():
            if hasattr(weather, key):
                setattr(weather, key, value)
        
        # Enhance lighting changes for better visibility
        if 'sun_altitude_angle' in params:
            angle = params['sun_altitude_angle']
            if angle > 0:
                # Daytime settings
                weather.sun_azimuth_angle = 180.0  # Sun position
                weather.cloudiness = 20.0  # Less cloudy for bright day
                weather.fog_density = 0.0   # Clear fog
                print(f"☀️  Daytime applied: {angle}° sun altitude")
            else:
                # Nighttime settings
                weather.sun_azimuth_angle = 180.0
                weather.cloudiness = 80.0   # More cloudy for darker night
                weather.fog_density = 10.0  # Some fog for atmosphere
                print(f"🌙 Nighttime applied: {angle}° sun altitude")
        
        # Apply weather to world
        self.world.set_weather(weather)
        
        # Force lighting update by ticking the world multiple times
        for _ in range(10):
            self.world.tick()
        
        # Print current lighting state for verification
        current_weather = self.world.get_weather()
        print(f"🌅 Lighting updated:")
        print(f"   Sun altitude: {current_weather.sun_altitude_angle:.1f}°")
        print(f"   Sun azimuth: {current_weather.sun_azimuth_angle:.1f}°")
        print(f"   Cloudiness: {current_weather.cloudiness:.1f}%")
        
        # Determine time of day
        if current_weather.sun_altitude_angle > 0:
            print(f"   Time: DAYTIME ☀️")
        else:
            print(f"   Time: NIGHTTIME 🌙")
    
    def _modify_vehicle_speeds(self, multiplier):
        """Modify vehicle speeds"""
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                velocity = vehicle.get_velocity()
                new_velocity = carla.Vector3D(
                    velocity.x * multiplier,
                    velocity.y * multiplier,
                    velocity.z * multiplier
                )
                vehicle.set_target_velocity(new_velocity)
        print(f"🚗 Speed multiplier: {multiplier}")
    
    def _add_vehicle(self):
        """Add new vehicle"""
        try:
            vehicle_bp = random.choice(self.blueprint_library.filter('vehicle.*'))
            spawn_points = self.world.get_map().get_spawn_points()
            spawn_point = random.choice(spawn_points)
            
            vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            vehicle.set_autopilot(True)
            self.vehicles.append(vehicle)
            print(f"➕ Added vehicle (total: {len(self.vehicles)})")
        except Exception as e:
            print(f"❌ Failed to add vehicle: {e}")
    
    def _remove_vehicle(self):
        """Remove vehicle"""
        if len(self.vehicles) > 2:  # Keep at least 2 vehicles for intersection activity
            vehicle = self.vehicles.pop()
            if vehicle.is_alive:
                vehicle.destroy()
            print(f"➖ Removed vehicle (total: {len(self.vehicles)})")
        else:
            print("⚠️  Minimum 2 vehicles required for intersection simulation")
    
    def start_interactive_simulation(self, duration=300):
        """Start interactive simulation with editing capabilities"""
        print("\n" + "="*60)
        print("🎬 EDITABLE CARLA SIMULATION WITH LATENCY TRACKING")
        print("="*60)
        print("📹 Camera Views: Intersection view & Third-person car following")
        print("🚗 Vehicles: Multiple vehicles with autopilot")
        print("📊 Recording: User-controlled frame capture")
        print("📊 Metrics: Real-time logging to txt files")
        print("="*60)
        print("🎮 CAMERA COMMANDS:")
        print("  • 'intersection view' - Static overhead camera")
        print("  • 'follow car' - Third-person car following")
        print("  • 'switch car' - Switch to next vehicle (in follow mode)")
        print("📹 RECORDING COMMANDS:")
        print("  • 'start recording' - Begin saving frames")
        print("  • 'stop recording' - Stop saving frames")
        print("🌦️  SCENE EDITING COMMANDS:")
        print("  • 'make it rain' / 'stop the rain'")
        print("  • 'change to night' / 'change to day'")
        print("  • 'heavy rain' / 'clear weather'")
        print("  • 'weather status' - Check current weather")
        print("  • 'make vehicles faster' / 'make vehicles slower'")
        print("  • 'add a car' / 'remove a car'")
        print("⚠️  OTHER:")
        print("  • 'quit' to exit")
        print("="*60)
        
        # Spawn scenario
        self.spawn_scenario()
        
        print(f"\n🎯 Current status:")
        print(f"   📹 Camera: {self.camera_mode}")
        print(f"   🔴 Recording: {'ON' if self.recording_enabled else 'OFF'}")
        if self.camera_mode == "follow_car" and self.followed_vehicle:
            print(f"   🚗 Following: {self.followed_vehicle.type_id}")
        print(f"\n💡 Type 'start recording' to begin capturing frames!")
        
        # Start command input thread
        command_thread = threading.Thread(target=self._command_input_loop, daemon=True)
        command_thread.start()
        
        start_time = time.time()
        last_movement_check = time.time()
        
        try:
            while time.time() - start_time < duration and self.editing_enabled:
                self.world.tick()
                
                # Process queued commands
                while self.command_queue:
                    command = self.command_queue.pop(0)
                    if command.lower() == 'quit':
                        self.editing_enabled = False
                        break
                    self.process_edit_command(command)
                
                # Periodically check and ensure vehicle movement (every 10 seconds)
                current_time = time.time()
                if current_time - last_movement_check > 10.0:
                    self._periodic_movement_check()
                    last_movement_check = current_time
                
                time.sleep(0.05)
                
        except KeyboardInterrupt:
            print("\n⏹️  Simulation interrupted")
        
        # Print final reports
        self._print_performance_report()
        self._save_detailed_logs()
        self._save_metrics_summary()
        
        self.cleanup()
    
    def _periodic_movement_check(self):
        """Periodically check if vehicles are moving and fix if needed"""
        # Check if followed vehicle is alive and moving
        if (self.camera_mode == "follow_car" and 
            self.followed_vehicle and 
            (not self.followed_vehicle.is_alive or 
             self._get_vehicle_speed(self.followed_vehicle) < 0.5)):
            
            print("🔄 Followed vehicle stopped/disappeared, switching to new vehicle...")
            self._select_and_monitor_follow_vehicle()
            if self.followed_vehicle:
                self._setup_camera_view()
        
        # Check all vehicles and fix stuck ones
        stuck_vehicles = []
        disappeared_vehicles = []
        
        for i, vehicle in enumerate(self.vehicles):
            if not vehicle.is_alive:
                disappeared_vehicles.append((i, vehicle))
                continue
                
            speed = self._get_vehicle_speed(vehicle)
            if speed < 0.3:  # Very low threshold for stuck detection
                stuck_vehicles.append((i, vehicle))
        
        # Handle disappeared vehicles
        if disappeared_vehicles:
            print(f"👻 {len(disappeared_vehicles)} vehicles disappeared, respawning...")
            self._respawn_disappeared_vehicles(disappeared_vehicles)
        
        # Handle stuck vehicles
        if stuck_vehicles:
            print(f"🔧 Found {len(stuck_vehicles)} stuck vehicles, fixing...")
            for i, vehicle in stuck_vehicles:
                self._unstuck_vehicle(vehicle, i)
    
    def _get_vehicle_speed(self, vehicle):
        """Get vehicle speed safely"""
        try:
            if vehicle.is_alive:
                velocity = vehicle.get_velocity()
                return (velocity.x**2 + velocity.y**2 + velocity.z**2)**0.5
        except:
            pass
        return 0.0
    
    def _respawn_disappeared_vehicles(self, disappeared_vehicles):
        """Respawn vehicles that have disappeared"""
        spawn_points = self.world.get_map().get_spawn_points()
        vehicle_blueprints = self.blueprint_library.filter('vehicle.*')
        
        # Remove disappeared vehicles from list
        for i, old_vehicle in disappeared_vehicles:
            if old_vehicle in self.vehicles:
                self.vehicles.remove(old_vehicle)
        
        # Spawn new vehicles
        for i, _ in disappeared_vehicles:
            try:
                vehicle_bp = random.choice(vehicle_blueprints)
                spawn_point = random.choice(spawn_points)
                
                new_vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
                
                # Configure for movement
                tm = self.client.get_trafficmanager()
                new_vehicle.set_autopilot(True, tm.get_port())
                tm.vehicle_percentage_speed_difference(new_vehicle, -30.0)
                tm.ignore_vehicles_percentage(new_vehicle, 70.0)
                tm.ignore_lights_percentage(new_vehicle, 100.0)  # Ignore traffic lights
                tm.ignore_signs_percentage(new_vehicle, 100.0)   # Ignore stop signs
                tm.auto_lane_change(new_vehicle, True)  # Enable lane changes
                
                # Initial boost
                forward = spawn_point.get_forward_vector()
                impulse = carla.Vector3D(forward.x * 15.0, forward.y * 15.0, 0.0)
                new_vehicle.add_impulse(impulse)
                
                self.vehicles.append(new_vehicle)
                print(f"  ✨ Respawned vehicle: {vehicle_bp.id}")
                
            except Exception as e:
                print(f"  ❌ Failed to respawn vehicle: {e}")
    
    def _unstuck_vehicle(self, vehicle, index):
        """Unstuck a specific vehicle with enhanced methods"""
        try:
            location = vehicle.get_location()
            
            # Method 1: Traffic manager boost
            tm = self.client.get_trafficmanager()
            tm.vehicle_percentage_speed_difference(vehicle, -60.0)  # Much faster
            tm.ignore_vehicles_percentage(vehicle, 95.0)  # Ignore almost everything
            tm.ignore_lights_percentage(vehicle, 100.0)  # Ignore traffic lights
            tm.ignore_signs_percentage(vehicle, 100.0)   # Ignore stop signs
            tm.auto_lane_change(vehicle, True)  # Enable lane changes
            
            # Method 2: Waypoint-based impulse
            waypoint = self.world.get_map().get_waypoint(location)
            if waypoint:
                # Try to find a clear path
                next_waypoints = waypoint.next(10.0)
                if next_waypoints:
                    target = next_waypoints[0].transform.location
                    direction = carla.Vector3D(
                        target.x - location.x,
                        target.y - location.y,
                        0.0
                    )
                    # Strong impulse
                    length = (direction.x**2 + direction.y**2)**0.5
                    if length > 0:
                        direction.x = (direction.x / length) * 25.0
                        direction.y = (direction.y / length) * 25.0
                        vehicle.add_impulse(direction)
            
            # Method 3: Teleport slightly forward if still stuck
            if self._get_vehicle_speed(vehicle) < 0.1:
                if waypoint and waypoint.next(5.0):
                    new_transform = waypoint.next(5.0)[0].transform
                    vehicle.set_transform(new_transform)
                    print(f"  📍 Teleported vehicle {index+1} forward")
            
            print(f"  🚀 Applied fixes to vehicle {index+1}")
                
        except Exception as e:
            print(f"  ❌ Failed to unstuck vehicle {index+1}: {e}")
    
    def _command_input_loop(self):
        """Handle command input"""
        while self.editing_enabled:
            try:
                # Show current status in prompt
                status_indicator = "🔴" if self.recording_enabled else "⚫"
                camera_mode = "📹INT" if self.camera_mode == "intersection" else "📹CAR"
                command = input(f"\n{status_indicator}{camera_mode} Command: ").strip()
                if command:
                    self.command_queue.append(command)
            except (EOFError, KeyboardInterrupt):
                break
    
    def _print_performance_report(self):
        """Print comprehensive performance report"""
        stats = self.latency_tracker.get_stats()
        edit_impact = self.latency_tracker.get_stats_around_edits()
        
        if not stats:
            return
        
        print("\n" + "="*60)
        print("📊 PERFORMANCE REPORT")
        print("="*60)
        
        print(f"\n📈 OVERALL PERFORMANCE:")
        print(f"   Total Frames: {stats['total_frames']}")
        print(f"   Total Edits: {stats['total_edits']}")
        print(f"   Average FPS: {stats['frame_rate']['avg_fps']:.2f}")
        print(f"   Processing Latency: {stats['processing_latency']['avg_ms']:.2f}ms avg")
        
        if edit_impact:
            print(f"\n🎯 EDIT PERFORMANCE IMPACT:")
            for impact in edit_impact:
                change = impact['performance_impact']
                symbol = "📈" if change > 0 else "📉" if change < 0 else "➡️"
                print(f"   {symbol} {impact['edit_type']}: {change:+.2f}ms impact")
        
        print("="*60)
    
    def _save_detailed_logs(self):
        """Save detailed logs in human-readable txt format"""
        # Save latency performance report
        latency_log_path = os.path.join(self.output_dir, 'latency_performance_report.txt')
        with open(latency_log_path, 'w') as f:
            f.write("="*60 + "\n")
            f.write("CARLA SIMULATION LATENCY PERFORMANCE REPORT\n")
            f.write("="*60 + "\n\n")
            
            stats = self.latency_tracker.get_stats()
            if stats:
                f.write(f"SIMULATION SUMMARY:\n")
                f.write(f"  • Total Frames Captured: {stats['total_frames']}\n")
                f.write(f"  • Total Scene Edits Made: {stats['total_edits']}\n")
                f.write(f"  • Average Frame Rate: {stats['frame_rate']['avg_fps']:.2f} FPS\n")
                f.write(f"  • Average Frame Interval: {stats['frame_rate']['avg_interval_ms']:.2f} ms\n\n")
                
                f.write(f"PROCESSING LATENCY METRICS:\n")
                f.write(f"  • Average Processing Time: {stats['processing_latency']['avg_ms']:.2f} ms\n")
                f.write(f"  • Minimum Processing Time: {stats['processing_latency']['min_ms']:.2f} ms\n")
                f.write(f"  • Maximum Processing Time: {stats['processing_latency']['max_ms']:.2f} ms\n")
                f.write(f"  • Standard Deviation: {stats['processing_latency']['std_ms']:.2f} ms\n")
                f.write(f"  • 95th Percentile: {stats['processing_latency']['p95_ms']:.2f} ms\n\n")
                
                f.write(f"CAPTURE-TO-PROCESS LATENCY:\n")
                f.write(f"  • Average Latency: {stats['capture_to_process_latency']['avg_ms']:.2f} ms\n")
                f.write(f"  • Minimum Latency: {stats['capture_to_process_latency']['min_ms']:.2f} ms\n")
                f.write(f"  • Maximum Latency: {stats['capture_to_process_latency']['max_ms']:.2f} ms\n")
                f.write(f"  • Standard Deviation: {stats['capture_to_process_latency']['std_ms']:.2f} ms\n")
                f.write(f"  • 95th Percentile: {stats['capture_to_process_latency']['p95_ms']:.2f} ms\n\n")
            
            # Edit impact analysis
            edit_impact = self.latency_tracker.get_stats_around_edits()
            if edit_impact:
                f.write(f"EDIT PERFORMANCE IMPACT ANALYSIS:\n")
                f.write(f"(Shows how each edit affected processing performance)\n\n")
                for i, impact in enumerate(edit_impact, 1):
                    change = impact['performance_impact']
                    impact_type = "INCREASE" if change > 0 else "DECREASE" if change < 0 else "NO CHANGE"
                    f.write(f"  Edit #{i}: {impact['edit_type'].upper()}\n")
                    f.write(f"    • Before Edit: {impact['before_avg_ms']:.2f} ms average\n")
                    f.write(f"    • After Edit: {impact['after_avg_ms']:.2f} ms average\n")
                    f.write(f"    • Performance Impact: {change:+.2f} ms ({impact_type})\n\n")
            else:
                f.write(f"EDIT PERFORMANCE IMPACT ANALYSIS:\n")
                f.write(f"  No edits made during simulation.\n\n")
            
            f.write("="*60 + "\n")
            f.write("End of Report\n")
            f.write("="*60 + "\n")
        
        # Save edit events log in readable format
        edit_log_path = os.path.join(self.output_dir, 'edit_events_log.txt')
        with open(edit_log_path, 'w') as f:
            f.write("SCENE EDIT EVENTS LOG\n")
            f.write("="*40 + "\n\n")
            
            with self.latency_tracker.lock:
                if self.latency_tracker.edit_events:
                    for i, edit in enumerate(self.latency_tracker.edit_events, 1):
                        timestamp_str = time.strftime('%H:%M:%S', time.localtime(edit['timestamp']))
                        f.write(f"Edit #{i}:\n")
                        f.write(f"  • Type: {edit['edit_type'].replace('_', ' ').title()}\n")
                        f.write(f"  • Time: {timestamp_str}\n")
                        f.write(f"  • At Frame: {edit['frame_count']}\n\n")
                else:
                    f.write("No edits were made during the simulation.\n")
        
        # Save detailed frame data
        frame_log_path = os.path.join(self.output_dir, 'frame_timing_details.txt')
        with open(frame_log_path, 'w') as f:
            f.write("DETAILED FRAME TIMING DATA\n")
            f.write("="*50 + "\n\n")
            f.write("Format: Frame ID | Processing Time | Capture-to-Process | Timestamp\n")
            f.write("-" * 70 + "\n")
            
            with self.latency_tracker.lock:
                for frame in self.latency_tracker.frame_timestamps:
                    timestamp_str = time.strftime('%H:%M:%S', time.localtime(frame['system_capture_time']))
                    f.write(f"{frame['frame_id']:>8} | {frame['processing_duration_ms']:>13.2f}ms | ")
                    f.write(f"{frame['capture_to_process_ms']:>15.2f}ms | {timestamp_str}\n")
        
        print(f"📄 Human-readable logs saved to {self.output_dir}")
    
    def _save_metrics_summary(self):
        """Save a comprehensive metrics summary file"""
        summary_path = os.path.join(self.output_dir, 'simulation_metrics_summary.txt')
        stats = self.latency_tracker.get_stats()
        
        if not stats:
            return
            
        with open(summary_path, 'w') as f:
            f.write("CARLA INTERSECTION SIMULATION - METRICS SUMMARY\n")
            f.write("="*60 + "\n\n")
            
            # Simulation setup info
            f.write("SIMULATION CONFIGURATION:\n")
            f.write(f"  • Camera Position: Static intersection view (15m elevation)\n")
            f.write(f"  • Camera FOV: 110° (wide intersection coverage)\n")
            f.write(f"  • Camera Angle: -30° downward pitch\n")
            f.write(f"  • Vehicles Spawned: {len(self.vehicles)}\n")
            f.write(f"  • Image Resolution: 800x600 pixels\n")
            f.write(f"  • Image Format: PNG\n\n")
            
            # Key performance metrics
            f.write("KEY PERFORMANCE INDICATORS:\n")
            f.write(f"  • Total Frames Captured: {stats['total_frames']:,}\n")
            f.write(f"  • Average Frame Rate: {stats['frame_rate']['avg_fps']:.2f} FPS\n")
            f.write(f"  • Average Processing Time: {stats['processing_latency']['avg_ms']:.2f} ms\n")
            f.write(f"  • Average Capture-to-Process Latency: {stats['capture_to_process_latency']['avg_ms']:.2f} ms\n")
            f.write(f"  • 95th Percentile Processing Time: {stats['processing_latency']['p95_ms']:.2f} ms\n")
            f.write(f"  • Total Scene Edits Applied: {stats['total_edits']}\n\n")
            
            # Performance classification
            avg_fps = stats['frame_rate']['avg_fps']
            avg_processing = stats['processing_latency']['avg_ms']
            
            f.write("PERFORMANCE ANALYSIS:\n")
            if avg_fps >= 20:
                f.write(f"  • Frame Rate: EXCELLENT ({avg_fps:.1f} FPS)\n")
            elif avg_fps >= 15:
                f.write(f"  • Frame Rate: GOOD ({avg_fps:.1f} FPS)\n")
            elif avg_fps >= 10:
                f.write(f"  • Frame Rate: ACCEPTABLE ({avg_fps:.1f} FPS)\n")
            else:
                f.write(f"  • Frame Rate: NEEDS IMPROVEMENT ({avg_fps:.1f} FPS)\n")
                
            if avg_processing <= 20:
                f.write(f"  • Processing Speed: EXCELLENT ({avg_processing:.1f} ms)\n")
            elif avg_processing <= 50:
                f.write(f"  • Processing Speed: GOOD ({avg_processing:.1f} ms)\n")
            elif avg_processing <= 100:
                f.write(f"  • Processing Speed: ACCEPTABLE ({avg_processing:.1f} ms)\n")
            else:
                f.write(f"  • Processing Speed: NEEDS IMPROVEMENT ({avg_processing:.1f} ms)\n")
            
            # Edit impact analysis
            edit_impact = self.latency_tracker.get_stats_around_edits()
            if edit_impact:
                f.write(f"\nEDIT IMPACT SUMMARY:\n")
                total_impact = sum(impact['performance_impact'] for impact in edit_impact)
                f.write(f"  • Number of Edits: {len(edit_impact)}\n")
                f.write(f"  • Average Impact per Edit: {total_impact/len(edit_impact):+.2f} ms\n")
                if total_impact > 5:
                    f.write(f"  • Overall Impact: SIGNIFICANT PERFORMANCE DECREASE\n")
                elif total_impact > 0:
                    f.write(f"  • Overall Impact: MINOR PERFORMANCE DECREASE\n")
                elif total_impact < -5:
                    f.write(f"  • Overall Impact: SIGNIFICANT PERFORMANCE IMPROVEMENT\n")
                else:
                    f.write(f"  • Overall Impact: MINIMAL IMPACT\n")
            else:
                f.write(f"\nEDIT IMPACT SUMMARY:\n")
                f.write(f"  • No scene edits were made during simulation\n")
            
            f.write(f"\n" + "="*60 + "\n")
            f.write(f"Report generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"="*60 + "\n")
        
        print(f"📊 Metrics summary saved: {summary_path}")
    
    def cleanup(self):
        """Clean up resources"""
        print("🧹 Cleaning up...")
        
        # Close metrics log file
        if self.metrics_file_handle:
            try:
                self.metrics_file_handle.write("\n" + "="*50 + "\n")
                self.metrics_file_handle.write(f"Ended at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                self.metrics_file_handle.write("="*50 + "\n")
                self.metrics_file_handle.close()
                print(f"📊 Metrics log saved and closed")
            except Exception as e:
                print(f"⚠️  Error closing metrics log: {e}")
        
        if self.camera and self.camera.is_alive:
            self.camera.stop()
            self.camera.destroy()
        
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        
        print("✅ Cleanup completed")


def main():
    """Main function"""
    print("🎮 CARLA Editable Simulation with Latency Tracking")
    print("📖 Talk2Traffic-style editing + Performance measurement")
    
    try:
        sim = EditableCameraSimulation(output_dir='editable_simulation_output')
        sim.start_interactive_simulation(duration=300)  # 5 minutes
        
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == '__main__':
    main()
