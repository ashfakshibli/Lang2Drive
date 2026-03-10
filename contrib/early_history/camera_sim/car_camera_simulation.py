#!/usr/bin/env python

"""
CARLA Car Camera Simulation Script
==================================

This script:
1. Spawns a vehicle in CARLA
2. Attaches a camera sensor to the vehicle
3. Sets the vehicle to autopilot mode
4. Captures and saves camera frames during simulation
5. Runs for a specified duration then cleans up

Usage:
    python car_camera_simulation.py [--duration SECONDS] [--output-dir PATH]
"""

import glob
import os
import sys
import time
import argparse
import numpy as np
import random

# Add the CARLA egg file to Python path
try:
    # Look for CARLA egg file in the CARLA installation directory
    carla_egg_path = r'C:\Carla\WindowsNoEditor\PythonAPI\carla\dist\carla-*.egg'
    egg_file = glob.glob(carla_egg_path)[0]
    print(f"Found CARLA egg file: {egg_file}")
    sys.path.append(egg_file)
    
    # Also add the carla directory itself
    carla_dir = r'C:\Carla\WindowsNoEditor\PythonAPI\carla'
    if carla_dir not in sys.path:
        sys.path.append(carla_dir)
        
except IndexError:
    print("ERROR: Could not find CARLA egg file!")
    print(f"Searched in: {carla_egg_path}")
    print("Trying alternative import methods...")
    
    # Try adding the PythonAPI directory
    carla_api_dir = r'C:\Carla\WindowsNoEditor\PythonAPI'
    if carla_api_dir not in sys.path:
        sys.path.append(carla_api_dir)

try:
    import carla
    print("✅ Successfully imported carla module!")
except ImportError as e:
    print(f"❌ Failed to import carla module: {e}")
    print("\n🔧 Troubleshooting:")
    print("1. Make sure CARLA server is running:")
    print("   cd C:\\Carla\\WindowsNoEditor")
    print("   .\\CarlaUE4.exe")
    print("2. Check if you have the right Python version (CARLA 0.9.15 works best with Python 3.7-3.8)")
    print("3. Try installing CARLA using pip: pip install carla")
    sys.exit(1)


class CarCameraSimulation:
    def __init__(self, host='localhost', port=2000, output_dir='output_images'):
        """Initialize the simulation"""
        self.client = None
        self.world = None
        self.vehicle = None
        self.camera = None
        self.output_dir = output_dir
        self.frame_count = 0
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Output directory: {os.path.abspath(self.output_dir)}")
        
        # Connect to CARLA
        try:
            self.client = carla.Client(host, port)
            self.client.set_timeout(10.0)
            self.world = self.client.get_world()
            print(f"✅ Connected to CARLA server at {host}:{port}")
            print(f"🗺️  Current map: {self.world.get_map().name}")
        except Exception as e:
            print(f"❌ Failed to connect to CARLA: {e}")
            sys.exit(1)
    
    def spawn_vehicle(self):
        """Spawn a vehicle in the world"""
        try:
            # Get the blueprint library
            blueprint_library = self.world.get_blueprint_library()
            
            # Choose a vehicle blueprint (Tesla Model 3)
            vehicle_bp = blueprint_library.filter('vehicle.tesla.model3')[0]
            print(f"🚗 Selected vehicle: {vehicle_bp.id}")
            
            # Get spawn points
            spawn_points = self.world.get_map().get_spawn_points()
            if not spawn_points:
                raise Exception("No spawn points available")
            
            # Choose a random spawn point
            spawn_point = random.choice(spawn_points)
            
            # Spawn the vehicle
            self.vehicle = self.world.spawn_actor(vehicle_bp, spawn_point)
            print(f"✅ Vehicle spawned at: {spawn_point.location}")
            
            # Set to autopilot
            self.vehicle.set_autopilot(True)
            print("🤖 Autopilot enabled")
            
        except Exception as e:
            print(f"❌ Failed to spawn vehicle: {e}")
            self.cleanup()
            sys.exit(1)
    
    def setup_camera(self):
        """Setup and attach camera to vehicle"""
        try:
            # Get camera blueprint
            blueprint_library = self.world.get_blueprint_library()
            camera_bp = blueprint_library.find('sensor.camera.rgb')
            
            # Set camera attributes
            camera_bp.set_attribute('image_size_x', '800')
            camera_bp.set_attribute('image_size_y', '600')
            camera_bp.set_attribute('fov', '90')
            
            # Set camera transform (position relative to vehicle)
            camera_transform = carla.Transform(
                carla.Location(x=2.0, z=1.4),  # 2m forward, 1.4m up
                carla.Rotation(pitch=0.0)      # Level camera
            )
            
            # Spawn camera attached to vehicle
            self.camera = self.world.spawn_actor(
                camera_bp, 
                camera_transform, 
                attach_to=self.vehicle
            )
            print("📷 Camera attached to vehicle")
            
            # Start listening for images
            self.camera.listen(self.process_image)
            print("🎬 Camera started recording")
            
        except Exception as e:
            print(f"❌ Failed to setup camera: {e}")
            self.cleanup()
            sys.exit(1)
    
    def process_image(self, image):
        """Process and save camera image"""
        try:
            # Convert image to numpy array
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]  # Remove alpha channel
            array = array[:, :, ::-1]  # Convert BGR to RGB
            
            # Save image
            filename = f"frame_{self.frame_count:06d}.png"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save using basic file operations (no PIL dependency)
            self.save_image_array(array, filepath)
            
            self.frame_count += 1
            if self.frame_count % 30 == 0:  # Print every 30 frames
                print(f"📸 Captured {self.frame_count} frames")
                
        except Exception as e:
            print(f"❌ Error processing image: {e}")
    
    def save_image_array(self, array, filepath):
        """Save numpy array as PNG file (simple implementation)"""
        try:
            # Try to use PIL if available
            try:
                from PIL import Image
                img = Image.fromarray(array)
                img.save(filepath)
            except ImportError:
                # Fallback: save as raw data with info
                print("⚠️  PIL not available, saving raw data instead")
                np.save(filepath.replace('.png', '.npy'), array)
        except Exception as e:
            print(f"❌ Error saving image: {e}")
    
    def run_simulation(self, duration=60):
        """Run the simulation for specified duration"""
        print(f"🚀 Starting simulation for {duration} seconds...")
        print("📸 Camera images will be saved to:", os.path.abspath(self.output_dir))
        print("🎯 Press Ctrl+C to stop early")
        
        start_time = time.time()
        
        try:
            while time.time() - start_time < duration:
                # Let CARLA process one step
                self.world.tick()
                time.sleep(0.05)  # Small delay to control frame rate
                
                # Print progress every 10 seconds
                elapsed = time.time() - start_time
                if int(elapsed) % 10 == 0 and elapsed > 0:
                    remaining = duration - elapsed
                    print(f"⏰ Running... {elapsed:.1f}s elapsed, {remaining:.1f}s remaining")
                    time.sleep(1)  # Avoid printing multiple times per second
                    
        except KeyboardInterrupt:
            print("\n⏹️  Simulation interrupted by user")
        
        print(f"✅ Simulation completed! Captured {self.frame_count} frames")
    
    def cleanup(self):
        """Clean up all spawned actors"""
        print("🧹 Cleaning up...")
        
        if self.camera is not None:
            self.camera.stop()
            self.camera.destroy()
            print("📷 Camera destroyed")
        
        if self.vehicle is not None:
            self.vehicle.destroy()
            print("🚗 Vehicle destroyed")
        
        print("✅ Cleanup completed")


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='CARLA Car Camera Simulation')
    parser.add_argument('--duration', type=int, default=60, 
                       help='Simulation duration in seconds (default: 60)')
    parser.add_argument('--output-dir', type=str, default='output_images',
                       help='Output directory for images (default: output_images)')
    parser.add_argument('--host', type=str, default='localhost',
                       help='CARLA server host (default: localhost)')
    parser.add_argument('--port', type=int, default=2000,
                       help='CARLA server port (default: 2000)')
    
    args = parser.parse_args()
    
    print("🎮 CARLA Car Camera Simulation")
    print("=" * 40)
    
    # Create simulation instance
    sim = CarCameraSimulation(
        host=args.host, 
        port=args.port, 
        output_dir=args.output_dir
    )
    
    try:
        # Setup simulation
        sim.spawn_vehicle()
        sim.setup_camera()
        
        # Run simulation
        sim.run_simulation(args.duration)
        
    finally:
        # Always cleanup
        sim.cleanup()


if __name__ == '__main__':
    main()
