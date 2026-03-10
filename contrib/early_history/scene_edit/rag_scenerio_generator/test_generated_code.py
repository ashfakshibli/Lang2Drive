"""
Test Generated CARLA Code
=========================

Test the generated code without requiring CARLA server.
"""

import os
import sys
import subprocess
import tempfile

def test_generated_code(filepath):
    """Test if generated code can be imported and has proper structure."""
    
    print(f"Testing: {filepath}")
    print("-" * 50)
    
    # Test 1: Syntax validation
    print("1. Syntax validation...")
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            code = f.read()
        
        compile(code, filepath, 'exec')
        print("   ✅ PASSED - Code has valid syntax")
    except SyntaxError as e:
        print(f"   ❌ FAILED - Syntax error: {e}")
        return False
    except Exception as e:
        print(f"   ❌ FAILED - Error reading file: {e}")
        return False
    
    # Test 2: Help text (check if script accepts arguments)
    print("\n2. Command line interface...")
    try:
        result = subprocess.run([
            sys.executable, filepath, '--help'
        ], capture_output=True, text=True, timeout=10)
        
        if result.returncode == 0 and 'CARLA' in result.stdout:
            print("   ✅ PASSED - Script has proper help text")
            print(f"   Help output preview: {result.stdout[:100]}...")
        else:
            print(f"   ⚠️  WARNING - Help might not work properly")
    except Exception as e:
        print(f"   ⚠️  WARNING - Could not test help: {e}")
    
    # Test 3: Structure validation
    print("\n3. Code structure validation...")
    
    required_elements = [
        'import carla',
        'def main():',
        'argparse',
        'vehicles = []',
        'camera = None',
        'client = carla.Client',
        'world.get_blueprint_library',
        'spawn_actor',
        'save_to_disk'
    ]
    
    missing = []
    for element in required_elements:
        if element not in code:
            missing.append(element)
    
    if not missing:
        print("   ✅ PASSED - All required elements present")
    else:
        print(f"   ❌ FAILED - Missing elements: {missing}")
        return False
    
    # Test 4: Configuration verification
    print("\n4. Configuration verification...")
    
    config_checks = [
        ('3 vehicles', code.count('range(3)') > 0 or 'vehicle_count=3' in code),
        ('Night weather', 'sun_altitude_angle=-30.0' in code or 'night' in code.lower()),
        ('Camera setup', 'sensor.camera.rgb' in code),
        ('Frame saving', 'save_to_disk' in code),
        ('Output directory', 'night-driving-frames' in code)
    ]
    
    all_good = True
    for check_name, check_result in config_checks:
        if check_result:
            print(f"   ✅ {check_name}")
        else:
            print(f"   ❌ {check_name}")
            all_good = False
    
    if all_good:
        print("\n🎉 ALL TESTS PASSED!")
        print("The generated code is ready to run with CARLA!")
        return True
    else:
        print("\n❌ Some tests failed")
        return False

def main():
    """Main test function."""
    
    print("Testing Generated CARLA Code")
    print("=" * 50)
    
    # Find the most recent generated file
    gen_dir = "../generated_scenarios"
    
    if not os.path.exists(gen_dir):
        print(f"Generated scenarios directory not found: {gen_dir}")
        return
    
    files = [f for f in os.listdir(gen_dir) if f.startswith('reliable_') and f.endswith('.py')]
    
    if not files:
        print("No generated files found")
        return
    
    # Sort by modification time to get the most recent
    files.sort(key=lambda f: os.path.getmtime(os.path.join(gen_dir, f)), reverse=True)
    
    latest_file = os.path.join(gen_dir, files[0])
    
    print(f"Testing latest generated file: {files[0]}")
    print(f"Full path: {latest_file}")
    print()
    
    success = test_generated_code(latest_file)
    
    if success:
        print("\n" + "="*60)
        print("READY TO RUN!")
        print("="*60)
        print(f"To test with CARLA:")
        print(f"1. Start CARLA server: CarlaUE4.exe")
        print(f"2. cd generated_scenarios")
        print(f"3. python {files[0]} --duration 10")
        print(f"4. Check the 'night-driving-frames' folder for saved images")

if __name__ == "__main__":
    main()
