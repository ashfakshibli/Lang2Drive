"""
Simple RAG Demo for CARLA Scenario Generation
=============================================

This is a simplified demo that shows how the RAG pipeline works
without all the complex imports. Perfect for testing and demonstration.
"""

import os
import json
import time
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

class SimpleCarlaRAG:
    """Simplified RAG implementation for CARLA scenarios."""
    
    def __init__(self, scenarios_db_path: str):
        """Initialize with scenarios database."""
        self.scenarios_db_path = scenarios_db_path
        
        # Configure Gemini API
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("GOOGLE_API_KEY not found in .env file")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.5-pro')
        
        # Load scenarios
        self.scenarios = self._load_scenarios()
        print(f"✅ Loaded {len(self.scenarios)} scenarios from database")
    
    def _load_scenarios(self):
        """Load scenarios from JSON file."""
        with open(self.scenarios_db_path, 'r') as f:
            data = json.load(f)
        return data.get('scenarios', [])
    
    def find_best_match(self, user_input: str):
        """Find the best matching scenario using Gemini."""
        
        # Create scenario descriptions for matching
        scenario_texts = []
        for i, scenario in enumerate(self.scenarios):
            text = f"Scenario {i}: {scenario['description']} - Keywords: {', '.join(scenario['keywords'])}"
            scenario_texts.append(text)
        
        # Use Gemini to find the best match
        prompt = f"""
        User request: "{user_input}"
        
        Available scenarios:
        {chr(10).join(scenario_texts)}
        
        Which scenario best matches the user's request? Return only the scenario number (0, 1, 2, etc.).
        """
        
        try:
            response = self.model.generate_content(prompt)
            scenario_index = int(response.text.strip())
            return self.scenarios[scenario_index]
        except:
            # Fallback to first scenario
            return self.scenarios[0]
    
    def understand_request(self, user_input: str):
        """Generate understanding of the user's request."""
        
        prompt = f"""
        Analyze this CARLA simulation request: "{user_input}"
        
        Explain in simple terms:
        1. What the user wants to see
        2. How many vehicles are needed
        3. What weather conditions
        4. What camera angle/perspective
        5. Any special requirements
        
        Be concise and clear.
        """
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def generate_carla_code(self, user_input: str, scenario: dict):
        """Generate CARLA Python code based on the scenario."""
        
        prompt = f"""
        Generate a complete, production-ready Python script for CARLA simulator based on:
        
        User Request: "{user_input}"
        Scenario: {scenario['description']}
        Parameters: {json.dumps(scenario['parameters'], indent=2)}
        
        The code MUST include:
        1. Proper error handling and connection testing
        2. Spawn {scenario['parameters'].get('vehicle_count', 1)} vehicles with retry logic
        3. Set weather to {scenario['parameters'].get('weather', 'clear')} with enhanced parameters
        4. Configure {scenario['parameters'].get('camera_mode', 'follow_car')} camera with proper attachment
        5. Run simulation with progress tracking and frame counting
        6. Save frames to output directory with organized naming
        7. Robust resource cleanup (vehicles, camera, etc.)
        8. Command-line arguments for duration, output directory, etc.
        9. Detailed logging and status messages
        10. Compatible with Python 3.8 and modern CARLA versions
        
        IMPORTANT REQUIREMENTS:
        - Include comprehensive error handling for all CARLA operations
        - Test CARLA connection before proceeding
        - Use try_spawn_actor with multiple attempts for vehicles
        - Attach camera properly to spawned vehicle (check if vehicle exists)
        - Create output directory if it doesn't exist
        - Print progress during frame saving (every 30 frames)
        - Include proper resource cleanup in finally block
        - Add detailed status messages for each step
        - Handle keyboard interruption gracefully
        - Include command-line arguments for customization
        
        Return a complete, runnable Python script with NO markdown formatting.
        The script should be robust enough to run in production.
        """
        
        response = self.model.generate_content(prompt)
        return response.text.strip()
    
    def process_request(self, user_input: str):
        """Complete pipeline to process a user request."""
        
        print(f"\n🎯 Processing: '{user_input}'")
        print("=" * 60)
        
        # Step 1: Understand the request
        print("🧠 Understanding your request...")
        understanding = self.understand_request(user_input)
        print(f"Understanding: {understanding}")
        
        # Step 2: Find best matching scenario
        print("\n🔍 Finding best matching scenario...")
        best_scenario = self.find_best_match(user_input)
        print(f"Best Match: {best_scenario['description']}")
        print(f"Parameters: {json.dumps(best_scenario['parameters'], indent=2)}")
        
        # Step 3: Generate CARLA code
        print("\n🔧 Generating CARLA code...")
        generated_code = self.generate_carla_code(user_input, best_scenario)
        
        # Step 4: Create explanation
        explanation = f"""
🤖 **RAG System Understanding & Plan**

📝 **Your Request:** "{user_input}"

🧠 **My Understanding:** {understanding}

🎯 **Selected Scenario:** {best_scenario['description']}

⚙️ **Configuration:**
- Vehicles: {best_scenario['parameters'].get('vehicle_count', 1)}
- Weather: {best_scenario['parameters'].get('weather', 'clear')}
- Camera: {best_scenario['parameters'].get('camera_mode', 'follow_car')}

🚀 **Execution Plan:**
1. Connect to CARLA server (localhost:2000)
2. Load map and spawn points
3. Configure weather and lighting
4. Spawn vehicles with AI behavior
5. Set up camera positioning
6. Run simulation and save frames
7. Clean up resources

✅ **Ready to run with Python 3.8 and CARLA!**
        """
        
        return {
            'explanation': explanation,
            'code': generated_code,
            'scenario': best_scenario,
            'understanding': understanding
        }

def save_code_to_file(code: str, filename: str = None):
    """Save generated code to a file."""
    if filename is None:
        timestamp = int(time.time())
        filename = f"generated_carla_{timestamp}.py"
    
    # Create output directory
    output_dir = Path("../generated_scenarios")
    output_dir.mkdir(exist_ok=True)
    
    # Clean up the code (remove markdown formatting if present)
    if code.startswith("```python"):
        code = code.replace("```python", "").replace("```", "").strip()
    
    file_path = output_dir / filename
    with open(file_path, 'w') as f:
        f.write(code)
    
    print(f"💾 Code saved to: {file_path}")
    return str(file_path)

def main():
    """Main demo function."""
    
    print("🚀 CARLA RAG Pipeline Demo")
    print("=" * 40)
    
    try:
        # Initialize RAG system
        scenarios_db = "../carla_scenarios_db.json"
        rag = SimpleCarlaRAG(scenarios_db)
        
        # Example requests
        example_requests = [
            "2 cars driving in rainy weather and camera following from third person view",
            "Multiple vehicles at an intersection with overhead camera",
            "Night driving with vehicle lights",
            "Single car on a straight road in sunny weather"
        ]
        
        print("\n📋 Example requests to try:")
        for i, req in enumerate(example_requests, 1):
            print(f"{i}. {req}")
        
        # Process user input
        while True:
            print("\n" + "="*60)
            user_input = input("\n🎯 Enter your scenario request (or 'quit' to exit): ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q']:
                print("👋 Goodbye!")
                break
            
            if not user_input:
                continue
            
            try:
                # Process the request
                result = rag.process_request(user_input)
                
                # Show explanation
                print("\n" + "="*60)
                print(result['explanation'])
                print("="*60)
                
                # Ask if user wants to save the code
                save = input("\n💾 Save generated code? (y/n): ").strip().lower()
                if save in ['y', 'yes']:
                    filename = f"scenario_{int(time.time())}.py"
                    saved_path = save_code_to_file(result['code'], filename)
                    print(f"✅ Code ready to run with Python 3.8!")
                    print(f"📁 File: {saved_path}")
                
            except Exception as e:
                print(f"❌ Error processing request: {e}")
                
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        print("Make sure you have:")
        print("1. GOOGLE_API_KEY in your .env file")
        print("2. carla_scenarios_db.json in the parent directory")

if __name__ == "__main__":
    main()
