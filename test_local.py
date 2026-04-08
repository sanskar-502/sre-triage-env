#test_local.py
import asyncio
from client import SREEnvClient, SREAction
from dotenv import load_dotenv  # Add this
load_dotenv()

async def test_container():
    print("🛰️ Connecting to local Docker container...")
    
    # Connect to the local running container
    env = SREEnvClient(base_url="http://localhost:8000")
    
    try:
        print("✅ Connected! Resetting environment...")
        result = await env.reset()
        print(f"📊 Initial Health State: {result.observation.system_health_check}")
        
        # Send a test action (ActionType must be a string or enum based on your models.py)
        print("⚙️ Sending test action: 'pm2 status'")
        action = SREAction(action_type="execute_command", command="pm2 status")
        result = await env.step(action)
        
        print(f"📂 Action Output: {result.observation.stdout}")
        print(f"💰 Reward: {result.reward}")
        print("🚀 Container communication is fully operational!")
    finally:
        await env.close()

if __name__ == "__main__":
    asyncio.run(test_container())