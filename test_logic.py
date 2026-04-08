#test_logic.py
import asyncio
from server.environment import SREEnvironment
from models import SREAction, ActionType
from dotenv import load_dotenv  # Add this
load_dotenv()

async def run_test():
    env = SREEnvironment(difficulty="medium")
    print(f"Initial Health: {env._run_health_check()}") # Should be 500
    
    # 1. Inspect the env file
    action1 = SREAction(action_type=ActionType.EXECUTE_COMMAND, command="cat .env")
    obs1 = env.step(action1)
    print(f"Step 1 (cat .env) -> {obs1.stdout}")
    
    # 2. Fix the env file
    action2 = SREAction(
        action_type=ActionType.WRITE_FILE, 
        file_path=".env", 
        file_content="PORT=3000\nMONGO_URI=mongodb://localhost:27017/app"
    )
    obs2 = env.step(action2)
    print(f"Step 2 (write_file) -> {obs2.stdout}")
    print(f"Health after Write (Still 500): {obs2.system_health_check}") 
    
    # 3. NEW: Restart the service to apply the fix
    print("⚙️ Issuing restart command...")
    action3 = SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 restart all")
    obs3 = env.step(action3)
    
    print(f"Step 3 (restart) -> {obs3.stdout}")
    print(f"Final Health: {obs3.system_health_check}") # NOW it will be 200 OK
    print(f"Is Done? {obs3.done}") # NOW it will be True

if __name__ == "__main__":
    asyncio.run(run_test())