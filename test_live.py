"""Test the live HF Space endpoints."""
import httpx
import json

BASE = "https://sanskar502-sre-triage-env.hf.space"

try:
    print("Testing Live HF Space: " + BASE)
    print("-" * 50)
    
    # Test /reset
    print("[1/3] POST /reset")
    r = httpx.post(f"{BASE}/reset", json={}, timeout=30)
    print(f"Status Code: {r.status_code}")
    data = r.json()
    obs = data.get("observation", data)
    print(f"Initial Health: {obs.get('system_health_check')}")
    print("-" * 50)

    # Test /step
    print("[2/3] POST /step")
    r2 = httpx.post(f"{BASE}/step", json={"action": {"action_type": "execute_command", "command": "pm2 status"}}, timeout=30)
    print(f"Status Code: {r2.status_code}")
    data2 = r2.json()
    obs2 = data2.get("observation", data2)
    stdout_preview = obs2.get('stdout', '').split('\n')[0]
    print(f"Stdout Output: {stdout_preview}...")
    print(f"Reward Received: {data2.get('reward')}")
    print("-" * 50)

    # Test /state
    print("[3/3] GET /state")
    r3 = httpx.get(f"{BASE}/state", timeout=30)
    print(f"Status Code: {r3.status_code}")
    print(f"Episode State: {json.dumps(r3.json())}")
    print("-" * 50)
    
    if r.status_code == 200 and r2.status_code == 200 and r3.status_code == 200:
        print("\n ALL LIVE ENDPOINTS ARE WORKING PERFECTLY! ")
        
except Exception as e:
    print(f"\n ERROR: {str(e)}")
