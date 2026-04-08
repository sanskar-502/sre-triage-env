# test_logic.py — Comprehensive unit tests for the SRE Triage Environment
from server.environment import SREEnvironment
from models import SREAction, ActionType

def test_difficulty(name, env, actions, expect_done=True):
    """Run a task and verify it resolves correctly."""
    print(f"\n{'='*50}")
    print(f"🔄 TEST: {name}")
    print(f"{'='*50}")
    obs = env.reset()
    print(f"  Initial Health: {obs.system_health_check}")

    for i, action in enumerate(actions, 1):
        obs = env.step(action)
        action_desc = f"{action.action_type.value}"
        if action.command:
            action_desc += f" '{action.command}'"
        if action.file_path:
            action_desc += f" ({action.file_path})"
        print(f"  Step {i} ({action_desc}) → Health: {obs.system_health_check}")

    print(f"  Done: {obs.done}")
    assert obs.done == expect_done, f"Expected done={expect_done}, got {obs.done}"
    if expect_done:
        assert obs.system_health_check == "HTTP 200 OK", f"Expected HTTP 200, got {obs.system_health_check}"
    print(f"  ✅ PASSED")

# ── Test Easy Mode ──
test_difficulty("Easy — Node service stopped", SREEnvironment(difficulty="easy"), [
    SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 status"),
    SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 start all"),
])

# ── Test Medium Mode ──
test_difficulty("Medium — Configuration drift", SREEnvironment(difficulty="medium"), [
    SREAction(action_type=ActionType.EXECUTE_COMMAND, command="cat logs/error.log"),
    SREAction(action_type=ActionType.WRITE_FILE, file_path=".env", file_content="PORT=3000\nMONGO_URI=mongodb://localhost:27017/app"),
    SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 restart all"),
])

# ── Test Hard Mode ──
test_difficulty("Hard — Hybrid failure", SREEnvironment(difficulty="hard"), [
    SREAction(action_type=ActionType.EXECUTE_COMMAND, command="cat logs/error.log"),
    SREAction(action_type=ActionType.WRITE_FILE, file_path=".env", file_content="PORT=3000\nMONGO_URI=mongodb://localhost:27017/app"),
    SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 restart all"),
    SREAction(action_type=ActionType.EXECUTE_COMMAND, command="kill -9 8891"),
])

# ── Test Dynamic Reset ──
print(f"\n{'='*50}")
print(f"🔄 TEST: Dynamic reset(difficulty)")
print(f"{'='*50}")
env = SREEnvironment()
for diff, expected_health in [("easy", "503"), ("medium", "500"), ("hard", "500")]:
    obs = env.reset(difficulty=diff)
    assert expected_health in obs.system_health_check, f"{diff}: expected {expected_health} in {obs.system_health_check}"
    if diff == "hard":
        ps_obs = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="ps aux"))
        assert "8891" in ps_obs.stdout, f"Hard mode should have rogue PID 8891"
        print(f"  Hard  → {obs.system_health_check} + rogue PID ✓")
    else:
        print(f"  {diff.capitalize():7s}→ {obs.system_health_check} ✓")
print(f"  ✅ PASSED")

# ── Test Step Penalty ──
print(f"\n{'='*50}")
print(f"🔄 TEST: Step penalty & repeat penalty")
print(f"{'='*50}")
env = SREEnvironment(difficulty="easy")
obs1 = env.step(SREAction(action_type=ActionType.CHECK_HEALTH))
assert obs1.reward < 0, f"Step cost should make check_health negative, got {obs1.reward}"
print(f"  ✅ Step penalty works (check_health reward: {obs1.reward:.2f})")

obs2 = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 status"))
obs3 = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 status"))
assert obs3.reward < obs2.reward, f"Repeat should be penalized more"
print(f"  ✅ Repeat penalty works (first: {obs2.reward:.2f}, repeat: {obs3.reward:.2f})")

# ── Test Red Herrings ──
print(f"\n{'='*50}")
print(f"🔄 TEST: Realistic logs with noise")
print(f"{'='*50}")
env = SREEnvironment(difficulty="medium")
obs = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="cat logs/error.log"))
assert "DeprecationWarning" in obs.stdout, "Logs should contain red herring warnings"
assert "2024-01-15" in obs.stdout, "Logs should contain timestamps"
print(f"  ✅ Logs contain timestamps and red herrings")
env2 = SREEnvironment(difficulty="medium")
obs2 = env2.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="cat .env"))
assert "staging migration" in obs2.stdout, ".env should have misleading comment"
print(f"  ✅ .env contains misleading comment about staging")

print(f"\n{'='*50}")
print(f"🎉 ALL TESTS PASSED")
print(f"{'='*50}")