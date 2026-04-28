from sre_triage.core.environment import SREEnvironment
from sre_triage.schemas import ActionType, SREAction


def test_easy_task_resolves():
    env = SREEnvironment(task_id="easy_node_down")
    obs = env.reset(task_id="easy_node_down")
    assert "503" in obs.system_health_check

    obs = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 start all"))
    assert obs.done is True
    assert obs.system_health_check == "HTTP 200 OK"


def test_medium_task_requires_env_fix_and_restart():
    env = SREEnvironment(task_id="medium_config_drift")
    env.reset(task_id="medium_config_drift")

    first = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="cat logs/error.log"))
    assert "27018" in first.stdout

    env.step(
        SREAction(
            action_type=ActionType.WRITE_FILE,
            file_path=".env",
            file_content="PORT=3000\nMONGO_URI=mongodb://localhost:27017/app\nJWT_SECRET=correct-horse-battery-staple",
        )
    )
    final = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 restart all"))
    assert final.done is True
    assert final.system_health_check == "HTTP 200 OK"


def test_hard_hybrid_failure_needs_process_kill():
    env = SREEnvironment(task_id="hard_hybrid_failure")
    env.reset(task_id="hard_hybrid_failure")
    env.step(
        SREAction(
            action_type=ActionType.WRITE_FILE,
            file_path=".env",
            file_content="PORT=3000\nMONGO_URI=mongodb://localhost:27017/app\nJWT_SECRET=correct-horse-battery-staple",
        )
    )
    mid = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 restart all"))
    assert "504" in mid.system_health_check
    final = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="kill -9 8891"))
    assert final.done is True
    assert final.system_health_check == "HTTP 200 OK"


def test_bad_secret_task_requires_secret_rotation():
    env = SREEnvironment(task_id="hard_bad_secret")
    env.reset(task_id="hard_bad_secret")
    log_obs = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="cat logs/error.log"))
    assert "invalid signature" in log_obs.stdout
    env.step(
        SREAction(
            action_type=ActionType.WRITE_FILE,
            file_path=".env",
            file_content="PORT=3000\nMONGO_URI=mongodb://localhost:27017/app\nJWT_SECRET=correct-horse-battery-staple",
        )
    )
    final = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 restart all"))
    assert final.done is True
    assert final.system_health_check == "HTTP 200 OK"


def test_disk_pressure_requires_log_rotation():
    env = SREEnvironment(task_id="hard_disk_pressure")
    env.reset(task_id="hard_disk_pressure")
    disk_obs = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="df -h"))
    assert "99%" in disk_obs.stdout
    final = env.step(
        SREAction(
            action_type=ActionType.EXECUTE_COMMAND,
            command="logrotate -f /etc/logrotate.d/mern-app",
        )
    )
    assert final.done is True
    assert final.system_health_check == "HTTP 200 OK"


def test_repeat_penalty_is_applied():
    env = SREEnvironment(task_id="easy_node_down")
    env.reset(task_id="easy_node_down")
    first = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 status"))
    second = env.step(SREAction(action_type=ActionType.EXECUTE_COMMAND, command="pm2 status"))
    assert second.reward < first.reward


def test_action_validation_rejects_invalid_shape():
    try:
        SREAction(action_type=ActionType.WRITE_FILE, file_path=".env")
    except ValueError as exc:
        assert "requires file_path and file_content" in str(exc)
    else:
        raise AssertionError("Expected write_file validation error")
