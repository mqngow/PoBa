"""Step 1 unit and integration tests for config and local Showdown connectivity."""

import asyncio
import importlib.util
import socket

import pytest

from poba.config import (
    BATTLE_FORMAT,
    PROJECT_ROOT,
    SHOWDOWN_HOST,
    SHOWDOWN_PORT,
    EnvConfig,
    PPOConfig,
    get_local_server_config,
)


def _load_run_benchmark():
    """Import run_benchmark from scripts/benchmark_baseline.py without sys.path hacks."""
    script_path = PROJECT_ROOT / "scripts" / "benchmark_baseline.py"
    spec = importlib.util.spec_from_file_location("benchmark_baseline", script_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.run_benchmark


def _is_port_open(host: str, port: int) -> bool:
    """Return True if a TCP connection succeeds on host:port within 1 second."""
    try:
        with socket.create_connection((host, port), timeout=1.0):
            return True
    except OSError:
        return False


def test_config_defaults() -> None:
    """Verify project paths, server configuration, and default dataclasses."""
    assert PROJECT_ROOT.exists()
    assert BATTLE_FORMAT == "gen9randombattle"

    # Check default config dataclasses
    env_cfg = EnvConfig()
    ppo_cfg = PPOConfig()
    assert env_cfg.battle_format == "gen9randombattle"
    assert env_cfg.use_prob_tracker is True
    assert ppo_cfg.learning_rate == 3e-4
    assert ppo_cfg.n_steps == 2048

    # Check both default and custom port server configurations
    default_srv = get_local_server_config()
    assert "localhost:8000/showdown/websocket" in default_srv.websocket_url
    custom_srv = get_local_server_config(port=8001)
    assert custom_srv.websocket_url == "ws://localhost:8001/showdown/websocket"


def test_local_showdown_100_battles() -> None:
    """Verify 100 local gen9randombattle matches complete cleanly with zero drops."""
    if not _is_port_open(SHOWDOWN_HOST, SHOWDOWN_PORT):
        pytest.skip("Local Showdown server not running on localhost:8000")

    run_benchmark = _load_run_benchmark()
    result = asyncio.run(run_benchmark(n_battles=100, concurrency=10))

    # Confirm every battle finished and MaxBasePowerPlayer outperforms RandomPlayer
    assert result["random_finished"] == 100
    assert result["max_power_finished"] == 100
    assert result["max_power_wins"] > result["random_wins"]
