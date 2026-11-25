"""Project configuration, paths, battle format settings, and PPO hyperparameters."""

from dataclasses import dataclass
from pathlib import Path

from poke_env import LocalhostServerConfiguration, ServerConfiguration

# Derive repository root from src/poba/config.py -> PoBa/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Standard project directories for datasets, checkpoints, logs, and local Showdown
DATA_DIR = PROJECT_ROOT / "data"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
LOGS_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
SHOWDOWN_DIR = PROJECT_ROOT / "pokemon-showdown"

# Public Gen 9 Random Battles usage & set statistics URL and local cache path
RANDBATS_STATS_URL = "https://pkmn.github.io/randbats/data/stats/gen9randombattle.json"
RANDBATS_STATS_PATH = DATA_DIR / "gen9randombattle_stats.json"

# Default Pokémon Showdown battle format and local server connection settings
BATTLE_FORMAT = "gen9randombattle"
SHOWDOWN_HOST = "localhost"
SHOWDOWN_PORT = 8000
SHOWDOWN_WS_URL = f"ws://{SHOWDOWN_HOST}:{SHOWDOWN_PORT}/showdown/websocket"
SHOWDOWN_AUTH_URL = "https://play.pokemonshowdown.com/action.php?"


def get_local_server_config(
    host: str = SHOWDOWN_HOST,
    port: int = SHOWDOWN_PORT,
) -> ServerConfiguration:
    """Return a poke-env ServerConfiguration targeting a local Showdown server."""
    # Reuse poke-env's built-in localhost config for the standard port
    if host == "localhost" and port == 8000:
        return LocalhostServerConfiguration

    # Build a custom WebSocket endpoint when running on a non-default host or port
    ws_url = f"ws://{host}:{port}/showdown/websocket"
    return ServerConfiguration(ws_url, SHOWDOWN_AUTH_URL)


@dataclass(frozen=True)
class EnvConfig:
    """Environment configuration for Gen 9 Random Battles."""

    battle_format: str = BATTLE_FORMAT
    use_prob_tracker: bool = True
    shaped_rewards: bool = True
    max_concurrent_battles: int = 10


@dataclass(frozen=True)
class PPOConfig:
    """Default MaskablePPO hyperparameters for training."""

    learning_rate: float = 3e-4
    n_steps: int = 2048
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    seed: int = 42
