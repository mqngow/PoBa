"""Setup and launch a local headless Pokémon Showdown server for RL simulation."""

import argparse
import shutil
import subprocess
from pathlib import Path

from poba.config import SHOWDOWN_DIR, SHOWDOWN_PORT

SHOWDOWN_REPO_URL = "https://github.com/smogon/pokemon-showdown.git"


def _resolve_npm() -> str:
    """Find the npm executable safely on both Windows (.cmd) and POSIX."""
    npm_exe = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm_exe:
        raise RuntimeError("npm was not found on PATH. Install Node.js before continuing.")
    return npm_exe


def setup_showdown(showdown_dir: Path = SHOWDOWN_DIR) -> None:
    """Clone pokemon-showdown, initialize config.js, and install Node dependencies."""
    # Clone a shallow copy of official pokemon-showdown if not already present
    if not showdown_dir.exists():
        print(f"Cloning {SHOWDOWN_REPO_URL} into {showdown_dir}...")
        subprocess.run(
            ["git", "clone", "--depth", "1", SHOWDOWN_REPO_URL, str(showdown_dir)],
            check=True,
        )

    # Copy the example server config so Showdown can start with local defaults
    config_file = showdown_dir / "config" / "config.js"
    example_config = showdown_dir / "config" / "config-example.js"
    if not config_file.exists():
        shutil.copy2(example_config, config_file)
        print(f"Created {config_file} from config-example.js.")

    # Install npm dependencies once so the local server can build and run
    node_modules = showdown_dir / "node_modules"
    if not node_modules.exists():
        npm_exe = _resolve_npm()
        print(f"Installing Node dependencies in {showdown_dir}...")
        subprocess.run([npm_exe, "install"], cwd=showdown_dir, check=True)


def start_showdown(showdown_dir: Path = SHOWDOWN_DIR, port: int = SHOWDOWN_PORT) -> None:
    """Start the local Showdown server in headless mode with --no-security."""
    cmd = ["node", "pokemon-showdown", "start", "--no-security", "--port", str(port)]
    print(f"Starting local Pokémon Showdown server on port {port}...")
    proc = subprocess.Popen(cmd, cwd=showdown_dir)

    try:
        # Block until the server exits or the user interrupts
        exit_code = proc.wait()
        if exit_code != 0:
            raise RuntimeError(f"Showdown server exited with status {exit_code}.")
    except KeyboardInterrupt:
        # Gracefully terminate the child Node process on Ctrl+C
        print("\nStopping local Pokémon Showdown server...")
        proc.terminate()
        proc.wait(timeout=10)


def main() -> None:
    """CLI entrypoint for setting up and starting the local Showdown server."""
    parser = argparse.ArgumentParser(description="Setup and run local Pokémon Showdown server.")
    parser.add_argument("--port", type=int, default=SHOWDOWN_PORT, help="WebSocket port to bind.")
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Clone and install dependencies without starting the server.",
    )
    args = parser.parse_args()

    setup_showdown(SHOWDOWN_DIR)
    if not args.setup_only:
        start_showdown(SHOWDOWN_DIR, port=args.port)


if __name__ == "__main__":
    main()
