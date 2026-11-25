"""Fetch and cache Gen 9 Random Battles usage and set statistics from pkmn/randbats."""

import argparse
import json
from pathlib import Path

import requests
from poke_env.data import to_id_str

from poba.config import RANDBATS_STATS_PATH, RANDBATS_STATS_URL


def fetch_stats(
    url: str = RANDBATS_STATS_URL,
    output_path: Path = RANDBATS_STATS_PATH,
    force: bool = False,
) -> Path:
    """Download randbats stats JSON and validate its schema before caching locally."""
    # Reuse existing local cache unless --force is requested
    if output_path.exists() and not force:
        print(f"Cached randbats stats already exist at {output_path}. Use --force to refresh.")
        return output_path

    # Ensure parent data directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Fetch stats JSON from pkmn/randbats
    print(f"Fetching Gen 9 Random Battles stats from {url}...")
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    payload = resp.json()

    # Basic schema sanity check: must be a non-empty dict of species -> {roles: ...}
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Fetched randbats stats payload is empty or not a JSON object.")
    sample = next(iter(payload.values()))
    if not isinstance(sample, dict) or "roles" not in sample:
        raise ValueError("Unexpected randbats stats schema: missing 'roles' key.")

    # Save validated JSON locally (ignored by git via .gitignore)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    # Compute summary stats for quick verification
    unique_moves = {
        to_id_str(m)
        for sp in payload.values()
        for role in sp.get("roles", {}).values()
        for m in role.get("moves", {})
    }
    unique_items = {
        to_id_str(i)
        for sp in payload.values()
        for role in sp.get("roles", {}).values()
        for i in role.get("items", {})
    }
    print(
        f"Saved {len(payload)} species, {len(unique_moves)} unique moves, "
        f"{len(unique_items)} unique items -> {output_path}"
    )
    return output_path


def main() -> None:
    """CLI entrypoint for downloading Gen 9 Random Battles stats."""
    parser = argparse.ArgumentParser(description="Fetch Gen 9 Random Battles stats JSON.")
    parser.add_argument("--url", type=str, default=RANDBATS_STATS_URL, help="Stats JSON URL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=RANDBATS_STATS_PATH,
        help="Destination JSON file path.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cached stats already exist.",
    )
    args = parser.parse_args()
    fetch_stats(url=args.url, output_path=args.output, force=args.force)


if __name__ == "__main__":
    main()
