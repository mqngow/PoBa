"""Benchmark local Showdown simulation speed with poke-env baseline bots."""

import argparse
import asyncio
import time

from poke_env.player import MaxBasePowerPlayer, RandomPlayer

from poba.config import BATTLE_FORMAT, SHOWDOWN_PORT, get_local_server_config


async def run_benchmark(
    n_battles: int = 100,
    concurrency: int = 10,
    battle_format: str = BATTLE_FORMAT,
    port: int = SHOWDOWN_PORT,
) -> dict:
    """Run local battles between RandomPlayer and MaxBasePowerPlayer and return metrics."""
    # Point both players at the local headless Showdown server
    server_config = get_local_server_config(port=port)

    random_player = RandomPlayer(
        battle_format=battle_format,
        server_configuration=server_config,
        max_concurrent_battles=concurrency,
    )
    max_power_player = MaxBasePowerPlayer(
        battle_format=battle_format,
        server_configuration=server_config,
        max_concurrent_battles=concurrency,
    )

    # Time the full batch of concurrent battles
    start_time = time.perf_counter()
    await random_player.battle_against(max_power_player, n_battles=n_battles)
    elapsed_sec = time.perf_counter() - start_time

    battles_per_sec = n_battles / elapsed_sec if elapsed_sec > 0 else 0.0
    return {
        "n_battles": n_battles,
        "random_finished": random_player.n_finished_battles,
        "max_power_finished": max_power_player.n_finished_battles,
        "random_wins": random_player.n_won_battles,
        "max_power_wins": max_power_player.n_won_battles,
        "elapsed_sec": elapsed_sec,
        "battles_per_sec": battles_per_sec,
    }


def main() -> None:
    """CLI entrypoint for running the baseline Showdown throughput benchmark."""
    parser = argparse.ArgumentParser(description="Benchmark local poke-env baseline battles.")
    parser.add_argument("--n-battles", type=int, default=100, help="Number of battles to run.")
    parser.add_argument("--concurrency", type=int, default=10, help="Max concurrent battles.")
    parser.add_argument("--format", type=str, default=BATTLE_FORMAT, help="Showdown battle format.")
    parser.add_argument("--port", type=int, default=SHOWDOWN_PORT, help="Local Showdown port.")
    args = parser.parse_args()

    result = asyncio.run(
        run_benchmark(
            n_battles=args.n_battles,
            concurrency=args.concurrency,
            battle_format=args.format,
            port=args.port,
        )
    )

    # Print a concise summary table of battle completion, win rates, and speed
    print("\n=== Baseline Showdown Benchmark ===")
    print(f"Format           : {args.format}")
    print(f"Requested        : {result['n_battles']}")
    print(f"Random Finished  : {result['random_finished']} (Wins: {result['random_wins']})")
    print(f"MaxPower Finished: {result['max_power_finished']} (Wins: {result['max_power_wins']})")
    print(f"Elapsed Time     : {result['elapsed_sec']:.2f}s")
    print(f"Throughput       : {result['battles_per_sec']:.2f} battles/sec")

    # Verify zero dropped battles
    assert result["random_finished"] == args.n_battles, "Some RandomPlayer battles did not finish."
    assert result["max_power_finished"] == args.n_battles, "Some MaxBasePowerPlayer battles did not finish."


if __name__ == "__main__":
    main()
