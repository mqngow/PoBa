# PoBa — Competitive Pokémon Showdown Deep RL Agent

**Author:** Magnus Lie (`vwe8wh`) — CS 4710 Final Project

PoBa trains a Deep Reinforcement Learning agent to play **Gen 9 Random Battles** (`gen9randombattle`) on Pokémon Showdown using `poke-env` and `sb3-contrib`'s `MaskablePPO`.

## Key Features

- **Opponent Move & Item Probability Tracking:** Uses public Smogon / Gen 9 Random Battle usage statistics to maintain posterior probabilities over unrevealed opponent moves and items instead of zero-filling unknowns.
- **Invalid Action Masking:** Masks illegal moves and switches (Choice locks, Encore, trapping, fainted teammates) before policy sampling via `MaskablePPO`.
- **Curriculum Warm-Up & Self-Play:** Trains first against built-in `poke-env` baselines (`RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`), then transitions to playing against historical checkpoints of itself.

## Repository Structure

```text
PoBa/
├── .gitignore
├── IMPLEMENTATION_PLAN.md
├── pyproject.toml
├── README.md
├── data/                   # Cached Smogon / Randbats usage stats (ignored raw/json)
├── scripts/                # CLI scripts for server setup, training, evaluation, ladder
├── src/
│   └── poba/
│       ├── __init__.py
│       ├── action_mask.py  # Discrete action mapping & legal action masks
│       ├── config.py       # Paths, battle format, and PPO hyperparameters
│       ├── env.py          # SinglesEnv & SingleAgentWrapper / ActionMasker
│       ├── evaluate.py     # Win-rate benchmarks, ablations, and ladder evaluation
│       ├── network.py      # Custom PyTorch feature extractor (embeddings + continuous)
│       ├── observation.py  # Battle state encoder (embed_battle)
│       ├── prob_tracker.py # Smogon / Randbats posterior belief tracker
│       ├── reward.py       # Shaped & sparse reward functions
│       └── self_play.py    # Frozen checkpoint opponent & self-play pool callback
└── tests/                  # Unit and integration tests
```

## Setup

Create a virtual environment and install the package in editable mode with development dependencies:

```bash
uv venv
# Activate: .venv\Scripts\activate (Windows) or source .venv/bin/activate (Linux/macOS)
uv pip install -e ".[dev]"
```

Or with standard `pip`:

```bash
python -m venv .venv
pip install -e ".[dev]"
```

## Roadmap

See [IMPLEMENTATION_PLAN.md](IMPLEMENTATION_PLAN.md) for the step-by-step build and verification plan.
