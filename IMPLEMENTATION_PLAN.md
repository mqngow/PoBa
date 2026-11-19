# PoBa Implementation Plan

**Project:** CS 4710 Final Project — Deep Reinforcement Learning for Competitive Pokémon Showdown via `poke-env`  
**Author:** Magnus Lie (`vwe8wh`)  
**Format:** Gen 9 Random Battles (`gen9randombattle`)  
**Core Stack:** `poke-env`, `sb3-contrib` (`MaskablePPO`), `stable-baselines3`, `gymnasium`, `PyTorch`

This plan breaks the system into **6 sequential, verifiable steps** so each component can be built, tested, and debugged in isolation before moving to the next.

---

## Step 1: Local Showdown Server & Baseline Connection

- **Targets:** `src/poba/config.py`, `scripts/` (server helper & connectivity check)
- **Goal:**
  - Clone and run a local headless Pokémon Showdown server on `localhost:8000` with `--no-security` enabled for fast WebSocket simulation.
  - Centralize project paths, `gen9randombattle` battle format constants, server URL configuration, and baseline PPO hyperparameters in `src/poba/config.py`.
  - Verify `poke-env` connectivity and benchmark battle simulation throughput locally.
- **Deliverable & Verification:**
  - A setup/benchmark script and test that runs 100 fast local `gen9randombattle` matches between `poke-env`'s built-in `RandomPlayer` and `MaxBasePowerPlayer` with zero connection drops.

---

## Step 2: Smogon / Randbats Probability Tracker

- **Targets:** `src/poba/prob_tracker.py`, `scripts/fetch_smogon_stats.py`, `data/`
- **Goal:**
  - Fetch and parse public Gen 9 Random Battles set/usage statistics from Smogon / Pokémon Showdown data sources into `data/`.
  - Implement a belief-state tracker that takes an opponent's species and observed reveals (moves used, held item triggered, ability revealed) and computes posterior probabilities over unrevealed moves and items instead of zero-filling unknowns.
  - Include an ablation toggle (e.g., `use_prob_tracker: bool`) that outputs zero vectors for unknown opponent moves/items when disabled.
- **Deliverable & Verification:**
  - Unit tests confirming valid prior distributions for sample species, Bayesian/conditional updates as moves/items are revealed mid-battle, and zeroed outputs when the ablation toggle is off.

---

## Step 3: Environment, State Vector, Action Masking & Rewards

- **Targets:** `src/poba/observation.py`, `src/poba/action_mask.py`, `src/poba/reward.py`, `src/poba/env.py`
- **Goal:**
  - **State Space (`observation.py`):** Implement `embed_battle(battle)` encoding:
    - Active Pokémon (species ID, types, base stats, stat boosts, status condition, HP%).
    - Team summary (HP%, alive/fainted flags, and status for all 6 slots).
    - Revealed opponent Pokémon state combined with posterior move/item probabilities from `prob_tracker.py`.
    - Field conditions (entry hazards, weather, terrain, Trick Room, Terastallization availability).
  - **Action Space & Masking (`action_mask.py`):** Map discrete actions (moves/Terastallization + switches) and compute boolean legal action masks from `battle.available_moves` and `battle.available_switches` to block impossible plays (Choice locks, Encore, Struggle, trapping, fainted teammates).
  - **Reward Function (`reward.py`):** Implement a configurable reward calculator supporting both:
    - *Shaped rewards*: turn-by-turn HP differential, KO/faint differential, status bonuses, and terminal `+1` win / `-1` loss.
    - *Sparse rewards* (ablation mode): terminal `+1` win / `-1` loss only.
  - **Gymnasium Environment (`env.py`):** Subclass `poke-env`'s `SinglesEnv`, wrap with `SingleAgentWrapper`, and expose `action_masks()` (via `sb3_contrib.common.wrappers.ActionMasker`) for `MaskablePPO`.
- **Deliverable & Verification:**
  - `gymnasium` space checks (`observation_space`, `action_space`, mask shape/dtype) and a 200-battle random legal rollout test verifying zero illegal action exceptions.

---

## Step 4: Actor-Critic Network & MaskablePPO Setup

- **Targets:** `src/poba/network.py`, `scripts/train.py`
- **Goal:**
  - Build a custom PyTorch `BaseFeaturesExtractor` with `nn.Embedding` layers for categorical features (Pokémon species IDs, move IDs/types) concatenated with normalized continuous features (HP%, stat modifiers, probability vectors).
  - Wire the feature extractor into `sb3-contrib`'s `MaskablePPO` (using `MultiInputPolicy` or custom `MaskableActorCriticPolicy`) with TensorBoard logging.
- **Deliverable & Verification:**
  - PyTorch forward-pass unit tests verifying output shapes and gradient flow, plus a short smoke-test training run (e.g., 2,048 environment steps) confirming TensorBoard loss/reward logs and checkpoint saving/loading.

---

## Step 5: Curriculum Warm-Up & Step-by-Step Self-Play

- **Targets:** `src/poba/self_play.py`, `scripts/train.py`
- **Goal:**
  - **Phase A (Warm-Up):** Train the masked PPO agent progressively against `poke-env` heuristic baselines: `RandomPlayer` → `MaxBasePowerPlayer` → `SimpleHeuristicsPlayer`.
  - **Phase B (Self-Play):** Implement `CheckpointOpponent` (a `poke-env` `Player` running frozen `MaskablePPO` policy checkpoints with action masking) and a training callback that periodically snapshots model weights and samples opponents from a historical checkpoint pool (e.g., 80% past checkpoints, 20% heuristic baselines to prevent forgetting).
- **Deliverable & Verification:**
  - Automated tests verifying checkpoint snapshotting, `CheckpointOpponent` inference with legal action masks, and smooth curriculum phase transitions.

---

## Step 6: Evaluation, Ablations & Live Ladder Testing

- **Targets:** `src/poba/evaluate.py`, `scripts/evaluate.py`, `scripts/play_ladder.py`
- **Goal:**
  - **Win-Rate Benchmarks:** Run 1,000-battle evaluation suites against `RandomPlayer`, `MaxBasePowerPlayer`, `SimpleHeuristicsPlayer`, and saved self-play checkpoints.
  - **Ablation Experiments:**
    1. *Smogon Probability Tracker* vs. *Zero-Filled Unknowns*.
    2. *Shaped Turn-by-Turn Rewards* vs. *Sparse Win/Loss-Only Rewards*.
  - **Live Ladder Evaluation:** Connect the final agent to the public Pokémon Showdown server (`sim3.psim.us`, `gen9randombattle`), play 50+ live rated games, record ELO progression, and plot win-rate / ELO curves with `matplotlib`.
- **Deliverable & Verification:**
  - Evaluation CLI script generating reproducible CSV/JSON metric tables and publication-ready graphs in `results/`.
