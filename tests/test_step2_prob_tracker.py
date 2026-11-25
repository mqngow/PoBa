"""Step 2 unit tests for the Smogon / Gen 9 Randbats probability tracker."""

import numpy as np
import pytest

from poba.config import RANDBATS_STATS_PATH
from poba.prob_tracker import ProbTracker

# Synthetic 2-role fixture so unit tests run deterministically without network or cache
SAMPLE_STATS = {
    "Charizard": {
        "level": 85,
        "roles": {
            "Setup Sweeper": {
                "weight": 0.5,
                "abilities": {"Blaze": 1.0},
                "items": {"Heavy-Duty Boots": 1.0},
                "moves": {
                    "Dragon Dance": 1.0,
                    "Flare Blitz": 1.0,
                    "Earthquake": 1.0,
                    "Outrage": 0.5,
                },
            },
            "Fast Attacker": {
                "weight": 0.5,
                "abilities": {"Solar Power": 1.0},
                "items": {"Choice Specs": 1.0},
                "moves": {
                    "Flamethrower": 1.0,
                    "Hurricane": 1.0,
                    "Focus Blast": 0.8,
                    "Earthquake": 0.6,
                },
            },
        },
    }
}


def test_prior_distributions() -> None:
    """Verify prior move/item probabilities and numpy vector shapes/dtypes."""
    tracker = ProbTracker(SAMPLE_STATS, enabled=True)

    move_probs = tracker.predict_moves("Charizard")
    item_probs = tracker.predict_items("Charizard")

    # Priors should average across the two 0.5-weighted roles
    assert move_probs["dragondance"] == pytest.approx(0.5)
    assert move_probs["flamethrower"] == pytest.approx(0.5)
    assert move_probs["earthquake"] == pytest.approx(0.8)
    assert item_probs["heavydutyboots"] == pytest.approx(0.5)
    assert item_probs["choicespecs"] == pytest.approx(0.5)

    # Vector outputs should have matching vocab dimensions and float32 dtype
    move_vec = tracker.get_move_vector("charizard")
    item_vec = tracker.get_item_vector("charizard")
    assert move_vec.dtype == np.float32
    assert item_vec.dtype == np.float32
    assert move_vec.shape == (tracker.move_vocab_size,)
    assert item_vec.shape == (tracker.item_vocab_size,)
    assert move_vec[tracker.move_to_idx["earthquake"]] == pytest.approx(0.8)


def test_bayesian_update_on_revealed_move() -> None:
    """Revealing a role-exclusive move should collapse the posterior onto that role."""
    tracker = ProbTracker(SAMPLE_STATS, enabled=True)

    # Flamethrower only exists on Fast Attacker -> Setup Sweeper drops to 0
    move_probs = tracker.predict_moves("Charizard", revealed_moves=["Flamethrower"])
    item_probs = tracker.predict_items("Charizard", revealed_moves=["Flamethrower"])

    assert move_probs["flamethrower"] == pytest.approx(1.0)
    assert move_probs.get("dragondance", 0.0) == pytest.approx(0.0)
    assert move_probs["hurricane"] == pytest.approx(1.0)
    assert move_probs["focusblast"] == pytest.approx(0.8)
    assert move_probs["earthquake"] == pytest.approx(0.6)
    assert item_probs["choicespecs"] == pytest.approx(1.0)
    assert item_probs.get("heavydutyboots", 0.0) == pytest.approx(0.0)


def test_four_revealed_moves_zeroes_unknowns() -> None:
    """Once all 4 moves are revealed, unrevealed moves must have probability 0."""
    tracker = ProbTracker(SAMPLE_STATS, enabled=True)

    revealed = ["Flamethrower", "Hurricane", "Focus Blast", "Earthquake"]
    move_probs = tracker.predict_moves("Charizard", revealed_moves=revealed)

    assert set(move_probs.keys()) == {"flamethrower", "hurricane", "focusblast", "earthquake"}
    assert all(p == pytest.approx(1.0) for p in move_probs.values())


def test_revealed_item_and_unexpected_reveal_fallback() -> None:
    """Revealing an item updates role posteriors, and unexpected reveals fall back safely."""
    tracker = ProbTracker(SAMPLE_STATS, enabled=True)

    # Revealing Heavy-Duty Boots pins the role to Setup Sweeper
    move_probs = tracker.predict_moves("Charizard", revealed_item="Heavy-Duty Boots")
    item_probs = tracker.predict_items("Charizard", revealed_item="Heavy-Duty Boots")
    assert item_probs == {"heavydutyboots": 1.0}
    assert move_probs["dragondance"] == pytest.approx(1.0)
    assert move_probs.get("flamethrower", 0.0) == pytest.approx(0.0)

    # Out-of-set item (e.g., from Trick/Switcheroo) should fall back cleanly to prior
    fallback_moves = tracker.predict_moves("Charizard", revealed_item="Leftovers")
    assert fallback_moves["dragondance"] == pytest.approx(0.5)
    assert fallback_moves["flamethrower"] == pytest.approx(0.5)


def test_ablation_disabled_zeroes_unknowns() -> None:
    """When enabled=False, unrevealed moves and items must be zero-filled."""
    tracker = ProbTracker(SAMPLE_STATS, enabled=False)

    move_probs = tracker.predict_moves("Charizard", revealed_moves=["Flamethrower"])
    item_probs = tracker.predict_items("Charizard", revealed_moves=["Flamethrower"])
    assert move_probs == {"flamethrower": 1.0}
    assert item_probs == {}

    move_vec = tracker.get_move_vector("Charizard", revealed_moves=["Flamethrower"])
    item_vec = tracker.get_item_vector("Charizard")
    assert move_vec.sum() == pytest.approx(1.0)
    assert item_vec.sum() == pytest.approx(0.0)


def test_from_real_cached_stats_file() -> None:
    """Validate ProbTracker against the cached Gen 9 Random Battles JSON dataset."""
    if not RANDBATS_STATS_PATH.exists():
        pytest.skip("Cached randbats stats JSON not present; run scripts/fetch_smogon_stats.py.")

    tracker = ProbTracker.from_file(RANDBATS_STATS_PATH, enabled=True)
    assert tracker.move_vocab_size > 300
    assert tracker.item_vocab_size > 40

    # Check real Charizard predictions update coherently when Flamethrower is revealed
    prior_moves = tracker.predict_moves("Charizard")
    post_moves = tracker.predict_moves("Charizard", revealed_moves=["Flamethrower"])
    assert 0.0 < prior_moves["flamethrower"] < 1.0
    assert post_moves["flamethrower"] == pytest.approx(1.0)
    assert post_moves.get("dragondance", 0.0) == pytest.approx(0.0)
