"""Smogon / Gen 9 Random Battles opponent move and item probability tracker."""

import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from poke_env.data import to_id_str

from poba.config import RANDBATS_STATS_PATH


@dataclass(frozen=True)
class _RoleStats:
    """Normalized set probabilities for one randbats role of a species."""

    weight: float
    moves: dict[str, float]
    items: dict[str, float]
    abilities: dict[str, float]


def _clean_optional(value: str | None) -> str | None:
    """Normalize an optional item or ability string, returning None if unknown."""
    if not value:
        return None
    normalized = to_id_str(value)
    # poke-env may represent unrevealed items/abilities as empty, 'none', or 'unknownitem'
    if normalized in {"", "unknownitem", "none"}:
        return None
    return normalized


class ProbTracker:
    """Tracks posterior probabilities over unrevealed opponent moves and items."""

    def __init__(self, stats: dict, enabled: bool = True) -> None:
        # When enabled=False (ablation mode), unrevealed moves/items are zero-filled
        self.enabled = enabled
        self._species_roles: dict[str, list[_RoleStats]] = {}

        all_moves: set[str] = set()
        all_items: set[str] = set()

        # Parse each species and its candidate roles using normalized poke-env IDs
        for species_name, species_data in stats.items():
            species_id = to_id_str(species_name)
            roles_list: list[_RoleStats] = []
            for role_data in species_data.get("roles", {}).values():
                moves = {to_id_str(m): float(p) for m, p in role_data.get("moves", {}).items()}
                items = {to_id_str(i): float(p) for i, p in role_data.get("items", {}).items()}
                abilities = {
                    to_id_str(a): float(p) for a, p in role_data.get("abilities", {}).items()
                }
                weight = float(role_data.get("weight", 1.0))

                roles_list.append(
                    _RoleStats(weight=weight, moves=moves, items=items, abilities=abilities)
                )
                all_moves.update(moves.keys())
                all_items.update(items.keys())

            if roles_list:
                self._species_roles[species_id] = roles_list

        # Build deterministic sorted vocabularies for fixed-length vector encoding
        self.move_to_idx: dict[str, int] = {m: idx for idx, m in enumerate(sorted(all_moves))}
        self.item_to_idx: dict[str, int] = {i: idx for idx, i in enumerate(sorted(all_items))}

    @property
    def move_vocab_size(self) -> int:
        """Number of unique moves in the randbats vocabulary."""
        return len(self.move_to_idx)

    @property
    def item_vocab_size(self) -> int:
        """Number of unique items in the randbats vocabulary."""
        return len(self.item_to_idx)

    @classmethod
    def from_file(cls, path: Path = RANDBATS_STATS_PATH, enabled: bool = True) -> "ProbTracker":
        """Load ProbTracker from a cached randbats stats JSON file."""
        if not path.exists():
            raise FileNotFoundError(
                f"Randbats stats file not found at {path}. "
                "Run `python scripts/fetch_smogon_stats.py` first."
            )
        with path.open("r", encoding="utf-8") as f:
            stats = json.load(f)
        return cls(stats=stats, enabled=enabled)

    def _role_posteriors(
        self,
        species_id: str,
        revealed_moves: set[str],
        revealed_item: str | None,
        revealed_ability: str | None,
    ) -> list[tuple[_RoleStats, float]]:
        """Compute Bayesian posterior weights across a species' candidate roles."""
        roles = self._species_roles.get(species_id)
        if not roles:
            return []

        unnormalized: list[float] = []
        for role in roles:
            # Start from role prior weight and multiply likelihoods of observed reveals
            likelihood = role.weight
            for move_id in revealed_moves:
                likelihood *= role.moves.get(move_id, 0.0)
            if revealed_item is not None:
                likelihood *= role.items.get(revealed_item, 0.0)
            if revealed_ability is not None:
                likelihood *= role.abilities.get(revealed_ability, 0.0)
            unnormalized.append(likelihood)

        total = sum(unnormalized)
        if total > 0.0:
            return [(role, w / total) for role, w in zip(roles, unnormalized, strict=True)]

        # Fallback to normalized priors if an unexpected reveal (e.g., Trick) zeroes all roles
        prior_total = sum(role.weight for role in roles)
        if prior_total > 0.0:
            return [(role, role.weight / prior_total) for role in roles]
        uniform = 1.0 / len(roles)
        return [(role, uniform) for role in roles]

    def predict_moves(
        self,
        species: str,
        revealed_moves: Iterable[str] = (),
        revealed_item: str | None = None,
        revealed_ability: str | None = None,
    ) -> dict[str, float]:
        """Return move -> probability in [0, 1] given revealed opponent information."""
        rev = {to_id_str(m) for m in revealed_moves if m}
        # In ablation mode or once all 4 moves are known, unknown moves have probability 0
        if not self.enabled or len(rev) >= 4:
            return {m: 1.0 for m in rev}

        species_id = to_id_str(species)
        item_id = _clean_optional(revealed_item)
        ability_id = _clean_optional(revealed_ability)
        posteriors = self._role_posteriors(species_id, rev, item_id, ability_id)

        # Revealed moves always have certainty 1.0
        probs: dict[str, float] = {m: 1.0 for m in rev}
        for role, post_weight in posteriors:
            if post_weight <= 0.0:
                continue
            for move_id, move_prob in role.moves.items():
                if move_id not in rev:
                    probs[move_id] = min(1.0, probs.get(move_id, 0.0) + post_weight * move_prob)
        return probs

    def predict_items(
        self,
        species: str,
        revealed_moves: Iterable[str] = (),
        revealed_item: str | None = None,
        revealed_ability: str | None = None,
    ) -> dict[str, float]:
        """Return item -> probability in [0, 1] given revealed opponent information."""
        item_id = _clean_optional(revealed_item)
        # If the item has already been revealed, it is known with certainty 1.0
        if item_id is not None:
            return {item_id: 1.0}
        # In ablation mode, unknown items are left empty (zero vector)
        if not self.enabled:
            return {}

        rev = {to_id_str(m) for m in revealed_moves if m}
        species_id = to_id_str(species)
        ability_id = _clean_optional(revealed_ability)
        posteriors = self._role_posteriors(species_id, rev, None, ability_id)

        probs: dict[str, float] = {}
        for role, post_weight in posteriors:
            if post_weight <= 0.0:
                continue
            for candidate_item, item_prob in role.items.items():
                probs[candidate_item] = min(
                    1.0, probs.get(candidate_item, 0.0) + post_weight * item_prob
                )
        return probs

    def get_move_vector(
        self,
        species: str,
        revealed_moves: Iterable[str] = (),
        revealed_item: str | None = None,
        revealed_ability: str | None = None,
    ) -> np.ndarray:
        """Encode predicted move probabilities into a fixed-length float32 vector."""
        vec = np.zeros(self.move_vocab_size, dtype=np.float32)
        for move_id, prob in self.predict_moves(
            species, revealed_moves, revealed_item, revealed_ability
        ).items():
            idx = self.move_to_idx.get(move_id)
            if idx is not None:
                vec[idx] = prob
        return vec

    def get_item_vector(
        self,
        species: str,
        revealed_moves: Iterable[str] = (),
        revealed_item: str | None = None,
        revealed_ability: str | None = None,
    ) -> np.ndarray:
        """Encode predicted item probabilities into a fixed-length float32 vector."""
        vec = np.zeros(self.item_vocab_size, dtype=np.float32)
        for item_id, prob in self.predict_items(
            species, revealed_moves, revealed_item, revealed_ability
        ).items():
            idx = self.item_to_idx.get(item_id)
            if idx is not None:
                vec[idx] = prob
        return vec

    def get_pokemon_vectors(self, pokemon) -> tuple[np.ndarray, np.ndarray]:
        """Extract move and item probability vectors directly from a poke-env Pokemon."""
        revealed_moves = pokemon.moves.keys() if getattr(pokemon, "moves", None) else ()
        return (
            self.get_move_vector(
                species=pokemon.species,
                revealed_moves=revealed_moves,
                revealed_item=getattr(pokemon, "item", None),
                revealed_ability=getattr(pokemon, "ability", None),
            ),
            self.get_item_vector(
                species=pokemon.species,
                revealed_moves=revealed_moves,
                revealed_item=getattr(pokemon, "item", None),
                revealed_ability=getattr(pokemon, "ability", None),
            ),
        )
