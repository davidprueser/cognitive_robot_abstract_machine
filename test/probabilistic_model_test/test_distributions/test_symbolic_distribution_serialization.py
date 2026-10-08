from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from krrood.adapters.json_serializer import from_json, to_json

from .hash_seed_processes import load_distribution, save_distribution
from .hash_seed_processes.string_backed_category import (
    StringBackedCategory,
    string_backed_distribution,
)


# %% helpers
def _run_module_with_hash_seed(module_name: str, hash_seed: int, *arguments: str):
    """
    Run *module_name* in a fresh interpreter whose string hashes are seeded with
    *hash_seed*.

    :param module_name: The fully qualified module to run.
    :param hash_seed: The value of ``PYTHONHASHSEED`` for that interpreter.
    :param arguments: Command line arguments handed to the module.
    """
    repository_root = Path(__file__).resolve().parents[3]
    subprocess.run(
        [sys.executable, "-m", module_name, *arguments],
        env={**os.environ, "PYTHONHASHSEED": str(hash_seed)},
        cwd=repository_root,
        check=True,
    )


# %% likelihood of string-backed categories
def test_likelihood_for_string_backed_symbolic_category() -> None:
    """
    A :class:`~enum.StrEnum` member's hash routinely exceeds float64's exact integer
    range, unlike a small :class:`~enum.IntEnum`, so its fitted probability is recovered
    only if likelihoods compare hashes exactly.
    """
    distribution = string_backed_distribution()

    likelihoods = distribution.likelihood(
        np.array(
            [[StringBackedCategory.ALPHA], [StringBackedCategory.BETA]], dtype=object
        )
    )

    assert likelihoods.tolist() == [
        distribution.probabilities[hash(StringBackedCategory.ALPHA)],
        distribution.probabilities[hash(StringBackedCategory.BETA)],
    ]


# %% serialization
def test_symbolic_distribution_round_trips_its_probabilities() -> None:
    distribution = string_backed_distribution()

    restored = from_json(json.loads(json.dumps(to_json(distribution))))

    assert dict(restored.probabilities) == dict(distribution.probabilities)
    assert restored.variable == distribution.variable


def test_symbolic_distribution_survives_a_different_hash_seed_process(
    tmp_path: Path,
) -> None:
    """
    A distribution exported by one process deserializes to the same probabilities in a
    process with a different ``PYTHONHASHSEED``.

    Members of a :class:`~enum.StrEnum` hash through Python's randomized string hash, so
    only two separate processes with different seeds can expose a hash-keyed export.
    """
    export_path = tmp_path / "symbolic_distribution.json"
    result_path = tmp_path / "probabilities.json"

    _run_module_with_hash_seed(save_distribution.__name__, 1, str(export_path))
    _run_module_with_hash_seed(
        load_distribution.__name__, 2, str(export_path), str(result_path)
    )

    expected = string_backed_distribution()
    assert json.loads(result_path.read_text()) == {
        member.value: expected.probabilities[hash(member)]
        for member in StringBackedCategory
    }
