"""Tests for pure helpers in ``flask_image_search.helper``.

Distance values use float32-exact fractions (sums of negative powers of two)
so that ``float(np.float32(x)) == x`` and equality comparisons are safe.

Coverage targets the four (sorted, take_top) cells in ``rank_results``:

    sorted=T, limit<n  → argpartition + stable argsort
    sorted=T, limit>=n → full argsort
    sorted=F, limit<n  → argpartition only (arbitrary k)
    sorted=F, no limit → identity early-return
"""

import numpy as np
from flask_image_search.helper import rank_results


def test_rank_results_empty():
    assert rank_results([], np.empty(0, dtype=np.float32)) == ()


# sorted=T, no limit → full argsort
def test_rank_results_sorts_by_distance():
    pks = [1, 2, 3]
    dists = np.array([0.5, 0.25, 0.75], dtype=np.float32)
    assert rank_results(pks, dists) == ((2, 0.25), (1, 0.5), (3, 0.75))


# sorted=T, limit>=n → full argsort, then no trimming
def test_rank_results_limit_larger_than_input():
    pks = [1, 2]
    dists = np.array([0.5, 0.25], dtype=np.float32)
    assert rank_results(pks, dists, limit=10) == ((2, 0.25), (1, 0.5))


# sorted=T, limit<n → argpartition + stable argsort
def test_rank_results_top_k_sorted():
    pks = [1, 2, 3, 4]
    dists = np.array([0.5, 0.125, 0.75, 0.25], dtype=np.float32)
    assert rank_results(pks, dists, limit=2) == ((2, 0.125), (4, 0.25))


# sorted=T → stable: equal distances preserve input order
def test_rank_results_stable_sort_for_ties():
    pks = [10, 20, 30]
    dists = np.array([0.5, 0.5, 0.125], dtype=np.float32)
    assert rank_results(pks, dists) == ((30, 0.125), (10, 0.5), (20, 0.5))


# sorted=F, no limit → identity
def test_rank_results_unsorted_preserves_order():
    pks = [1, 2, 3]
    dists = np.array([0.5, 0.25, 0.75], dtype=np.float32)
    assert rank_results(pks, dists, sorted=False) == (
        (1, 0.5),
        (2, 0.25),
        (3, 0.75),
    )


# sorted=F, limit<n → top-k by distance, order unspecified
def test_rank_results_top_k_unsorted_returns_smallest():
    pks = [1, 2, 3, 4]
    dists = np.array([0.5, 0.125, 0.75, 0.25], dtype=np.float32)
    results = rank_results(pks, dists, sorted=False, limit=2)
    assert len(results) == 2
    assert set(results) == {(2, 0.125), (4, 0.25)}


# Distances must be Python floats, not numpy scalars (sqlalchemy literal()
# in case_from_results expects native floats).
def test_rank_results_distances_are_python_floats():
    pks = [1]
    dists = np.array([0.5], dtype=np.float32)
    results = rank_results(pks, dists)
    assert isinstance(results[0][1], float)
    assert not isinstance(results[0][1], np.floating)


# Same float-type contract on the early-return (sorted=False) path.
def test_rank_results_unsorted_distances_are_python_floats():
    pks = [1, 2]
    dists = np.array([0.5, 0.25], dtype=np.float32)
    results = rank_results(pks, dists, sorted=False)
    assert all(
        isinstance(d, float) and not isinstance(d, np.floating) for _, d in results
    )
