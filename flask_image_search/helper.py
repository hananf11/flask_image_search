"""Pure helpers for Flask-Image-Search.

Stateless utilities used across the package.  No Flask app context, no
runtime I/O -- everything here can be exercised in a unit test without
spinning up an app.
"""

import numpy as np
import torch
from sqlalchemy import case as sa_case
from sqlalchemy import column as sa_column
from sqlalchemy import literal


def extract_features(model, preprocess, image, device):
    """Run ``image`` through ``model`` and return the L2-normalised feature vector.

    Free function so offline tools (fixture generation, batch jobs) can produce
    vectors that match runtime exactly without instantiating ``ImageSearch``.
    """
    tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(tensor)[0].cpu().numpy()
    return feature / np.linalg.norm(feature)


def vector_table_name(tablename, namespace, suffix="vectors"):
    """Build the per-namespace sibling-table name used by every backend."""
    return f"{tablename}_{suffix}__{namespace}"


def encode_vector(vector, dim):
    """Validate and pack a 1-D vector to raw float32 bytes."""
    arr = np.asarray(vector, dtype=np.float32)
    if arr.shape != (dim,):
        raise ValueError(f"expected vector of shape ({dim},), got {arr.shape}")
    return arr.tobytes()


def case_from_results(column, results):
    """Build a SQL CASE expression mapping pks to distances.

    Used by every backend's ``distance_expr`` whenever it falls back to
    materialising search results into a CASE statement.  Returns None when
    there are no results so callers can skip ordering.
    """
    if isinstance(column, str):
        column = sa_column(column)
    whens = [(column == literal(pk), literal(float(d))) for pk, d in results]
    if not whens:
        return None
    return sa_case(*whens, else_=literal(float("inf")))


def rank_results(pks, dists, sorted=True, limit=None):
    """Sort/limit a (pks, dists) pair into the public results tuple shape."""
    if not pks:
        return ()
    dists = np.asarray(dists)
    n = len(dists)
    take_top = limit is not None and limit < n

    if take_top and sorted:
        # Top-k via partial sort (O(N)), then stable-sort the k survivors.
        top = np.argpartition(dists, limit)[:limit]
        order = top[np.argsort(dists[top], kind="stable")]
    elif take_top:
        order = np.argpartition(dists, limit)[:limit]
    elif sorted:
        order = np.argsort(dists, kind="stable")
    else:
        return tuple(zip(pks, dists.tolist()))

    out_pks = [pks[int(i)] for i in order]
    out_dists = dists[order].tolist()
    return tuple(zip(out_pks, out_dists))
