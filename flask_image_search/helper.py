"""Pure helpers for Flask-Image-Search.

Stateless utilities used across the package.  No Flask app context, no
runtime I/O -- everything here can be exercised in a unit test without
spinning up an app.
"""

import hashlib
import re

import numpy as np
from sqlalchemy import case as sa_case
from sqlalchemy import column as sa_column
from sqlalchemy import literal

_NAMESPACE_SAFE = re.compile(r"[^A-Za-z0-9_]+")


def derive_namespace(model, dim=None):
    """Derive a stable, table-name-safe namespace from a torch.nn.Module.

    Any change to architecture or output dimension flows through to a different
    namespace, so switching backbones never silently reuses stale vectors.
    ``str(model)`` produces a stable layer-dump that changes with architecture.
    """
    if model is None:
        return "default"

    name = model.__class__.__name__.lower()
    short = hashlib.sha1(str(model).encode("utf-8")).hexdigest()[:8]
    raw = f"{name}_{dim or 'x'}_{short}"
    return _NAMESPACE_SAFE.sub("_", raw).strip("_") or "default"


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

    if sorted:
        order = np.argsort(np.asarray(dists), kind="stable")
    else:
        order = range(len(dists))

    results = tuple((pks[i], float(dists[i])) for i in order)
    if limit is not None:
        results = results[:limit]
    return results
