"""Vector storage backends for Flask-Image-Search.

Three implementations are auto-dispatched based on the SQLAlchemy dialect
and what's installed:

* :class:`GenericBackend` -- always works, brute-force L2 in NumPy.
* :class:`SqliteVecBackend` -- KNN inside SQLite via the ``sqlite-vec``
  extension; opt-in via ``pip install flask-image-search[sqlite]``.
* :class:`PgVectorBackend` -- native ``vector`` column in Postgres via
  ``pgvector``; opt-in via ``pip install flask-image-search[postgres]``.

Pass ``backend=`` to :class:`~flask_image_search.ImageSearch` to override
auto-selection.
"""

from types import SimpleNamespace

import numpy as np
from sqlalchemy import (
    Column,
    LargeBinary,
    Table,
    event,
    func,
    literal_column,
    select,
    text,
)
from sqlalchemy import (
    case as sa_case,
)
from sqlalchemy import (
    column as sa_column,
)

from flask_image_search.helper import encode_vector, rank_results, vector_table_name


class VectorBackend:
    """Storage backend for indexed feature vectors."""

    def register(self, image_search, model, dim):
        raise NotImplementedError

    def upsert(self, connection, model, pk, vector):
        raise NotImplementedError

    def delete(self, connection, model, pk):
        raise NotImplementedError

    def search(self, model, query_vector, sorted=True, limit=None):
        raise NotImplementedError

    def distance_expr(self, model, query_vector, column, limit=None):
        raise NotImplementedError

    def count_indexed(self, model):
        raise NotImplementedError

    def is_indexed(self, connection, model, pk):
        """Return True if pk already has a vector stored.

        Used by ImageSearch to skip re-indexing when replace=False. Backends
        that use INSERT OR REPLACE semantics can keep the default (False) and
        always let upsert run.
        """
        return False


class GenericBackend(VectorBackend):
    """Dialect-agnostic backend.

    Vectors are stored as raw float32 bytes in a LargeBinary column on a
    sibling table ``<tablename>_vectors__<namespace>(pk, embedding)``.
    Queries stream the corpus in chunks and run a brute-force L2 scan.
    """

    # Rows per streaming chunk during search. At dim=4096 float32, a chunk of
    # 4096 holds ~64 MB -- bounded regardless of corpus size. Override in a
    # subclass for constrained hosts (smaller) or fat corpora (larger).
    _CHUNK_SIZE = 4096

    def __init__(self):
        self._stores = {}

    def register(self, image_search, model, dim):
        tablename = model.__tablename__
        pk_attr = image_search.models[tablename].id
        pk_col = getattr(model, pk_attr)
        pk_type = pk_col.property.columns[0].type.copy()

        metadata = model.metadata
        vec_tablename = vector_table_name(tablename, image_search.namespace)

        if vec_tablename in metadata.tables:
            table = metadata.tables[vec_tablename]
        else:
            table = Table(
                vec_tablename,
                metadata,
                Column("pk", pk_type, primary_key=True),
                Column("embedding", LargeBinary, nullable=False),
            )

        self._stores[tablename] = SimpleNamespace(
            table=table,
            dim=dim,
            image_search=image_search,
            created=False,
        )
        self._ensure_created(tablename)

    def _ensure_created(self, tablename):
        store = self._stores[tablename]
        if store.created:
            return
        try:
            engine = store.image_search.db.engine
        except Exception:
            return
        store.table.create(bind=engine, checkfirst=True)
        store.created = True

    def upsert(self, connection, model, pk, vector):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        blob = encode_vector(vector, store.dim)
        table = store.table
        connection.execute(table.delete().where(table.c.pk == pk))
        connection.execute(table.insert().values(pk=pk, embedding=blob))

    def delete(self, connection, model, pk):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        connection.execute(store.table.delete().where(store.table.c.pk == pk))

    def search(self, model, query_vector, sorted=True, limit=None):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        query = np.asarray(query_vector, dtype=np.float32)

        pks = []
        dists = []
        chunk = np.empty((self._CHUNK_SIZE, store.dim), dtype=np.float32)

        with store.image_search.db.engine.connect() as conn:
            result = conn.execute(
                select(store.table.c.pk, store.table.c.embedding).order_by(store.table.c.pk)
            )
            for rows in iter(lambda: result.fetchmany(self._CHUNK_SIZE), []):
                n = len(rows)
                buf = chunk[:n] if n == self._CHUNK_SIZE else np.empty((n, store.dim), dtype=np.float32)
                for i, (_, blob) in enumerate(rows):
                    buf[i] = np.frombuffer(blob, dtype=np.float32)
                chunk_dists = np.linalg.norm(buf - query, axis=1)
                pks.extend(r[0] for r in rows)
                dists.extend(chunk_dists.tolist())

        return rank_results(pks, dists, sorted=sorted, limit=limit)

    def distance_expr(self, model, query_vector, column, limit=None):
        results = self.search(model, query_vector, sorted=limit is not None, limit=limit)

        if isinstance(column, str):
            column = sa_column(column)

        whens = []
        for pk, distance in results:
            if limit and len(whens) >= limit:
                break
            whens.append(
                (column == literal_column(_sql_literal(pk)), literal_column(str(distance)))
            )

        if not whens:
            return None
        return sa_case(*whens, else_=literal_column("9999"))

    def count_indexed(self, model):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        with store.image_search.db.engine.connect() as conn:
            return conn.execute(select(func.count()).select_from(store.table)).scalar() or 0

    def is_indexed(self, connection, model, pk):
        store = self._stores[model.__tablename__]
        self._ensure_created(model.__tablename__)
        return connection.execute(
            select(store.table.c.pk).where(store.table.c.pk == pk)
        ).first() is not None


def _sql_literal(value):
    if isinstance(value, (int, float)):
        return str(value)
    escaped = str(value).replace("'", "''")
    return f"'{escaped}'"


class SqliteVecBackend(VectorBackend):
    """SQLite backend using the ``sqlite-vec`` extension.

    Requires ``pip install sqlite-vec`` and a Python sqlite3 built with
    ``SQLITE_ENABLE_LOAD_EXTENSION`` (not available on all distributions --
    check with ``hasattr(sqlite3.connect(':memory:'), 'enable_load_extension')``).
    Vectors are stored in a ``vec0`` virtual table
    ``<tablename>_vec__<namespace>(pk, embedding float[dim])``.
    KNN search is delegated entirely to SQLite -- no numpy matrix is loaded
    into memory.
    """

    def __init__(self):
        self._stores = {}
        self._engines_patched = set()

    @staticmethod
    def available():
        """Return True if sqlite-vec can be loaded on this Python build."""
        import sqlite3

        import sqlite_vec
        con = sqlite3.connect(":memory:")
        if not hasattr(con, "enable_load_extension"):
            return False
        try:
            con.enable_load_extension(True)
            sqlite_vec.load(con)
            return True
        except Exception:
            return False
        finally:
            con.close()

    def _patch_engine(self, engine):
        if id(engine) in self._engines_patched:
            return
        import sqlite_vec

        @event.listens_for(engine, "connect")
        def _load_ext(dbapi_conn, _record):
            dbapi_conn.enable_load_extension(True)
            sqlite_vec.load(dbapi_conn)
            dbapi_conn.enable_load_extension(False)

        self._engines_patched.add(id(engine))

    def register(self, image_search, model, dim):
        tablename = model.__tablename__
        pk_attr = image_search.models[tablename].id
        pk_col = getattr(model, pk_attr)
        pk_type = pk_col.property.columns[0].type.copy()

        self._stores[tablename] = SimpleNamespace(
            dim=dim,
            vec_tablename=f"{tablename}_vec__{image_search.namespace}",
            image_search=image_search,
            pk_type=pk_type,
            created=False,
        )
        self._ensure_created(tablename)

    def _ensure_created(self, tablename):
        store = self._stores[tablename]
        if store.created:
            return
        try:
            engine = store.image_search.db.engine
        except Exception:
            return
        self._patch_engine(engine)
        conn = store.image_search.db.session.connection()
        conn.execute(text(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS {store.vec_tablename} "
            f"USING vec0(pk INTEGER PRIMARY KEY, embedding float[{store.dim}])"
        ))
        store.created = True

    def _serialize(self, vector):
        import sqlite_vec
        return sqlite_vec.serialize_float32(np.asarray(vector, dtype=np.float32).tolist())

    def upsert(self, connection, model, pk, vector):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        blob = self._serialize(vector)
        connection.execute(
            text(f"DELETE FROM {store.vec_tablename} WHERE pk = :pk"), {"pk": pk}
        )
        connection.execute(
            text(f"INSERT INTO {store.vec_tablename}(pk, embedding) VALUES (:pk, :emb)"),
            {"pk": pk, "emb": blob},
        )

    def delete(self, connection, model, pk):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        connection.execute(
            text(f"DELETE FROM {store.vec_tablename} WHERE pk = :pk"), {"pk": pk}
        )

    def search(self, model, query_vector, sorted=True, limit=None):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        blob = self._serialize(query_vector)
        conn = store.image_search.db.session.connection()

        if limit is not None:
            # Fast KNN path -- sqlite-vec uses its ANN index. `distance` here is
            # squared L2; sqrt it on the way out so results match GenericBackend
            # and PgVectorBackend (true L2). Ordering stays on raw `distance`
            # since sqrt is monotonic -- same ranking, cheaper sort.
            rows = conn.execute(
                text(
                    f"SELECT pk, sqrt(distance) FROM {store.vec_tablename} "
                    f"WHERE embedding MATCH :q ORDER BY distance LIMIT :lim"
                ),
                {"q": blob, "lim": limit},
            ).fetchall()
        else:
            order_clause = "ORDER BY distance" if sorted else ""
            rows = conn.execute(
                text(
                    f"SELECT pk, vec_distance_L2(embedding, :q) AS distance "
                    f"FROM {store.vec_tablename} {order_clause}"
                ),
                {"q": blob},
            ).fetchall()

        return tuple((row[0], float(row[1])) for row in rows)

    def distance_expr(self, model, query_vector, column, limit=None):
        results = self.search(model, query_vector, sorted=limit is not None, limit=limit)

        if isinstance(column, str):
            column = sa_column(column)

        whens = []
        for pk, distance in results:
            if limit and len(whens) >= limit:
                break
            whens.append(
                (column == literal_column(_sql_literal(pk)), literal_column(str(distance)))
            )

        if not whens:
            return None
        return sa_case(*whens, else_=literal_column("9999"))

    def count_indexed(self, model):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        conn = store.image_search.db.session.connection()
        row = conn.execute(text(f"SELECT COUNT(*) FROM {store.vec_tablename}")).fetchone()
        return row[0]


class PgVectorBackend(VectorBackend):
    """PostgreSQL backend using the ``pgvector`` extension.

    Requires ``pip install pgvector`` and the ``vector`` extension installed
    in the database (``CREATE EXTENSION IF NOT EXISTS vector``).  Vectors are
    stored in a sibling table ``<tablename>_vectors__<namespace>(pk, embedding vector(dim))``.
    Distance ordering uses pgvector's ``<->`` L2 operator, which benefits from
    IVFFlat / HNSW indexes on large corpora.
    """

    def __init__(self):
        self._stores = {}

    def register(self, image_search, model, dim):
        from pgvector.sqlalchemy import Vector

        tablename = model.__tablename__
        pk_attr = image_search.models[tablename].id
        pk_col = getattr(model, pk_attr)
        pk_type = pk_col.property.columns[0].type.copy()

        metadata = model.metadata
        vec_tablename = vector_table_name(tablename, image_search.namespace)

        if vec_tablename in metadata.tables:
            table = metadata.tables[vec_tablename]
        else:
            table = Table(
                vec_tablename,
                metadata,
                Column("pk", pk_type, primary_key=True),
                Column("embedding", Vector(dim), nullable=False),
            )

        self._stores[tablename] = SimpleNamespace(
            table=table,
            dim=dim,
            image_search=image_search,
            created=False,
        )
        self._ensure_created(tablename)

    def _ensure_created(self, tablename):
        store = self._stores[tablename]
        if store.created:
            return
        try:
            engine = store.image_search.db.engine
        except Exception:
            return
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            conn.commit()
        store.table.create(bind=engine, checkfirst=True)
        store.created = True

    def upsert(self, connection, model, pk, vector):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        vec = np.asarray(vector, dtype=np.float32).tolist()
        table = store.table
        connection.execute(table.delete().where(table.c.pk == pk))
        connection.execute(table.insert().values(pk=pk, embedding=vec))

    def delete(self, connection, model, pk):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        table = store.table
        connection.execute(table.delete().where(table.c.pk == pk))

    def search(self, model, query_vector, sorted=True, limit=None):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        vec = np.asarray(query_vector, dtype=np.float32).tolist()
        table = store.table

        dist_expr = table.c.embedding.op("<->")(text(f"'{vec}'::vector"))
        q = select(table.c.pk, dist_expr.label("distance"))
        if sorted:
            q = q.order_by("distance")
        if limit is not None:
            q = q.limit(limit)

        conn = store.image_search.db.session.connection()
        rows = conn.execute(q).fetchall()
        return tuple((row[0], float(row[1])) for row in rows)

    def distance_expr(self, model, query_vector, column, limit=None):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        vec = np.asarray(query_vector, dtype=np.float32).tolist()
        table = store.table

        if isinstance(column, str):
            column = sa_column(column)

        # Correlated subquery: evaluated per-row by PostgreSQL, benefits from
        # pgvector index when used with LIMIT in the outer query.
        dist = (
            select(table.c.embedding.op("<->")(text(f"'{vec}'::vector")))
            .where(table.c.pk == column)
            .scalar_subquery()
        )

        if limit is not None:
            # Fall back to CASE to honour the limit contract
            results = self.search(model, query_vector, sorted=True, limit=limit)
            whens = [
                (column == literal_column(_sql_literal(pk)), literal_column(str(distance)))
                for pk, distance in results
            ]
            if not whens:
                return None
            return sa_case(*whens, else_=literal_column("9999"))

        return dist

    def count_indexed(self, model):
        self._ensure_created(model.__tablename__)
        store = self._stores[model.__tablename__]
        conn = store.image_search.db.session.connection()
        row = conn.execute(select(func.count()).select_from(store.table)).fetchone()
        return row[0]


def auto_backend(dialect_name):
    """Pick the best available backend for ``dialect_name``."""
    if dialect_name == "sqlite":
        try:
            import sqlite_vec  # noqa: F401
            if SqliteVecBackend.available():
                return SqliteVecBackend()
        except ImportError:
            pass
    elif dialect_name == "postgresql":
        try:
            from pgvector.sqlalchemy import Vector  # noqa: F401
            return PgVectorBackend()
        except ImportError:
            pass
    return GenericBackend()
