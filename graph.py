from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import sqlite_vec

OPS = {"=", "!=", ">", ">=", "<", "<=", "LIKE", "IN", "COSINE_GTE"}


@dataclass(frozen=True)
class Pred:
    target: str
    ref: str
    prop: str
    op: str
    value: Any


class _ExprOps:
    def _bin(self, op, other):
        return Expr(op, (self, other))

    def _rbin(self, op, other):
        return Expr(op, (other, self))

    def concat(self, *xs):
        return Expr("concat", (self, *xs))

    def lower(self):
        return Expr("lower", (self,))

    def upper(self):
        return Expr("upper", (self,))

    def strip(self):
        return Expr("strip", (self,))

    def replace(self, old, new):
        return Expr("replace", (self, old, new))

    def strlen(self):
        return Expr("strlen", (self,))


@dataclass(frozen=True)
class Expr(_ExprOps):
    op: str
    args: tuple[Any, ...]


for _name, _op in (
    ("__add__", "add"),
    ("__sub__", "sub"),
    ("__mul__", "mul"),
    ("__truediv__", "div"),
    ("__floordiv__", "floordiv"),
    ("__mod__", "mod"),
    ("__pow__", "pow"),
):
    setattr(_ExprOps, _name, lambda s, o, _op=_op: s._bin(_op, o))
    setattr(_ExprOps, "__r" + _name[2:], lambda s, o, _op=_op: s._rbin(_op, o))
setattr(_ExprOps, "__neg__", lambda s: Expr("neg", (s,)))
setattr(_ExprOps, "__pos__", lambda s: Expr("pos", (s,)))


@dataclass(frozen=True)
class Field:
    target: str
    ref: str
    path: str

    def __getattr__(self, prop):
        if prop.startswith("_"):
            raise AttributeError(prop)
        return Field(self.target, self.ref, f"{self.path}.{prop}")

    def __getitem__(self, prop):
        return Field(self.target, self.ref, f"{self.path}.{prop}")

    def _pred(self, op, value):
        return Pred(self.target, self.ref, self.path, op, value)

    def in_(self, values):
        return self._pred("IN", list(values))

    def contains(self, value):
        return self._pred("LIKE", f"%{value}%")

    def startswith(self, value):
        return self._pred("LIKE", f"{value}%")

    def endswith(self, value):
        return self._pred("LIKE", f"%{value}")

    def similar(self, query, min_score=0.0):
        return self._pred("COSINE_GTE", {"query": _vec(query), "min": float(min_score)})


for _name, _op in (
    ("__eq__", "="),
    ("__ne__", "!="),
    ("__gt__", ">"),
    ("__ge__", ">="),
    ("__lt__", "<"),
    ("__le__", "<="),
):
    setattr(Field, _name, lambda s, v, _op=_op: s._pred(_op, v))


@dataclass(frozen=True)
class Var(_ExprOps):
    name: str

    def __getattr__(self, prop):
        if prop.startswith("_"):
            raise AttributeError(prop)
        return Field("node", self.name, prop)

    def __getitem__(self, prop):
        return Field("node", self.name, str(prop))


@dataclass(frozen=True)
class Edge:
    index: int

    def __getattr__(self, prop):
        if prop.startswith("_"):
            raise AttributeError(prop)
        return Field("edge", str(self.index), prop)

    def __getitem__(self, prop):
        return Field("edge", str(self.index), str(prop))


@dataclass(frozen=True)
class Const(_ExprOps):
    value: str


@dataclass(frozen=True)
class Command:
    lhs: tuple[Any, ...] = ()
    where_clauses: tuple[Any, ...] = ()
    limit: int | None = None
    random: bool = False
    rhs: tuple[Any, ...] | None = None
    namespace: str | None = None

    def where(self, *predicates):
        return replace(self, where_clauses=self.where_clauses + tuple(predicates))

    def rewrite(self, rhs, random=None, limit=None):
        if not isinstance(rhs, (list, tuple)):
            raise TypeError("rewrite(rhs) expects a list/tuple of rhs terms")
        return replace(
            self,
            rhs=tuple(rhs),
            random=self.random if random is None else bool(random),
            limit=self.limit if limit is None else int(limit),
        )


@dataclass(frozen=True)
class Namespace:
    name: str

    def __post_init__(self):
        object.__setattr__(self, "name", _normalize_ns(self.name))

    def edge(self, *nodes, rel="_", embedding=None, data=None, temp=False, **props):
        return edge(*nodes, rel=rel, embedding=embedding, data=data, temp=temp, **props)

    def node(self, node_id, embedding=None, data=None, temp=False, **props):
        return node(node_id, embedding=embedding, data=data, temp=temp, **props)

    def match(self, *lhs, limit=None, random=False):
        return replace(match(*lhs, limit=limit, random=random), namespace=self.name)

    def rewrite(self, *lhs, to, random=False, limit=None):
        return self.match(*lhs, limit=limit, random=random).rewrite(to)


class Result(list[dict[str, Any]]):
    def map(self, fn):
        return [fn(r) for r in self]

    def first(self, default=None):
        return self[0] if self else default

    def bindings(self, *keys, default=None):
        return _project([dict(r.get("bindings") or r) for r in self], keys, default)

    def edge_data(self, *keys, index=1, default=None):
        if index < 1:
            raise ValueError("index must be >= 1")
        rows = []
        for row in self:
            edges = row.get("hyperedges") or []
            edge_row = edges[index - 1] if index <= len(edges) else {}
            rows.append(dict((edge_row or {}).get("data") or {}))
        return _project(rows, keys, default)

    def rows(self, *keys, default=None):
        return _project([dict(r) for r in self], keys, default)


class _Engine:
    def __init__(self, db_path="graph.db"):
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        self.db.execute("PRAGMA temp_store = MEMORY")
        self.db.enable_load_extension(True)
        try:
            sqlite_vec.load(self.db)
        finally:
            self.db.enable_load_extension(False)
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS hyperedges(
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                namespace TEXT NOT NULL DEFAULT '',
                relation TEXT NOT NULL,
                nodes_json TEXT NOT NULL,
                arity INTEGER NOT NULL,
                node0 TEXT,
                node1 TEXT,
                node2 TEXT,
                data TEXT NOT NULL DEFAULT '{}',
                embedding TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS uq_hyper_ns ON hyperedges(namespace, relation, nodes_json);
            CREATE INDEX IF NOT EXISTS idx_hyper_ns_arity ON hyperedges(namespace, arity);
            CREATE INDEX IF NOT EXISTS idx_hyper_ns_rel_arity ON hyperedges(namespace, relation, arity);
            CREATE INDEX IF NOT EXISTS idx_hyper_ns_rel_arity_n0 ON hyperedges(namespace, relation, arity, node0);
            CREATE INDEX IF NOT EXISTS idx_hyper_ns_rel_n0_n1 ON hyperedges(namespace, relation, node0, node1);
            CREATE INDEX IF NOT EXISTS idx_hyper_ns_rel_n0 ON hyperedges(namespace, relation, node0);
            """
        )
        self.db.commit()

    @contextmanager
    def _tx(self, rollback=False):
        if self.db.in_transaction:
            if not rollback:
                yield
                return
            sp = f"s_{uuid.uuid4().hex[:8]}"
            self.db.execute(f"SAVEPOINT {sp}")
            try:
                yield
            finally:
                self.db.execute(f"ROLLBACK TO {sp}")
                self.db.execute(f"RELEASE {sp}")
            return

        self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            if rollback:
                self.db.rollback()
            else:
                self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def run(self, *commands, mem=None, temp=False):
        if not commands or not all(isinstance(c, Command) for c in commands):
            raise TypeError("run(command[, command2, ...], mem=...)")
        mem_terms = [_as_term(t, pattern=False) for t in _coerce_terms(mem)]
        rollback = bool(temp or mem_terms)
        out: list[Result] = []
        with self._tx(rollback=rollback):
            for cmd in commands:
                if mem_terms:
                    ns = cmd.namespace or ""
                    for term in mem_terms:
                        self._upsert_term(ns, term, {})
                rows = self._rewrite(cmd) if cmd.rhs is not None else self._match(cmd)
                out.append(Result(rows))
        return out[0] if len(out) == 1 else out

    def _compile_query(self, lhs, filters, limit, random_order, namespace):
        if not lhs:
            raise ValueError("match requires at least one lhs term")

        ns = namespace or ""
        where, params, var_col = [], [], {}
        from_sql = "FROM " + ", ".join(f"hyperedges h{i}" for i in range(1, len(lhs) + 1))

        for i, term in enumerate(lhs, 1):
            a = f"h{i}"
            where += [f"{a}.namespace = ?", f"{a}.arity = ?"]
            params += [ns, len(term["nodes"])]
            if term["relation"] != "_":
                where.append(f"{a}.relation = ?")
                params.append(term["relation"])
            where += [f"{a}.edge_id <> h{k}.edge_id" for k in range(1, i)]

            for j, tok in enumerate(term["nodes"]):
                col = f"{a}.node{j}" if j < 3 else f"json_extract({a}.nodes_json, '$[{j}]')"
                if tok["kind"] == "const":
                    where.append(f"{col} = ?")
                    params.append(tok["value"])
                else:
                    prev = var_col.get(tok["value"])
                    if prev is None:
                        var_col[tok["value"]] = col
                    else:
                        where.append(f"{col} = {prev}")

        for pred in filters:
            if pred["target"] == "node":
                ref = str(pred["ref"])
                col = var_col.get(ref)
                if col is None:
                    raise ValueError(f"unknown node variable in filter: {ref}")
                clause, vals = self._filter_sql(pred, "np.data", "np.embedding")
                where.append(
                    "EXISTS (SELECT 1 FROM hyperedges np "
                    "WHERE np.namespace = ? AND np.relation = '__node__' "
                    "AND np.arity = 1 "
                    f"AND np.node0 = {col} AND {clause})"
                )
                params.append(ns)
                params.extend(vals)
            else:
                idx = int(pred["ref"])
                if idx < 1 or idx > len(lhs):
                    raise ValueError("edge filter index out of range")
                clause, vals = self._filter_sql(pred, f"h{idx}.data", f"h{idx}.embedding")
                where.append(clause)
                params.extend(vals)

        vars_sorted = sorted(var_col)
        edge_cols = [f"e{i}" for i in range(1, len(lhs) + 1)]
        select = [f"h{i}.edge_id AS e{i}" for i in range(1, len(lhs) + 1)]
        select += [f"{var_col[v]} AS v_{v}" for v in vars_sorted]
        sql = f"SELECT {', '.join(select)} {from_sql} WHERE {' AND '.join(where)}"
        if random_order:
            sql += " ORDER BY RANDOM()"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return sql, params, vars_sorted, edge_cols

    def _filter_sql(self, pred, json_col, emb_col):
        if pred["op"] == "COSINE_GTE":
            if pred["prop"] != "embedding":
                raise ValueError("similar() is only supported on the embedding field")
            query = _vec_json(pred["value"]["query"])
            min_score = float(pred["value"].get("min", 0.0))
            return f"{emb_col} IS NOT NULL AND vec_distance_cosine({emb_col}, ?) <= ?", [query, 1.0 - min_score]

        op, path, val = pred["op"], f"$.{pred['prop']}", pred["value"]
        expr = f"json_extract({json_col}, ?)"
        if op == "IN":
            vals = list(val)
            if not vals:
                return "1 = 0", []
            return f"{expr} IN ({','.join(['?'] * len(vals))})", [path] + vals
        if val is None and op == "=":
            return f"{expr} IS NULL", [path]
        if val is None and op == "!=":
            return f"{expr} IS NOT NULL", [path]
        return f"{expr} {op} ?", [path, val]

    def _match(self, command):
        sql, params, vars_sorted, edge_cols = self._compile_query(
            [_as_term(t, pattern=True) for t in command.lhs],
            [_as_pred(p) for p in command.where_clauses],
            command.limit,
            command.random,
            command.namespace,
        )
        rows = self.db.execute(sql, params).fetchall()
        edge_ids = list(dict.fromkeys(int(r[c]) for r in rows for c in edge_cols))
        edge_by_id = self._fetch_edges(edge_ids)
        return [
            {
                "bindings": {v: str(r[f"v_{v}"]) for v in vars_sorted},
                "hyperedges": [edge_by_id[eid] for eid in (int(r[c]) for c in edge_cols) if eid in edge_by_id],
            }
            for r in rows
        ]

    def _rewrite(self, command):
        lhs = [_as_term(t, pattern=True) for t in command.lhs]
        rhs = [_as_term(t, pattern=bool(lhs)) for t in (command.rhs or ())]
        if command.limit is not None and command.limit < 1:
            raise ValueError("limit must be >= 1")
        if not lhs:
            return [self._emit(rhs, {}, command.namespace)] if rhs else []

        sql, params, vars_sorted, edge_cols = self._compile_query(
            lhs,
            [_as_pred(p) for p in command.where_clauses],
            1,
            command.random,
            command.namespace,
        )
        delete_sql = f"DELETE FROM hyperedges WHERE edge_id IN ({','.join(['?'] * len(edge_cols))})"

        out = []
        while command.limit is None or len(out) < command.limit:
            row = self.db.execute(sql, params).fetchone()
            if row is None:
                break
            edge_ids = [int(row[c]) for c in edge_cols]
            self.db.execute(delete_sql, edge_ids)
            env = {v: str(row[f"v_{v}"]) for v in vars_sorted}
            out.append(self._emit(rhs, env, command.namespace))
        return out

    def _emit(self, terms, env, namespace):
        ns = namespace or ""
        edges = [self._upsert_term(ns, term, env) for term in terms]
        return {"bindings": dict(env), "hyperedges": edges}

    def _upsert_term(self, namespace, term, env):
        nodes = []
        for tok in term["nodes"]:
            kind, value = tok["kind"], tok["value"]
            if kind == "const":
                nodes.append(str(value))
            elif kind == "expr":
                nodes.append(str(_eval(value, env)))
            else:
                name = str(value)
                env.setdefault(name, f"n_{uuid.uuid4().hex[:12]}")
                nodes.append(env[name])

        relation = term["relation"]
        data = _resolve(term["data"], env)
        emb = _resolve(term["embedding"], env) if term["embedding"] is not None else None
        edge_id = self._upsert_edge(namespace, relation, nodes, data, emb)
        return _edge_obj(edge_id, relation, nodes, data, emb)

    def _fetch_edges(self, edge_ids):
        if not edge_ids:
            return {}
        rows = self.db.execute(
            f"SELECT edge_id, relation, nodes_json, data, embedding FROM hyperedges WHERE edge_id IN ({','.join(['?'] * len(edge_ids))})",
            edge_ids,
        ).fetchall()
        out = {}
        for row in rows:
            out[int(row["edge_id"])] = _edge_obj(
                row["edge_id"],
                row["relation"],
                _json_load(row["nodes_json"], default=[]),
                _json_load(row["data"], default={}),
                _json_load(row["embedding"], default=None),
            )
        return out

    def _upsert_edge(self, namespace, relation, nodes, data, embedding):
        arity = len(nodes)
        row = self.db.execute(
            "INSERT INTO hyperedges(namespace, relation, nodes_json, arity, node0, node1, node2, data, embedding) VALUES (?,?,?,?,?,?,?,?,?) "
            "ON CONFLICT(namespace, relation, nodes_json) DO UPDATE SET "
            "arity=excluded.arity, node0=excluded.node0, node1=excluded.node1, node2=excluded.node2, "
            "data=excluded.data, embedding=excluded.embedding RETURNING edge_id",
            (
                namespace,
                relation,
                json.dumps(nodes, separators=(",", ":")),
                arity,
                nodes[0] if arity > 0 else None,
                nodes[1] if arity > 1 else None,
                nodes[2] if arity > 2 else None,
                json.dumps(dict(data or {}), separators=(",", ":")),
                None if embedding is None else _vec_json(embedding),
            ),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to upsert edge")
        return int(row["edge_id"])


def _project(rows, keys, default):
    return rows if not keys else [{k: row.get(k, default) for k in keys} for row in rows]


def _json_load(raw, default):
    return default if raw is None else json.loads(raw)


def _edge_obj(edge_id, relation, nodes, data, embedding):
    return {
        "edge_id": int(edge_id),
        "relation": relation,
        "nodes": list(nodes),
        "data": dict(data),
        "embedding": None if embedding is None else _vec(embedding),
    }


def _coerce_terms(raw):
    return [] if raw is None else (list(raw) if isinstance(raw, (list, tuple)) else [raw])


def _as_term(raw, pattern):
    if not isinstance(raw, dict):
        raise ValueError("term must be created with graph.edge(...) or graph.node(...)")
    nodes = list(raw["nodes"])
    if not nodes:
        raise ValueError("term nodes must be non-empty")
    return {
        "relation": str(raw["relation"]),
        "nodes": [_as_tok(tok, pattern) for tok in nodes],
        "data": dict(raw.get("data") or {}),
        "embedding": raw.get("embedding"),
    }


def _as_tok(tok, pattern):
    if isinstance(tok, Const):
        return {"kind": "const", "value": tok.value}
    if isinstance(tok, Var):
        return {"kind": "var" if pattern else "const", "value": tok.name}
    if isinstance(tok, Expr):
        if not pattern:
            raise ValueError("expression tokens are only supported when rewrite has a non-empty lhs")
        return {"kind": "expr", "value": tok}
    if pattern:
        if not isinstance(tok, str):
            raise ValueError("pattern tokens must be strings, Var, Const, or Expr")
        return {"kind": "var", "value": tok}
    return {"kind": "const", "value": str(tok)}


def _as_pred(item):
    if not isinstance(item, Pred):
        raise ValueError("where entries must be predicates built from Var/Edge fields")
    out = {
        "target": item.target.lower(),
        "ref": item.ref,
        "prop": item.prop,
        "op": item.op,
        "value": item.value,
    }
    if out["target"] not in {"node", "edge"} or out["op"] not in OPS:
        raise ValueError("invalid where predicate")
    return out


def _num(value):
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return float(text)


def _add(a, b):
    try:
        return _num(a) + _num(b)
    except Exception:
        return f"{a}{b}"


BIN_EVAL = {
    "add": _add,
    "sub": lambda a, b: _num(a) - _num(b),
    "mul": lambda a, b: _num(a) * _num(b),
    "div": lambda a, b: _num(a) / _num(b),
    "floordiv": lambda a, b: _num(a) // _num(b),
    "mod": lambda a, b: _num(a) % _num(b),
    "pow": lambda a, b: _num(a) ** _num(b),
    "replace": lambda a, b, c: str(a).replace(str(b), str(c)),
}
UN_EVAL = {
    "neg": lambda a: -_num(a),
    "pos": lambda a: +_num(a),
    "lower": lambda a: str(a).lower(),
    "upper": lambda a: str(a).upper(),
    "strip": lambda a: str(a).strip(),
    "strlen": lambda a: len(str(a)),
}


def _resolve(value, env):
    if isinstance(value, Expr):
        return _eval(value, env)
    if isinstance(value, Var):
        if value.name not in env:
            raise ValueError(f"unbound variable in expression: {value.name}")
        return env[value.name]
    if isinstance(value, Const):
        return str(value.value)
    if isinstance(value, (list, tuple)):
        return [_resolve(v, env) for v in value]
    if isinstance(value, dict):
        return {str(k): _resolve(v, env) for k, v in value.items()}
    return value


def _eval(expr, env):
    args = [_resolve(a, env) for a in expr.args]
    if expr.op == "concat":
        return "".join(str(v) for v in args)
    if expr.op in BIN_EVAL:
        return BIN_EVAL[expr.op](*args)
    if expr.op in UN_EVAL:
        if len(args) != 1:
            raise ValueError(f"expression op {expr.op} expects one arg")
        return UN_EVAL[expr.op](args[0])
    raise ValueError(f"unknown expression op: {expr.op}")


def _vec(value):
    raw = _json_load(value, default=None) if isinstance(value, str) else value
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("embedding must be a non-empty list/tuple of numbers")
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding values must be numeric") from exc


def _vec_json(value):
    return json.dumps(_vec(value), separators=(",", ":"))


def _normalize_ns(namespace):
    value = str(namespace).strip()
    if not value:
        raise ValueError("namespace must be non-empty")
    if ":" in value:
        raise ValueError("invalid namespace; ':' is reserved")
    return value


def normalize_ns(namespace):
    return _normalize_ns(namespace)


def validate_term(term):
    _as_term(term, pattern=False)
    return term


def vars(names):
    if not isinstance(names, str):
        raise TypeError("vars(names) expects a space-delimited string")
    return tuple(Var(name) for name in names.split() if name)


def const(value):
    return Const(str(value))


def on(index):
    return Edge(int(index))


def edge(*nodes, rel="_", embedding=None, data=None, temp=False, **props):
    if not nodes:
        raise ValueError("term requires at least one node")
    payload = dict(data or {})
    payload.update(props)
    return {
        "relation": str(rel),
        "nodes": tuple(nodes),
        "data": payload,
        "embedding": embedding,
        "temp": bool(temp),
    }


def node(node_id, embedding=None, data=None, temp=False, **props):
    return edge(node_id, rel="__node__", embedding=embedding, data=data, temp=temp, **props)


def match(*lhs, limit=None, random=False):
    return Command(lhs=tuple(lhs), limit=None if limit is None else int(limit), random=bool(random))


def rewrite(*lhs, to, random=False, limit=None):
    return match(*lhs, limit=limit, random=random).rewrite(to)


def ns(name):
    return Namespace(name)


_ENGINE: _Engine | None = None


def get_engine() -> _Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _Engine()
    return _ENGINE


@contextmanager
def using_engine(engine: _Engine | None):
    if engine is None:
        yield get_engine()
        return
    global _ENGINE
    prev = _ENGINE
    _ENGINE = engine
    try:
        yield engine
    finally:
        _ENGINE = prev


__all__ = [
    "_Engine",
    "vars",
    "const",
    "on",
    "edge",
    "node",
    "ns",
    "match",
    "rewrite",
    "Result",
    "normalize_ns",
    "validate_term",
    "get_engine",
    "using_engine",
]
