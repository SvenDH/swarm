from __future__ import annotations

import json
import os
import sqlite3
import uuid
from dataclasses import dataclass, replace
from typing import Any

OPS = {"=", "!=", ">", ">=", "<", "<=", "LIKE", "IN", "COSINE_GTE"}


@dataclass(frozen=True)
class Pred:
    target: str
    ref: str
    prop: str
    op: str
    value: Any


class _ExprOps:
    def _bin(self, op: str, other: Any) -> Expr:
        return Expr(op, (self, other))
    def _rbin(self, op: str, other: Any) -> Expr:
        return Expr(op, (other, self))
    def concat(self, *others: Any) -> Expr:
        return Expr("concat", (self, *others))
    def lower(self) -> Expr:
        return Expr("lower", (self,))
    def upper(self) -> Expr:
        return Expr("upper", (self,))
    def strip(self) -> Expr:
        return Expr("strip", (self,))
    def replace(self, old: Any, new: Any) -> Expr:
        return Expr("replace", (self, old, new))
    def strlen(self) -> Expr:
        return Expr("strlen", (self,))


@dataclass(frozen=True)
class Expr(_ExprOps):
    op: str
    args: tuple[Any, ...]


def _make_bin(op: str):
    return lambda self, other: self._bin(op, other)


def _make_rbin(op: str):
    return lambda self, other: self._rbin(op, other)


for _name, _op in (
    ("__add__", "add"),
    ("__sub__", "sub"),
    ("__mul__", "mul"),
    ("__truediv__", "div"),
    ("__floordiv__", "floordiv"),
    ("__mod__", "mod"),
    ("__pow__", "pow"),
):
    setattr(_ExprOps, _name, _make_bin(_op))
    setattr(_ExprOps, "__r" + _name[2:], _make_rbin(_op))

setattr(_ExprOps, "__neg__", lambda self: Expr("neg", (self,)))
setattr(_ExprOps, "__pos__", lambda self: Expr("pos", (self,)))


@dataclass(frozen=True)
class Field:
    target: str
    ref: str
    path: str

    def __getattr__(self, prop: str) -> Field:
        if prop.startswith("_"):
            raise AttributeError(prop)
        return Field(self.target, self.ref, f"{self.path}.{prop}")

    def __getitem__(self, prop: Any) -> Field:
        return Field(self.target, self.ref, f"{self.path}.{prop}")

    def _pred(self, op: str, value: Any) -> Pred:
        return Pred(self.target, self.ref, self.path, op, value)

    def in_(self, values: list[Any] | tuple[Any, ...] | set[Any]) -> Pred:
        return self._pred("IN", list(values))

    def contains(self, value: Any) -> Pred:
        return self._pred("LIKE", f"%{value}%")

    def startswith(self, value: Any) -> Pred:
        return self._pred("LIKE", f"{value}%")

    def endswith(self, value: Any) -> Pred:
        return self._pred("LIKE", f"%{value}")

    def similar(self, query: Any, min_score: float = 0.0) -> Pred:
        return self._pred("COSINE_GTE", {"query": _vec(query), "min": float(min_score)})


def _cmp(op: str):
    return lambda self, value: self._pred(op, value)


for _name, _op in (
    ("__eq__", "="),
    ("__ne__", "!="),
    ("__gt__", ">"),
    ("__ge__", ">="),
    ("__lt__", "<"),
    ("__le__", "<="),
):
    setattr(Field, _name, _cmp(_op))


@dataclass(frozen=True)
class Var(_ExprOps):
    name: str

    def __getattr__(self, prop: str) -> Field:
        if prop.startswith("_"):
            raise AttributeError(prop)
        return Field("node", self.name, prop)

    def __getitem__(self, prop: Any) -> Field:
        return Field("node", self.name, str(prop))


@dataclass(frozen=True)
class Edge:
    index: int

    def __getattr__(self, prop: str) -> Field:
        if prop.startswith("_"):
            raise AttributeError(prop)
        return Field("edge", str(self.index), prop)

    def __getitem__(self, prop: Any) -> Field:
        return Field("edge", str(self.index), str(prop))


@dataclass(frozen=True)
class Const(_ExprOps):
    value: str


@dataclass(frozen=True)
class Command:
    lhs: tuple[Any, ...] = ()
    where_clauses: tuple[Any, ...] = ()
    limit: int = 100
    rhs: tuple[Any, ...] | None = None
    mode: str = "first"
    rewrite_limit: int | None = None
    namespace: str | None = None

    def where(self, *predicates: Any) -> Command:
        return replace(self, where_clauses=self.where_clauses + tuple(predicates))

    def rewrite(self, *rhs: Any, mode: str = "first", limit: int | None = None) -> Command:
        terms = rhs[0] if len(rhs) == 1 and isinstance(rhs[0], (list, tuple)) else rhs
        return replace(
            self,
            rhs=tuple(terms),
            mode=str(mode),
            rewrite_limit=None if limit is None else int(limit),
        )


@dataclass(frozen=True)
class Namespace:
    name: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _normalize_ns(self.name))

    def id(self, node_id: Any) -> str:
        return _ns_node(self.name, node_id)

    def edge(self, *nodes: Any, rel: str = "_", data: dict[str, Any] | None = None, **props: Any) -> dict[str, Any]:
        return edge(*(self._tok(v) for v in nodes), rel=rel, data=data, **props)

    def node(self, node_id: Any, data: dict[str, Any] | None = None, **props: Any) -> dict[str, Any]:
        return node(self.id(node_id), data=data, **props)

    def match(self, *lhs: Any, limit: int = 100, mode: str = "first") -> Command:
        return replace(match(*lhs, limit=limit, mode=mode), namespace=self.name)

    def rewrite(self, *lhs: Any, to: list[Any] | tuple[Any, ...], mode: str = "first", limit: int = 100) -> Command:
        return self.match(*lhs).rewrite(to, mode=mode, limit=limit)

    def _tok(self, token: Any) -> Any:
        if isinstance(token, Var):
            return token
        if isinstance(token, Const):
            return Const(_ns_node(self.name, token.value))
        return _ns_node(self.name, token)


class Result(list[dict[str, Any]]):
    """List-like query/rewrite result with small data-wrangling helpers."""

    def map(self, fn: Any) -> list[Any]:
        return [fn(row) for row in self]

    def first(self, default: Any = None) -> dict[str, Any] | Any:
        return self[0] if self else default

    def bindings(self, *keys: str, default: Any = None) -> list[dict[str, Any]]:
        rows = [dict(r.get("bindings") or r) for r in self]
        return self._project(rows, keys, default)

    def edge_data(self, *keys: str, index: int = 1, default: Any = None) -> list[dict[str, Any]]:
        if index < 1:
            raise ValueError("index must be >= 1")
        rows = []
        for row in self:
            edges = row.get("hyperedges") or []
            edge = edges[index - 1] if index <= len(edges) else {}
            rows.append(dict((edge or {}).get("data") or {}))
        return self._project(rows, keys, default)

    def rows(self, *keys: str, default: Any = None) -> list[dict[str, Any]]:
        return self._project([dict(r) for r in self], keys, default)

    @staticmethod
    def _project(rows: list[dict[str, Any]], keys: tuple[str, ...], default: Any) -> list[dict[str, Any]]:
        if not keys:
            return rows
        return [{k: row.get(k, default) for k in keys} for row in rows]


class _Engine:
    def __init__(self, db_path: str = "graph.db") -> None:
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        self.db.execute("PRAGMA temp_store = MEMORY")
        self.db.create_function("py_cosine", 2, _sql_cosine, deterministic=True)
        self.vec_backend = "python"
        self._init_sqlite_vec()
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS hyperedges(
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                relation TEXT NOT NULL,
                nodes_json TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_hyper ON hyperedges(relation);
            CREATE UNIQUE INDEX IF NOT EXISTS uq_hyper ON hyperedges(relation,nodes_json);
            """
        )
        self.db.commit()

    def run(self, command: Command | list[Command] | tuple[Command, ...]) -> Result | list[Result]:
        if isinstance(command, Command):
            return Result(self._rewrite(command) if command.rhs is not None else self._match(command))
        if not isinstance(command, (list, tuple)) or not all(isinstance(c, Command) for c in command):
            raise TypeError("run(command_or_commands)")
        self.db.execute("BEGIN IMMEDIATE")
        try:
            out = [Result(self._rewrite(cmd) if cmd.rhs is not None else self._match(cmd)) for cmd in command]
            self.db.commit()
            return out
        except Exception:
            self.db.rollback()
            raise

    def _match(self, command: Command) -> list[dict[str, Any]]:
        mode = command.mode.lower()
        if mode not in {"first", "all", "random"}:
            raise ValueError("match mode must be 'first', 'all', or 'random'")
        sql_text, params, vars_sorted, edge_cols = self._compile_query(
            [_as_term(t, pattern=True) for t in command.lhs],
            [_as_pred(pred) for pred in command.where_clauses],
            1 if mode == "random" else command.limit,
            mode == "random",
            command.namespace,
        )
        rows = self.db.execute(sql_text, params).fetchall()
        if not rows:
            return []
        edge_ids = list(dict.fromkeys(int(row[col]) for row in rows for col in edge_cols))
        edge_by_id = self._fetch_edges(edge_ids)
        return [
            {
                "bindings": {v: str(row[f"v_{v}"]) for v in vars_sorted},
                "hyperedges": [edge_by_id[eid] for eid in (int(row[col]) for col in edge_cols) if eid in edge_by_id],
            }
            for row in rows
        ]

    def _rewrite(self, command: Command) -> list[dict[str, Any]]:
        lhs = [_as_term(t, pattern=True) for t in command.lhs]
        filters = [_as_pred(pred) for pred in command.where_clauses]
        rhs = [_as_term(t, pattern=bool(lhs)) for t in (command.rhs or ())]
        mode = command.mode.lower()
        limit = command.limit if command.rewrite_limit is None else command.rewrite_limit
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if mode not in {"first", "all", "random"}:
            raise ValueError("mode must be 'first', 'all', or 'random'")

        own_tx = not self.db.in_transaction
        if own_tx:
            self.db.execute("BEGIN IMMEDIATE")
        try:
            if not lhs:
                out = [self._emit_terms(rhs, {}, command.namespace)] if rhs else []
                if own_tx:
                    self.db.commit()
                return out

            sql_text, params, vars_sorted, edge_cols = self._compile_query(
                lhs, filters, limit, mode == "random", command.namespace
            )
            max_steps = limit if mode == "all" else 1
            out: list[dict[str, Any]] = []

            while len(out) < max_steps:
                row = self.db.execute(sql_text, params).fetchone()
                if row is None:
                    break
                edge_ids = [int(row[col]) for col in edge_cols]
                if edge_ids:
                    placeholders = ",".join(["?"] * len(edge_ids))
                    self.db.execute(f"DELETE FROM hyperedges WHERE edge_id IN ({placeholders})", edge_ids)
                env = {v: str(row[f"v_{v}"]) for v in vars_sorted}
                out.append(self._emit_terms(rhs, env, command.namespace))

            if mode == "all" and len(out) == max_steps:
                raise ValueError("rewrite reached limit; possible non-terminating rule")
            if own_tx:
                self.db.commit()
            return out
        except Exception:
            if own_tx:
                self.db.rollback()
            raise

    def _compile_query(
        self,
        lhs: list[dict[str, Any]],
        filters: list[dict[str, Any]],
        limit: int,
        random_order: bool = False,
        namespace: str | None = None,
    ) -> tuple[str, list[Any], list[str], list[str]]:
        if not lhs:
            raise ValueError("match requires at least one lhs term")
        joins, where = [], []
        join_params, where_params = [], []
        var_col: dict[str, str] = {}
        edge_aliases: list[str] = []

        for i, term in enumerate(lhs, 1):
            alias = f"h{i}"
            edge_aliases.append(alias)

            if i == 1:
                joins.append("FROM hyperedges h1")
                where.append("json_array_length(h1.nodes_json) = ?")
                where_params.append(len(term["nodes"]))
                if term["relation"] != "_":
                    where.append("h1.relation = ?")
                    where_params.append(term["relation"])
            else:
                on = [f"json_array_length({alias}.nodes_json) = ?"]
                on.extend(f"{alias}.edge_id <> h{k}.edge_id" for k in range(1, i))
                join_params.append(len(term["nodes"]))
                if term["relation"] != "_":
                    on.append(f"{alias}.relation = ?")
                    join_params.append(term["relation"])
                joins.append(f"JOIN hyperedges {alias} ON {' AND '.join(on)}")

            for j, tok in enumerate(term["nodes"]):
                col = f"json_extract({alias}.nodes_json, '$[{j}]')"
                if tok["kind"] == "const":
                    where.append(f"{col} = ?")
                    where_params.append(_ns_node(namespace, tok["value"]))
                    continue
                prev = var_col.get(tok["value"])
                if prev is None:
                    var_col[tok["value"]] = col
                else:
                    where.append(f"{col} = {prev}")

        for pred in filters:
            if pred["target"] == "node":
                ref = str(pred["ref"])
                if ref not in var_col:
                    raise ValueError(f"unknown node variable in filter: {ref}")
                clause, values = self._filter_clause("np.data", pred)
                where.append(
                    "EXISTS (SELECT 1 FROM hyperedges np "
                    "WHERE np.relation = '__node__' "
                    "AND json_array_length(np.nodes_json)=1 "
                    f"AND json_extract(np.nodes_json, '$[0]') = {var_col[ref]} "
                    f"AND {clause})"
                )
                where_params.extend(values)
                continue

            idx = int(pred["ref"])
            if idx < 1 or idx > len(edge_aliases):
                raise ValueError("edge filter index out of range")
            clause, values = self._filter_clause(f"{edge_aliases[idx - 1]}.data", pred)
            where.append(clause)
            where_params.extend(values)

        vars_sorted = sorted(var_col)
        if namespace:
            where.extend(f"{var_col[v]} LIKE ?" for v in vars_sorted)
            where_params.extend(f"{namespace}:%" for _ in vars_sorted)

        edge_cols = [f"e{i}" for i in range(1, len(lhs) + 1)]
        select = [f"h{i}.edge_id AS e{i}" for i in range(1, len(lhs) + 1)]
        select.extend(f"{var_col[v]} AS v_{v}" for v in vars_sorted)
        order = " ORDER BY RANDOM()" if random_order else ""
        sql_text = f"SELECT {', '.join(select)} {' '.join(joins)} WHERE {' AND '.join(where)}{order} LIMIT {int(limit)}"
        return sql_text, join_params + where_params, vars_sorted, edge_cols

    def _filter_clause(self, json_col: str, pred: dict[str, Any]) -> tuple[str, list[Any]]:
        path = f"$.{pred['prop']}"
        op = pred["op"]
        val = pred["value"]
        if op == "COSINE_GTE":
            query = _vec_json(val["query"])
            min_score = float(val.get("min", 0.0))
            if self.vec_backend == "sqlite-vec":
                # Native sqlite-vec distance: cosine distance in [0, 2], lower is closer.
                max_dist = 1.0 - min_score
                return (
                    f"json_type({json_col}, ?)='array' AND vec_distance_cosine(json_extract({json_col}, ?), ?) <= ?",
                    [path, path, query, max_dist],
                )
            return (
                f"json_type({json_col}, ?)='array' AND py_cosine(json_extract({json_col}, ?), ?) >= ?",
                [path, path, query, min_score],
            )
        if op == "IN":
            vals = list(val)
            if not vals:
                return "1 = 0", []
            return f"json_extract({json_col}, ?) IN ({','.join(['?'] * len(vals))})", [path] + vals
        if val is None and op == "=":
            return f"json_extract({json_col}, ?) IS NULL", [path]
        if val is None and op == "!=":
            return f"json_extract({json_col}, ?) IS NOT NULL", [path]
        return f"json_extract({json_col}, ?) {op} ?", [path, val]

    def _emit_terms(self, terms: list[dict[str, Any]], env: dict[str, str], namespace: str | None = None) -> dict[str, Any]:
        resolved: list[tuple[str, list[str], dict[str, Any]]] = []
        for term in terms:
            node_ids: list[str] = []
            for tok in term["nodes"]:
                if tok["kind"] == "const":
                    node_ids.append(_ns_node(namespace, tok["value"]))
                    continue
                if tok["kind"] == "expr":
                    node_ids.append(_ns_node(namespace, _eval_expr(tok["value"], env, namespace)))
                    continue
                name = str(tok["value"])
                if name not in env:
                    env[name] = _ns_node(namespace, f"n_{uuid.uuid4().hex[:12]}")
                node_ids.append(env[name])
            resolved.append((term["relation"], node_ids, _resolve_value(term["data"], env, namespace)))

        edge_ids: list[int] = []
        for rel, node_ids, data in resolved:
            edge_id = self._insert_edge(rel, node_ids, data)
            edge_ids.append(edge_id)
        if not edge_ids:
            return {"bindings": dict(env), "hyperedges": []}

        edge_by_id = self._fetch_edges(edge_ids)
        return {
            "bindings": dict(env),
            "hyperedges": [edge_by_id[eid] for eid in edge_ids if eid in edge_by_id],
        }

    def _fetch_edges(self, edge_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not edge_ids:
            return {}
        placeholders = ",".join(["?"] * len(edge_ids))
        rows = self.db.execute(
            "SELECT edge_id, relation, nodes_json, data "
            f"FROM hyperedges WHERE edge_id IN ({placeholders})",
            edge_ids,
        ).fetchall()
        return {
            int(row["edge_id"]): {
                "edge_id": int(row["edge_id"]),
                "relation": row["relation"],
                "nodes": json.loads(row["nodes_json"]),
                "data": json.loads(row["data"]),
            }
            for row in rows
        }

    def _insert_edge(self, relation: str, node_ids: list[str], data: dict[str, Any]) -> int:
        nodes_json = json.dumps(node_ids, separators=(",", ":"))
        payload = dict(data or {})
        if "embedding" in payload and payload["embedding"] is not None:
            payload["embedding"] = _vec(payload["embedding"])
        data_json = json.dumps(payload, separators=(",", ":"))
        row = self.db.execute(
            "INSERT INTO hyperedges(relation,nodes_json,data) VALUES (?,?,?) "
            "ON CONFLICT(relation,nodes_json) DO UPDATE SET data=excluded.data "
            "RETURNING edge_id",
            (relation, nodes_json, data_json),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to resolve edge id after upsert")
        return int(row["edge_id"])

    def _init_sqlite_vec(self) -> None:
        path = os.getenv("SQLITE_VEC_PATH")
        if path:
            try:
                self.db.enable_load_extension(True)
                self.db.load_extension(path)
            except sqlite3.Error:
                pass
            finally:
                try:
                    self.db.enable_load_extension(False)
                except sqlite3.Error:
                    pass
        try:
            import sqlite_vec  # type: ignore

            self.db.enable_load_extension(True)
            sqlite_vec.load(self.db)
            self.db.enable_load_extension(False)
        except Exception:
            try:
                self.db.enable_load_extension(False)
            except sqlite3.Error:
                pass
        try:
            self.db.execute("SELECT vec_distance_cosine('[1,0]', '[1,0]')").fetchone()
            self.vec_backend = "sqlite-vec"
        except sqlite3.Error:
            self.vec_backend = "python"


def _as_term(raw: Any, pattern: bool) -> dict[str, Any]:
    if isinstance(raw, dict):
        relation = str(raw.get("relation", "_"))
        nodes = list(raw.get("nodes", ()))
        data = dict(raw.get("data") or {})
    elif isinstance(raw, (tuple, list)):
        if len(raw) == 2 and isinstance(raw[0], str) and isinstance(raw[1], (tuple, list)):
            relation, nodes = str(raw[0]), list(raw[1])
        else:
            relation, nodes = "_", list(raw)
        data = {}
    else:
        raise ValueError("term must be dict/list/tuple")

    if not nodes:
        raise ValueError("term nodes must be non-empty")
    return {"relation": relation, "nodes": [_as_token(node, pattern) for node in nodes], "data": data}


def _as_token(tok: Any, pattern: bool) -> dict[str, Any]:
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


def _num(value: Any) -> int | float:
    if isinstance(value, (int, float)):
        return value
    text = str(value).strip()
    try:
        return int(text)
    except ValueError:
        return float(text)


def _add(a: Any, b: Any) -> Any:
    try:
        return _num(a) + _num(b)
    except Exception:
        return f"{a}{b}"


def _vec(value: Any) -> list[float]:
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, (list, tuple)) or not raw:
        raise ValueError("embedding must be a non-empty list/tuple of numbers")
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError) as exc:
        raise ValueError("embedding values must be numeric") from exc


def _vec_json(value: Any) -> str:
    return json.dumps(_vec(value), separators=(",", ":"))


def _sql_cosine(a: Any, b: Any) -> float:
    try:
        va, vb = _vec(a), _vec(b)
    except Exception:
        return -1.0
    if len(va) != len(vb):
        return -1.0
    dot = sum(x * y for x, y in zip(va, vb))
    na = sum(x * x for x in va) ** 0.5
    nb = sum(y * y for y in vb) ** 0.5
    if na == 0.0 or nb == 0.0:
        return -1.0
    return dot / (na * nb)


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


def _resolve_value(value: Any, env: dict[str, str], namespace: str | None) -> Any:
    if isinstance(value, Expr):
        return _eval_expr(value, env, namespace)
    if isinstance(value, Var):
        if value.name not in env:
            raise ValueError(f"unbound variable in expression: {value.name}")
        return env[value.name]
    if isinstance(value, Const):
        return _ns_node(namespace, value.value)
    if isinstance(value, list):
        return [_resolve_value(v, env, namespace) for v in value]
    if isinstance(value, tuple):
        return [_resolve_value(v, env, namespace) for v in value]
    if isinstance(value, dict):
        return {str(k): _resolve_value(v, env, namespace) for k, v in value.items()}
    return value


def _eval_expr(expr: Expr, env: dict[str, str], namespace: str | None) -> Any:
    op, argc = expr.op, len(expr.args)
    args = [_resolve_value(arg, env, namespace) for arg in expr.args]
    if op == "concat":
        return "".join(str(v) for v in args)
    if op in BIN_EVAL:
        return BIN_EVAL[op](*args)
    if op in UN_EVAL:
        if argc != 1:
            raise ValueError(f"expression op {op} expects one arg")
        return UN_EVAL[op](args[0])
    raise ValueError(f"unknown expression op: {op}")


def _as_pred(item: Any) -> dict[str, Any]:
    if not isinstance(item, Pred):
        raise ValueError("where entries must be predicates built from Var/Edge fields")
    pred = {
        "target": item.target.lower(),
        "ref": item.ref,
        "prop": item.prop,
        "op": item.op,
        "value": item.value,
    }
    if pred["target"] not in {"node", "edge"} or pred["op"] not in OPS:
        raise ValueError("invalid where predicate")
    return pred


def match(*lhs: Any, limit: int = 100, mode: str = "first") -> Command:
    return Command(lhs=tuple(lhs), where_clauses=(), limit=int(limit), mode=str(mode).lower(), namespace=None)


def rewrite(*lhs: Any, to: list[Any] | tuple[Any, ...], mode: str = "first", limit: int = 100) -> Command:
    return match(*lhs).rewrite(to, mode=mode, limit=limit)


def vars(names: str | list[str] | tuple[str, ...]) -> tuple[Var, ...]:
    raw = names.replace(",", " ").split() if isinstance(names, str) else [str(x) for x in names]
    return tuple(Var(name) for name in raw if name)


def on(index: int) -> Edge:
    return Edge(int(index))


def const(value: Any) -> Const:
    return Const(str(value))


def edge(*nodes: Any, rel: str = "_", data: dict[str, Any] | None = None, **props: Any) -> dict[str, Any]:
    if not nodes:
        raise ValueError("term requires at least one node")
    payload = dict(data or {})
    payload.update(props)
    return {"relation": str(rel), "nodes": tuple(nodes), "data": payload}


def node(node_id: Any, data: dict[str, Any] | None = None, **props: Any) -> dict[str, Any]:
    return edge(node_id, rel="__node__", data=data, **props)


def _ns_node(namespace: str | None, node_id: Any) -> str:
    raw = str(node_id)
    if not namespace:
        return raw
    prefix = f"{namespace}:"
    return raw if raw.startswith(prefix) else f"{prefix}{raw}"


def ns(name: str) -> Namespace:
    return Namespace(name)

def _normalize_ns(namespace: str) -> str:
    value = str(namespace).strip()
    if not value:
        raise ValueError("namespace must be non-empty")
    if ":" in value:
        raise ValueError("invalid namespace; ':' is reserved")
    return value


__all__ = [
    "vars",
    "const",
    "on",
    "edge",
    "node",
    "ns",
    "match",
    "rewrite",
    "exec",
    "Result",
]

_ENGINE: _Engine | None = None


def _engine() -> _Engine:
    global _ENGINE
    if _ENGINE is None:
        _ENGINE = _Engine()
    return _ENGINE


def exec(*commands: Any) -> Result | list[Result]:  # noqa: A001
    if len(commands) == 1 and isinstance(commands[0], Command):
        return _engine().run(commands[0])
    if len(commands) == 1 and isinstance(commands[0], (list, tuple)):
        return _engine().run(commands[0])
    if commands:
        return _engine().run(list(commands))
    raise TypeError("exec(command[, command2, ...])")
