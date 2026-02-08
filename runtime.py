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
    def _bin(self, op, other): return Expr(op, (self, other))
    def _rbin(self, op, other): return Expr(op, (other, self))
    def concat(self, *xs): return Expr("concat", (self, *xs))
    def lower(self): return Expr("lower", (self,))
    def upper(self): return Expr("upper", (self,))
    def strip(self): return Expr("strip", (self,))
    def replace(self, old, new): return Expr("replace", (self, old, new))
    def strlen(self): return Expr("strlen", (self,))


@dataclass(frozen=True)
class Expr(_ExprOps):
    op: str
    args: tuple[Any, ...]


for _name, _op in (
    ("__add__", "add"), ("__sub__", "sub"), ("__mul__", "mul"), ("__truediv__", "div"),
    ("__floordiv__", "floordiv"), ("__mod__", "mod"), ("__pow__", "pow"),
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
        if prop.startswith("_"): raise AttributeError(prop)
        return Field(self.target, self.ref, f"{self.path}.{prop}")

    def __getitem__(self, prop): return Field(self.target, self.ref, f"{self.path}.{prop}")
    def _pred(self, op, value): return Pred(self.target, self.ref, self.path, op, value)
    def in_(self, values): return self._pred("IN", list(values))
    def contains(self, value): return self._pred("LIKE", f"%{value}%")
    def startswith(self, value): return self._pred("LIKE", f"{value}%")
    def endswith(self, value): return self._pred("LIKE", f"%{value}")
    def similar(self, query, min_score=0.0): return self._pred("COSINE_GTE", {"query": _vec(query), "min": float(min_score)})


for _name, _op in (("__eq__", "="), ("__ne__", "!="), ("__gt__", ">"), ("__ge__", ">="), ("__lt__", "<"), ("__le__", "<=")):
    setattr(Field, _name, lambda s, v, _op=_op: s._pred(_op, v))


@dataclass(frozen=True)
class Var(_ExprOps):
    name: str
    def __getattr__(self, prop):
        if prop.startswith("_"): raise AttributeError(prop)
        return Field("node", self.name, prop)
    def __getitem__(self, prop): return Field("node", self.name, str(prop))


@dataclass(frozen=True)
class Edge:
    index: int
    def __getattr__(self, prop):
        if prop.startswith("_"): raise AttributeError(prop)
        return Field("edge", str(self.index), prop)
    def __getitem__(self, prop): return Field("edge", str(self.index), str(prop))


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
            limit=self.limit if limit is None else _as_limit(limit),
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
    def map(self, fn): return [fn(r) for r in self]
    def first(self, default=None): return self[0] if self else default

    def bindings(self, *keys, default=None):
        return _project([dict(r.get("bindings") or r) for r in self], keys, default)

    def edge_data(self, *keys, index=1, default=None):
        if index < 1: raise ValueError("index must be >= 1")
        rows = []
        for row in self:
            edges = row.get("hyperedges") or []
            edge = edges[index - 1] if index <= len(edges) else {}
            rows.append(dict((edge or {}).get("data") or {}))
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
            """
        )
        self.db.commit()

    @contextmanager
    def _tx(self):
        own = not self.db.in_transaction
        if own: self.db.execute("BEGIN IMMEDIATE")
        try:
            yield
            if own: self.db.commit()
        except Exception:
            if own: self.db.rollback()
            raise

    @contextmanager
    def _rollback_tx(self):
        own = not self.db.in_transaction
        if own:
            self.db.execute("BEGIN IMMEDIATE")
            try:
                yield
            finally:
                self.db.rollback()
            return
        with self._savepoint(True):
            yield

    @contextmanager
    def _savepoint(self, enabled):
        if not enabled:
            yield
            return
        name = f"m_{uuid.uuid4().hex[:8]}"
        self.db.execute(f"SAVEPOINT {name}")
        try:
            yield
        finally:
            self.db.execute(f"ROLLBACK TO {name}")
            self.db.execute(f"RELEASE {name}")

    def run(self, *commands, mem=None, temp=False):
        commands, single = _as_command_list(commands, "run(command[, command2, ...], mem=...)")
        mem_terms = [_as_term(t, pattern=False) for t in _coerce_terms(mem)]
        scope = self._rollback_tx if temp else self._tx
        with scope():
                out = []
                for command in commands:
                    with self._savepoint(bool(mem_terms)):
                        if mem_terms:
                            ns = command.namespace or ""
                            for term in mem_terms:
                                self._upsert_term(ns, term, {})
                        rows = self._match(command) if command.rhs is None else self._rewrite(command)
                        out.append(Result(rows))
        return out[0] if single else out

    def _compile_query(self, lhs, filters, limit, random_order, namespace):
        if not lhs: raise ValueError("match requires at least one lhs term")
        ns, where, params, var_col = namespace or "", [], [], {}
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
                    if prev is None: var_col[tok["value"]] = col
                    else: where.append(f"{col} = {prev}")

        for pred in filters:
            if pred["target"] == "node":
                ref = str(pred["ref"])
                if ref not in var_col: raise ValueError(f"unknown node variable in filter: {ref}")
                clause, vals = self._filter_sql(pred, "np.data", "np.embedding")
                where.append(
                    "EXISTS (SELECT 1 FROM hyperedges np "
                    "WHERE np.namespace = ? AND np.relation = '__node__' "
                    "AND np.arity = 1 "
                    f"AND np.node0 = {var_col[ref]} AND {clause})"
                )
                params.append(ns)
                params.extend(vals)
            else:
                idx = int(pred["ref"])
                if idx < 1 or idx > len(lhs): raise ValueError("edge filter index out of range")
                clause, vals = self._filter_sql(pred, f"h{idx}.data", f"h{idx}.embedding")
                where.append(clause)
                params.extend(vals)

        vars_sorted = sorted(var_col)
        edge_cols = [f"e{i}" for i in range(1, len(lhs) + 1)]
        select = [f"h{i}.edge_id AS e{i}" for i in range(1, len(lhs) + 1)]
        select += [f"{var_col[v]} AS v_{v}" for v in vars_sorted]
        sql = f"SELECT {', '.join(select)} {from_sql} WHERE {' AND '.join(where)}"
        if random_order: sql += " ORDER BY RANDOM()"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        return sql, params, vars_sorted, edge_cols

    def _filter_sql(self, pred, json_col, emb_col):
        if pred["op"] == "COSINE_GTE":
            if pred["prop"] != "embedding": raise ValueError("similar() is only supported on the embedding field")
            q = _vec_json(pred["value"]["query"])
            min_score = float(pred["value"].get("min", 0.0))
            return f"{emb_col} IS NOT NULL AND vec_distance_cosine({emb_col}, ?) <= ?", [q, 1.0 - min_score]
        op, path, val = pred["op"], f"$.{pred['prop']}", pred["value"]
        ex = f"json_extract({json_col}, ?)"
        if op == "IN":
            vals = list(val)
            return ("1 = 0", []) if not vals else (f"{ex} IN ({','.join(['?'] * len(vals))})", [path] + vals)
        if val is None and op == "=": return f"{ex} IS NULL", [path]
        if val is None and op == "!=": return f"{ex} IS NOT NULL", [path]
        return f"{ex} {op} ?", [path, val]

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
                "bindings": {v: str(row[f"v_{v}"]) for v in vars_sorted},
                "hyperedges": [edge_by_id[eid] for eid in (int(row[c]) for c in edge_cols) if eid in edge_by_id],
            }
            for row in rows
        ]

    def _rewrite(self, command):
        lhs = [_as_term(t, pattern=True) for t in command.lhs]
        rhs = [_as_term(t, pattern=bool(lhs)) for t in (command.rhs or ())]
        filters = [_as_pred(p) for p in command.where_clauses]
        random_order, limit = command.random, command.limit
        if limit is not None and limit < 1: raise ValueError("limit must be >= 1")

        if not lhs: return [self._emit(rhs, {}, command.namespace)] if rhs else []
        sql, params, vars_sorted, edge_cols = self._compile_query(lhs, filters, 1, random_order, command.namespace)
        out = []
        delete_sql = f"DELETE FROM hyperedges WHERE edge_id IN ({','.join(['?'] * len(edge_cols))})"
        while limit is None or len(out) < limit:
            row = self.db.execute(sql, params).fetchone()
            if row is None: break
            ids = [int(row[c]) for c in edge_cols]
            self.db.execute(delete_sql, ids)
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
            if tok["kind"] == "const":
                nodes.append(str(tok["value"]))
                continue
            if tok["kind"] == "expr":
                nodes.append(str(_eval(tok["value"], env)))
                continue
            name = str(tok["value"])
            if name not in env: env[name] = f"n_{uuid.uuid4().hex[:12]}"
            nodes.append(env[name])
        relation = term["relation"]
        data = _resolve(term["data"], env)
        emb = _resolve(term["embedding"], env) if term["embedding"] is not None else None
        edge_id = self._upsert_edge(namespace, relation, nodes, data, emb)
        return _edge_obj(edge_id, relation, nodes, data, emb)

    def _fetch_edges(self, edge_ids):
        if not edge_ids: return {}
        rows = self.db.execute(
            f"SELECT edge_id, relation, nodes_json, data, embedding FROM hyperedges WHERE edge_id IN ({','.join(['?'] * len(edge_ids))})",
            edge_ids,
        ).fetchall()
        loads = json.loads
        out = {}
        for r in rows:
            eid = int(r["edge_id"])
            out[eid] = _edge_obj(
                eid,
                r["relation"],
                loads(r["nodes_json"]),
                loads(r["data"]),
                None if r["embedding"] is None else loads(r["embedding"]),
            )
        return out

    def _upsert_edge(self, namespace, relation, nodes, data, embedding):
        payload = dict(data or {})
        emb = None if embedding is None else _vec_json(embedding)
        arity = len(nodes)
        n0 = nodes[0] if arity > 0 else None
        n1 = nodes[1] if arity > 1 else None
        n2 = nodes[2] if arity > 2 else None
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
                n0,
                n1,
                n2,
                json.dumps(payload, separators=(",", ":")),
                emb,
            ),
        ).fetchone()
        if row is None: raise RuntimeError("failed to upsert edge")
        return int(row["edge_id"])

def _project(rows, keys, default):
    return rows if not keys else [{k: row.get(k, default) for k in keys} for row in rows]


def _edge_obj(edge_id, relation, nodes, data, embedding):
    return {
        "edge_id": int(edge_id),
        "relation": relation,
        "nodes": list(nodes),
        "data": dict(data),
        "embedding": None if embedding is None else _vec(embedding),
    }


def _coerce_terms(raw):
    if raw is None: return []
    if isinstance(raw, (list, tuple)): return list(raw)
    return [raw]


def _as_term(raw, pattern):
    if not isinstance(raw, dict):
        raise ValueError("term must be created with runtime.edge(...) or runtime.node(...)")
    rel = str(raw["relation"])
    nodes = list(raw["nodes"])
    data = dict(raw.get("data") or {})
    embedding = raw.get("embedding")
    temp = bool(raw.get("temp", False))
    if not nodes: raise ValueError("term nodes must be non-empty")
    return {"relation": rel, "nodes": [_as_tok(n, pattern) for n in nodes], "data": data, "embedding": embedding, "temp": temp}


def _as_tok(tok, pattern):
    if isinstance(tok, Const): return {"kind": "const", "value": tok.value}
    if isinstance(tok, Var): return {"kind": "var" if pattern else "const", "value": tok.name}
    if isinstance(tok, Expr):
        if not pattern: raise ValueError("expression tokens are only supported when rewrite has a non-empty lhs")
        return {"kind": "expr", "value": tok}
    if pattern:
        if not isinstance(tok, str): raise ValueError("pattern tokens must be strings, Var, Const, or Expr")
        return {"kind": "var", "value": tok}
    return {"kind": "const", "value": str(tok)}


def _as_pred(item):
    if not isinstance(item, Pred): raise ValueError("where entries must be predicates built from Var/Edge fields")
    out = {"target": item.target.lower(), "ref": item.ref, "prop": item.prop, "op": item.op, "value": item.value}
    if out["target"] not in {"node", "edge"} or out["op"] not in OPS: raise ValueError("invalid where predicate")
    return out


def _num(value):
    if isinstance(value, (int, float)): return value
    text = str(value).strip()
    try: return int(text)
    except ValueError: return float(text)


def _add(a, b):
    try: return _num(a) + _num(b)
    except Exception: return f"{a}{b}"


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
    "neg": lambda a: -_num(a), "pos": lambda a: +_num(a), "lower": lambda a: str(a).lower(),
    "upper": lambda a: str(a).upper(), "strip": lambda a: str(a).strip(), "strlen": lambda a: len(str(a)),
}


def _resolve(value, env):
    if isinstance(value, Expr): return _eval(value, env)
    if isinstance(value, Var):
        if value.name not in env: raise ValueError(f"unbound variable in expression: {value.name}")
        return env[value.name]
    if isinstance(value, Const): return str(value.value)
    if isinstance(value, (list, tuple)): return [_resolve(v, env) for v in value]
    if isinstance(value, dict): return {str(k): _resolve(v, env) for k, v in value.items()}
    return value


def _eval(expr, env):
    args = [_resolve(a, env) for a in expr.args]
    if expr.op == "concat": return "".join(str(v) for v in args)
    if expr.op in BIN_EVAL: return BIN_EVAL[expr.op](*args)
    if expr.op in UN_EVAL:
        if len(args) != 1: raise ValueError(f"expression op {expr.op} expects one arg")
        return UN_EVAL[expr.op](args[0])
    raise ValueError(f"unknown expression op: {expr.op}")


def _vec(value):
    raw = json.loads(value) if isinstance(value, str) else value
    if not isinstance(raw, (list, tuple)) or not raw: raise ValueError("embedding must be a non-empty list/tuple of numbers")
    try: return [float(x) for x in raw]
    except (TypeError, ValueError) as exc: raise ValueError("embedding values must be numeric") from exc


def _vec_json(value): return json.dumps(_vec(value), separators=(",", ":"))


def vars(names):
    if not isinstance(names, str): raise TypeError("vars(names) expects a space-delimited string")
    return tuple(Var(name) for name in names.split() if name)


def const(value): return Const(str(value))
def on(index): return Edge(int(index))


def edge(*nodes, rel="_", embedding=None, data=None, temp=False, **props):
    if not nodes: raise ValueError("term requires at least one node")
    payload = dict(data or {})
    payload.update(props)
    return {"relation": str(rel), "nodes": tuple(nodes), "data": payload, "embedding": embedding, "temp": bool(temp)}


def node(node_id, embedding=None, data=None, temp=False, **props):
    return edge(node_id, rel="__node__", embedding=embedding, data=data, temp=temp, **props)


def _as_limit(limit):
    if limit is None: return None
    return int(limit)


def match(*lhs, limit=None, random=False):
    return Command(lhs=tuple(lhs), where_clauses=(), limit=_as_limit(limit), random=bool(random), namespace=None)


def rewrite(*lhs, to, random=False, limit=None):
    return match(*lhs, limit=limit, random=random).rewrite(to)


def ns(name): return Namespace(name)


def _normalize_ns(namespace):
    value = str(namespace).strip()
    if not value: raise ValueError("namespace must be non-empty")
    if ":" in value: raise ValueError("invalid namespace; ':' is reserved")
    return value


_ENGINE: _Engine | None = None


def _engine():
    global _ENGINE
    if _ENGINE is None: _ENGINE = _Engine()
    return _ENGINE


def _as_command_list(raw, err):
    if not raw: raise TypeError(err)
    items, single = list(raw), len(raw) == 1
    if items and all(isinstance(c, Command) for c in items): return items, single
    raise TypeError(err)


def _split_exec_args(raw):
    if not raw: raise TypeError("exec(command[, command2, ...], ...)")
    commands, inline_seed = [], []
    for item in raw:
        if isinstance(item, Command):
            commands.append(item)
            continue
        try:
            term = _as_term(item, pattern=False)
        except Exception as exc:
            raise TypeError("exec() arguments must be commands, or temp edge/node terms") from exc
        if not term["temp"]:
            raise TypeError("non-command exec args must be edge/node with temp=True")
        inline_seed.append(item)
    if not commands: raise TypeError("exec() requires at least one command")
    return commands, inline_seed


def exec(*commands, temp=False):  # noqa: A001
    cmds, inline_seed = _split_exec_args(commands)
    return _engine().run(*cmds, mem=inline_seed, temp=temp)


__all__ = ["vars", "const", "on", "edge", "node", "ns", "match", "rewrite", "exec", "Result"]
