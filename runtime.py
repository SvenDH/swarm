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
    limit: int = 100
    random: bool = False
    rhs: tuple[Any, ...] | None = None
    namespace: str | None = None

    def where(self, *predicates):
        return replace(self, where_clauses=self.where_clauses + tuple(predicates))

    def rewrite(self, *rhs, random=None, limit=None):
        terms = rhs[0] if len(rhs) == 1 and isinstance(rhs[0], (list, tuple)) else rhs
        return replace(
            self,
            rhs=tuple(terms),
            random=self.random if random is None else bool(random),
            limit=self.limit if limit is None else int(limit),
        )


@dataclass(frozen=True)
class Namespace:
    name: str

    def __post_init__(self): object.__setattr__(self, "name", _normalize_ns(self.name))

    def edge(self, *nodes, rel="_", embedding=None, data=None, temp=False, **props):
        return edge(
            *[n if isinstance(n, (Var, Expr, Const)) else str(n) for n in nodes],
            rel=rel,
            embedding=embedding,
            data=data,
            temp=temp,
            **props,
        )

    def node(self, node_id, embedding=None, data=None, temp=False, **props):
        return node(str(node_id), embedding=embedding, data=data, temp=temp, **props)
    def match(self, *lhs, limit=100, random=False): return replace(match(*lhs, limit=limit, random=random), namespace=self.name)
    def rewrite(self, *lhs, to, random=False, limit=100): return self.match(*lhs, limit=limit, random=random).rewrite(to)


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

    def rows(self, *keys, default=None): return _project([dict(r) for r in self], keys, default)


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
                data TEXT NOT NULL DEFAULT '{}',
                embedding TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_hyper_ns_rel ON hyperedges(namespace, relation);
            CREATE UNIQUE INDEX IF NOT EXISTS uq_hyper_ns ON hyperedges(namespace, relation, nodes_json);
            """
        )
        cols = {r["name"] for r in self.db.execute("PRAGMA table_info(hyperedges)").fetchall()}
        if "embedding" not in cols: self.db.execute("ALTER TABLE hyperedges ADD COLUMN embedding TEXT")
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

    def run(self, *commands, mem=None):
        commands, single = _commands(commands, "run(command[, command2, ...], mem=...)")
        mem_terms = [_as_term(t, pattern=False) for t in _coerce_terms(mem)]
        with self._tx():
            out = [Result(self._run_one(c, mem_terms)) for c in commands]
        return out[0] if single else out

    def _run_one(self, command, mem_terms):
        with self._savepoint(bool(mem_terms)):
            if mem_terms:
                ns = command.namespace or ""
                for term in mem_terms:
                    self._upsert_term(ns, term, {})
            return self._match(command) if command.rhs is None else self._rewrite(command)

    def _compile_query(self, lhs, filters, limit, random_order, namespace):
        if not lhs: raise ValueError("match requires at least one lhs term")
        ns, where, params, var_col = namespace or "", [], [], {}
        from_sql = "FROM " + ", ".join(f"hyperedges h{i}" for i in range(1, len(lhs) + 1))

        for i, term in enumerate(lhs, 1):
            a = f"h{i}"
            where += [f"{a}.namespace = ?", f"json_array_length({a}.nodes_json) = ?"]
            params += [ns, len(term["nodes"])]
            if term["relation"] != "_":
                where.append(f"{a}.relation = ?")
                params.append(term["relation"])
            where += [f"{a}.edge_id <> h{k}.edge_id" for k in range(1, i)]

            for j, tok in enumerate(term["nodes"]):
                col = f"json_extract({a}.nodes_json, '$[{j}]')"
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
                    "AND json_array_length(np.nodes_json) = 1 "
                    f"AND json_extract(np.nodes_json, '$[0]') = {var_col[ref]} AND {clause})"
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
        sql += f" LIMIT {int(limit)}"
        return sql, params, vars_sorted, edge_cols

    def _pred_sql(self, json_col, pred):
        op, path, val = pred["op"], f"$.{pred['prop']}", pred["value"]
        ex = f"json_extract({json_col}, ?)"
        if op == "IN":
            vals = list(val)
            return ("1 = 0", []) if not vals else (f"{ex} IN ({','.join(['?'] * len(vals))})", [path] + vals)
        if val is None and op == "=": return f"{ex} IS NULL", [path]
        if val is None and op == "!=": return f"{ex} IS NOT NULL", [path]
        return f"{ex} {op} ?", [path, val]

    def _filter_sql(self, pred, json_col, emb_col):
        if pred["op"] == "COSINE_GTE":
            if pred["prop"] != "embedding": raise ValueError("similar() is only supported on the embedding field")
            return self._embed_sql(emb_col, pred)
        return self._pred_sql(json_col, pred)

    def _embed_sql(self, col, pred):
        q, m = _vec_json(pred["value"]["query"]), float(pred["value"].get("min", 0.0))
        return f"{col} IS NOT NULL AND vec_distance_cosine({col}, ?) <= ?", [q, 1.0 - m]

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
        if limit < 1: raise ValueError("limit must be >= 1")

        if not lhs: return [self._emit(rhs, {}, command.namespace)] if rhs else []
        sql, params, vars_sorted, edge_cols = self._compile_query(lhs, filters, 1, random_order, command.namespace)
        out = []
        while len(out) < limit:
            row = self.db.execute(sql, params).fetchone()
            if row is None: break
            ids = [int(row[c]) for c in edge_cols]
            if ids:
                self.db.execute(f"DELETE FROM hyperedges WHERE edge_id IN ({','.join(['?'] * len(ids))})", ids)
            env = {v: str(row[f"v_{v}"]) for v in vars_sorted}
            out.append(self._emit(rhs, env, command.namespace))
        return out

    def _emit(self, terms, env, namespace):
        ns = namespace or ""
        edge_ids = [self._upsert_term(ns, term, env) for term in terms]
        edge_by_id = self._fetch_edges(edge_ids)
        return {"bindings": dict(env), "hyperedges": [edge_by_id[eid] for eid in edge_ids if eid in edge_by_id]}

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
        emb = _resolve(term["embedding"], env) if term["embedding"] is not None else None
        return self._upsert_edge(namespace, term["relation"], nodes, _resolve(term["data"], env), emb)

    def _fetch_edges(self, edge_ids):
        if not edge_ids: return {}
        rows = self.db.execute(
            f"SELECT edge_id, relation, nodes_json, data, embedding FROM hyperedges WHERE edge_id IN ({','.join(['?'] * len(edge_ids))})",
            edge_ids,
        ).fetchall()
        return {
            int(r["edge_id"]): {
                "edge_id": int(r["edge_id"]),
                "relation": r["relation"],
                "nodes": json.loads(r["nodes_json"]),
                "data": json.loads(r["data"]),
                "embedding": None if r["embedding"] is None else json.loads(r["embedding"]),
            }
            for r in rows
        }

    def _upsert_edge(self, namespace, relation, nodes, data, embedding):
        payload = dict(data or {})
        emb = None if embedding is None else _vec_json(embedding)
        row = self.db.execute(
            "INSERT INTO hyperedges(namespace, relation, nodes_json, data, embedding) VALUES (?,?,?,?,?) "
            "ON CONFLICT(namespace, relation, nodes_json) DO UPDATE SET data=excluded.data, embedding=excluded.embedding RETURNING edge_id",
            (namespace, relation, json.dumps(nodes, separators=(",", ":")), json.dumps(payload, separators=(",", ":")), emb),
        ).fetchone()
        if row is None: raise RuntimeError("failed to upsert edge")
        return int(row["edge_id"])

def _project(rows, keys, default):
    return rows if not keys else [{k: row.get(k, default) for k in keys} for row in rows]


def _coerce_terms(raw):
    if raw is None: return []
    if isinstance(raw, list): return raw
    if isinstance(raw, tuple) and (not raw or isinstance(raw[0], (dict, list, tuple))): return list(raw)
    return [raw]


def _as_term(raw, pattern):
    if isinstance(raw, dict):
        rel, nodes, data = str(raw.get("relation", "_")), list(raw.get("nodes", ())), dict(raw.get("data") or {})
        embedding = raw.get("embedding")
        temp = bool(raw.get("temp", False))
    elif isinstance(raw, (tuple, list)):
        if len(raw) == 2 and isinstance(raw[0], str) and isinstance(raw[1], (tuple, list)): rel, nodes = str(raw[0]), list(raw[1])
        else: rel, nodes = "_", list(raw)
        data = {}
        embedding = None
        temp = False
    else:
        raise ValueError("term must be dict/list/tuple")
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
    raw = names.replace(",", " ").split() if isinstance(names, str) else [str(n) for n in names]
    return tuple(Var(name) for name in raw if name)


def const(value): return Const(str(value))
def on(index): return Edge(int(index))


def edge(*nodes, rel="_", embedding=None, data=None, temp=False, **props):
    if not nodes: raise ValueError("term requires at least one node")
    payload = dict(data or {})
    payload.update(props)
    return {"relation": str(rel), "nodes": tuple(nodes), "data": payload, "embedding": embedding, "temp": bool(temp)}


def node(node_id, embedding=None, data=None, temp=False, **props):
    return edge(node_id, rel="__node__", embedding=embedding, data=data, temp=temp, **props)


def match(*lhs, limit=100, random=False):
    return Command(lhs=tuple(lhs), where_clauses=(), limit=int(limit), random=bool(random), namespace=None)


def rewrite(*lhs, to, random=False, limit=100): return match(*lhs, limit=limit, random=random).rewrite(to)


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


def _commands(raw, err):
    if not raw: raise TypeError(err)
    if len(raw) == 1:
        one = raw[0]
        if isinstance(one, Command): return [one], True
        if isinstance(one, (list, tuple)) and all(isinstance(c, Command) for c in one): return list(one), False
    if all(isinstance(c, Command) for c in raw): return list(raw), len(raw) == 1
    raise TypeError(err)


def _split_exec_args(raw):
    if not raw: raise TypeError("exec(command[, command2, ...], ...)")
    if len(raw) == 1 and isinstance(raw[0], (list, tuple)) and all(isinstance(c, Command) for c in raw[0]):
        return list(raw[0]), True, []
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
    return commands, len(commands) == 1, inline_seed


def exec(*commands, temp=False):  # noqa: A001
    cmds, single, inline_seed = _split_exec_args(commands)
    if not temp: return _engine().run(cmds if not single else cmds[0], mem=inline_seed)
    eng = _Engine(":memory:")
    try:
        return eng.run(cmds if not single else cmds[0], mem=inline_seed)
    finally: eng.db.close()


__all__ = ["vars", "const", "on", "edge", "node", "ns", "match", "rewrite", "exec", "Result"]
