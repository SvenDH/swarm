from __future__ import annotations
import json
import sqlite3
import uuid
from dataclasses import dataclass, replace
from typing import Any

OPS = {"=", "!=", ">", ">=", "<", "<=", "LIKE", "IN"}

@dataclass(frozen=True)
class Pred:
    target: str
    ref: str
    prop: str
    op: str
    value: Any

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
    def __eq__(self, value: Any) -> Pred:  # type: ignore[override]
        return self._pred("=", value)
    def __ne__(self, value: Any) -> Pred:  # type: ignore[override]
        return self._pred("!=", value)
    def __gt__(self, value: Any) -> Pred:
        return self._pred(">", value)
    def __ge__(self, value: Any) -> Pred:
        return self._pred(">=", value)
    def __lt__(self, value: Any) -> Pred:
        return self._pred("<", value)
    def __le__(self, value: Any) -> Pred:
        return self._pred("<=", value)
    def in_(self, values: list[Any] | tuple[Any, ...] | set[Any]) -> Pred:
        return self._pred("IN", list(values))
    def contains(self, value: Any) -> Pred:
        return self._pred("LIKE", f"%{value}%")
    def startswith(self, value: Any) -> Pred:
        return self._pred("LIKE", f"{value}%")
    def endswith(self, value: Any) -> Pred:
        return self._pred("LIKE", f"%{value}")

@dataclass(frozen=True)
class Var:
    name: str
    def __getattr__(self, prop: str) -> Field:
        if prop.startswith("_"):
            raise AttributeError(prop)
        return Field("node", self.name, prop)
    def __getitem__(self, prop: Any) -> Field:
        return Field("node", self.name, str(prop))
    @classmethod
    def many(cls, names: str | list[str] | tuple[str, ...]) -> tuple[Var, ...]:
        raw = names.replace(",", " ").split() if isinstance(names, str) else [str(x) for x in names]
        return tuple(cls(name) for name in raw if name)

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
class Const:
    value: str

@dataclass(frozen=True)
class Term:
    nodes: tuple[Any, ...]
    relation: str = "_"
    data: dict[str, Any] | None = None
    def __init__(self, *nodes: Any, rel: str = "_", data: dict[str, Any] | None = None) -> None:
        if not nodes:
            raise ValueError("term requires at least one node")
        object.__setattr__(self, "nodes", tuple(nodes))
        object.__setattr__(self, "relation", str(rel))
        object.__setattr__(self, "data", dict(data or {}))

@dataclass(frozen=True)
class Command:
    lhs: tuple[Any, ...] = ()
    where_clauses: tuple[Any, ...] = ()
    limit: int = 100
    rhs: tuple[Any, ...] | None = None
    mode: str = "first"
    rewrite_limit: int | None = None
    
    def where(self, *predicates: Any) -> Command:
        # Immutable chain API.
        return replace(self, where_clauses=self.where_clauses + tuple(predicates))
    
    def update(self, rhs: list[Any] | tuple[Any, ...], *, mode: str = "first", limit: int | None = None) -> Command:
        # Rewrite is expressed as Match(lhs).update(rhs,...).
        return replace(
            self,
            rhs=tuple(rhs),
            mode=str(mode),
            rewrite_limit=None if limit is None else int(limit),
        )

class _Engine:
    def __init__(self, db_path: str = "graphs.db") -> None:
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        # Favor rewrite throughput with durable-enough defaults for local workloads.
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        self.db.execute("PRAGMA temp_store = MEMORY")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS hyperedges(
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                nodes_json TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}'
            );
            CREATE INDEX IF NOT EXISTS idx_hyper ON hyperedges(graph_id,relation);
            CREATE UNIQUE INDEX IF NOT EXISTS uq_hyper ON hyperedges(graph_id,relation,nodes_json);
            """
        )
        self.db.commit()

    def run(self, graph_id: str, command: Command) -> list[dict[str, Any]]:
        return self._rewrite(graph_id, command) if command.rhs is not None else self._match(graph_id, command)

    def _match(self, graph_id: str, command: Command) -> list[dict[str, Any]]:
        mode = command.mode.lower()
        if mode not in {"first", "random"}:
            raise ValueError("match mode must be 'first' or 'random'")
        qlimit = 1 if mode == "random" else command.limit
        lhs = [_as_term(t, pattern=True) for t in command.lhs]
        filters = [_as_pred(pred) for pred in command.where_clauses]
        sql_text, params, vars_sorted, edge_cols = self._compile_query(graph_id, lhs, filters, qlimit, mode == "random")
        
        rows = self.db.execute(sql_text, params).fetchall()
        if not rows:
            return []
        unique_edge_ids = list(dict.fromkeys(int(row[col]) for row in rows for col in edge_cols))
        edge_by_id = self._fetch_edges(graph_id, unique_edge_ids)
        return [
            {
                "bindings": {v: str(row[f"v_{v}"]) for v in vars_sorted},
                "hyperedges": [edge_by_id[eid] for eid in (int(row[col]) for col in edge_cols) if eid in edge_by_id],
            }
            for row in rows
        ]

    def _rewrite(self, graph_id: str, command: Command) -> list[dict[str, Any]]:
        lhs = [_as_term(t, pattern=True) for t in command.lhs]
        filters = [_as_pred(pred) for pred in command.where_clauses]
        rhs = [_as_term(t, pattern=bool(lhs)) for t in (command.rhs or ())]
        mode = command.mode.lower()
        limit = command.limit if command.rhs is None or command.rewrite_limit is None else command.rewrite_limit
        if limit < 1:
            raise ValueError("limit must be >= 1")
        if mode not in {"first", "all", "random"}:
            raise ValueError("mode must be 'first', 'all', or 'random'")

        # Single transaction for the full rewrite loop.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            if not lhs:
                rewritten = [self._emit_terms(graph_id, rhs, {})] if rhs else []
                self.db.commit()
                return rewritten

            random_order = mode == "random"
            sql_text, params, vars_sorted, edge_cols = self._compile_query(graph_id, lhs, filters, limit, random_order)
            max_steps = limit if mode == "all" else 1
            count = 0
            rewritten: list[dict[str, Any]] = []
            while count < max_steps:
                row = self.db.execute(sql_text, params).fetchone()
                if row is None:
                    break
                rewritten.append(self._apply_row(graph_id, row, vars_sorted, edge_cols, rhs))
                count += 1
            if mode == "all" and count == max_steps:
                raise ValueError("rewrite reached limit; possible non-terminating rule")

            self.db.commit()
            return rewritten
        except Exception:
            self.db.rollback()
            raise

    def _compile_query(
        self,
        graph_id: str,
        lhs: list[dict[str, Any]],
        filters: list[dict[str, Any]],
        limit: int,
        random_order: bool = False,
    ) -> tuple[str, list[Any], list[str], list[str]]:
        if not lhs:
            raise ValueError("match requires at least one lhs term")
        joins: list[str] = []
        where: list[str] = []
        join_params: list[Any] = []
        where_params: list[Any] = []
        var_col: dict[str, str] = {}
        edge_aliases: list[str] = []

        for i, term in enumerate(lhs, 1):
            edge_alias = f"h{i}"
            edge_aliases.append(edge_alias)
            if i == 1:
                joins.append("FROM hyperedges h1")
                where.extend(["h1.graph_id = ?", "json_array_length(h1.nodes_json) = ?"])
                where_params.extend([graph_id, len(term["nodes"])])
                if term["relation"] != "_":
                    where.append("h1.relation = ?")
                    where_params.append(term["relation"])
            else:
                on = [f"{edge_alias}.graph_id = h1.graph_id", f"json_array_length({edge_alias}.nodes_json) = ?"]
                on.extend(f"{edge_alias}.edge_id <> h{k}.edge_id" for k in range(1, i))
                join_params.append(len(term["nodes"]))
                if term["relation"] != "_":
                    on.append(f"{edge_alias}.relation = ?")
                    join_params.append(term["relation"])
                joins.append(f"JOIN hyperedges {edge_alias} ON {' AND '.join(on)}")

            for j, tok in enumerate(term["nodes"]):
                col = f"json_extract({edge_alias}.nodes_json, '$[{j}]')"
                if tok["kind"] == "const":
                    where.append(f"{col} = ?")
                    where_params.append(tok["value"])
                if tok["kind"] == "var":
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
                    f"WHERE np.graph_id = h1.graph_id "
                    "AND np.relation = '__node__' "
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
        edge_cols = [f"e{i}" for i in range(1, len(lhs) + 1)]
        select = [f"h{i}.edge_id AS e{i}" for i in range(1, len(lhs) + 1)]
        select.extend(f"{var_col[v]} AS v_{v}" for v in vars_sorted)
        order = " ORDER BY RANDOM()" if random_order else ""
        sql_text = f"SELECT {', '.join(select)} {' '.join(joins)} WHERE {' AND '.join(where)}{order} LIMIT {int(limit)}"
        return sql_text, join_params + where_params, vars_sorted, edge_cols

    @staticmethod
    def _filter_clause(json_col: str, pred: dict[str, Any]) -> tuple[str, list[Any]]:
        path = f"$.{pred['prop']}"
        op = pred["op"]
        val = pred["value"]
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

    def _apply_row(self, graph_id: str, row: sqlite3.Row, vars_sorted: list[str], edge_cols: list[str], rhs: list[dict[str, Any]]) -> dict[str, Any]:
        edge_ids = [int(row[col]) for col in edge_cols]
        if edge_ids:
            placeholders = ",".join(["?"] * len(edge_ids))
            self.db.execute(
                f"DELETE FROM hyperedges WHERE graph_id=? AND edge_id IN ({placeholders})",
                [graph_id] + edge_ids,
            )
        env = {v: str(row[f"v_{v}"]) for v in vars_sorted}
        return self._emit_terms(graph_id, rhs, env)

    def _emit_terms(self, graph_id: str, terms: list[dict[str, Any]], env: dict[str, str]) -> dict[str, Any]:
        resolved_terms: list[tuple[str, list[str], dict[str, Any]]] = []
        for term in terms:
            node_ids: list[str] = []
            for tok in term["nodes"]:
                if tok["kind"] == "const":
                    node_ids.append(str(tok["value"]))
                    continue
                name = str(tok["value"])
                if name not in env:
                    env[name] = f"n_{uuid.uuid4().hex[:12]}"
                node_ids.append(env[name])
            resolved_terms.append((term["relation"], node_ids, term["data"]))
        edge_ids = [self._insert_edge(graph_id, relation, node_ids, data) for relation, node_ids, data in resolved_terms]

        if not edge_ids:
            return {"bindings": dict(env), "hyperedges": []}

        edge_by_id = self._fetch_edges(graph_id, edge_ids)
        return {
            "bindings": dict(env),
            "hyperedges": [edge_by_id[edge_id] for edge_id in edge_ids if edge_id in edge_by_id],
        }

    def _fetch_edges(self, graph_id: str, edge_ids: list[int]) -> dict[int, dict[str, Any]]:
        if not edge_ids:
            return {}
        placeholders = ",".join(["?"] * len(edge_ids))
        rows = self.db.execute(
            "SELECT edge_id, relation, nodes_json, data "
            f"FROM hyperedges WHERE graph_id=? AND edge_id IN ({placeholders})",
            [graph_id] + edge_ids,
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

    def _insert_edge(self, graph_id: str, relation: str, node_ids: list[str], data: dict[str, Any]) -> int:
        nodes_json = json.dumps(node_ids, separators=(",", ":"))
        data_json = json.dumps(data or {}, separators=(",", ":"))
        cur = self.db.execute(
            "INSERT OR IGNORE INTO hyperedges(graph_id,relation,nodes_json,data) VALUES (?,?,?,?)",
            (graph_id, relation, nodes_json, data_json),
        )
        if cur.rowcount == 1:
            return int(cur.lastrowid)

        row = self.db.execute(
            "SELECT edge_id FROM hyperedges WHERE graph_id=? AND relation=? AND nodes_json=?",
            (graph_id, relation, nodes_json),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to resolve edge id after upsert")
        return int(row["edge_id"])


def _as_term(raw: Any, pattern: bool) -> dict[str, Any]:
    if isinstance(raw, Term):
        relation, nodes, data = raw.relation, list(raw.nodes), dict(raw.data or {})
    elif isinstance(raw, (tuple, list)):
        if len(raw) == 2 and isinstance(raw[0], str) and isinstance(raw[1], (tuple, list)):
            relation, nodes = str(raw[0]), list(raw[1])
        else:
            relation, nodes = "_", list(raw)
        data = {}
    else:
        raise ValueError("term must be Term/list/tuple")
    if not nodes:
        raise ValueError("term nodes must be non-empty")
    return {"relation": relation, "nodes": [_as_token(node, pattern) for node in nodes], "data": data}


def _as_token(tok: Any, pattern: bool) -> dict[str, str]:
    if isinstance(tok, Const):
        return {"kind": "const", "value": tok.value}
    if isinstance(tok, Var):
        return {"kind": "var" if pattern else "const", "value": tok.name}
    if pattern:
        if not isinstance(tok, str):
            raise ValueError("pattern tokens must be strings, Var, or Const")
        return {"kind": "var", "value": tok}
    return {"kind": "const", "value": str(tok)}


def _as_pred(item: Any) -> dict[str, Any]:
    if not isinstance(item, Pred):
        raise ValueError("where entries must be predicates built from Var/Edge fields")
    pred = {"target": item.target.lower(), "ref": item.ref, "prop": item.prop, "op": item.op, "value": item.value}
    if pred["target"] not in {"node", "edge"} or pred["op"] not in OPS:
        raise ValueError("invalid where predicate")
    return pred


def select(
    *lhs: Any,
    where: tuple[Any, ...] | list[Any] | None = None,
    limit: int = 100,
    mode: str = "first",
) -> Command:
    return Command(
        lhs=tuple(lhs),
        where_clauses=tuple(where or ()),
        limit=int(limit),
        mode=str(mode),
    )


def update(rhs: list[Any] | tuple[Any, ...], **kwargs: Any) -> Command:
    return select().update(rhs, **kwargs)


_ENGINE = _Engine()


def exec(graph_id: str, command: Command) -> list[dict[str, Any]]:  # noqa: A001
    return _ENGINE.run(graph_id, command)
