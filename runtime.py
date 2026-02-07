from __future__ import annotations
import json
import sqlite3
import uuid
from dataclasses import dataclass
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
    
    def __init__(
        self,
        *lhs: Any,
        where: tuple[Any, ...] | list[Any] | None = None,
        limit: int = 100,
        mode: str = "first",
    ) -> None:
        object.__setattr__(self, "lhs", tuple(lhs))
        object.__setattr__(self, "where_clauses", tuple(where or ()))
        object.__setattr__(self, "limit", int(limit))
        object.__setattr__(self, "rhs", None)
        object.__setattr__(self, "mode", str(mode))
        object.__setattr__(self, "rewrite_limit", None)
    
    def where(self, *predicates: Any) -> Command:
        # Immutable chain API.
        out = Command(*self.lhs, where=self.where_clauses + tuple(predicates), limit=self.limit, mode=self.mode)
        object.__setattr__(out, "rhs", self.rhs)
        object.__setattr__(out, "mode", self.mode)
        object.__setattr__(out, "rewrite_limit", self.rewrite_limit)
        return out
    
    def update(self, rhs: list[Any] | tuple[Any, ...], *, mode: str = "first", limit: int | None = None) -> Command:
        # Rewrite is expressed as Match(lhs).update(rhs,...).
        out = Command(*self.lhs, where=self.where_clauses, limit=self.limit, mode=self.mode)
        object.__setattr__(out, "rhs", tuple(rhs))
        object.__setattr__(out, "mode", str(mode))
        object.__setattr__(out, "rewrite_limit", None if limit is None else int(limit))
        return out

class _Engine:
    def __init__(self, db_path: str = "graphs.db") -> None:
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        # Favor rewrite throughput with durable-enough defaults for local workloads.
        self.db.execute("PRAGMA journal_mode = WAL")
        self.db.execute("PRAGMA synchronous = NORMAL")
        self.db.execute("PRAGMA temp_store = MEMORY")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS graphs(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS nodes(
                graph_id TEXT NOT NULL,
                id TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(graph_id,id),
                FOREIGN KEY(graph_id) REFERENCES graphs(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS hyperedges(
                edge_id INTEGER PRIMARY KEY AUTOINCREMENT,
                graph_id TEXT NOT NULL,
                relation TEXT NOT NULL,
                arity INTEGER NOT NULL,
                nodes_json TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                FOREIGN KEY(graph_id) REFERENCES graphs(id) ON DELETE CASCADE
            );
            CREATE TABLE IF NOT EXISTS edge_nodes(
                graph_id TEXT NOT NULL,
                edge_id INTEGER NOT NULL,
                pos INTEGER NOT NULL,
                node_id TEXT NOT NULL,
                PRIMARY KEY(graph_id,edge_id,pos),
                FOREIGN KEY(edge_id) REFERENCES hyperedges(edge_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_hyper ON hyperedges(graph_id,relation,arity);
            CREATE INDEX IF NOT EXISTS idx_hyper_graph_arity ON hyperedges(graph_id,arity);
            CREATE INDEX IF NOT EXISTS idx_edge_nodes ON edge_nodes(graph_id,node_id,edge_id,pos);
            CREATE INDEX IF NOT EXISTS idx_edge_nodes_edgepos ON edge_nodes(graph_id,edge_id,pos,node_id);
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
        lhs, filters = self._compile_inputs(command)
        sql_text, params, vars_sorted, edge_cols = self._compile_query(
            graph_id,
            lhs,
            filters,
            qlimit,
            random_order=mode == "random",
        )
        rows = self.db.execute(sql_text, params).fetchall()
        if not rows:
            return []
        unique_edge_ids = list(dict.fromkeys(eid for row in rows for eid in self._edge_ids_from_row(row, edge_cols)))
        edge_by_id = self._fetch_edges(graph_id, unique_edge_ids)
        return [
            {
                "bindings": {v: str(row[f"v_{v}"]) for v in vars_sorted},
                "hyperedges": [edge_by_id[eid] for eid in self._edge_ids_from_row(row, edge_cols) if eid in edge_by_id],
            }
            for row in rows
        ]

    def _rewrite(self, graph_id: str, command: Command) -> list[dict[str, Any]]:
        lhs, filters = self._compile_inputs(command)
        rhs = [_as_term(t, pattern=bool(lhs)) for t in (command.rhs or ())]
        mode = command.mode.lower()
        limit = self._effective_limit(command)
        if mode not in {"first", "all", "random"}:
            raise ValueError("mode must be 'first', 'all', or 'random'")

        # Single transaction for the full rewrite loop.
        self.db.execute("BEGIN IMMEDIATE")
        try:
            self._ensure_graph(graph_id)
            if not lhs:
                rewritten = [self._emit_terms(graph_id, rhs, {})] if rhs else []
                self.db.commit()
                return rewritten

            random_order = mode == "random"
            sql_text, params, vars_sorted, edge_cols = self._compile_query(
                graph_id,
                lhs,
                filters,
                limit,
                random_order=random_order,
            )
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
        # Compile pattern + predicates into one SQL statement used by match/rewrite.
        if not lhs:
            raise ValueError("match requires at least one lhs term")
        
        planned = sorted(enumerate(lhs, start=1), key=lambda item: self._term_rank(item[1]), reverse=True)
        joins: list[str] = []
        where: list[str] = []
        join_params: list[Any] = []
        where_params: list[Any] = []
        var_col: dict[str, str] = {}
        alias_by_orig: dict[int, str] = {}

        for i, (orig_idx, term) in enumerate(planned, 1):
            edge_alias = f"h{i}"
            alias_by_orig[orig_idx] = edge_alias
            if i == 1:
                joins.append("FROM hyperedges h1")
                where.extend(["h1.graph_id = ?", "h1.arity = ?"])
                where_params.extend([graph_id, len(term["nodes"])])
                if term["relation"] != "_":
                    where.append("h1.relation = ?")
                    where_params.append(term["relation"])
            else:
                on = [f"{edge_alias}.graph_id = h1.graph_id", f"{edge_alias}.arity = ?"]
                on.extend(f"{edge_alias}.edge_id <> h{k}.edge_id" for k in range(1, i))
                join_params.append(len(term["nodes"]))
                if term["relation"] != "_":
                    on.append(f"{edge_alias}.relation = ?")
                    join_params.append(term["relation"])
                joins.append(f"JOIN hyperedges {edge_alias} ON {' AND '.join(on)}")

            for j, tok in enumerate(term["nodes"]):
                en = f"en{i}_{j}"
                on = [f"{en}.graph_id={edge_alias}.graph_id", f"{en}.edge_id={edge_alias}.edge_id", f"{en}.pos={j}"]
                if tok["kind"] == "const":
                    on.append(f"{en}.node_id = ?")
                    join_params.append(tok["value"])
                joins.append(f"JOIN edge_nodes {en} ON {' AND '.join(on)}")
                col = f"{en}.node_id"
                if tok["kind"] == "var" and tok["value"] in var_col:
                    where.append(f"{col} = {var_col[tok['value']]}")
                elif tok["kind"] == "var":
                    var_col[tok["value"]] = col

        edge_aliases = [alias_by_orig[i] for i in range(1, len(lhs) + 1)]

        for pred in filters:
            if pred["target"] == "node":
                ref = str(pred["ref"])
                if ref not in var_col:
                    raise ValueError(f"unknown node variable in filter: {ref}")
                clause, values = self._filter_clause("n.data", pred)
                where.append(
                    "EXISTS (SELECT 1 FROM nodes n "
                    f"WHERE n.graph_id = h1.graph_id AND n.id = {var_col[ref]} AND {clause})"
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
        select = [f"{alias_by_orig[i]}.edge_id AS e{i}" for i in range(1, len(lhs) + 1)]
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

    @staticmethod
    def _term_rank(term: dict[str, Any]) -> tuple[int, int, int]:
        relation_fixed = 1 if term["relation"] != "_" else 0
        const_count = sum(1 for tok in term["nodes"] if tok["kind"] == "const")
        # Prefer selective terms first: fixed relation, more constants, then lower arity.
        return relation_fixed, const_count, -len(term["nodes"])

    def _apply_row(self, graph_id: str, row: sqlite3.Row, vars_sorted: list[str], edge_cols: list[str], rhs: list[dict[str, Any]]) -> dict[str, Any]:
        self._delete_edges(graph_id, self._edge_ids_from_row(row, edge_cols))
        env = {v: str(row[f"v_{v}"]) for v in vars_sorted}
        return self._emit_terms(graph_id, rhs, env)

    def _emit_terms(self, graph_id: str, terms: list[dict[str, Any]], env: dict[str, str]) -> dict[str, Any]:
        resolved_terms: list[tuple[str, list[str], dict[str, Any]]] = []
        all_node_ids: list[str] = []
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
            all_node_ids.extend(node_ids)

        self._ensure_nodes(graph_id, all_node_ids)
        edge_ids: list[int] = []
        edge_by_id: dict[int, dict[str, Any]] = {}
        existing_ids: list[int] = []
        for relation, node_ids, data in resolved_terms:
            edge_id, inserted = self._insert_edge(graph_id, relation, node_ids, data)
            edge_ids.append(edge_id)
            if inserted:
                edge_by_id[edge_id] = {
                    "edge_id": edge_id,
                    "relation": relation,
                    "nodes": list(node_ids),
                    "data": dict(data or {}),
                }
            else:
                existing_ids.append(edge_id)

        if not edge_ids:
            return {"bindings": dict(env), "hyperedges": []}

        if existing_ids:
            edge_by_id.update(self._fetch_edges(graph_id, existing_ids))
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

    def _delete_edges(self, graph_id: str, edge_ids: list[int]) -> None:
        if not edge_ids:
            return
        placeholders = ",".join(["?"] * len(edge_ids))
        self.db.execute(
            f"DELETE FROM hyperedges WHERE graph_id=? AND edge_id IN ({placeholders})",
            [graph_id] + edge_ids,
        )

    @staticmethod
    def _edge_ids_from_row(row: sqlite3.Row, edge_cols: list[str]) -> list[int]:
        return [int(row[col]) for col in edge_cols]

    def _insert_edge(self, graph_id: str, relation: str, node_ids: list[str], data: dict[str, Any]) -> tuple[int, bool]:
        nodes_json = json.dumps(node_ids, separators=(",", ":"))
        data_json = json.dumps(data or {}, separators=(",", ":"))
        cur = self.db.execute(
            "INSERT OR IGNORE INTO hyperedges(graph_id,relation,arity,nodes_json,data) VALUES (?,?,?,?,?)",
            (graph_id, relation, len(node_ids), nodes_json, data_json),
        )
        if cur.rowcount == 1:
            edge_id = int(cur.lastrowid)
            self.db.executemany(
                "INSERT INTO edge_nodes(graph_id,edge_id,pos,node_id) VALUES (?,?,?,?)",
                [(graph_id, edge_id, i, node_id) for i, node_id in enumerate(node_ids)],
            )
            return edge_id, True

        row = self.db.execute(
            "SELECT edge_id FROM hyperedges WHERE graph_id=? AND relation=? AND nodes_json=?",
            (graph_id, relation, nodes_json),
        ).fetchone()
        if row is None:
            raise RuntimeError("failed to resolve edge id after upsert")
        return int(row["edge_id"]), False

    def _ensure_nodes(self, graph_id: str, node_ids: list[str]) -> None:
        unique = list(dict.fromkeys(node_ids))
        if unique:
            self.db.executemany("INSERT OR IGNORE INTO nodes(graph_id,id,data) VALUES (?,?,'{}')", [(graph_id, node_id) for node_id in unique])

    def _ensure_graph(self, graph_id: str) -> None:
        self.db.execute("INSERT OR IGNORE INTO graphs(id) VALUES (?)", (graph_id,))

    @staticmethod
    def _effective_limit(command: Command) -> int:
        limit = command.limit if command.rhs is None or command.rewrite_limit is None else command.rewrite_limit
        if limit < 1:
            raise ValueError("limit must be >= 1")
        return limit

    @staticmethod
    def _compile_inputs(command: Command) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        return [_as_term(t, pattern=True) for t in command.lhs], [_as_pred(pred) for pred in command.where_clauses]


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


select = Command  # Alias for nicer DSL.


def update(rhs: list[Any] | tuple[Any, ...], **kwargs: Any) -> Command:
    return Command().update(rhs, **kwargs)


_ENGINE = _Engine()


def exec(graph_id: str, command: Command) -> list[dict[str, Any]]:  # noqa: A001
    return _ENGINE.run(graph_id, command)
