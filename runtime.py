from __future__ import annotations

import json
import sqlite3
import uuid
from typing import Any


class Const:
    def __init__(self, value: Any) -> None:
        self.value = str(value)


class Field:
    def __init__(self, target: str, ref: str, path: str) -> None:
        self.target = target
        self.ref = ref
        self.path = path

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

    def contains(self, value: Any) -> Pred:
        return self._pred("LIKE", f"%{value}%")

    def startswith(self, value: Any) -> Pred:
        return self._pred("LIKE", f"{value}%")

    def endswith(self, value: Any) -> Pred:
        return self._pred("LIKE", f"%{value}")

    def in_(self, values: list[Any] | tuple[Any, ...] | set[Any]) -> Pred:
        return self._pred("IN", list(values))


class Var:
    def __init__(self, name: str) -> None:
        self.name = str(name)

    def __getattr__(self, prop: str) -> Field:
        if prop.startswith("_"):
            raise AttributeError(prop)
        return Field("node", self.name, prop)

    def __getitem__(self, prop: Any) -> Field:
        return Field("node", self.name, str(prop))

    @classmethod
    def many(cls, names: str | list[str] | tuple[str, ...]) -> tuple[Var, ...]:
        if isinstance(names, str):
            raw = [x for x in names.replace(",", " ").split() if x]
        else:
            raw = [str(x) for x in names]
        return tuple(cls(n) for n in raw)


class Edge:
    def __init__(self, index: int) -> None:
        self.index = int(index)

    def __getattr__(self, prop: str) -> Field:
        if prop.startswith("_"):
            raise AttributeError(prop)
        return Field("edge", str(self.index), prop)

    def __getitem__(self, prop: Any) -> Field:
        return Field("edge", str(self.index), str(prop))


class Pred:
    def __init__(self, target: str, ref: str, prop: str, op: str, value: Any) -> None:
        self.target = str(target)
        self.ref = str(ref)
        self.prop = str(prop)
        self.op = str(op)
        self.value = value


class Term:
    def __init__(self, *nodes: Any, rel: str = "_", data: dict[str, Any] | None = None) -> None:
        if not nodes:
            raise ValueError("term requires at least one node")
        self.relation = str(rel)
        self.nodes = tuple(nodes)
        self.data = dict(data or {})


class Match:
    def __init__(self, *lhs: Any, where: list[Any] | tuple[Any, ...] | None = None, limit: int = 100) -> None:
        self.lhs = tuple(lhs)
        self.where = tuple(where or ())
        self.limit = int(limit)


class Rewrite:
    def __init__(
        self,
        lhs: list[Any] | tuple[Any, ...],
        rhs: list[Any] | tuple[Any, ...],
        *,
        where: list[Any] | tuple[Any, ...] | None = None,
        mode: str = "first",
        limit: int = 100,
    ) -> None:
        self.lhs = tuple(lhs)
        self.rhs = tuple(rhs)
        self.where = tuple(where or ())
        self.mode = str(mode)
        self.limit = int(limit)


class _Engine:
    def __init__(self, db_path: str = "graphs.db") -> None:
        self.db = sqlite3.connect(db_path)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA foreign_keys = ON")
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS graphs(id TEXT PRIMARY KEY);
            CREATE TABLE IF NOT EXISTS nodes(
                graph_id TEXT NOT NULL,
                id TEXT NOT NULL,
                data TEXT NOT NULL DEFAULT '{}',
                PRIMARY KEY(graph_id, id),
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
                PRIMARY KEY(graph_id, edge_id, pos),
                FOREIGN KEY(edge_id) REFERENCES hyperedges(edge_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_hyper ON hyperedges(graph_id, relation, arity);
            CREATE INDEX IF NOT EXISTS idx_edge_nodes ON edge_nodes(graph_id, node_id, edge_id, pos);
            CREATE UNIQUE INDEX IF NOT EXISTS uq_hyper ON hyperedges(graph_id, relation, nodes_json);
            """
        )
        self._graph_cols = {str(r["name"]) for r in self.db.execute("PRAGMA table_info(graphs)")}
        self._sync_edge_nodes()
        self.db.commit()

    def run(self, graph_id: str, command: Match | Rewrite | dict[str, Any]) -> dict[str, Any]:
        if not graph_id:
            raise ValueError("graph_id is required")

        if isinstance(command, Match):
            cmd = {
                "op": "match",
                "lhs": list(command.lhs),
                "where": list(command.where),
                "limit": command.limit,
            }
        elif isinstance(command, Rewrite):
            cmd = {
                "op": "rewrite",
                "lhs": list(command.lhs),
                "rhs": list(command.rhs),
                "where": list(command.where),
                "mode": command.mode,
                "limit": command.limit,
            }
        elif isinstance(command, dict):
            cmd = command
        else:
            raise ValueError("command must be Match, Rewrite, or dict")

        op = str(cmd.get("op", "")).lower()
        if op == "match":
            return self._match(graph_id, cmd)
        if op == "rewrite":
            return self._rewrite(graph_id, cmd)
        raise ValueError("command op must be 'match' or 'rewrite'")

    def _match(self, graph_id: str, cmd: dict[str, Any]) -> dict[str, Any]:
        lhs = [_parse_term(t, pattern=True) for t in list(cmd.get("lhs", []))]
        where = _parse_where(list(cmd.get("where", [])))
        limit = int(cmd.get("limit", 100))
        if limit < 1:
            raise ValueError("limit must be >= 1")

        sql, params, vars_sorted, edge_cols = self._compile_match(graph_id, lhs, where, limit)
        rows = self.db.execute(sql, params).fetchall()
        return {
            "graph_id": graph_id,
            "kind": "match",
            "match_count": len(rows),
            "matched_subgraphs": [self._subgraph(graph_id, row, vars_sorted, edge_cols) for row in rows],
        }

    def _rewrite(self, graph_id: str, cmd: dict[str, Any]) -> dict[str, Any]:
        lhs_raw = list(cmd.get("lhs", []))
        rhs_raw = list(cmd.get("rhs", []))
        lhs = [_parse_term(t, pattern=True) for t in lhs_raw]
        rhs = [_parse_term(t, pattern=bool(lhs)) for t in rhs_raw]
        where = _parse_where(list(cmd.get("where", [])))
        mode = str(cmd.get("mode", "first")).lower()
        limit = int(cmd.get("limit", 100))
        if mode not in {"first", "all"}:
            raise ValueError("mode must be 'first' or 'all'")
        if limit < 1:
            raise ValueError("limit must be >= 1")

        with self.db:
            self._ensure_graph(graph_id)

            if not lhs:
                if rhs:
                    self._emit_rhs(graph_id, rhs, {})
                return {
                    "graph_id": graph_id,
                    "kind": "rewrite",
                    "rewrite_count": 1 if rhs else 0,
                    "matched_subgraphs": [],
                }

            sql, params, vars_sorted, edge_cols = self._compile_match(graph_id, lhs, where, limit)
            max_steps = 1 if mode == "first" else limit
            matched: list[dict[str, Any]] = []

            for _ in range(max_steps):
                row = self.db.execute(sql, params).fetchone()
                if row is None:
                    break
                matched.append(self._subgraph(graph_id, row, vars_sorted, edge_cols))
                self._apply_one(graph_id, row, vars_sorted, edge_cols, rhs)

            if mode == "all" and len(matched) == max_steps:
                raise ValueError("rewrite reached limit; possible non-terminating rule")

        return {
            "graph_id": graph_id,
            "kind": "rewrite",
            "rewrite_count": len(matched),
            "matched_subgraphs": matched,
        }

    def _compile_match(
        self,
        graph_id: str,
        lhs: list[dict[str, Any]],
        filters: list[dict[str, Any]],
        limit: int,
    ) -> tuple[str, list[Any], list[str], list[str]]:
        if not lhs:
            raise ValueError("match requires at least one lhs term")

        joins: list[str] = []
        where: list[str] = []
        params: list[Any] = []
        var_cols: dict[str, str] = {}
        edge_aliases: list[str] = []

        for i, term in enumerate(lhs, start=1):
            h = f"h{i}"
            edge_aliases.append(h)
            joins.append("FROM hyperedges h1" if i == 1 else f"JOIN hyperedges {h} ON {h}.graph_id = h1.graph_id")
            where += [f"{h}.graph_id = ?", f"{h}.arity = ?"]
            params += [graph_id, len(term["nodes"])]
            if term["relation"] != "_":
                where.append(f"{h}.relation = ?")
                params.append(term["relation"])

            for j, tok in enumerate(term["nodes"]):
                en = f"en{i}_{j}"
                joins.append(
                    f"JOIN edge_nodes {en} ON {en}.graph_id = {h}.graph_id AND {en}.edge_id = {h}.edge_id AND {en}.pos = {j}"
                )
                col = f"{en}.node_id"
                if tok["kind"] == "const":
                    where.append(f"{col} = ?")
                    params.append(tok["value"])
                elif tok["value"] in var_cols:
                    where.append(f"{col} = {var_cols[tok['value']]}")
                else:
                    var_cols[tok["value"]] = col

        for i in range(len(edge_aliases)):
            for j in range(i + 1, len(edge_aliases)):
                where.append(f"{edge_aliases[i]}.edge_id <> {edge_aliases[j]}.edge_id")

        node_joins: dict[str, str] = {}
        for f in filters:
            if f["target"] == "node":
                ref = str(f["ref"])
                if ref not in var_cols:
                    raise ValueError(f"unknown node variable in filter: {ref}")
                alias = node_joins.get(ref)
                if alias is None:
                    alias = f"n{len(node_joins)+1}"
                    node_joins[ref] = alias
                    joins.append(f"JOIN nodes {alias} ON {alias}.graph_id = h1.graph_id AND {alias}.id = {var_cols[ref]}")
                self._append_filter(where, params, f"{alias}.data", f)
            else:
                idx = int(f["ref"])
                if idx < 1 or idx > len(edge_aliases):
                    raise ValueError("edge filter index out of range")
                self._append_filter(where, params, f"{edge_aliases[idx-1]}.data", f)

        vars_sorted = sorted(var_cols)
        edge_cols = [f"e{i}" for i in range(1, len(edge_aliases) + 1)]
        select = [f"{h}.edge_id AS e{i}" for i, h in enumerate(edge_aliases, start=1)]
        select += [f"{var_cols[v]} AS v_{v}" for v in vars_sorted]
        sql = (
            "SELECT "
            + ", ".join(select)
            + " "
            + " ".join(joins)
            + " WHERE "
            + " AND ".join(where)
            + f" LIMIT {int(limit)}"
        )
        return sql, params, vars_sorted, edge_cols

    @staticmethod
    def _append_filter(where: list[str], params: list[Any], json_col: str, f: dict[str, Any]) -> None:
        path = f"$.{f['prop']}"
        op = str(f["op"])
        value = f["value"]

        if op == "IN":
            vals = list(value)
            if not vals:
                where.append("1 = 0")
                return
            where.append(f"json_extract({json_col}, ?) IN ({','.join(['?'] * len(vals))})")
            params.append(path)
            params.extend(vals)
            return

        if value is None and op == "=":
            where.append(f"json_extract({json_col}, ?) IS NULL")
            params.append(path)
            return

        if value is None and op == "!=":
            where.append(f"json_extract({json_col}, ?) IS NOT NULL")
            params.append(path)
            return

        where.append(f"json_extract({json_col}, ?) {op} ?")
        params.extend([path, value])

    def _subgraph(
        self,
        graph_id: str,
        row: sqlite3.Row,
        vars_sorted: list[str],
        edge_cols: list[str],
    ) -> dict[str, Any]:
        edge_ids = [int(row[c]) for c in edge_cols]
        rows = self.db.execute(
            f"SELECT edge_id, relation, nodes_json, data FROM hyperedges WHERE graph_id = ? AND edge_id IN ({','.join(['?'] * len(edge_ids))})",
            [graph_id] + edge_ids,
        ).fetchall()
        return {
            "bindings": {v: str(row[f"v_{v}"]) for v in vars_sorted},
            "hyperedges": [
                {
                    "edge_id": int(r["edge_id"]),
                    "relation": r["relation"],
                    "nodes": json.loads(r["nodes_json"]),
                    "data": json.loads(r["data"]),
                }
                for r in rows
            ],
        }

    def _apply_one(
        self,
        graph_id: str,
        row: sqlite3.Row,
        vars_sorted: list[str],
        edge_cols: list[str],
        rhs: list[dict[str, Any]],
    ) -> None:
        edge_ids = [int(row[c]) for c in edge_cols]
        self.db.execute(
            f"DELETE FROM hyperedges WHERE graph_id = ? AND edge_id IN ({','.join(['?'] * len(edge_ids))})",
            [graph_id] + edge_ids,
        )
        env = {v: str(row[f"v_{v}"]) for v in vars_sorted}
        self._emit_rhs(graph_id, rhs, env)

    def _emit_rhs(self, graph_id: str, rhs: list[dict[str, Any]], env: dict[str, str]) -> None:
        for term in rhs:
            ids: list[str] = []
            for tok in term["nodes"]:
                if tok["kind"] == "const":
                    nid = str(tok["value"])
                else:
                    name = str(tok["value"])
                    nid = env.get(name)
                    if nid is None:
                        nid = f"n_{uuid.uuid4().hex[:12]}"
                        env[name] = nid
                ids.append(nid)
            self._ensure_nodes(graph_id, ids)
            self._insert_edge(graph_id, term["relation"], ids, term["data"])

    def _insert_edge(self, graph_id: str, relation: str, node_ids: list[str], data: dict[str, Any]) -> int:
        cur = self.db.execute(
            "INSERT OR IGNORE INTO hyperedges(graph_id, relation, arity, nodes_json, data) VALUES (?, ?, ?, ?, ?)",
            (
                graph_id,
                relation,
                len(node_ids),
                json.dumps(node_ids, separators=(",", ":")),
                json.dumps(data or {}, separators=(",", ":")),
            ),
        )
        if cur.rowcount != 1:
            return 0
        edge_id = int(cur.lastrowid)
        self.db.executemany(
            "INSERT INTO edge_nodes(graph_id, edge_id, pos, node_id) VALUES (?, ?, ?, ?)",
            [(graph_id, edge_id, i, nid) for i, nid in enumerate(node_ids)],
        )
        return 1

    def _ensure_nodes(self, graph_id: str, ids: list[str]) -> None:
        uniq = list(dict.fromkeys(ids))
        if uniq:
            self.db.executemany(
                "INSERT OR IGNORE INTO nodes(graph_id, id, data) VALUES (?, ?, '{}')",
                [(graph_id, nid) for nid in uniq],
            )

    def _ensure_graph(self, graph_id: str) -> None:
        if "name" in self._graph_cols and "payload" in self._graph_cols:
            self.db.execute(
                "INSERT OR IGNORE INTO graphs(id, name, payload) VALUES (?, ?, '{}')",
                (graph_id, graph_id),
            )
            return
        if "name" in self._graph_cols:
            self.db.execute("INSERT OR IGNORE INTO graphs(id, name) VALUES (?, ?)", (graph_id, graph_id))
            return
        self.db.execute("INSERT OR IGNORE INTO graphs(id) VALUES (?)", (graph_id,))

    def _sync_edge_nodes(self) -> None:
        expected = int(self.db.execute("SELECT COALESCE(SUM(arity),0) FROM hyperedges").fetchone()[0])
        actual = int(self.db.execute("SELECT COUNT(*) FROM edge_nodes").fetchone()[0])
        if expected == actual:
            return
        self.db.execute("DELETE FROM edge_nodes")
        for r in self.db.execute("SELECT edge_id, graph_id, nodes_json FROM hyperedges"):
            nodes = json.loads(r["nodes_json"])
            self.db.executemany(
                "INSERT INTO edge_nodes(graph_id, edge_id, pos, node_id) VALUES (?, ?, ?, ?)",
                [(r["graph_id"], int(r["edge_id"]), i, str(nid)) for i, nid in enumerate(nodes)],
            )


def _parse_term(raw: Any, pattern: bool) -> dict[str, Any]:
    if isinstance(raw, Term):
        relation = raw.relation
        nodes = list(raw.nodes)
        data = dict(raw.data)
    elif isinstance(raw, dict):
        relation = str(raw.get("relation", raw.get("rel", "_")))
        nodes = list(raw.get("nodes", []))
        data = raw.get("data", {}) if isinstance(raw.get("data", {}), dict) else {}
    elif isinstance(raw, (tuple, list)):
        if len(raw) == 2 and isinstance(raw[0], str) and isinstance(raw[1], (tuple, list)):
            relation, nodes = str(raw[0]), list(raw[1])
        else:
            relation, nodes = "_", list(raw)
        data = {}
    else:
        raise ValueError("term must be Term/dict/list/tuple")

    if not nodes:
        raise ValueError("term nodes must be non-empty")
    return {"relation": relation, "nodes": [_parse_token(t, pattern) for t in nodes], "data": data}


def _parse_token(tok: Any, pattern: bool) -> dict[str, str]:
    if isinstance(tok, Const):
        return {"kind": "const", "value": tok.value}
    if isinstance(tok, Var):
        return {"kind": "var" if pattern else "const", "value": tok.name}
    if isinstance(tok, dict) and "const" in tok:
        return {"kind": "const", "value": str(tok["const"])}
    if isinstance(tok, (tuple, list)) and len(tok) == 2 and tok[0] == "const":
        return {"kind": "const", "value": str(tok[1])}
    if pattern:
        if not isinstance(tok, str):
            raise ValueError("pattern tokens must be strings, Var, or Const")
        return {"kind": "var", "value": tok}
    return {"kind": "const", "value": str(tok)}


def _parse_where(where: list[Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for w in where:
        if isinstance(w, Pred):
            f = {
                "target": w.target.lower(),
                "ref": w.ref,
                "prop": w.prop,
                "op": w.op,
                "value": w.value,
            }
        elif isinstance(w, dict):
            f = {
                "target": str(w["target"]).lower(),
                "ref": str(w["ref"]),
                "prop": str(w["prop"]),
                "op": str(w["op"]),
                "value": w["value"],
            }
        elif isinstance(w, (tuple, list)) and len(w) == 5:
            f = {
                "target": str(w[0]).lower(),
                "ref": str(w[1]),
                "prop": str(w[2]),
                "op": str(w[3]),
                "value": w[4],
            }
        else:
            raise ValueError("where entries must be Pred, dict, or 5-item tuple/list")

        if f["target"] not in {"node", "edge"}:
            raise ValueError("where target must be node or edge")
        if f["op"] not in {"=", "!=", ">", ">=", "<", "<=", "LIKE", "IN"}:
            raise ValueError("unsupported where operator")
        out.append(f)
    return out


_ENGINE = _Engine()


def exec(graph_id: str, command: Match | Rewrite | dict[str, Any]) -> dict[str, Any]:  # noqa: A001
    return _ENGINE.run(graph_id, command)
