from __future__ import annotations

import argparse
import json
from typing import Any

import acl
import names
import graph

MAX_HIERARCHY_DEPTH = 3


def _default_engine() -> graph._Engine:
    eng = getattr(graph, "_ENGINE", None)
    if eng is None:
        eng = graph._Engine()
        graph._ENGINE = eng
    return eng


def _load_json(value: str | None) -> dict[str, Any]:
    return json.loads(value) if value else {}


def _node_data(db, namespace: str, node_id: str) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT data FROM hyperedges WHERE namespace=? AND relation='__node__' AND arity=1 AND node0=?",
        (namespace, node_id),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["data"]) if row["data"] else {}


def _children(db, namespace: str, parent_id: str) -> list[str]:
    rows = db.execute(
        "SELECT node1 FROM hyperedges WHERE namespace=? AND relation='controls' AND node0=? ORDER BY node1",
        (namespace, parent_id),
    ).fetchall()
    return [str(row["node1"]) for row in rows]


def _new_node_id(db, namespace: str, used: set[str] | None = None) -> str:
    used = used or set()
    while True:
        candidate = f"node:{names.gen_name()}"
        if candidate in used:
            continue
        if _node_data(db, namespace, candidate) is None:
            used.add(candidate)
            return candidate


def _node_terms(
    g: graph.Namespace,
    *,
    namespace: str,
    node_id: str,
    layer: int,
    shared_graph_id: str,
    parent_id: str | None = None,
    x: int | None = None,
    y: int | None = None,
) -> list[dict[str, Any]]:
    props: dict[str, Any] = {"kind": "node", "layer": int(layer)}
    if parent_id is not None:
        props["parent"] = parent_id
    if x is not None:
        props["x"] = x
    if y is not None:
        props["y"] = y

    ctx, mem, interp = f"ctx:{node_id}", f"mem:{node_id}", f"interp:{node_id}"
    terms = [
        g.node(node_id, **props),
        g.node(ctx, kind="context", layer=layer, node=node_id),
        g.node(mem, kind="memory_graph", layer=layer, node=node_id),
        g.node(interp, kind="interpreter", layer=layer, node=node_id, runtime="python"),
        g.edge(node_id, ctx, rel="has_context"),
        g.edge(node_id, mem, rel="has_memory_graph"),
        g.edge(node_id, interp, rel="uses_interpreter"),
        g.edge(interp, shared_graph_id, rel="shares_graph"),
        g.edge(ctx, shared_graph_id, rel="publishes_to_graph", mode="auto"),
        g.edge(ctx, mem, rel="consults_memory", mode="auto"),
        g.edge(ctx, mem, rel="writes_memory", mode="auto"),
        *acl.terms(
            namespace,
            node_id,
            layer=layer,
            context=ctx,
            memory=mem,
            graph=shared_graph_id,
        ),
    ]
    if parent_id is not None:
        parent_ctx = f"ctx:{parent_id}"
        terms += [
            g.edge(parent_id, node_id, rel="controls"),
            g.edge(parent_ctx, ctx, rel="commands_down_ctx", mode="auto"),
            g.edge(ctx, parent_ctx, rel="answers_up_ctx", mode="auto"),
        ]
    return terms


def _sibling_lateral_terms(
    g: graph.Namespace,
    child_id: str,
    siblings: list[str],
    *,
    layer: int,
    parent_id: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    payload: dict[str, Any] = {"layer": int(layer), "parent": parent_id}
    for sibling in siblings:
        out.append(g.edge(child_id, sibling, rel="lateral", **payload))
        out.append(g.edge(sibling, child_id, rel="lateral", **payload))
        out.append(g.edge(f"ctx:{child_id}", f"ctx:{sibling}", rel="answers_lateral_ctx", mode="auto", **payload))
        out.append(g.edge(f"ctx:{sibling}", f"ctx:{child_id}", rel="answers_lateral_ctx", mode="auto", **payload))
    return out


def spawn_node(
    parent_id: str | None = None,
    child_id: str | None = None,
    *,
    namespace: str = "hierarchy",
    x: int | None = None,
    y: int | None = None,
    engine: graph._Engine | None = None,
) -> dict[str, Any]:
    """Create one hierarchy node (or child under `parent_id`) with sibling links."""
    eng = engine or _default_engine()
    ns = graph._normalize_ns(namespace)
    g = graph.ns(ns)
    shared_graph_id = f"graph:{ns}:shared"

    with eng._tx():
        terms: list[dict[str, Any]] = [g.node(shared_graph_id, kind="shared_graph")]
        siblings: list[str] = []
        child_layer = 0
        if parent_id is not None:
            parent = _node_data(eng.db, ns, parent_id)
            if parent is None or parent.get("kind") != "node":
                raise ValueError(f"parent node not found: {parent_id}")
            child_layer = int(parent.get("layer", 0)) + 1
            if child_layer > MAX_HIERARCHY_DEPTH:
                raise ValueError(
                    f"max hierarchy depth exceeded for {parent_id}: "
                    f"{child_layer}>{MAX_HIERARCHY_DEPTH}"
                )
            siblings = _children(eng.db, ns, parent_id)
        child = child_id or _new_node_id(eng.db, ns)
        if child_id is not None and _node_data(eng.db, ns, child) is not None:
            raise ValueError(f"child node already exists: {child}")

        terms += _node_terms(
            g,
            namespace=ns,
            node_id=child,
            layer=child_layer,
            shared_graph_id=shared_graph_id,
            parent_id=parent_id,
            x=x,
            y=y,
        )
        if parent_id is not None and siblings:
            terms += _sibling_lateral_terms(
                g,
                child,
                siblings,
                layer=child_layer,
                parent_id=parent_id,
            )
        eng.run(g.rewrite(to=terms))

    return {
        "namespace": ns,
        "parent": parent_id,
        "node": child,
        "child": child,
        "layer": child_layer,
        "max_depth": MAX_HIERARCHY_DEPTH,
        "connected_siblings": len(siblings),
    }


def has_permission(
    subject: str,
    target: str,
    *,
    action: str = "read",
    namespace: str = "hierarchy",
    engine: graph._Engine | None = None,
) -> bool:
    """Check ACL permission in the hierarchy namespace."""
    return acl.check(subject, target, action=action, namespace=namespace, engine=engine)


def route_messages_once(
    namespace: str = "hierarchy",
    *,
    limit: int = 1000,
    engine: graph._Engine | None = None,
) -> dict[str, int]:
    """Route one batch of outbound context messages to recipients and memory graphs."""
    if limit < 1:
        raise ValueError("limit must be >= 1")

    eng = engine or _default_engine()
    ns = graph._normalize_ns(namespace)
    routes = [
        ("out_answer", ("answers_up_ctx", "answers_lateral_ctx"), "in_answer", "answer", True, "answer_delivered"),
        ("out_command", ("commands_down_ctx",), "in_command", "command", True, "command_delivered"),
        ("out_graph", ("publishes_to_graph",), "in_graph", "graph", False, "graph_delivered"),
    ]
    stats = {
        "answer_delivered": 0,
        "command_delivered": 0,
        "graph_delivered": 0,
        "memory_items_written": 0,
        "messages_consumed": 0,
    }

    with eng._tx():
        for out_rel, via_rels, in_rel, channel, write_memory, stat_key in routes:
            rel_sql = ",".join("?" for _ in via_rels)
            rows = eng.db.execute(
                f"""
                SELECT m.edge_id, m.node1 AS msg, m.data, m.embedding, r.node1 AS dst
                FROM hyperedges m
                JOIN hyperedges r
                  ON r.namespace = m.namespace
                 AND r.node0 = m.node0
                 AND r.relation IN ({rel_sql})
                WHERE m.namespace = ? AND m.relation = ?
                LIMIT ?
                """,
                (*via_rels, ns, out_rel, limit),
            ).fetchall()

            mem_by_ctx: dict[str, list[str]] = {}
            if write_memory and rows:
                dsts = sorted({str(r["dst"]) for r in rows})
                qmarks = ",".join("?" for _ in dsts)
                mem_rows = eng.db.execute(
                    f"""
                    SELECT node0, node1
                    FROM hyperedges
                    WHERE namespace=? AND relation='consults_memory' AND node0 IN ({qmarks})
                    """,
                    (ns, *dsts),
                ).fetchall()
                for mem_row in mem_rows:
                    mem_by_ctx.setdefault(str(mem_row["node0"]), []).append(str(mem_row["node1"]))

            consumed: list[int] = []
            for row in rows:
                data = _load_json(row["data"])
                emb = None if row["embedding"] is None else json.loads(row["embedding"])
                eng._upsert_edge(ns, in_rel, [row["dst"], row["msg"]], data, emb)
                consumed.append(int(row["edge_id"]))

                if write_memory:
                    mem_targets = mem_by_ctx.get(str(row["dst"]), [])
                    for mem_id in mem_targets:
                        eng._upsert_edge(ns, "memory_item", [mem_id, row["msg"]], {"channel": channel}, emb)
                    stats["memory_items_written"] += len(mem_targets)

            if consumed:
                eng.db.execute(
                    f"DELETE FROM hyperedges WHERE edge_id IN ({','.join('?' for _ in consumed)})",
                    tuple(consumed),
                )
            stats["messages_consumed"] += len(consumed)
            stats[stat_key] += len(consumed)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Create one hierarchy node (optionally under an existing parent).")
    parser.add_argument("--parent-id", default=None, help="Parent node id. Omit to create a root node.")
    parser.add_argument("--child-id", default=None, help="Optional explicit id for the new node.")
    parser.add_argument("--x", type=int, default=None, help="Optional x coordinate metadata.")
    parser.add_argument("--y", type=int, default=None, help="Optional y coordinate metadata.")
    parser.add_argument("--namespace", default="hierarchy", help="Graph namespace.")
    parser.add_argument("--db-path", default="hierarchy.db", help="SQLite DB path for runtime engine.")
    args = parser.parse_args()

    engine = graph._Engine(args.db_path)
    try:
        print(
            json.dumps(
                spawn_node(
                    args.parent_id,
                    args.child_id,
                    namespace=args.namespace,
                    x=args.x,
                    y=args.y,
                    engine=engine,
                ),
                indent=2,
            )
        )
    finally:
        engine.db.close()


if __name__ == "__main__":
    main()
