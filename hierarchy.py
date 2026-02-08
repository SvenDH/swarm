from __future__ import annotations

import argparse
import json
from typing import Any

import acl
import names
import runtime


def _neighbors(i: int, j: int, size: int):
    if i > 0:
        yield "north", i - 1, j
    if i + 1 < size:
        yield "south", i + 1, j
    if j > 0:
        yield "west", i, j - 1
    if j + 1 < size:
        yield "east", i, j + 1


def _node_data(db, namespace: str, node_id: str) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT data FROM hyperedges WHERE namespace=? AND relation='__node__' AND arity=1 AND node0=?",
        (namespace, node_id),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["data"]) if row["data"] else {}


def _child_count(db, namespace: str, parent_id: str) -> int:
    return int(
        db.execute(
            "SELECT COUNT(*) FROM hyperedges WHERE namespace=? AND relation='controls' AND node0=?",
            (namespace, parent_id),
        ).fetchone()[0]
    )


def _new_agent_id(db, namespace: str, used: set[str] | None = None) -> str:
    used = used or set()
    while True:
        candidate = f"agent:{names.gen_name()}"
        if candidate in used:
            continue
        if _node_data(db, namespace, candidate) is None:
            used.add(candidate)
            return candidate


def _agent_terms(
    g: runtime.Namespace,
    *,
    namespace: str,
    agent_id: str,
    layer: int,
    max_children: int,
    shared_graph_id: str,
    parent_id: str | None = None,
    x: int | None = None,
    y: int | None = None,
) -> list[dict[str, Any]]:
    props: dict[str, Any] = {"kind": "agent", "layer": int(layer), "max_children": int(max_children)}
    if parent_id is not None:
        props["parent"] = parent_id
    if x is not None:
        props["x"] = x
    if y is not None:
        props["y"] = y

    ctx, mem, interp = f"ctx:{agent_id}", f"mem:{agent_id}", f"interp:{agent_id}"
    terms = [
        g.node(agent_id, **props),
        g.node(ctx, kind="context", layer=layer, agent=agent_id),
        g.node(mem, kind="memory_graph", layer=layer, agent=agent_id),
        g.node(interp, kind="interpreter", layer=layer, agent=agent_id, runtime="python"),
        g.edge(agent_id, ctx, rel="has_context"),
        g.edge(agent_id, mem, rel="has_memory_graph"),
        g.edge(agent_id, interp, rel="uses_interpreter"),
        g.edge(interp, shared_graph_id, rel="shares_graph"),
        g.edge(ctx, shared_graph_id, rel="publishes_to_graph", mode="auto"),
        g.edge(ctx, mem, rel="consults_memory", mode="auto"),
        g.edge(ctx, mem, rel="writes_memory", mode="auto"),
        *acl.terms(
            namespace,
            agent_id,
            layer=layer,
            context=ctx,
            memory=mem,
            graph=shared_graph_id,
        ),
    ]
    if parent_id is not None:
        parent_ctx = f"ctx:{parent_id}"
        terms += [
            g.edge(parent_id, agent_id, rel="controls"),
            g.edge(parent_ctx, ctx, rel="commands_down_ctx", mode="auto"),
            g.edge(ctx, parent_ctx, rel="answers_up_ctx", mode="auto"),
        ]
    return terms


def _lateral_terms(
    g: runtime.Namespace,
    ids: dict[tuple[int, int], str],
    *,
    size: int,
    layer: int,
    parent_id: str | None = None,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for (i, j), src in ids.items():
        for direction, ni, nj in _neighbors(i, j, size):
            dst = ids[(ni, nj)]
            payload: dict[str, Any] = {"layer": int(layer), "direction": direction}
            if parent_id is not None:
                payload["parent"] = parent_id
            out.append(g.edge(src, dst, rel="lateral", **payload))
            out.append(g.edge(f"ctx:{src}", f"ctx:{dst}", rel="answers_lateral_ctx", mode="auto", **payload))
    return out


def spawn_agent(
    parent_id: str,
    child_id: str | None = None,
    *,
    namespace: str = "agent_hierarchy",
    layer: int | None = None,
    x: int | None = None,
    y: int | None = None,
    max_children: int = 0,
    engine: runtime._Engine | None = None,
) -> dict[str, Any]:
    eng = engine or runtime._engine()
    ns = runtime._normalize_ns(namespace)
    g = runtime.ns(ns)

    with eng._tx():
        parent = _node_data(eng.db, ns, parent_id)
        if parent is None or parent.get("kind") != "agent":
            raise ValueError(f"parent agent not found: {parent_id}")

        used = _child_count(eng.db, ns, parent_id)
        budget = int(parent.get("max_children", 0))
        if used >= budget:
            raise ValueError(f"spawn budget exceeded for {parent_id}: {used}/{budget}")

        child = child_id or _new_agent_id(eng.db, ns)
        if child_id is not None and _node_data(eng.db, ns, child) is not None:
            raise ValueError(f"child agent already exists: {child}")

        child_layer = int(parent.get("layer", 0)) + 1 if layer is None else int(layer)
        terms = _agent_terms(
            g,
            namespace=ns,
            agent_id=child,
            layer=child_layer,
            max_children=max_children,
            shared_graph_id=f"graph:{ns}:shared",
            parent_id=parent_id,
            x=x,
            y=y,
        )
        eng.run(g.rewrite(to=terms))

    return {
        "namespace": ns,
        "parent": parent_id,
        "child": child,
        "layer": child_layer,
        "parent_budget": budget,
        "parent_children_used": used + 1,
        "parent_children_remaining": budget - (used + 1),
    }


def create_agent_hierarchy(
    n: int,
    m: int,
    namespace: str = "agent_hierarchy",
    *,
    root_budget: int | None = None,
    child_budget: int | None = None,
    engine: runtime._Engine | None = None,
) -> dict[str, Any]:
    if n < 1 or m < 1:
        raise ValueError("n and m must be >= 1")

    eng = engine or runtime._engine()
    ns = runtime._normalize_ns(namespace)
    g = runtime.ns(ns)
    used_ids: set[str] = set()

    root_id = _new_agent_id(eng.db, ns, used_ids)
    shared_graph_id = f"graph:{ns}:shared"
    root_max = n * n if root_budget is None else int(root_budget)
    mid_max = m * m if child_budget is None else int(child_budget)

    terms: list[dict[str, Any]] = [g.node(shared_graph_id, kind="shared_graph")]
    terms += _agent_terms(
        g,
        namespace=ns,
        agent_id=root_id,
        layer=0,
        max_children=root_max,
        shared_graph_id=shared_graph_id,
    )

    level1: dict[tuple[int, int], str] = {}
    for i in range(n):
        for j in range(n):
            aid = _new_agent_id(eng.db, ns, used_ids)
            level1[(i, j)] = aid
            terms += _agent_terms(
                g,
                namespace=ns,
                agent_id=aid,
                layer=1,
                max_children=mid_max,
                shared_graph_id=shared_graph_id,
                parent_id=root_id,
                x=i,
                y=j,
            )
    terms += _lateral_terms(g, level1, size=n, layer=1)

    for parent in level1.values():
        level2: dict[tuple[int, int], str] = {}
        for i in range(m):
            for j in range(m):
                aid = _new_agent_id(eng.db, ns, used_ids)
                level2[(i, j)] = aid
                terms += _agent_terms(
                    g,
                    namespace=ns,
                    agent_id=aid,
                    layer=2,
                    max_children=0,
                    shared_graph_id=shared_graph_id,
                    parent_id=parent,
                    x=i,
                    y=j,
                )
        terms += _lateral_terms(g, level2, size=m, layer=2, parent_id=parent)

    eng.run(g.rewrite(to=terms))

    agents_total = 1 + n * n + n * n * m * m
    users_total = agents_total
    groups_total = 4
    edges_can_agent = agents_total * 3
    edges_can_user = agents_total * 2
    edges_can_group = groups_total
    edges_can_total = edges_can_agent + edges_can_user + edges_can_group

    return {
        "namespace": ns,
        "n": n,
        "m": m,
        "agents_total": agents_total,
        "users_total": users_total,
        "groups_total": groups_total,
        "root_budget": root_max,
        "child_budget": mid_max,
        "edges_can": edges_can_total,
        "edges_can_agent": edges_can_agent,
        "edges_can_user": edges_can_user,
        "edges_can_group": edges_can_group,
        "edges_can_read": edges_can_total,
        "edges_can_write": edges_can_total,
        "edges_member_of": users_total * 2,
        "edges_uses_identity": users_total,
        "root_agent": root_id,
        "group_all": f"group:{ns}:all",
        "graph_node": shared_graph_id,
        "terms_upserted": len(terms),
    }


def has_permission(
    subject: str,
    target: str,
    *,
    action: str = "read",
    namespace: str = "agent_hierarchy",
    engine: runtime._Engine | None = None,
) -> bool:
    return acl.check(subject, target, action=action, namespace=namespace, engine=engine)


def route_messages_once(
    namespace: str = "agent_hierarchy",
    *,
    limit: int = 1000,
    engine: runtime._Engine | None = None,
) -> dict[str, int]:
    if limit < 1:
        raise ValueError("limit must be >= 1")

    eng = engine or runtime._engine()
    ns = runtime._normalize_ns(namespace)
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

            consumed: list[int] = []
            for row in rows:
                data = json.loads(row["data"]) if row["data"] else {}
                emb = None if row["embedding"] is None else json.loads(row["embedding"])
                eng._upsert_edge(ns, in_rel, [row["dst"], row["msg"]], data, emb)
                consumed.append(int(row["edge_id"]))

                if write_memory:
                    mem_rows = eng.db.execute(
                        "SELECT node1 FROM hyperedges WHERE namespace=? AND relation='consults_memory' AND node0=?",
                        (ns, row["dst"]),
                    ).fetchall()
                    for mem in mem_rows:
                        eng._upsert_edge(ns, "memory_item", [mem["node1"], row["msg"]], {"channel": channel}, emb)
                    stats["memory_items_written"] += len(mem_rows)

            if consumed:
                eng.db.execute(
                    f"DELETE FROM hyperedges WHERE edge_id IN ({','.join('?' for _ in consumed)})",
                    tuple(consumed),
                )
            stats["messages_consumed"] += len(consumed)
            stats[stat_key] += len(consumed)

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(description="Create an agent hierarchy graph in runtime.")
    parser.add_argument("--n", type=int, default=3, help="Grid size N for layer-1 (NxN).")
    parser.add_argument("--m", type=int, default=2, help="Grid size M for each layer-2 child grid (MxM).")
    parser.add_argument("--namespace", default="agent_hierarchy", help="Graph namespace.")
    parser.add_argument("--db-path", default="agent_hierarchy.db", help="SQLite DB path for runtime engine.")
    parser.add_argument("--root-budget", type=int, default=None, help="Max number of children controlled by root.")
    parser.add_argument("--child-budget", type=int, default=None, help="Max number of children controlled by layer-1 agents.")
    args = parser.parse_args()

    engine = runtime._Engine(args.db_path)
    try:
        print(
            json.dumps(
                create_agent_hierarchy(
                    args.n,
                    args.m,
                    namespace=args.namespace,
                    root_budget=args.root_budget,
                    child_budget=args.child_budget,
                    engine=engine,
                ),
                indent=2,
            )
        )
    finally:
        engine.db.close()


if __name__ == "__main__":
    main()
