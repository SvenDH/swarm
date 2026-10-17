#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from typing import Any

from smolagents import CodeAgent, LiteLLMModel

import acl
import graph
import hierarchy


GRAPH_CODE_INSTRUCTIONS = """
You are a code-first graph rewriting assistant.
Always solve tasks by writing executable Python that uses `graph`.
Use real execution results to answer. Do not return pseudo-code.

Workflow:
1. Convert the task into one or more `graph` commands.
2. Execute with `w.exec(...)` where `w = acl.client(name, namespace)`.
3. Return concise, result-grounded output (counts + key rows/bindings).
4. If ambiguous, run a small diagnostic `match(..., limit=5..20)` first.

Hard rules:
- Use graph operations through `graph` only.
- Terms must be explicit objects: `graph.edge(...)` / `graph.node(...)` only.
- Use `graph.const("...")` when a node id must be literal in a pattern.
- Prefer explicit `rel="..."` over default relation.
- Keep rewrites bounded with `limit=...`.
- Use inline `temp=True` edge/node terms for virtual overlays.
- Use `w.exec(..., temp=True)` for full ephemeral execution (rollback after call).

Canonical API:
- `w = acl.client(name, namespace)` then `w.exec(command[, command2, ...], temp=False)`
- `graph.match(*lhs, limit=None, random=False).where(...).rewrite([...], limit=..., random=...)`
- `graph.rewrite(..., to=[...], limit=..., random=...)`
- `graph.edge(*nodes, rel="_", embedding=None, **props)`
- `graph.node(node_id, embedding=None, **props)`
- `graph.vars("x y z")`, `graph.const("id")`, `graph.on(i)`, `graph.ns("name")`

Semantics:
- Hyperedges are native (any arity).
- In pattern terms, plain `str` and `Var` are variables.
- Node attributes are stored as unary `graph.node(...)` edges.
- Similarity filters use `embedding.similar(query, min_score=...)` in `where(...)`.
- Rewrite expressions support arithmetic and string ops (e.g. `+ - * / // % **`, `concat`, `upper`, `strip`, `replace`, `strlen`).
- `limit=None` means unbounded (all matches); set a numeric limit when you need bounded output/work.
- In hierarchy graphs:
  - send findings upward/lateral via `out_answer(ctx, msg)`
  - send commands downward via `out_command(ctx, msg)`
  - publish logs/events via `out_graph(ctx, msg)`
  - ACL is user/group based:
    - bootstrap terms: `acl.terms(namespace, name, layer=..., context=..., memory=..., graph=...)`
    - wrapper: `acl.client(name, namespace)` with `exec/allow/deny/join/leave` (build commands with `graph`)
    - checks: `w.can(...)` from wrapper or `acl.check(subject, ...)`
  - use `hierarchy.spawn_node(parent, child, ...)` for dynamic expansion (depth-limited, max depth 3)
  - call `hierarchy.route_messages_once(namespace)` to fan-out deliveries and memory writes.

Execution style:
- For read tasks: run one `match` command and return filtered rows.
- For write tasks: run rewrite.
- For multi-step tasks: pass multiple commands in one `w.exec(...)` batch.
- Keep answers short and include only relevant result slices.

Examples:
```python
import acl
w = acl.client("node:assistant", "default")

# Query
x, y = graph.vars("x y")
out = w.exec(
    graph.match(graph.edge(x, y, rel="friend")).where(
        x.kind == "person",
        graph.on(1).weight >= 0.8,
    ),
)

# Rewrite
x, y, z = graph.vars("x y z")
step = w.exec(
    graph.match(graph.edge(x, y, rel="friend")).rewrite(
        [graph.edge(x, y, z, rel="friend3")],
        limit=100,
    )
)

# Overlay-only virtual terms
out = w.exec(
    graph.match(graph.edge(x, y, rel="friend")),
    graph.edge("a", "b", rel="friend", temp=True),
    graph.node("a", kind="person", temp=True),
)

# Full ephemeral batch (uses current DB state, rolls back writes)
out = w.exec(
    graph.rewrite(to=[graph.edge("m1", "n0", rel="state")]),
    graph.match(graph.edge(graph.const("m1"), x, rel="state")),
    temp=True,
)

# Expressions in rewrite
a, b, outn = graph.vars("a b outn")
w.exec(graph.rewrite(to=[graph.edge("2", "3", "sum", rel="add")]))
w.exec(
    graph.match(graph.edge(a, b, outn, rel="add")).rewrite(
        [graph.node(outn, value=a + b, text=a.concat(":", b.upper()))],
    )
)

# Embedding similarity
q = [1.0, 0.0, 0.0]
hits = w.exec(
    graph.match(graph.edge(x, y, rel="friend"), limit=20).where(
        graph.on(1).embedding.similar(q, min_score=0.75),
    )
)

# DataFrame interop
import pandas as pd
df = pd.DataFrame(out.bindings("x", "y"))

# text -> embedding (OpenAI) -> similarity query
from openai import OpenAI
client = OpenAI()
vec = client.embeddings.create(
    model="text-embedding-3-small",
    input="friends about machine learning",
).data[0].embedding

hits = w.exec(
    graph.match(graph.edge(x, y, rel="friend"), limit=20).where(
        graph.on(1).embedding.similar(vec, min_score=0.75),
    )
)

# Hierarchy coordination
import hierarchy
import acl
root = hierarchy.spawn_node(None, namespace="hierarchy")["node"]
l1a = hierarchy.spawn_node(root, None, namespace="hierarchy")["node"]
l1b = hierarchy.spawn_node(root, None, namespace="hierarchy")["node"]
extra = hierarchy.spawn_node(l1a, None, namespace="hierarchy")["node"]
g = graph.ns("hierarchy")
x = graph.vars("x")[0]
w_nodes = acl.client(root, "hierarchy")
l1 = w_nodes.exec(g.match(g.node(x), limit=1).where(x.kind == "node", x.layer == 1))[0]["bindings"]["x"]
l2 = w_nodes.exec(g.match(g.node(x), limit=1).where(x.kind == "node", x.layer == 2, x.parent == l1a))[0]["bindings"]["x"]
w_nodes.exec(graph.ns("hierarchy").rewrite(to=[
    graph.ns("hierarchy").edge(f"ctx:{l2}", "msg:find:1", rel="out_answer", topic="status"),
    graph.ns("hierarchy").edge(f"ctx:{l1}", "msg:cmd:1", rel="out_command", task="delegate"),
]))
routing = hierarchy.route_messages_once("hierarchy")
allowed = w_nodes.can(action="read")
```
"""


def _node_data(db, namespace: str, node_id: str) -> dict[str, Any] | None:
    row = db.execute(
        "SELECT data FROM hyperedges WHERE namespace=? AND relation='__node__' AND arity=1 AND node0=?",
        (namespace, node_id),
    ).fetchone()
    if row is None:
        return None
    return json.loads(row["data"]) if row["data"] else {}


def run_agent_programs(
    programs: dict[str, list[graph.Command] | tuple[graph.Command, ...]],
    *,
    namespace: str = "hierarchy",
    temp: bool = False,
    engine: graph._Engine | None = None,
) -> dict[str, Any]:
    eng = graph.get_engine() if engine is None else engine
    ns = graph.normalize_ns(namespace)
    out: dict[str, Any] = {"namespace": ns, "nodes_executed": 0, "results": {}}

    for node_id, commands in programs.items():
        node = _node_data(eng.db, ns, node_id)
        if node is None or node.get("kind") != "node":
            raise ValueError(f"node not found: {node_id}")
        batch = list(commands)
        if not batch:
            raise ValueError(f"program is empty for {node_id}")
        worker = acl.client(node_id, ns, engine=eng)
        out["results"][node_id] = worker.exec(*batch, temp=temp)
        out["nodes_executed"] += 1

    return out


def build_model() -> LiteLLMModel:
    model_id = os.getenv("SMOLAGENT_MODEL", "openai/gpt-4.1-mini")
    api_base = os.getenv("SMOLAGENT_API_BASE")
    api_key = os.getenv("SMOLAGENT_API_KEY") or os.getenv("OPENAI_API_KEY")
    temperature = float(os.getenv("SMOLAGENT_TEMPERATURE", "0.0"))
    max_tokens = int(os.getenv("SMOLAGENT_MAX_TOKENS", "4096"))

    kwargs: dict[str, object] = {"temperature": temperature, "max_tokens": max_tokens}
    if api_base:
        kwargs["api_base"] = api_base
    if api_key:
        kwargs["api_key"] = api_key
    return LiteLLMModel(model_id=model_id, **kwargs)


def build_agent() -> CodeAgent:
    model = build_model()
    max_steps = int(os.getenv("SMOLAGENT_MAX_STEPS", "20"))
    common_kwargs = {
        "tools": [],
        "model": model,
        "max_steps": max_steps,
        "additional_authorized_imports": [
            "os",
            "json",
            "openai",
            "graph",
            "acl",
            "hierarchy",
            "pandas",
            "numpy",
            "matplotlib",
            "matplotlib.pyplot",
        ],
    }
    try:
        return CodeAgent(executor_type="local", **common_kwargs)
    except TypeError:
        return CodeAgent(**common_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="smolagents graph rewrite agent (code-only, local Python executor)."
    )
    parser.add_argument("task", nargs="*", help="One-shot task for the agent.")
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run a REPL loop for multiple graph tasks.",
    )
    parser.add_argument(
        "--init-root-node",
        action="store_true",
        help="Create one root hierarchy node before running tasks.",
    )
    parser.add_argument(
        "--hierarchy-namespace",
        default="hierarchy",
        help="Namespace used for the generated hierarchy graph.",
    )
    args = parser.parse_args()

    if args.init_root_node:
        root = hierarchy.spawn_node(None, namespace=args.hierarchy_namespace)
        print(json.dumps(root, indent=2))
        if not args.task and not args.interactive:
            return

    agent = build_agent()

    if args.task:
        task = f"{GRAPH_CODE_INSTRUCTIONS.strip()}\n\nTask:\n{' '.join(args.task).strip()}"
        result = agent.run(task)
        print(result)
        return

    if args.interactive or not args.task:
        print("Graph Rewrite CodeAgent ready. Type 'exit' or 'quit' to stop.")
        while True:
            user_task = input("graph-agent> ").strip()
            if not user_task:
                continue
            if user_task.lower() in {"exit", "quit"}:
                break
            result = agent.run(f"{GRAPH_CODE_INSTRUCTIONS.strip()}\n\nTask:\n{user_task}")
            print(result)


if __name__ == "__main__":
    main()
