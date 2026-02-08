#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os

from smolagents import CodeAgent, LiteLLMModel

from hierarchy import create_agent_hierarchy


GRAPH_CODE_INSTRUCTIONS = """
You are a code-first graph rewriting assistant.
Always solve tasks by writing executable Python that uses `runtime`.
Use real execution results to answer. Do not return pseudo-code.

Workflow:
1. Convert the task into one or more `runtime` commands.
2. Execute with `w.exec(...)` where `w = acl.client(name).ns(namespace)`.
3. Return concise, result-grounded output (counts + key rows/bindings).
4. If ambiguous, run a small diagnostic `match(..., limit=5..20)` first.

Hard rules:
- Use graph operations through `runtime` only.
- Terms must be explicit objects: `runtime.edge(...)` / `runtime.node(...)` only.
- Use `runtime.const("...")` when a node id must be literal in a pattern.
- Prefer explicit `rel="..."` over default relation.
- Keep rewrites bounded with `limit=...`.
- Use inline `temp=True` edge/node terms for virtual overlays.
- Use `w.exec(..., temp=True)` for full ephemeral execution (rollback after call).

Canonical API:
- `w = acl.client(name).ns(namespace)` then `w.exec(command[, command2, ...], temp=False)`
- `runtime.match(*lhs, limit=None, random=False).where(...).rewrite([...], limit=..., random=...)`
- `runtime.rewrite(..., to=[...], limit=..., random=...)`
- `runtime.edge(*nodes, rel="_", embedding=None, **props)`
- `runtime.node(node_id, embedding=None, **props)`
- `runtime.vars("x y z")`, `runtime.const("id")`, `runtime.on(i)`, `runtime.ns("name")`

Semantics:
- Hyperedges are native (any arity).
- In pattern terms, plain `str` and `Var` are variables.
- Node attributes are stored as unary `runtime.node(...)` edges.
- Similarity filters use `embedding.similar(query, min_score=...)` in `where(...)`.
- Rewrite expressions support arithmetic and string ops (e.g. `+ - * / // % **`, `concat`, `upper`, `strip`, `replace`, `strlen`).
- `limit=None` means unbounded (all matches); set a numeric limit when you need bounded output/work.
- In agent hierarchies:
  - send findings upward/lateral via `out_answer(ctx, msg)`
  - send commands downward via `out_command(ctx, msg)`
  - publish logs/events via `out_graph(ctx, msg)`
  - ACL is user/group based:
    - bootstrap terms: `acl.terms(namespace, name, layer=..., context=..., memory=..., graph=...)`
    - wrapper: `acl.client(name).ns(namespace)` with `exec/allow/deny/join/leave` (build commands with `runtime`)
    - checks: `w.can(...)` from wrapper or `acl.check(subject, ...)`
  - use `hierarchy.spawn_agent(parent, child, ...)` for dynamic expansion (budget-enforced)
  - call `hierarchy.route_messages_once(namespace)` to fan-out deliveries and memory writes.

Execution style:
- For read tasks: run one `match` command and return filtered rows.
- For write tasks: run rewrite.
- For multi-step tasks: pass multiple commands in one `w.exec(...)` batch.
- Keep answers short and include only relevant result slices.

Examples:
```python
import acl
w = acl.client("agent:assistant").ns("default")

# Query
x, y = runtime.vars("x y")
out = w.exec(
    runtime.match(runtime.edge(x, y, rel="friend")).where(
        x.kind == "person",
        runtime.on(1).weight >= 0.8,
    ),
)

# Rewrite
x, y, z = runtime.vars("x y z")
step = w.exec(
    runtime.match(runtime.edge(x, y, rel="friend")).rewrite(
        [runtime.edge(x, y, z, rel="friend3")],
        limit=100,
    )
)

# Overlay-only virtual terms
out = w.exec(
    runtime.match(runtime.edge(x, y, rel="friend")),
    runtime.edge("a", "b", rel="friend", temp=True),
    runtime.node("a", kind="person", temp=True),
)

# Full ephemeral batch (uses current DB state, rolls back writes)
out = w.exec(
    runtime.rewrite(to=[runtime.edge("m1", "n0", rel="state")]),
    runtime.match(runtime.edge(runtime.const("m1"), x, rel="state")),
    temp=True,
)

# Expressions in rewrite
a, b, outn = runtime.vars("a b outn")
w.exec(runtime.rewrite(to=[runtime.edge("2", "3", "sum", rel="add")]))
w.exec(
    runtime.match(runtime.edge(a, b, outn, rel="add")).rewrite(
        [runtime.node(outn, value=a + b, text=a.concat(":", b.upper()))],
    )
)

# Embedding similarity
q = [1.0, 0.0, 0.0]
hits = w.exec(
    runtime.match(runtime.edge(x, y, rel="friend"), limit=20).where(
        runtime.on(1).embedding.similar(q, min_score=0.75),
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
    runtime.match(runtime.edge(x, y, rel="friend"), limit=20).where(
        runtime.on(1).embedding.similar(vec, min_score=0.75),
    )
)

# Hierarchy coordination
import hierarchy
import acl
summary = hierarchy.create_agent_hierarchy(2, 2, namespace="agents", root_budget=6, child_budget=4)
root = summary["root_agent"]
extra = hierarchy.spawn_agent(root, None, namespace="agents", max_children=2)
g = runtime.ns("agents")
x = runtime.vars("x")[0]
w_agents = acl.client(root).ns("agents")
l1 = w_agents.exec(g.match(g.node(x), limit=1).where(x.kind == "agent", x.layer == 1))[0]["bindings"]["x"]
l2 = w_agents.exec(g.match(g.node(x), limit=1).where(x.kind == "agent", x.layer == 2, x.parent == l1))[0]["bindings"]["x"]
w_agents.exec(runtime.ns("agents").rewrite(to=[
    runtime.ns("agents").edge(f"ctx:{l2}", "msg:find:1", rel="out_answer", topic="status"),
    runtime.ns("agents").edge(f"ctx:{l1}", "msg:cmd:1", rel="out_command", task="delegate"),
]))
routing = hierarchy.route_messages_once("agents")
allowed = w_agents.can(action="read")
```
"""


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
            "runtime",
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
        "--init-agent-hierarchy",
        action="store_true",
        help="Create an entry->NxN->MxM agent hierarchy graph before running tasks.",
    )
    parser.add_argument(
        "--hierarchy-n",
        type=int,
        default=3,
        help="N for the layer-1 NxN agent grid.",
    )
    parser.add_argument(
        "--hierarchy-m",
        type=int,
        default=2,
        help="M for each layer-2 MxM agent grid under every layer-1 agent.",
    )
    parser.add_argument(
        "--hierarchy-namespace",
        default="agent_hierarchy",
        help="Namespace used for the generated hierarchy graph.",
    )
    args = parser.parse_args()

    if args.init_agent_hierarchy:
        summary = create_agent_hierarchy(
            args.hierarchy_n,
            args.hierarchy_m,
            namespace=args.hierarchy_namespace,
        )
        print(json.dumps(summary, indent=2))
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
