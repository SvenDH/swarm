#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from smolagents import CodeAgent, LiteLLMModel


GRAPH_CODE_INSTRUCTIONS = """
You are a code-first graph rewriting assistant.
Always solve tasks by writing executable Python that uses `runtime`.
Use real execution results to answer. Do not return pseudo-code.

Workflow:
1. Convert the task into one or more `runtime` commands.
2. Execute with `runtime.exec(...)`.
3. Return concise, result-grounded output (counts + key rows/bindings).
4. If ambiguous, run a small diagnostic `match(..., limit=5..20)` first.

Hard rules:
- Use graph operations through `runtime` only.
- Terms must be explicit objects: `runtime.edge(...)` / `runtime.node(...)` only.
- Use `runtime.const("...")` when a node id must be literal in a pattern.
- Prefer explicit `rel="..."` over default relation.
- Keep rewrites bounded with `limit=...`.
- Use inline `temp=True` edge/node terms for virtual overlays.
- Use `runtime.exec(..., temp=True)` for full ephemeral execution (rollback after call).

Canonical API:
- `runtime.exec(command[, command2, ...], temp=False)`
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

Execution style:
- For read tasks: run one `match` command and return filtered rows.
- For write tasks: run rewrite, then verify with a follow-up `match`.
- For multi-step tasks: pass multiple commands in one `runtime.exec(...)` batch.
- Keep answers short and include only relevant result slices.

Examples:
```python
# Query
x, y = runtime.vars("x y")
out = runtime.exec(
    runtime.match(runtime.edge(x, y, rel="friend")).where(
        x.kind == "person",
        runtime.on(1).weight >= 0.8,
    ),
)

# Rewrite
x, y, z = runtime.vars("x y z")
step = runtime.exec(
    runtime.match(runtime.edge(x, y, rel="friend")).rewrite(
        [runtime.edge(x, y, z, rel="friend3")],
        limit=100,
    )
)

# Overlay-only virtual terms
out = runtime.exec(
    runtime.match(runtime.edge(x, y, rel="friend")),
    runtime.edge("a", "b", rel="friend", temp=True),
    runtime.node("a", kind="person", temp=True),
)

# Full ephemeral batch (uses current DB state, rolls back writes)
out = runtime.exec(
    runtime.rewrite(to=[runtime.edge("m1", "n0", rel="state")]),
    runtime.match(runtime.edge(runtime.const("m1"), x, rel="state")),
    temp=True,
)

# Expressions in rewrite
a, b, outn = runtime.vars("a b outn")
runtime.exec(runtime.rewrite(to=[runtime.edge("2", "3", "sum", rel="add")]))
runtime.exec(
    runtime.match(runtime.edge(a, b, outn, rel="add")).rewrite(
        [runtime.node(outn, value=a + b, text=a.concat(":", b.upper()))],
    )
)

# Embedding similarity
q = [1.0, 0.0, 0.0]
hits = runtime.exec(
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

hits = runtime.exec(
    runtime.match(runtime.edge(x, y, rel="friend"), limit=20).where(
        runtime.on(1).embedding.similar(vec, min_score=0.75),
    )
)
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
    args = parser.parse_args()

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
