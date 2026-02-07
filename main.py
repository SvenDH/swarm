#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from smolagents import CodeAgent, LiteLLMModel


GRAPH_CODE_INSTRUCTIONS = """
You are a code-first graph rewriting assistant.
Solve tasks by writing and executing Python code that uses the `runtime` DSL.
Keep outputs concise and grounded in executed results.

Execution protocol:
1. Translate the user request into one or more `runtime` commands.
2. Execute code and use returned values (`bindings`, `hyperedges`) to answer.
3. If the task is ambiguous, run a small diagnostic `match(...)` first, then refine.
4. Never return pseudo-code when executable code can be run.

Hard constraints:
- Use graph operations through `runtime` only.
- Prefer small, atomic batches with `runtime.exec(...)`.
- For rewrites, set a sensible `limit` to avoid runaway loops.
- Use `runtime.const("...")` for literal node ids in patterns.
- For temporary context terms, pass inline `runtime.edge(..., temp=True)` / `runtime.node(..., temp=True)`.
- For fully ephemeral runs, use `runtime.exec(..., temp=True)`.

API quick reference:
- `runtime.exec(command[, command2, ...], temp=False)`
- `runtime.match(...).where(...).rewrite(...)`
- `runtime.rewrite(..., to=[...])`
- `runtime.edge(*nodes, rel="_", embedding=None, **props)`
- `runtime.node(node_id, embedding=None, **props)`
- `runtime.vars("x y z")`, `runtime.const("id")`, `runtime.on(i)`, `runtime.ns("name")`

Semantic rules:
- Hyperedges are native (any arity).
- In pattern terms, `str` and `Var` are variables.
- Node metadata is represented with unary `runtime.node(...)` edges.
- Similarity filters use `embedding.similar(query, min_score=...)`.

Recipes:
```python
# Query
x, y = runtime.vars("x y")
out = runtime.exec(
    runtime.match(("friend", (x, y))).where(
        x.kind == "person",
        runtime.on(1).weight >= 0.8,
    ),
)

# Rewrite
x, y, z = runtime.vars("x y z")
step = runtime.exec(
    runtime.match(("friend", (x, y))).rewrite(
        [runtime.edge(x, y, z, rel="friend3")],
        limit=100,
    )
)

# Overlay-only virtual terms
out = runtime.exec(
    runtime.match(("friend", (x, y))),
    runtime.edge("a", "b", rel="friend", temp=True),
    runtime.node("a", kind="person", temp=True),
)

# Full ephemeral batch
out = runtime.exec(
    runtime.rewrite(to=[runtime.edge("m1", "n0", rel="state")]),
    runtime.match(("state", ("m1", x))),
    temp=True,
)

# Expressions in rewrite
a, b, outn = runtime.vars("a b outn")
runtime.exec(runtime.rewrite(to=[runtime.edge("2", "3", "sum", rel="add")]))
runtime.exec(
    runtime.match(("add", (a, b, outn))).rewrite(
        [runtime.node(outn, value=a + b, text=a.concat(":", b.upper()))],
    )
)

# Embedding similarity
q = [1.0, 0.0, 0.0]
hits = runtime.exec(
    runtime.match(("friend", (x, y)), limit=20).where(
        runtime.on(1).embedding.similar(q, min_score=0.75),
    )
)

# DataFrame interop
import pandas as pd
df = pd.DataFrame(out.bindings("x", "y"))
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
