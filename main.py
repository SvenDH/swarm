#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from smolagents import CodeAgent, LiteLLMModel


GRAPH_CODE_INSTRUCTIONS = """
You are a code-first graph rewriting assistant.
Write and run Python code that uses only the `runtime` DSL.
Prefer concise answers backed by executed code.

Use:
- `runtime.exec(command[, command2, ...])` for atomic execution
- `runtime.match(...).where(...).rewrite(...)`
- `runtime.edge(...)`, `runtime.node(...)`, `runtime.vars(...)`
- `runtime.ns("name")` for namespace-scoped work

Core patterns:
```python
x, y, z, u, v = runtime.vars("x y z u v")

# match + where
cmd = runtime.match(("friend", (x, y))).where(
    x.kind == "person",
    runtime.on(1).weight >= 0.5,
)
out = runtime.exec(cmd)

# rewrite
step = runtime.exec(
    runtime.match((x, x, y), (y, z, u)).rewrite(
        [runtime.edge(x, v, u), runtime.edge(y, v, z), runtime.edge(v, v, u)],
        mode="first",
        limit=100,
    )
)
```

Execution operators (for program trees / computed rewrites):
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `+`/`-`
- String ops: `.concat(...)`, `.lower()`, `.upper()`, `.strip()`, `.replace(old,new)`, `.strlen()`

```python
a, b, out = runtime.vars("a b out")
runtime.exec(runtime.rewrite(to=[runtime.edge("2", "3", "sum", rel="add")]))
runtime.exec(
    runtime.match(("add", (a, b, out))).rewrite(
        [runtime.node(out, value=a + b, diff=a - b, text=a.concat(":", b.upper()))],
        mode="first",
    )
)
```

Vector similarity (inside `where`):
```python
q = [1.0, 0.0, 0.0]
x, y = runtime.vars("x y")
out = runtime.exec(
    runtime.match(("friend", (x, y)), mode="all", limit=20).where(
        runtime.on(1).embedding.similar(q, min_score=0.75),
        x.embedding.similar(q, min_score=0.70),
    )
)
```

Text -> vector (OpenAI embeddings) and use in `where`:
```python
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
q = client.embeddings.create(
    model="text-embedding-3-small",
    input="Who works at OpenAI?",
).data[0].embedding

chunk, ent = runtime.vars("chunk ent")
hits = runtime.exec(
    runtime.match(("mentions", (chunk, ent)), mode="all", limit=20).where(
        chunk.embedding.similar(q, min_score=0.75)
    )
)
```

DataFrame interop:
```python
import pandas as pd
df_bindings = pd.DataFrame(out.bindings("x", "y"))
df_edge_data = pd.DataFrame(out.edge_data())   # edge properties/data
df_rows = pd.DataFrame(out.rows())             # raw row dicts
```

Program execution (state transition):
```python
pc, nxt = runtime.vars("pc nxt")
runtime.exec(runtime.rewrite(to=[
    runtime.edge("m1", "n0", rel="state"),
    runtime.edge("n0", "n1", rel="step"),
]))
runtime.exec(
    runtime.match(("state", ("m1", pc)), ("step", (pc, nxt))).rewrite(
        [runtime.edge("m1", nxt, rel="state")],
        mode="first",
    )
)
```

Bayesian network inference (rule-based over CPT edges):
```python
rain, sprinkler, wet = runtime.vars("rain sprinkler wet")
runtime.exec(runtime.rewrite(to=[
    runtime.edge("T", "T", "T", rel="cpt_wetgrass", p=0.99),
    runtime.edge("T", "F", "T", rel="cpt_wetgrass", p=0.90),
    runtime.edge("F", "T", "T", rel="cpt_wetgrass", p=0.80),
    runtime.edge("F", "F", "F", rel="cpt_wetgrass", p=1.00),
    runtime.edge("Rain", "T", rel="evidence"),
    runtime.edge("Sprinkler", "F", rel="evidence"),
]))
post = runtime.exec(
    runtime.match(
        ("evidence", ("Rain", rain)),
        ("evidence", ("Sprinkler", sprinkler)),
        ("cpt_wetgrass", (rain, sprinkler, wet)),
    ).where(runtime.on(3).p >= 0.8).rewrite(
        [runtime.edge("WetGrass", wet, rel="belief", source="cpt")],
        mode="first",
    )
)
```

GraphRAG network + query:
```python
chunk, ent, nbr = runtime.vars("chunk ent nbr")
runtime.exec(runtime.rewrite(to=[
    runtime.node("chunk:1", text="Alice works at OpenAI in SF", embedding=[0.95, 0.05, 0.0]),
    runtime.node("chunk:2", text="Bob lives in NYC", embedding=[0.10, 0.90, 0.0]),
    runtime.node("Alice", kind="entity"),
    runtime.node("OpenAI", kind="entity"),
    runtime.edge("chunk:1", "Alice", rel="mentions"),
    runtime.edge("chunk:1", "OpenAI", rel="mentions"),
    runtime.edge("Alice", "OpenAI", rel="related", weight=0.9),
]))
q = [1.0, 0.0, 0.0]
hits = runtime.exec(
    runtime.match(("mentions", (chunk, ent)), mode="all", limit=20).where(
        chunk.embedding.similar(q, min_score=0.75)
    )
)
expanded = runtime.exec(
    runtime.match(("related", (ent, nbr)), mode="all", limit=20).where(
        runtime.on(1).weight >= 0.7
    )
)
```

Rules:
- Hyperedges are native (any arity).
- In patterns, strings and `Var` are variables.
- Use `runtime.const("id")` for literal node ids in patterns.
- Node properties are unary node-meta edges via `runtime.node(...)`.
- Keep rewrites bounded (`mode` + `limit`) and deterministic when possible.
- If data may be missing, check and seed before rewriting.
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
