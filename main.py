#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from smolagents import CodeAgent, LiteLLMModel


GRAPH_CODE_INSTRUCTIONS = """
For graph work, use:
- runtime.exec(command)
- runtime.match(...).where(...).rewrite(...)
- runtime.edge(...), runtime.node(...), runtime.vars(...)
- runtime.ns("name") for namespace-scoped ids/match/rewrite

Core DSL:
- Variables:
  x, y, z, u, v = runtime.vars("x y z u v")
- Match:
  runtime.match(("friend", (x, y))).where(x.kind == "person", runtime.on(1).weight >= 0.5)
- Rewrite:
  runtime.match((x, x, y), (y, z, u)).rewrite(
      [runtime.edge(x, v, u), runtime.edge(y, v, z), runtime.edge(v, v, u)],
      mode="all",
      limit=100,
  )

Seeding data (empty LHS rewrite):
runtime.exec(runtime.rewrite(to=[
    runtime.edge("alice", "bob", rel="friend", weight=0.9),
    runtime.edge("bob", "carol", rel="friend", weight=0.8),
    runtime.node("alice", kind="person", age=34),
    runtime.node("bob", kind="person", age=29),
]))

Search/filter example:
x, y = runtime.vars("x y")
out = runtime.exec(
    runtime.match(("friend", (x, y))).where(
        x.kind == "person",
        runtime.on(1).weight >= 0.8,
    )
)
records = [row["bindings"] for row in out]

Inference example (2-hop friend closure):
x, y, z = runtime.vars("x y z")
runtime.exec(
    runtime.match(("friend", (x, y)), ("friend", (y, z))).rewrite(
        [runtime.edge(x, z, rel="friend2", rule="two_hop")],
        mode="all",
        limit=1000,
    )
)

Program execution example (state transition):
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

Interop examples:
- pandas:
  import pandas as pd
  df = pd.DataFrame([row["bindings"] for row in out])
  print(df.head())
- numpy:
  import numpy as np
  arr = np.array([[row["bindings"].get("x"), row["bindings"].get("y")] for row in out], dtype=object)
- matplotlib:
  import matplotlib.pyplot as plt
  df["x"].value_counts().plot(kind="bar")
  plt.show()

Namespace isolation (single DB, multiple logical graphs):
- g1 = runtime.ns("g1")
- Use g1.id("alice"), g1.edge(...), g1.node(...), g1.match(...), g1.rewrite(...)

Rules:
- Hyperedges are native (any arity).
- In patterns: strings/Var are variables; use runtime.const("id") for literal node ids.
- Node properties come from unary node meta edges: runtime.node("id", ...).
- Keep rewrites bounded with mode/limit.
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
            "json",
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
