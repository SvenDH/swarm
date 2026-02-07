#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os

from smolagents import CodeAgent, LiteLLMModel


GRAPH_CODE_INSTRUCTIONS = """
For graph work, import `runtime` and use only:
- runtime.exec(command)  # default graph
- runtime.exec(graph_id, command)

DSL command format:
- object DSL:
  x, y, z, u, v = runtime.vars("x y z u v")
  runtime.match((x, x, y), (y, z, u)).where(x.kind == "person", runtime.Edge(1).weight >= 0.5)
  runtime.match((x, x, y), (y, z, u)).rewrite(
      [(x, v, u), (y, v, z), (v, v, u)],
      mode="all",
      limit=100,
  )

Examples:
- seed graph data (empty LHS):
  runtime.exec("demo", runtime.rewrite(to=[
      runtime.edge("a", "a", "b"),
      runtime.edge("b", "c", "d"),
      runtime.edge("a", "b", "c", rel="friend"),
  ]))
- run query:
  x, y, z, u = runtime.vars("x y z u")
  runtime.exec("demo", runtime.match((x, x, y), (y, z, u)))
- run rewrite:
  x, y, z, u, v = runtime.vars("x y z u v")
  runtime.exec("demo", runtime.match((x, x, y), (y, z, u)).rewrite(
      [(x, v, u), (y, v, z), (v, v, u)],
      mode="all",
      limit=100,
  ))

Rules:
- Use hyperedges (not binary-only edges).
- In patterns, strings/Var are variables. Use runtime.Const("node_id") for literal node ids.
- Node property filters (like x.kind == "person") read unary node-meta edges: runtime.node("node_id", kind="person").
- General modeling recipe:
  - Put structural links in edges: runtime.edge(src, dst, rel="...").
  - Put node attributes in unary node edges: runtime.node(id, ...).
  - Put edge attributes in edge properties: runtime.edge(..., weight=..., confidence=...).
- Common graph conventions:
  - KG: runtime.edge(head, tail, rel="predicate"), runtime.node(entity, label="Type")
  - Bayesian: runtime.edge(parent, child, rel="parent"), runtime.node(var, kind="bayes", states=[...], cpt={...})
  - Causal: runtime.edge(cause, effect, rel="causes", strength=..., sign=...)
  - AST/program graph: runtime.edge(parent, child, rel="child", slot="body", idx=0), runtime.node(n, kind="ast", type="If")
  - Factor/hypergraph: runtime.edge(v1, v2, v3, rel="factor", fn="phi_name")
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
        "additional_authorized_imports": ["json", "runtime"],
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
