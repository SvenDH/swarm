# SQLite Hypergraph Rewriter

Single API function:

- `runtime.exec(graph_id, command)`
- `runtime.sql(graph_id, command)` to inspect compiled SQL + bind params

It executes an object-based DSL command and returns the result.

## Object DSL

Core objects:

- `runtime.Var("x")` or `runtime.Var.many("x y z")`
- `runtime.Const("node_id")`
- `runtime.Edge(1)` for edge-property filters
- `runtime.Term(*nodes, rel="_")`
- `runtime.Match(...)`

Rewrite is expressed as `Match(...).update(...)`.

## Usage

```python
import json
import runtime

# Seed graph: empty LHS rewrite via Match().update(...)
runtime.exec(
    "demo",
    runtime.Match().update(
        [
            runtime.Term("a", "a", "b"),
            runtime.Term("b", "c", "d"),
            runtime.Term("a", "b", "c", rel="friend", data={"weight": 0.8}),
        ]
    ),
)

x, y, z, u = runtime.Var.many("x y z u")

# Query
m = runtime.exec(
    "demo",
    runtime.Match((x, x, y), (y, z, u)).where(x.kind == "person", runtime.Edge(1).weight >= 0.5),
)
print(json.dumps(m, indent=2))

# Rewrite
x, y, z, u, v = runtime.Var.many("x y z u v")
r = runtime.exec(
    "demo",
    runtime.Match((x, x, y), (y, z, u)).update(
        [(x, v, u), (y, v, z), (v, v, u)],
        mode="first",
        limit=100,
    ),
)
print(json.dumps(r, indent=2))
```

## Notes

- `Match.where(...)` is chainable.
- Pattern strings and `Var` are variables.
- Literal node ids in patterns use `Const("node_id")`.
- New RHS variables create fresh nodes.
- Rewrites execute atomically in one SQLite transaction.
- `python runtime.py` runs a built-in example that prints seed/match/rewrite SQL and outputs.
