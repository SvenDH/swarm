# SQLite Hypergraph Rewriter

Single API function:

- `runtime.exec(graph_id, command)`

It executes an object-based DSL command and returns the result.

## Object DSL

Core objects:

- `runtime.Var("x")` or `runtime.Var.many("x y z")`
- `runtime.Const("node_id")`
- `runtime.Edge(1)` for edge-property filters
- `runtime.Term(*nodes, rel="_")`
- `runtime.Match(...)`
- `runtime.Rewrite(...)`

Match command object:

```python
x, y, z, u = runtime.Var.many("x y z u")
cmd = runtime.Match(
    (x, x, y),
    (y, z, u),
    where=[x.kind == "person", runtime.Edge(1).weight >= 0.5],
    limit=100,
)
```

Rewrite command object:

```python
x, y, z, u, v = runtime.Var.many("x y z u v")
cmd = runtime.Rewrite(
    lhs=[(x, x, y), (y, z, u)],
    rhs=[(x, v, u), (y, v, z), (v, v, u)],
    mode="all",
    limit=100,
)
```

## Usage

```python
import json
import runtime

# Seed graph: empty LHS + RHS inserts hyperedges.
runtime.exec(
    "demo",
    runtime.Rewrite(
        lhs=[],
        rhs=[
            runtime.Term("a", "a", "b"),
            runtime.Term("b", "c", "d"),
            runtime.Term("a", "b", "c", rel="friend", data={"weight": 0.8}),
        ],
    ),
)

x, y, z, u = runtime.Var.many("x y z u")
m = runtime.exec("demo", runtime.Match((x, x, y), (y, z, u)))
print(json.dumps(m, indent=2))

x, y, z, u, v = runtime.Var.many("x y z u v")
r = runtime.exec(
    "demo",
    runtime.Rewrite(
        lhs=[(x, x, y), (y, z, u)],
        rhs=[(x, v, u), (y, v, z), (v, v, u)],
        mode="first",
    ),
)
print(json.dumps(r, indent=2))
```

## Notes

- Pattern strings and `Var` are variables.
- Literal node ids in patterns use `Const("node_id")`.
- New RHS variables create fresh nodes.
- Rewrites execute atomically in one SQLite transaction.
