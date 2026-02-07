# SQLite Hypergraph Rewriter

Single API function:

- `runtime.exec(command)` for the default graph

It executes an object-based DSL command and returns the result.

## Object DSL

Core objects:

- `runtime.vars("x y z")` (friendly alias)
- `runtime.const("node_id")` for literal node ids in patterns
- `runtime.on(1)` for edge-property filters
- `runtime.edge(*nodes, rel="_", **props)` (friendly term builder)
- `runtime.node("id", **props)` for node metadata edges
- `runtime.ns("name")` namespace handle
- `runtime.match(...)`

Rewrite is expressed as `match(...).rewrite(...)` or top-level `rewrite(..., to=[...])`.

## Usage

```python
import json
import runtime

# Seed graph: empty LHS rewrite
runtime.exec(runtime.rewrite(to=[
    runtime.edge("a", "a", "b"),
    runtime.edge("b", "c", "d", rel="r2"),
    runtime.edge("a", "b", "c", rel="friend", weight=0.8),
    runtime.node("a", kind="person"),
]))

x, y, z, u = runtime.vars("x y z u")

# Query
m = runtime.exec(
    runtime.match((x, x, y), (y, z, u)).where(x.kind == "person", runtime.on(1).weight >= 0.5),
)
print(json.dumps(m, indent=2))

# Rewrite
x, y, z, u, v = runtime.vars("x y z u v")
r = runtime.exec(
    runtime.match((x, x, y), (y, z, u)).rewrite(
        [(x, v, u), (y, v, z), (v, v, u)],
        mode="first",
        limit=100,
    ),
)
print(json.dumps(r, indent=2))
```

## Modeling Different Graph Types

Use `rel` + edge/node properties to encode graph semantics.

### General Recipe (Any Domain)

```python
# structural links
runtime.edge("src", "dst", rel="link_type", weight=0.9)

# node attributes
runtime.node("src", kind="entity", label="Person")

# n-ary relations
runtime.edge("a", "b", "c", rel="event", ts=1700000000)
```

### Segregating Different Graphs (Namespace Prefixes)

Use a namespace handle so one DB can host many logical graphs safely.

```python
g1 = runtime.ns("g1")
g2 = runtime.ns("g2")

runtime.exec(runtime.rewrite(to=[
    g1.edge("alice", "bob", rel="friend"),
    g2.edge("alice", "bob", rel="friend"),
]))
```

Namespace-aware matching/rewrite:

```python
x, y = runtime.vars("x y")
g1 = runtime.ns("g1")

# only matches nodes whose ids start with "g1:"
g1_matches = runtime.exec(g1.match(("friend", (x, y)), mode="all", limit=100))
```

When namespace is set on a rewrite, fresh variables are created in that namespace automatically.

### Knowledge Graph (KG)

```python
runtime.exec(runtime.rewrite(to=[
    runtime.edge("alice", "openai", rel="works_at", source="hr"),
    runtime.edge("alice", "sf", rel="lives_in"),
    runtime.node("alice", label="Person"),
    runtime.node("openai", label="Org"),
]))
```

### Bayesian Network

```python
runtime.exec(runtime.rewrite(to=[
    runtime.edge("Rain", "WetGrass", rel="parent"),
    runtime.edge("Sprinkler", "WetGrass", rel="parent"),
    runtime.node("Rain", kind="bayes", states=["T", "F"], cpt={"T": 0.2, "F": 0.8}),
    runtime.node("Sprinkler", kind="bayes", states=["T", "F"], cpt={"T": 0.4, "F": 0.6}),
    runtime.node("WetGrass", kind="bayes", states=["T", "F"], cpt={
        "Rain=T,Sprinkler=T": {"T": 0.99, "F": 0.01},
        "Rain=T,Sprinkler=F": {"T": 0.9, "F": 0.1},
        "Rain=F,Sprinkler=T": {"T": 0.8, "F": 0.2},
        "Rain=F,Sprinkler=F": {"T": 0.0, "F": 1.0},
    }),
]))
```

### Causal Graph

```python
runtime.exec(runtime.rewrite(to=[
    runtime.edge("Smoking", "Cancer", rel="causes", strength=0.7),
    runtime.edge("Tar", "Cancer", rel="mediates"),
    runtime.edge("Tax", "Smoking", rel="intervenes", effect="-"),
    runtime.node("Smoking", kind="variable"),
    runtime.node("Cancer", kind="outcome"),
]))
```

### AST / Program Graph

```python
runtime.exec(runtime.rewrite(to=[
    runtime.node("n_if", kind="ast", type="If"),
    runtime.node("n_cond", kind="ast", type="Compare"),
    runtime.node("n_then", kind="ast", type="Assign"),
    runtime.edge("n_if", "n_cond", rel="child", slot="test", idx=0),
    runtime.edge("n_if", "n_then", rel="child", slot="body", idx=0),
]))
```

### Factor / Hypergraph Style

```python
runtime.exec(runtime.rewrite(to=[
    runtime.edge("X1", "X2", "X3", rel="factor", fn="phi1"),
    runtime.edge("X2", "X4", rel="factor", fn="phi2"),
]))
```

## Common Workflows

### Search / Filtering

```python
x, y = runtime.vars("x y")

# find friend edges where source node is a Person
out = runtime.exec(
    runtime.match(("friend", (x, y))).where(
        runtime.on(1).weight >= 0.5,
        x.label == "Person",
    )
)
```

### Inference (Forward Rule Application)

```python
x, y, z = runtime.vars("x y z")

# if friend(x,y) and friend(y,z), infer friend2(x,z)
derived = runtime.exec(
    runtime.match(("friend", (x, y)), ("friend", (y, z))).rewrite(
        [runtime.edge(x, z, rel="friend2", rule="two_hop")],
        mode="all",
        limit=1000,
    )
)
```

### Program Execution / State Transition

```python
pc, nxt = runtime.vars("pc nxt")

# state(machine, pc) + step(pc,nxt) -> state(machine, nxt)
runtime.exec(runtime.rewrite(to=[
    runtime.edge("m1", "n0", rel="state"),
    runtime.edge("n0", "n1", rel="step"),
    runtime.edge("n1", "n2", rel="step"),
]))

step1 = runtime.exec(
    runtime.match(("state", ("m1", pc)), ("step", (pc, nxt))).rewrite(
        [runtime.edge("m1", nxt, rel="state")],
        mode="first",
    )
)
```

## Result Interop

`runtime.exec(...)` returns a list of rewrite/match results with:
- `bindings`: variable -> node id
- `hyperedges`: rewritten/matched edges for that result

Use helper conversions:

```python
out = runtime.exec(runtime.match(("friend", (x, y)), mode="all", limit=100))

# list[dict] for normal Python processing
records = [row["bindings"] for row in out]

# pandas DataFrame
import pandas as pd
df = pd.DataFrame(records)

# numpy array (column order is explicit here)
import numpy as np
arr = np.array([[r.get("x"), r.get("y")] for r in records], dtype=object)

# matplotlib (counts per value in x)
import matplotlib.pyplot as plt
df["x"].value_counts().plot(kind="bar")
plt.show()
```

Optional dependencies:
- `pip install pandas numpy matplotlib`

## Notes

- `match(...).where(...)` is chainable.
- Pattern strings and `Var` are variables.
- Literal node ids in patterns use `runtime.const("node_id")`.
- New RHS variables create fresh nodes.
- Rewrites execute atomically in one SQLite transaction.
