# SQLite Hypergraph Rewriter

Single API function:

- `runtime.exec(command[, command2, ...])` for the default graph

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
- `field.similar(query, min_score=0.0)` for embedding similarity in `where(...)`

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
    runtime.node("a", kind="person", embedding=[0.9, 0.1, 0.0]),
]))

# Atomic batch (equivalent to exec([cmd1, cmd2, ...]))
batch = runtime.exec(
    runtime.rewrite(to=[runtime.edge("u", "v", rel="seed")]),
    runtime.match(("seed", ("u", "v"))),
)

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

## Optional Vector Embeddings In `where(...)`

Store embeddings on either nodes or edges using the reserved `embedding` property.

```python
runtime.exec(runtime.rewrite(to=[
    runtime.node("alice", kind="person", embedding=[0.9, 0.1, 0.0]),
    runtime.node("bob", kind="person", embedding=[0.85, 0.15, 0.0]),
    runtime.edge("alice", "bob", rel="friend", embedding=[0.8, 0.2, 0.0]),
]))
```

Search:

```python
q = [1.0, 0.0, 0.0]

# Edge embedding similarity
x, y = runtime.vars("x y")
near_edges = runtime.exec(
    runtime.match(("friend", (x, y)), mode="all", limit=5).where(
        runtime.on(1).embedding.similar(q, min_score=0.75)
    )
)

# Namespace-scoped node embedding similarity
g1 = runtime.ns("g1")
u, v = runtime.vars("u v")
g1_hits = runtime.exec(
    g1.match(("friend", (u, v)), mode="all", limit=5).where(
        u.embedding.similar(q, min_score=0.7)
    )
)
```

Environment notes:
- If `sqlite-vec` is available (`import sqlite_vec`) or `SQLITE_VEC_PATH` points to the extension, runtime uses `vec_distance_cosine`.
- Otherwise it falls back to a deterministic Python cosine function in SQLite.

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

### Bayesian Network Inference (Rule-Based)

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

posterior = runtime.exec(
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

### Program Tree Operators

RHS node/data values can use expression operators built from bound vars:

```python
a, b, out = runtime.vars("a b out")
runtime.exec(runtime.rewrite(to=[runtime.edge("2", "3", "sum", rel="add")]))

step = runtime.exec(
    runtime.match(("add", (a, b, out))).rewrite(
        [runtime.node(out, value=a + b, diff=a - b, prod=a * b, quo=b // a)],
        mode="first",
    )
)
```

Supported expression operators:
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `+`/`-`
- String: `.concat(...)`, `.lower()`, `.upper()`, `.strip()`, `.replace(old,new)`, `.strlen()`

## Result Interop

`runtime.exec(...)` returns a `runtime.Result` (list-compatible) with:
- `bindings`: variable -> node id
- `hyperedges`: rewritten/matched edges for that result

Use helper conversions:

```python
out = runtime.exec(runtime.match(("friend", (x, y)), mode="all", limit=100))

# Map rows
pairs = out.map(lambda row: (row["bindings"]["x"], row["bindings"]["y"]))

# list[dict] from bindings or edge properties
records = out.bindings("x", "y")
weights = out.edge_data("weight")

# pandas DataFrame
import pandas as pd
df = pd.DataFrame(records)
df_edge = pd.DataFrame(weights)
df_rows = pd.DataFrame(out.rows())

# numpy array (column order is explicit here)
import numpy as np
arr = np.array([[r["x"], r["y"]] for r in records], dtype=object)

# matplotlib (counts per value in x)
import matplotlib.pyplot as plt
df["x"].value_counts().plot(kind="bar")
plt.show()
```

Most useful `Result` helpers:
- `out.first()`
- `out.bindings(...)`
- `out.edge_data(...)`
- `out.rows(...)`

## GraphRAG Example

Build:

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
```

Query (semantic retrieval + expansion):

```python
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

Optional dependencies:
- `pip install pandas numpy matplotlib`

## Notes

- `match(...).where(...)` is chainable.
- Pattern strings and `Var` are variables.
- Literal node ids in patterns use `runtime.const("node_id")`.
- New RHS variables create fresh nodes.
- Rewrites execute atomically in one SQLite transaction.
