# SQLite Hypergraph Rewriter

Single API function:

- `runtime.exec(command[, command2, ...])` for the default graph

It executes an object-based DSL command and returns the result.

No backward-compat migrations are applied. If the schema changes, recreate the DB file.

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
- `runtime.exec(..., temp=True)` for ephemeral execution on current DB state (rollback at end)
- `runtime.edge(..., temp=True)` / `runtime.node(..., temp=True)` for inline virtual seed terms
- Terms are explicit: use `runtime.edge(...)` / `runtime.node(...)` only (tuple syntax is not supported)

Rewrite is expressed as `match(...).rewrite(...)` or top-level `rewrite(..., to=[...])`.
`limit=None` is the default and means "all matches".

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

# Atomic batch
batch = runtime.exec(
    runtime.rewrite(to=[runtime.edge("u", "v", rel="seed")]),
    runtime.match(runtime.edge("u", "v", rel="seed")),
)

x, y, z, u = runtime.vars("x y z u")

# Query
m = runtime.exec(
    runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u)).where(x.kind == "person", runtime.on(1).weight >= 0.5),
)
print(json.dumps(m, indent=2))

# Rewrite
x, y, z, u, v = runtime.vars("x y z u v")
r = runtime.exec(
    runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u)).rewrite(
        [runtime.edge(x, v, u), runtime.edge(y, v, z), runtime.edge(v, v, u)],
        limit=100,
    ),
)
print(json.dumps(r, indent=2))
```

## In-Memory Subgraphs (Virtual Queries / Execution)

Use inline `temp=True` edges/nodes in `runtime.exec(...)` to inject virtual terms for a command.
These temp terms are overlay-only and rolled back after returning results.

```python
x, y = runtime.vars("x y")
virtual = runtime.exec(
    runtime.match(runtime.edge(x, y, rel="friend"), limit=10).where(
        x.kind == "person", runtime.on(1).weight >= 0.8
    ),
    runtime.edge("a", "b", rel="friend", weight=0.9, temp=True),
    runtime.node("a", kind="person", temp=True),
)
```

Virtual program step:

```python
pc, nxt = runtime.vars("pc nxt")
step = runtime.exec(
    runtime.match(
        runtime.edge(runtime.const("m1"), pc, rel="state"),
        runtime.edge(pc, nxt, rel="step"),
    ).rewrite([runtime.edge(runtime.const("m1"), nxt, rel="state")]),
    runtime.edge("m1", "n0", rel="state", temp=True),
    runtime.edge("n0", "n1", rel="step", temp=True),
)
```

`temp=True` inline terms are always ephemeral (result-only):

```python
runtime.exec(
    runtime.match(runtime.edge(x, y, rel="friend")).rewrite([runtime.edge(x, y, rel="friend2")]),
    runtime.edge("a", "b", rel="friend", temp=True),
)
```

Ephemeral program state (single call, rollback at end):

```python
pc, nxt, x, y = runtime.vars("pc nxt x y")
out = runtime.exec(
    runtime.rewrite(to=[
        runtime.edge("m1", "n0", rel="state"),
        runtime.edge("n0", "n1", rel="step"),
    ]),
    runtime.match(runtime.edge(runtime.const("m1"), pc, rel="state"), runtime.edge(pc, nxt, rel="step")).rewrite(
        [runtime.edge("m1", nxt, rel="state")],
    ),
    runtime.match(runtime.edge(x, y, rel="state")),
    temp=True,
)
```

## Optional Vector Embeddings In `where(...)`

Store embeddings on either nodes or edges using the first-class `embedding=` field.

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
    runtime.match(runtime.edge(x, y, rel="friend"), limit=5).where(
        runtime.on(1).embedding.similar(q, min_score=0.75)
    )
)

# Namespace-scoped node embedding similarity
g1 = runtime.ns("g1")
u, v = runtime.vars("u v")
g1_hits = runtime.exec(
    g1.match(g1.edge(u, v, rel="friend"), limit=5).where(
        u.embedding.similar(q, min_score=0.7)
    )
)
```

Environment notes:
- `sqlite-vec` is required and used directly via `vec_distance_cosine`.

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

### Segregating Different Graphs (Namespaces)

Use a namespace handle so one DB can host many logical graphs safely.

```python
g1 = runtime.ns("g1")
g2 = runtime.ns("g2")

runtime.exec(
    g1.rewrite(to=[g1.edge("alice", "bob", rel="friend")]),
    g2.rewrite(to=[g2.edge("alice", "bob", rel="friend")]),
)
```

Namespace-aware matching/rewrite:

```python
x, y = runtime.vars("x y")
g1 = runtime.ns("g1")

# only matches rows stored in namespace "g1"
g1_matches = runtime.exec(g1.match(g1.edge(x, y, rel="friend"), limit=100))
```

Namespaces are stored in a dedicated SQL column, so node ids can be reused across namespaces.

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
    runtime.match(runtime.edge(x, y, rel="friend")).where(
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
    runtime.match(runtime.edge(x, y, rel="friend"), runtime.edge(y, z, rel="friend")).rewrite(
        [runtime.edge(x, z, rel="friend2", rule="two_hop")],
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
        runtime.edge(runtime.const("Rain"), rain, rel="evidence"),
        runtime.edge(runtime.const("Sprinkler"), sprinkler, rel="evidence"),
        runtime.edge(rain, sprinkler, wet, rel="cpt_wetgrass"),
    ).where(runtime.on(3).p >= 0.8).rewrite(
        [runtime.edge("WetGrass", wet, rel="belief", source="cpt")],
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
    runtime.match(runtime.edge(runtime.const("m1"), pc, rel="state"), runtime.edge(pc, nxt, rel="step")).rewrite(
        [runtime.edge("m1", nxt, rel="state")],
    )
)
```

### Program Tree Operators

RHS node/data values can use expression operators built from bound vars:

```python
a, b, out = runtime.vars("a b out")
runtime.exec(runtime.rewrite(to=[runtime.edge("2", "3", "sum", rel="add")]))

step = runtime.exec(
    runtime.match(runtime.edge(a, b, out, rel="add")).rewrite(
        [runtime.node(out, value=a + b, diff=a - b, prod=a * b, quo=b // a)],
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
out = runtime.exec(runtime.match(runtime.edge(x, y, rel="friend"), limit=100))

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
    runtime.match(runtime.edge(chunk, ent, rel="mentions"), limit=20).where(
        chunk.embedding.similar(q, min_score=0.75)
    )
)
expanded = runtime.exec(
    runtime.match(runtime.edge(ent, nbr, rel="related"), limit=20).where(
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
