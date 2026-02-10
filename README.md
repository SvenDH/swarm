# SQLite Hypergraph Rewriter

Single API function:

- `graph.exec(command[, command2, ...])` for the default graph

It executes an object-based DSL command and returns the result.

No backward-compat migrations are applied. If the schema changes, recreate the DB file.

## Object DSL

Core objects:

- `graph.vars("x y z")` (friendly alias)
- `graph.const("node_id")` for literal node ids in patterns
- `graph.on(1)` for edge-property filters
- `graph.edge(*nodes, rel="_", **props)` (friendly term builder)
- `graph.node("id", **props)` for node metadata edges
- `graph.ns("name")` namespace handle
- `graph.match(...)`
- `field.similar(query, min_score=0.0)` for embedding similarity in `where(...)`
- `graph.exec(..., temp=True)` for ephemeral execution on current DB state (rollback at end)
- `graph.edge(..., temp=True)` / `graph.node(..., temp=True)` for inline virtual seed terms
- Terms are explicit: use `graph.edge(...)` / `graph.node(...)` only (tuple syntax is not supported)

Rewrite is expressed as `match(...).rewrite(...)` or top-level `rewrite(..., to=[...])`.
`limit=None` is the default and means "all matches".

## Usage

```python
import json
import graph

# Seed graph: empty LHS rewrite
graph.exec(graph.rewrite(to=[
    graph.edge("a", "a", "b"),
    graph.edge("b", "c", "d", rel="r2"),
    graph.edge("a", "b", "c", rel="friend", weight=0.8),
    graph.node("a", kind="person", embedding=[0.9, 0.1, 0.0]),
]))

# Atomic batch
batch = graph.exec(
    graph.rewrite(to=[graph.edge("u", "v", rel="seed")]),
    graph.match(graph.edge("u", "v", rel="seed")),
)

x, y, z, u = graph.vars("x y z u")

# Query
m = graph.exec(
    graph.match(graph.edge(x, x, y), graph.edge(y, z, u)).where(x.kind == "person", graph.on(1).weight >= 0.5),
)
print(json.dumps(m, indent=2))

# Rewrite
x, y, z, u, v = graph.vars("x y z u v")
r = graph.exec(
    graph.match(graph.edge(x, x, y), graph.edge(y, z, u)).rewrite(
        [graph.edge(x, v, u), graph.edge(y, v, z), graph.edge(v, v, u)],
        limit=100,
    ),
)
print(json.dumps(r, indent=2))
```

## In-Memory Subgraphs (Virtual Queries / Execution)

Use inline `temp=True` edges/nodes in `graph.exec(...)` to inject virtual terms for a command.
These temp terms are overlay-only and rolled back after returning results.

```python
x, y = graph.vars("x y")
virtual = graph.exec(
    graph.match(graph.edge(x, y, rel="friend"), limit=10).where(
        x.kind == "person", graph.on(1).weight >= 0.8
    ),
    graph.edge("a", "b", rel="friend", weight=0.9, temp=True),
    graph.node("a", kind="person", temp=True),
)
```

Virtual program step:

```python
pc, nxt = graph.vars("pc nxt")
step = graph.exec(
    graph.match(
        graph.edge(graph.const("m1"), pc, rel="state"),
        graph.edge(pc, nxt, rel="step"),
    ).rewrite([graph.edge(graph.const("m1"), nxt, rel="state")]),
    graph.edge("m1", "n0", rel="state", temp=True),
    graph.edge("n0", "n1", rel="step", temp=True),
)
```

`temp=True` inline terms are always ephemeral (result-only):

```python
graph.exec(
    graph.match(graph.edge(x, y, rel="friend")).rewrite([graph.edge(x, y, rel="friend2")]),
    graph.edge("a", "b", rel="friend", temp=True),
)
```

Ephemeral program state (single call, rollback at end):

```python
pc, nxt, x, y = graph.vars("pc nxt x y")
out = graph.exec(
    graph.rewrite(to=[
        graph.edge("m1", "n0", rel="state"),
        graph.edge("n0", "n1", rel="step"),
    ]),
    graph.match(graph.edge(graph.const("m1"), pc, rel="state"), graph.edge(pc, nxt, rel="step")).rewrite(
        [graph.edge("m1", nxt, rel="state")],
    ),
    graph.match(graph.edge(x, y, rel="state")),
    temp=True,
)
```

## Optional Vector Embeddings In `where(...)`

Store embeddings on either nodes or edges using the first-class `embedding=` field.

```python
graph.exec(graph.rewrite(to=[
    graph.node("alice", kind="person", embedding=[0.9, 0.1, 0.0]),
    graph.node("bob", kind="person", embedding=[0.85, 0.15, 0.0]),
    graph.edge("alice", "bob", rel="friend", embedding=[0.8, 0.2, 0.0]),
]))
```

Search:

```python
q = [1.0, 0.0, 0.0]

# Edge embedding similarity
x, y = graph.vars("x y")
near_edges = graph.exec(
    graph.match(graph.edge(x, y, rel="friend"), limit=5).where(
        graph.on(1).embedding.similar(q, min_score=0.75)
    )
)

# Namespace-scoped node embedding similarity
g1 = graph.ns("g1")
u, v = graph.vars("u v")
g1_hits = graph.exec(
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
graph.edge("src", "dst", rel="link_type", weight=0.9)

# node attributes
graph.node("src", kind="entity", label="Person")

# n-ary relations
graph.edge("a", "b", "c", rel="event", ts=1700000000)
```

### Segregating Different Graphs (Namespaces)

Use a namespace handle so one DB can host many logical graphs safely.

```python
g1 = graph.ns("g1")
g2 = graph.ns("g2")

graph.exec(
    g1.rewrite(to=[g1.edge("alice", "bob", rel="friend")]),
    g2.rewrite(to=[g2.edge("alice", "bob", rel="friend")]),
)
```

Namespace-aware matching/rewrite:

```python
x, y = graph.vars("x y")
g1 = graph.ns("g1")

# only matches rows stored in namespace "g1"
g1_matches = graph.exec(g1.match(g1.edge(x, y, rel="friend"), limit=100))
```

Namespaces are stored in a dedicated SQL column, so node ids can be reused across namespaces.

### Knowledge Graph (KG)

```python
graph.exec(graph.rewrite(to=[
    graph.edge("alice", "openai", rel="works_at", source="hr"),
    graph.edge("alice", "sf", rel="lives_in"),
    graph.node("alice", label="Person"),
    graph.node("openai", label="Org"),
]))
```

### Bayesian Network

```python
graph.exec(graph.rewrite(to=[
    graph.edge("Rain", "WetGrass", rel="parent"),
    graph.edge("Sprinkler", "WetGrass", rel="parent"),
    graph.node("Rain", kind="bayes", states=["T", "F"], cpt={"T": 0.2, "F": 0.8}),
    graph.node("Sprinkler", kind="bayes", states=["T", "F"], cpt={"T": 0.4, "F": 0.6}),
    graph.node("WetGrass", kind="bayes", states=["T", "F"], cpt={
        "Rain=T,Sprinkler=T": {"T": 0.99, "F": 0.01},
        "Rain=T,Sprinkler=F": {"T": 0.9, "F": 0.1},
        "Rain=F,Sprinkler=T": {"T": 0.8, "F": 0.2},
        "Rain=F,Sprinkler=F": {"T": 0.0, "F": 1.0},
    }),
]))
```

### Causal Graph

```python
graph.exec(graph.rewrite(to=[
    graph.edge("Smoking", "Cancer", rel="causes", strength=0.7),
    graph.edge("Tar", "Cancer", rel="mediates"),
    graph.edge("Tax", "Smoking", rel="intervenes", effect="-"),
    graph.node("Smoking", kind="variable"),
    graph.node("Cancer", kind="outcome"),
]))
```

### AST / Program Graph

```python
graph.exec(graph.rewrite(to=[
    graph.node("n_if", kind="ast", type="If"),
    graph.node("n_cond", kind="ast", type="Compare"),
    graph.node("n_then", kind="ast", type="Assign"),
    graph.edge("n_if", "n_cond", rel="child", slot="test", idx=0),
    graph.edge("n_if", "n_then", rel="child", slot="body", idx=0),
]))
```

### Factor / Hypergraph Style

```python
graph.exec(graph.rewrite(to=[
    graph.edge("X1", "X2", "X3", rel="factor", fn="phi1"),
    graph.edge("X2", "X4", rel="factor", fn="phi2"),
]))
```

## Common Workflows

### Search / Filtering

```python
x, y = graph.vars("x y")

# find friend edges where source node is a Person
out = graph.exec(
    graph.match(graph.edge(x, y, rel="friend")).where(
        graph.on(1).weight >= 0.5,
        x.label == "Person",
    )
)
```

### Inference (Forward Rule Application)

```python
x, y, z = graph.vars("x y z")

# if friend(x,y) and friend(y,z), infer friend2(x,z)
derived = graph.exec(
    graph.match(graph.edge(x, y, rel="friend"), graph.edge(y, z, rel="friend")).rewrite(
        [graph.edge(x, z, rel="friend2", rule="two_hop")],
        limit=1000,
    )
)
```

### Bayesian Network Inference (Rule-Based)

```python
rain, sprinkler, wet = graph.vars("rain sprinkler wet")
graph.exec(graph.rewrite(to=[
    graph.edge("T", "T", "T", rel="cpt_wetgrass", p=0.99),
    graph.edge("T", "F", "T", rel="cpt_wetgrass", p=0.90),
    graph.edge("F", "T", "T", rel="cpt_wetgrass", p=0.80),
    graph.edge("F", "F", "F", rel="cpt_wetgrass", p=1.00),
    graph.edge("Rain", "T", rel="evidence"),
    graph.edge("Sprinkler", "F", rel="evidence"),
]))

posterior = graph.exec(
    graph.match(
        graph.edge(graph.const("Rain"), rain, rel="evidence"),
        graph.edge(graph.const("Sprinkler"), sprinkler, rel="evidence"),
        graph.edge(rain, sprinkler, wet, rel="cpt_wetgrass"),
    ).where(graph.on(3).p >= 0.8).rewrite(
        [graph.edge("WetGrass", wet, rel="belief", source="cpt")],
    )
)
```

### Program Execution / State Transition

```python
pc, nxt = graph.vars("pc nxt")

# state(machine, pc) + step(pc,nxt) -> state(machine, nxt)
graph.exec(graph.rewrite(to=[
    graph.edge("m1", "n0", rel="state"),
    graph.edge("n0", "n1", rel="step"),
    graph.edge("n1", "n2", rel="step"),
]))

step1 = graph.exec(
    graph.match(graph.edge(graph.const("m1"), pc, rel="state"), graph.edge(pc, nxt, rel="step")).rewrite(
        [graph.edge("m1", nxt, rel="state")],
    )
)
```

### Program Tree Operators

RHS node/data values can use expression operators built from bound vars:

```python
a, b, out = graph.vars("a b out")
graph.exec(graph.rewrite(to=[graph.edge("2", "3", "sum", rel="add")]))

step = graph.exec(
    graph.match(graph.edge(a, b, out, rel="add")).rewrite(
        [graph.node(out, value=a + b, diff=a - b, prod=a * b, quo=b // a)],
    )
)
```

Supported expression operators:
- Arithmetic: `+`, `-`, `*`, `/`, `//`, `%`, `**`, unary `+`/`-`
- String: `.concat(...)`, `.lower()`, `.upper()`, `.strip()`, `.replace(old,new)`, `.strlen()`

## Result Interop

`graph.exec(...)` returns a `graph.Result` (list-compatible) with:
- `bindings`: variable -> node id
- `hyperedges`: rewritten/matched edges for that result

Use helper conversions:

```python
out = graph.exec(graph.match(graph.edge(x, y, rel="friend"), limit=100))

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

## Benchmarks

General match/rewrite benchmark:

```bash
.venv/bin/python benchmarks/benchmark_graph.py
```

`temp=True` rewrite throughput benchmark:

```bash
.venv/bin/python benchmarks/benchmark_temp_rewrite.py --pairs 500 --runs 40 --limit 500
```

## GraphRAG Example

Build:

```python
chunk, ent, nbr = graph.vars("chunk ent nbr")
graph.exec(graph.rewrite(to=[
    graph.node("chunk:1", text="Alice works at OpenAI in SF", embedding=[0.95, 0.05, 0.0]),
    graph.node("chunk:2", text="Bob lives in NYC", embedding=[0.10, 0.90, 0.0]),
    graph.node("Alice", kind="entity"),
    graph.node("OpenAI", kind="entity"),
    graph.edge("chunk:1", "Alice", rel="mentions"),
    graph.edge("chunk:1", "OpenAI", rel="mentions"),
    graph.edge("Alice", "OpenAI", rel="related", weight=0.9),
]))
```

Query (semantic retrieval + expansion):

```python
q = [1.0, 0.0, 0.0]
hits = graph.exec(
    graph.match(graph.edge(chunk, ent, rel="mentions"), limit=20).where(
        chunk.embedding.similar(q, min_score=0.75)
    )
)
expanded = graph.exec(
    graph.match(graph.edge(ent, nbr, rel="related"), limit=20).where(
        graph.on(1).weight >= 0.7
    )
)
```

Optional dependencies:
- `pip install pandas numpy matplotlib`

## Agent Hierarchy Example

Create a root agent, then let agents spawn their own connected subagents.
Every agent has its own interpreter node that shares one single graph node.

The hierarchy creates one context node and one memory-graph node per agent.
Parent/child structure is depth-limited (`max depth = 3`).

Context-level auto-routing channels:
- `commands_down_ctx`: parent-context -> child-context
- `answers_up_ctx`: child-context -> parent-context
- `answers_lateral_ctx`: sibling-context -> sibling-context
- `publishes_to_graph`: context -> shared graph node
- `consults_memory`: context -> local memory graph
- `writes_memory`: context -> local memory graph

ACL edges (user/group based):
- Bootstrap ACL terms: `acl.terms(namespace, name, layer=..., context=..., memory=..., graph=...)`
- Wrapper: `acl.client(name, namespace)` with `exec/allow/deny/join/leave` (build commands with `graph`)
- Check access: `w.can(...)` from wrapper or `acl.check(subject, ...)`

```bash
.venv/bin/python hierarchy.py --namespace agents --db-path agent_hierarchy.db
```

Programmatic usage:

```python
from hierarchy import route_messages_once, spawn_agent
import acl
import graph

root = spawn_agent(None, namespace="agents")["agent"]
print(root)
l1a = spawn_agent(root, None, namespace="agents")["agent"]
l1b = spawn_agent(root, None, namespace="agents")["agent"]

# dynamic spawn by parent, depth-limited (max depth=3)
leaf = spawn_agent(l1a, None, namespace="agents")["agent"]
print(leaf)

g = graph.ns("agents")
x = graph.vars("x")[0]
l2 = graph.exec(g.match(g.node(x), limit=1).where(x.kind == "agent", x.layer == 2, x.parent == l1a))[0]["bindings"]["x"]
graph.exec(g.rewrite(to=[
    g.edge(f"ctx:{l2}", "msg:1", rel="out_answer", topic="status"),
    g.edge(f"ctx:{root}", "msg:2", rel="out_command", task="delegate"),
    g.edge(f"ctx:{l2}", "msg:3", rel="out_graph", kind="log"),
]))
print(route_messages_once("agents"))

# ACL check (direct + group inherited)
print(acl.client(root, "agents").can(action="read"))

# ACL wrapper around graph with access checks
w = acl.client(root, "agents")
print(w.exec(g.match(g.node(x), limit=1).where(x.kind == "agent")))
```

Routing conventions:
- outbound: `out_answer`, `out_command`, `out_graph`
- delivered: `in_answer`, `in_command`, `in_graph`
- memory writes: `memory_item(memory_graph, message)`

## Notes

- `match(...).where(...)` is chainable.
- Pattern strings and `Var` are variables.
- Literal node ids in patterns use `graph.const("node_id")`.
- New RHS variables create fresh nodes.
- Rewrites execute atomically in one SQLite transaction.
