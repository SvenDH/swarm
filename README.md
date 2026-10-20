# Monty agents

A small, asynchronous code-agent worker with a Rich REPL. The runtime lives in
`agents.py`, presentation in `main.py`, and reviewed library adapters in `wrappers/`.

## Run

```sh
uv sync --extra libraries
# Set LLM_API_KEY, LLM_BASE_URL and LLM_MODEL in .env or the environment.
uv run main.py --with-worker
```

Or run `uv run main.py --worker` and connect with `uv run main.py`.
Use `/new`, `/sessions`, `/use ID`, `/resume`, and `/quit`.
Escape stops the active agent without discarding its resumable work.

Assistant text, provider-supplied reasoning, code, output and errors stream in
separate blocks. Reasoning fields are saved verbatim for provider continuation;
private reasoning is never invented or requested.

## Python usage

Use ordinary imports for installed libraries:

```python
import numpy as np
from PIL import Image

rng = np.random.default_rng(42)
pixels = rng.integers(0, 256, size=(64, 64, 3), dtype="uint8")
Image.fromarray(pixels).save("noise.png")
display("noise.png")
del pixels
```

Supported packages are discovered from the host, but imports and operations go
through explicit allowlists. Host modules, attributes, paths, arbitrary callbacks,
pickle and subprocess access are not exposed.

Keep code simple:

- Run scripts with `exec(open("script.py").read())`; explicit namespace dictionaries
  are supported too.
- Release large values with `del`. `globals()` provides a read-only snapshot;
  use `globals().copy()` when a mutable namespace is needed.
- Supported introspection includes `dir`, `vars`, `getattr`, `hasattr` and `callable`.
- Prefer vectorized operations and short blocks over many library round trips.
- Use `np.random.default_rng(seed)` or the supported legacy NumPy random APIs.
  `Image.point` accepts lookup tables and callbacks evaluated inside Monty.
- Keep task progress and durable facts in `MEMORY.md`. It is loaded before each
  model request. Long conversations are compacted into summaries while preserving
  recent messages and complete tool-call batches.

Reviewed standard-library adapters provide session-scoped operations, not host
introspection or arbitrary imports. This is a supported Python subset, not a full
CPython replacement. Legacy command-style library APIs and automatic repair of
old provider histories are not supported.

Ordinary and dynamic imports share the approved module set. Native Monty modules
include `sys`, `json`, `time`, `random`, `base64`, and `binascii`; introspection of
these modules stays inside the interpreter. Additional adapters provide `zlib`
compression/checksums, `struct` packing, `hashlib` hashes, typed `array.array`
values, and in-memory `io.BytesIO`/`StringIO` streams. These objects stay in RAM,
not in checkpoints. Binary streams also work with
Pillow's open/save operations.

`isinstance(image, Image.Image)` and `isinstance(array, np.ndarray)` are supported
without exposing host classes or constructors. General `type()` identity remains
that of the facade. The adapters expose reviewed subsets, not every stdlib API:
file descriptors, file-backed streams, hash file readers, raw buffers and arbitrary
host imports are unavailable. Known output allocations are checked against the
worker budget, including decompression and variable-length digests.

## State and recovery

One worker owns the database. Agents have isolated homes at `/home/<name>/`
and disk checkouts beneath `workspaces/<name>/`.
Files from `template/` are copied into each new home.

SQLite stores conversation history, interpreter checkpoints, file metadata,
pending prompts and executions. File bytes live in immutable objects outside
SQLite, accessed through the `Storage` interface. Checkouts are disposable.

Native arrays, images, frames, figures, buffers and similar objects live in a
worker-wide RAM store. Monty holds only small session-authenticated references;
operations do not copy native payloads through the interpreter. Views and aliases
refer to the same native data. Explicit exports such as `np.save` or image saving
still write durable files. `zip`, `iter`, and `enumerate` accept array facades.
Fitted estimators stay native in RAM. Plot/document commands retain RAM-only
copies of their input data so later edits do not change previously recorded commands;
there is no native checkpoint encoding or recipe decoding.

An execution is recorded before it starts. Only a successful checkpoint and
its associated files commit. A failed block rolls back ordinary variables and files,
and invalidates that session's native objects rather than preserving partial mutations;
printed output remains visible but does not prove a write committed.
After a restart, pending work resumes from the last committed state. Native
objects are deliberately not restored: their old references raise an error
instructing the agent to recompute or reload them. Python code, ordinary values,
conversation history and saved files remain durable. Merely reopening a session
handle in the same worker preserves its resident native objects.
Completed execution IDs are deduplicated. Escape persists a paused state until
explicitly resumed.

This version has **no backward compatibility or automatic migrations**.
Old schemas and incompatible interpreter snapshots are rejected, not rewritten
or deleted. Use a fresh database and workspace root when upgrading:

```sh
uv run main.py --with-worker --database fresh.db --workspaces fresh-workspaces
```

## Shared memory

The default execution budget is **1 GiB per worker**, not per agent. Interpreter
checkpoints stay on disk; native objects stay in the shared, budgeted RAM store.
Least-recently-used native references may be evicted to make space and must then
be recomputed or reloaded. Shared NumPy backing buffers are counted once.
Code blocks queue for one shared heap reservation;
model requests remain concurrent.

Configure it with `--memory-mb 1024`, `SWARM_MEMORY_BYTES=1073741824`, or
`Runtime(..., memory_budget=1024**3)`. There are no per-session overrides.
The budget is fixed for a database because Monty embeds it in checkpoints;
changing it requires a fresh database.

Checkpointing reserves space for two serialized copies plus 25% headroom.
Known native allocations are preflighted and retained payloads are budgeted with
checkpoint headroom, but arbitrary native internals, scratch memory, package
caches and host serialization are not fully accounted for. This is
**not a hard process-RSS quota**; use OS-level isolation for that.

General allocation errors are plain `MemoryError`; NumPy allocation errors
retain its requested-size, shape and dtype diagnostic.

## Execution deadlines

Python blocks have a 60-second wall-clock budget, including interpreter loading,
host-library calls and checkpointing. Set `--execution-seconds 120` on the worker
or `Runtime(..., execution_timeout=120)` to change it. The REPL shows elapsed time
and library-call counts during long blocks, and identifies waiting agents.

Timeouts roll back the block and are saved as completed tool errors, not replayed
as interrupted executions after a restart. The model can use the error to split
its work. Escape still pauses the turn and leaves interrupted work resumable.

An in-flight native operation cannot safely be killed inside the shared Python
process: timeout/cancellation stops further callbacks, then waits for that operation
to drain before releasing the shared memory slot. That drain can exceed the
deadline and is shown in the REPL. This is not a hard native-code kill deadline.
Native operations use RAM references; explicitly fetching large plain lists or
bytes still transfers that data to the interpreter.

## Embedded use

```python
import asyncio
from agents import Runtime

async def main():
    async with Runtime("agents.db") as runtime:
        agent = await runtime.agent()
        print(await agent.run("Create a small chart and save it."))

asyncio.run(main())
```

Share one `Runtime` per database. It owns the client, IO threads, execution
pool and per-agent locks. Blocking IO and library work stay off the event loop.

## Tests

```sh
PYTHONPATH=tests uv run --extra libraries python -m unittest discover -s tests -q
```

Tests use real SQLite, Monty, files and worker sockets with a mocked model
provider. They do not establish live-provider compatibility.
