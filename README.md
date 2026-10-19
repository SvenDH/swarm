# Monty agents

Standard-library compatibility is session-scoped:

- `os`/`os.path` and `stat`: path operations, directory listing/walking, file status,
  removal/rename and the session environment. `chdir`, process APIs and host file
  descriptors are not supported. Environment snapshots are read-only.
- `shutil`: `copyfile`, `copy`, `copytree`, `move`, `rmtree`, over the same files as
  `open()`/`Path`. Timestamp/ownership preservation (`copy2`) is not implemented.
- `tempfile`: `gettempdir`, `gettempprefix`, `mkdtemp`, using session `/tmp`.
- `importlib`: `import_module` and `util.find_spec` for the supported registry,
  plus native `math`, `pathlib`, `datetime`. No unrestricted host imports.
- `resource`: read-only `RLIMIT_AS` reports the configured session budget, not a
  real process address-space quota. Limit changes and `getrusage` are unsupported.
- `platform.system()` reports the Linux execution profile. No host machine details
  are exposed. `gc.collect()` explicitly reports unsupported; it does not collect
  the shared host process or fabricate counters. Native `sys` remains unchanged.

`getattr`, `hasattr`, `callable`, `dir`, and `vars` understand supported library
objects; basic object-field discovery and caller-local `vars()`/`dir()` are also
available. These are partial compatibility APIs: live global namespace mutation,
CPython code objects (`compile`) and buffer-protocol `memoryview` remain unsupported.
`globals()` returns a read-only snapshot of module bindings, including inside
functions. It supports lookup, membership, iteration, `len`, `get`, `keys`,
`values`, `items`, and `copy`. Internal helper names and deleted names are omitted.
Snapshots retain their original bindings; referenced mutable values are not frozen.
Operations use the existing checkpoint/rollback and filesystem isolation. Filesystem
metadata is the current filesystem's metadata, not host inode/ownership information.

Reusable code can live in session files and run with `exec(open("script.py").read())`.
Dynamic `exec()` source, including nested calls, receives the same import and
library-operation translation as top-level code. Execution remains in the
interpreter with the same caller scope, filesystem isolation, resource limits,
and atomic rollback; source is never executed by host Python. NumPy/PIL version
queries work, and agent-facing failures use normal Python exception names rather
than interpreter-specific prefixes. This is still a supported Python subset,
not a promise of full CPython compatibility.

General allocation failures raise `MemoryError` with no message. NumPy allocation
failures retain NumPy's standard requested-size, shape and dtype message, including
preflight rejections. Internal session-budget/worker details are not exposed;
other exception types and saved history are unchanged.

`del` supports variable names (including multiple/unpacked targets), list/dict
entries, list slices and pandas columns, including inside `exec()`. Name deletion
releases the value and subsequent translated reads raise `NameError`; deletion
state survives restart and participates in atomic rollback. Internally names use
a checkpoint-safe deleted marker rather than native symbol-table removal; this
is not full CPython namespace introspection. Attribute deletion is not supported.

Agents can inspect images with `display(...)` inside the Python tool:

```python
from PIL import Image

image = Image.open("scene.png")
display(image)          # a PIL image
display("scene.png")   # or a session file path (pathlib.Path also works)
```

This requires Pillow on the host and a vision-capable model/provider. Explicit
`display()` calls send JPEG previews (at most 1024 pixels per side) using the
[image input format](https://developers.openai.com/api/docs/guides/images-vision).
Original files and image objects remain unchanged. Previews are stored outside
SQLite; only object references commit with successful code execution. They
survive restart and expand into image messages when calling the provider, after
all tool results in a batch. Context compaction summarizes older history rather
than retaining all old images. The REPL shows a display confirmation, not the
image pixels; last-expression image objects are not automatically displayed.

Each agent can keep durable preferences, corrections and reusable lessons in
`/home/<name>/MEMORY.md`, editing it with the existing Python file tools. No
extra service or memory database is required. Notes use the same atomic commits,
rollback, isolated storage and restart recovery as other files. Delete or edit
the file to forget or correct a memory; new sessions do not share notes. Seed
`template/MEMORY.md` if every new session should start with initial notes.

Before each normal model request, committed notes are reloaded as reference data
(not system instructions), outside conversation history. This keeps them fresh
after tool calls, compaction and restart without accumulating duplicate messages.
Automatic previews are limited to 8,000 characters; files over 32 KiB receive a
read-file notice instead. Longer details can live in other files linked from
MEMORY.md and be read on demand. Keep notes concise, verify stale facts, and do
not save secrets. Memory writing is agent-directed, not a guaranteed automatic
post-turn extraction step.

This deliberately small design borrows bounded notes from
[Hermes](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory/)
and file-based progressive disclosure from
[SandboxAgent](https://openai.github.io/openai-agents-js/guides/sandbox-agents/memory/).
[DeepSeek Harness](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/user/guide/mcp-memory.md)
also documents optional external memory integrations; those services are not
needed here.

Run the worker and REPL in one process:

```sh
uv run main.py --with-worker
```

In this mode `/quit` also shuts down the worker; unfinished work resumes on
the next start. Other REPL clients can connect while this process is running.

Or start the session worker, then connect from another terminal:

```sh
uv run main.py --worker
uv run main.py
```

Configure `LLM_API_KEY`, `LLM_BASE_URL` and `LLM_MODEL` on the worker, in the
environment or `.env`. With a separate worker, `/quit` or disconnecting does
not stop agents. Reconnect with `--session ID`. `/new`, `/sessions`, `/use ID`,
`/resume` and `/quit` are available. Both commands accept `--database` and
`--socket`; execution settings such as `--workers`, `--template` and
`--context-chars` belong on the worker command.

Press Escape during a REPL run to stop that session's worker task. The session
stays paused across worker restarts and reconnects; use `/resume` (or submit a new
prompt) to continue. Committed state is preserved. An in-flight native operation
may need to settle before cancellation finishes. Arrow/function keys do not
normally trigger a stop. Ordinary client disconnection still leaves work running.

The worker resumes every unfinished, non-paused session at startup, not just the session
selected in a REPL. Prompts are durably queued per session and run in order;
different sessions share the runtime's concurrency limits. Model/infrastructure
failures retry with backoff (up to 30 seconds); step budgets automatically yield
and continue. Recovered completed turns are not submitted again. Streaming output
is live, not an event archive; reconnecting observes remaining output and the
saved final answer.

Provider-supplied `reasoning_content` or `reasoning` text deltas stream into a
separate dimmed, magenta-bordered Reasoning panel. The exact supplied fields
are saved with assistant messages and sent back on continuation requests, as
required by thinking-mode providers. No reasoning is fabricated or requested
from providers that do not expose it. Summarization excludes these fields from
its input history. If old history is rejected for missing reasoning, the agent
rebases it through a fresh continuation summary, preserving files/globals and
completed tool effects. Other HTTP 400 errors stop rather than retry forever.

After the startup recovery scan, submissions wake only their own session; there
is no inbox polling or global rescan per completed prompt. Idle sessions retain
no runner task. The REPL renders streamed panels at its refresh rate, not once
per token.

Exactly one worker may own a database/socket. The local Unix socket is restricted
to its owner (`0600`); it is a trusted local interface, not a network API. Restart
the worker to recover after a crash. Run it under a service manager if it should
restart automatically when the process dies. Multiple REPL clients are supported,
but multiple execution-worker processes for one database are not.

For embedded execution (without the worker service), share one runtime across agents:

```python
import asyncio
from agents import Runtime

async def main():
    async with Runtime("agent.db", workers=4, model_concurrency=128) as runtime:
        agents = await asyncio.gather(*(runtime.agent() for _ in range(1000)))
        results = await asyncio.gather(*(agent.run("What is 6 * 7?") for agent in agents))

asyncio.run(main())
```

`await runtime.agent(session_id=id)` reopens a saved agent. `await agent.run()`
continues an interrupted turn. `execute(code)` runs code without a model request.
`state()`, `files()`, `read_file(path)` and `sessions()` are also awaitable.
`await CodeAgent.open(...)` is convenient for one agent; close it with
`await agent.close()`. Use one shared `Runtime` per database for concurrent agents.
Do not open an embedded runtime against a database owned by a worker.

Programmatic worker clients use `WorkerClient(database).request('agent')`,
`request('sessions')` and `request('run', session_id=id, prompt='...')`.
Pass `on_output(kind, text)` to receive streamed blocks. `Worker(...)` is an async
context manager for embedding the service; `await worker.serve()` hosts it.

Sessions automatically summarize older conversation history before model requests
once its serialized text reaches 120,000 characters. Recent messages and complete
tool-call/result batches are retained; earlier summaries are folded into the next
one. Configure `Runtime(..., context_chars=120_000)` or `--context-chars` on the worker;
use `context_chars=None` to disable it. This is a soft character threshold, not an
exact model token limit; unusually large individual messages can still exceed a
provider's context window.

Summaries stream in a **Context memory** panel and replace old messages atomically
in SQLite. A failed/interrupted summary leaves the original history available for
retry; a committed summary survives restart. Python globals, files and pending
executions are unchanged. Summarization is lossy: the original conversation prefix
is replaced, not archived.

Use Python's `open()` and `pathlib.Path` for virtual filesystem operations.
There are no direct `pwd`, `cd`, `ls`, `mkdir`, `read`, `write`, `edit`, `glob`,
`grep`, `mv` or `rm` host functions. The working directory remains the session home.

## Crash recovery

There is no event log or replay layer. SQLite stores only:

- `sessions`: the latest interpreter checkpoint, working directory and conversation.
- `files`: current paths and immutable storage keys, not file contents.
- `executions`: pending commands and their results, allowing safe retries.
- `prompts`: the worker's durable inbox, started state, final answers and retry errors.
- `runtime_secrets`: one host-only key used to authenticate library state across restarts.

A command is saved before execution. Its interpreter checkpoint, file links and
model-tool result commit in one SQLite transaction. Failed code rolls back its
files/globals; interrupted infrastructure leaves a pending command to retry.
Use `execution_id` with `execute()` when retrying an explicit command; model tool
IDs are deduplicated automatically. Incomplete model streams are never executed.
Model requests can repeat after a crash.

The inbox entry is marked started in the same transaction that appends its user
message. After restart, a started entry resumes existing history instead of
appending the prompt again, including a crash between saving the final assistant
message and saving the inbox result. The inbox table is created automatically
for existing checkpoint databases.

File objects are durably published before the database commit. Visible directories
under `workspaces/<name>/` are disposable copies, repaired when a session reopens.
Editing those directories directly does not update the agent. The storage
interface remains replaceable with a blob backend. Back up the database and
object storage together: checkpoints recover interrupted work, not a lost database.

This schema intentionally does not migrate event-sourced databases. Existing data
is not deleted; select a fresh database and workspace root, for example:

```sh
uv run main.py --database checkpoint.db --workspaces checkpoint-workspaces
```

## Starter files and execution

Files in `session_template/` are copied into `/home/<name>/` for each new session.
Nested/empty directories, hidden files and binary files are supported. Existing
sessions never consult the source again. Change the source with `--template PATH`
or `Runtime(..., template=PATH)`; use an empty directory for no starter files.
Symlinks/special files are rejected. Starter files must fit the session's
configured memory budget.

The runtime shares one SQLite connection, HTTP client, bounded Monty pool and IO
workers. Histories/checkpoints load only when admitted, not for every waiting
agent. Blocking IO and native-library callbacks run off-loop. Wrappers in
`wrappers/` import their libraries lazily. Output callbacks run on the event-loop
thread and should return promptly; all application printing/Rich styling is in
`main.py`.

## Library syntax

Library packages are optional. Install all supported packages and run with:

```sh
uv run --extra libraries main.py
```

Alternatively install only the packages you need in the host Python environment.
The agent discovers packages once per process, advertises only installed ones,
and rejects imports for missing packages. Restart after installing/removing a
package. Discovery does not import heavy libraries; actual imports remain lazy.
The default dependency set needs none of these libraries to run the agent.

Use normal import syntax in tool code:

```python
import numpy as np
import pandas as pd
from PIL import Image
from openpyxl import Workbook, load_workbook
from pypdf import PdfReader, PdfWriter
from docx import Document

array = np.arange(6).reshape(2, 3)
frame = pd.DataFrame({'value': [1, 2, 3]})
frame['double'] = frame['value'] * 2
frame.to_excel('data.xlsx', index=False)
Image.new('RGB', (100, 100), 'blue').save('image.png')
```

These are **curated compatibility APIs**, not arbitrary native imports. Monty
1.1 does not load these packages, so `wrappers/imports.py` translates imports,
method calls, indexing and operators. Standard-library imports remain native.
Supported constructors include arrays, DataFrames/Series/Indexes, images,
workbooks, PDF readers/writers and Word documents. Common reductions, slicing,
filtering, groupby/aggregation, file conversion and document editing are exposed
through explicit allowlists in the corresponding files under `wrappers/`.
`np.random.default_rng(seed)` creates an independently checkpointed generator.

Coverage is expanded explicitly, never by forwarding arbitrary public native
attributes. `wrappers/api.py` selects reviewed names and converts document enum
constants to plain integers; unavailable names are omitted for older host versions.
This is not an assertion that every unexposed API is unsafe: some require a new
virtual-IO, aliasing or durable-state adapter before they can be supported.

Expanded data-only APIs include:

- NumPy numeric trigonometric/hyperbolic/bitwise ufuncs, sets, bounded `pad` and
  `take`, machine limits (`finfo/iinfo`) and additional checkpointed RNG distributions.
- pandas pivots/melting/crosstabs, categoricals, date/timedelta indexes, `str`/`dt`
  accessors, cumulative statistics and deterministic sampling. Pivot reducers use
  an allowlist, not arbitrary method names such as `to_pickle` or `eval`.
- Pillow image enums, `ImageColor`, `ImageChops`, compositing, histogram/pixel
  helpers and checkpointed `ImageFont.load_default(size=...)`. Fonts remain
  embedded-only: no `truetype` host font lookup or arbitrary font objects.
- SciPy numeric `special` ufuncs, physical constants, more distributions,
  `ndimage` filters/morphology/zoom, linear algebra, signal functions/windows.
  Ufunc `out` arguments and uninitialized masked output remain blocked.
- Matplotlib named colors, `cm.<name>(values)`, additional axes styling and
  figure legends; Seaborn categorical/regression/ECDF plots, despine and palettes.
- openpyxl styles/number formats, docx/pptx integer enum namespaces and length
  units, PDF transformations/constants/search, ReportLab named colors/page sizes
  and drawing helpers, HTML traversal and XlsxWriter cell/range utilities.
- More sklearn numeric metrics and preprocessing, and SymPy typed elementary
  functions, constants, sets, linear solving and matrix methods. Symbolic string
  evaluation, code generation and arbitrary native constructors remain blocked.

The explicit export/member maps in each adapter are the source of truth, not
the complete API documentation of the host library. Optional packages are still
loaded only when installed. Restart the worker after changing wrappers.

Additional optional adapters expose these common subsets:

- `numpy`: bounded `tile/repeat/eye/diag/meshgrid`, reshaping and live array views,
  cumulative and NaN-aware reductions, quantiles, histograms, interpolation,
  comparisons and linear algebra (SVD, QR, eigenvalues, Cholesky, least squares).
  `numpy.fft` includes bounded `fft/ifft/rfft/irfft`, frequencies and shifts.
  `mgrid/ogrid` accept integer, floating-point and complex-step slice syntax;
  `indices`, `broadcast_to` and `bincount` check output sizes before allocation.
  Arithmetic/comparison/logical ufuncs include `add/subtract/multiply/divide` and
  `reduce/accumulate/reduceat/outer/at`. Indexed `at` updates persist through array
  aliases and restarts. Keyword `out=` accepts authenticated array targets,
  including views and masked ufunc calls with initialized outputs. Numeric dtype
  aliases such as `np.float32` are callable constructors and valid dtype arguments.
  `empty`/`empty_like` allocate initialized zeros to avoid exposing host memory.
- `PIL.ImageOps`: flipping, color/grayscale adjustments, fit/pad/contain/cover and
  bounded borders. `PIL.ImageEnhance` exposes Brightness/Contrast/Color/Sharpness
  with checkpointed source-image snapshots. `PIL.ImageFilter` includes bounded
  Gaussian/box blur, median/min/max/mode filters and common fixed filters.
  `PIL.ImageDraw.Draw(image)` supports shapes, lines, text and text measurements.
  Drawing contexts remain linked to their original image across restarts and
  failed executions roll back drawings. Text uses the embedded default font;
  host font files are unavailable, and font/stroke sizes and text masks are bounded.
- `matplotlib.pyplot`: `figure/subplots`, line/scatter/bar/histogram/image plots,
  step/stairs/stack/contour plots, spans, text/annotations, axes styling,
  figure labels/layout and PNG/SVG/PDF `savefig`. Plot calls record commands;
  native artist handles and global styles are not exposed.
  `matplotlib.colors` supports RGB/RGBA/hex and RGB/HSV conversions.
  `matplotlib.use('Agg')` is accepted without changing host-global settings;
  `get_backend()` returns `Agg`. GUI/custom backends are not exposed.
- `seaborn`: `scatterplot`, `lineplot`, `histplot`, `boxplot`, `barplot`,
  `heatmap`, `violinplot`, `kdeplot`. Pass `ax=` to draw into an existing figure;
  otherwise save through the returned axes' `.figure.savefig(...)`.
- `scipy`: distributions and statistical tests, linear algebra, signal/FFT,
  interpolation, linear programming, assignment and distances; ranks, spread,
  normality tests and `special` functions (softmax, sigmoid, gamma, erf).
  `scipy.integrate` supports sampled-data trapezoid/Simpson integration and their
  cumulative variants (not callback-based quadrature). `scipy.spatial.transform`
  exposes checkpointed `Rotation` constructors, conversions, apply and inverse.
  No Python callbacks.
- `pptx`: presentations, layouts/slides, text boxes, paragraphs/runs/fonts,
  tables and pictures; `pptx.util.Inches/Cm/Pt` and virtual `save`.
- `reportlab`: `pdfgen.canvas.Canvas`, text, standard PDF fonts, lines/shapes,
  images, page sizes/colors and `showPage/save`.
- `pdfplumber`: virtual-file `open`, pages, text/words/tables and cropping.
- `pymupdf` (also `fitz`): PDF `open`, pages, text extraction/insertion,
  bounded raster rendering, page creation and virtual `save`.
- `bs4`: `BeautifulSoup(..., 'html.parser')`, search/CSS selectors,
  text/attributes and tag edits. No network fetching or XML parsers.
- `sympy`: symbols, arithmetic, equations/solving, calculus, simplification
  and matrices. Build expressions from symbols, not strings; no `sympify`,
  `parse_expr` or `lambdify` evaluation.
- `sklearn`: scalers/encoders, train/test splitting, common metrics,
  LinearRegression/Ridge/LogisticRegression, KMeans and PCA. Fitted state is
  explicit primitive/array data, not pickle, and inference does not refit models.
- `xlsxwriter`: workbooks, sheets, formatting, cells/rows/formulas, charts
  and images; `close()` writes the XLSX into the virtual filesystem.

Figures and write-only Canvas/XlsxWriter objects checkpoint bounded drawing/write
recipes and render to memory on save. Matplotlib uses local `Figure` objects,
not shared pyplot state; rendering is serialized for native thread safety.
Figure axes collections are Python lists (nested lists for 2D subplots), not
NumPy arrays. Artist handles, interactive displays, arbitrary styles/fonts,
global Seaborn themes, ReportLab Platypus, and APIs not listed are not exposed.
Recipes are limited to 1,000 operations per figure/canvas/sheet/chart.

The compatibility objects store safe data/formats (not pickle or live host
handles), so they survive calls, restarts and transaction rollbacks. Workbook
cells, document paragraphs/runs and PDF pages remain views of their parent.
All library file IO—including `Path` arguments—uses the virtual filesystem.
Read-only operations avoid reserializing the original object. Libraries still
load lazily on the Python worker pool, never through an unrestricted host importer.
Native extensions are initialized there before Monty callback threads use them,
including when reopening a checkpoint in a fresh process.

This is intentionally not full NumPy/pandas/Python emulation: wildcard imports,
full package introspection/type checks, custom callbacks (`apply`), `query`/`eval`,
pickle IO and compatibility-object context managers are not supported. Imports
inside dynamic `exec` strings are translated; imports inside `eval` strings or
separate Python modules are not supported.
Names beginning `_library_` are reserved. Unsupported functions fail explicitly;
there is no fallback to arbitrary host attributes or host filesystem access.
Use numeric dtype constructors or names such as `'float64'`, and vectorized methods. There are
no fixed element-count or pixel-count caps: each session gets a byte budget,
50 MiB by default. Set the default for **new** sessions with:

```sh
uv run main.py --with-worker --memory-mb 512
```

Or set `SWARM_MEMORY_BYTES=536870912` in the worker environment/`.env`.
Embedded callers can use `Runtime(..., memory_budget=512 * 1024**2)` and override
individual new sessions with `await runtime.agent(memory_budget=128 * 1024**2)`.
`WorkerClient.request('agent', memory_budget=...)` also accepts bytes for a new
session. Budgets are saved in SQLite and survive restarts; reopening an existing
session retains its budget. Existing databases gain this field without losing
sessions or checkpoints. The old ARRAY/IMAGE/MONTY size settings are removed.

Monty enforces the budget across its live heap, including globals and serialized
library descriptors. Copies, wrapper overhead and checkpointing need headroom;
the sum of array `nbytes` is not all the memory required. Free unused variables
with `del`, reuse buffers and prefer smaller dtypes or chunks. Wrapper
allocation preflights use this same session budget, based on dtype/sample bytes,
not arbitrary array/image dimensions. Context-local checks isolate concurrent
sessions. Object dtypes remain prohibited; native shape constraints and iteration
limits remain in place.

This is **not a hard process RSS quota**. Native extension scratch allocations,
Python-side decoding/serialization, shared package caches and virtual-filesystem
buffers are outside Monty's heap accounting. Native preflights cover known
allocation shapes, not every temporary. Concurrent sessions can therefore use
more host memory than their interpreter heaps; use OS isolation/resource limits
when hard memory enforcement is required.

Direct library method calls use one authenticated bridge operation rather than
separate lookup/call operations. Read-only array receivers use buffer views over
validated NPY data instead of writable copies; mutations restore a writable copy.
The NPY restore path avoids the extra `np.load`/stream-read payload copy. These
changes reduce transient copying, but do not remove checkpoint headroom or make
the configured byte budget equal to usable array payload size.

Each Python checkout still permits 1,000 suspensions (external calls, lookups and
virtual OS operations). The short system prompt simply asks the model to avoid
too many library calls per code block, prefer vectorized operations, or split
work across blocks; it does not describe the suspension mechanism or counts.

NumPy supports `uint8` and other numeric dtype aliases, ellipsis indexing,
`nbytes` and `tobytes()`. PIL exposes resampling constants, `merge`, `split`,
`getchannel` and `putalpha`. Matplotlib exposes bounded `fig.add_axes` and
compression/quality-only `savefig(pil_kwargs=...)`. Monty does not support `del`
or `import gc`; use `value = None` to release references, and
`sorted(paths, key=str)` to order virtual paths.

## Native-library boundary

Native packages deliberately execute in the host process, outside Monty. The
bridge authenticates serialized objects and view paths before restoring or
replaying them. Its HMAC key stays on the host in SQLite, with separate derived
keys for each session. Knowing or overwriting the `_library_` Python helpers does
not grant permission to forge native state. Keep the database and backups private.

User dictionaries are escaped separately from protocol/recipe markers. Native
callbacks cannot be smuggled through object references, and Monty class proxies
are rejected before they reach native constructors (including legacy helpers).
Symbolic trees require typed nodes and ordinary identifier symbol names. SymPy
expression strings are rejected in constructors, matrix writes and operators.
Positional options receive the same checks as keyword
options. Legacy Pillow helpers use the same restricted image formats as imports.

Old, unsigned library checkpoints are deliberately rejected; create a new session
to use libraries after this upgrade. Saved databases/files are not deleted or
silently trusted/re-signed. New authenticated checkpoints survive normal restarts.

These checks restrict the exposed capability API, not the operating system.
Native-code vulnerabilities, parser/decompression bombs, and CPU/memory exhaustion
can still affect the host process. Use OS-isolated workers with resource limits
if executing actively hostile workloads; passing regression tests is not a
formal security guarantee.

## Tests

```sh
uv run --extra libraries python -m unittest discover -s tests -q
```

Tests cover wrappers, streaming, recovery after abrupt process death, cancellation,
session isolation, starter files and 1,000 agents sharing bounded resources.
Provider tests use mocked HTTP, not a live model.
