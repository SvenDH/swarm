# Monty agents

`uv run main.py` starts the async REPL. Configure `LLM_API_KEY`, `LLM_BASE_URL`
and `LLM_MODEL` in the environment or `.env`. Reopen with `--session ID`;
unfinished work continues automatically. `/new`, `/sessions`, `/use ID`,
`/resume` and `/quit` are available in the REPL.

Share one runtime across agents:

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
Multi-process hosting is not supported.

Use Python's `open()` and `pathlib.Path` for virtual filesystem operations.
There are no direct `pwd`, `cd`, `ls`, `mkdir`, `read`, `write`, `edit`, `glob`,
`grep`, `mv` or `rm` host functions. The working directory remains the session home.

## Crash recovery

There is no event log or replay layer. SQLite stores only:

- `sessions`: the latest interpreter checkpoint, working directory and conversation.
- `files`: current paths and immutable storage keys, not file contents.
- `executions`: pending commands and their results, allowing safe retries.
- `runtime_secrets`: one host-only key used to authenticate library state across restarts.

A command is saved before execution. Its interpreter checkpoint, file links and
model-tool result commit in one SQLite transaction. Failed code rolls back its
files/globals; interrupted infrastructure leaves a pending command to retry.
Use `execution_id` with `execute()` when retrying an explicit command; model tool
IDs are deduplicated automatically. Incomplete model streams are never executed.
Model requests can repeat after a crash.

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
Symlinks/special files are rejected and individual files are limited to 32 MiB.

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

Additional optional adapters expose these common subsets:

- `matplotlib.pyplot`: `figure/subplots`, line/scatter/bar/histogram/image plots,
  axes labels/scales/legends and PNG/SVG/PDF `savefig`.
- `seaborn`: `scatterplot`, `lineplot`, `histplot`, `boxplot`, `barplot`,
  `heatmap`, `violinplot`, `kdeplot`. Pass `ax=` to draw into an existing figure;
  otherwise save through the returned axes' `.figure.savefig(...)`.
- `scipy`: distributions and statistical tests, linear algebra, signal/FFT,
  interpolation, linear programming, assignment and distances. No Python callbacks.
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
package introspection/type checks, custom callbacks (`apply`), NumPy `out=`, `query`/`eval`,
pickle IO and compatibility-object context managers are not supported. Imports
inside dynamic `exec`/`eval` strings or separate Python modules are not translated.
Names beginning `_library_` are reserved. Unsupported functions fail explicitly;
there is no fallback to arbitrary host attributes or host filesystem access.
Use dtype names such as `'float64'` and explicit/vectorized methods. Arrays are
limited to 1,000,000 elements/32 MiB, images to 4,000,000 pixels and library
iteration to 10,000 items. Native-library work remains outside Monty's resource
accounting; these wrappers are not a separate OS-level resource sandbox.

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
