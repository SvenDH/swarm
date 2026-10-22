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
OpenCV (`import cv2`) supports reviewed image-processing operations, NumPy arrays,
and workspace-only image reads/writes. The optional `libraries` extra installs
`opencv-python-headless`; discovery also accepts an existing host OpenCV install.
GUI, cameras/video, model loaders and host file APIs are not exposed.

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

An execution is recorded before it starts. Ordinary Python exceptions commit the
partial interpreter and file state, and keep native mutations made before the error.
The error is recorded as the tool result, so recovery does not replay that block.
Crashes, timeouts, cancellations and failed checkpoint writes preserve the previous
durable checkpoint and invalidate native objects. Printed output alone does not
prove a write committed.
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

## Context compaction

Compaction adapts the prune-first, protected-tail and structured-handoff approach
from [Hermes Agent](https://github.com/NousResearch/hermes-agent/blob/71633f762d2d74faebe604d0b8a2a2e6cf48346a/agent/context_compressor.py)
(MIT), without importing its framework. Agent instructions are unchanged.

Use `--context-tokens 32000` (or `Runtime(context_tokens=32000)`) to set the
context-window budget; `context_tokens=None` disables automatic compaction.
This replaces `--context-chars`. Reserve the smaller of 8,192 tokens or one quarter
of the window for the response; compression starts at 80% of the remaining input
budget. Set the window for your actual model/provider, not just its model name.
`--response-tokens` / `Runtime(response_tokens=...)` changes the response cap.
DeepSeek summary requests explicitly disable thinking; normal agent calls are unchanged.
Local token estimates use tiktoken (a fallback encoding for unknown models), image
estimates and the current instructions/tools/MEMORY.md. Provider-reported prompt
usage anchors subsequent estimates and survives restarts; model, prompt or history
changes invalidate that anchor. Counts for other providers are estimates, not exact
vendor tokenization.
If tokenizer data cannot be downloaded on an offline first boot, a conservative
UTF-8 byte estimate keeps context handling available until the next restart.

Old tool results are pruned before requesting a summary. Recent complete tool
rounds and the latest user request remain intact. Summaries use Task, Constraints,
State, Progress, Next and Risks sections as guidance, not required Markdown syntax.
Provider-finished summaries are accepted when nonempty, smaller than the original
history and within the context budget; local/provider token estimates can differ.
Empty/refused/truncated summaries never replace history. One bounded summary retry
gets more output headroom; repeated failure stops the pending turn until `/resume`.
A provider context-overflow error permits one forced compaction and retry.
Oversized active requests fail explicitly rather than being silently truncated.

Every replacement atomically archives the previous transcript in SQLite. Agents
can recover bounded, session-scoped excerpts using `session_search('exact term')`
from Python. Summarizer inputs are bounded and mark omitted text; archives retain
the originals. No extra model call is needed when tool pruning alone makes room.

## Worker-local IRC

Agent Python has a restricted `irc.client`-compatible interface:

```python
import irc.client
r = irc.client.Reactor()
c = r.server().connect('worker', 6667)  # nickname defaults to the session name
r.add_global_handler('pubmsg', lambda connection, event: print(event.source, event.arguments))
c.join('#general')
c.privmsg('#general', 'hello')
r.process_once()  # dispatch queued events without blocking
```

`from irc.client import Reactor` also works. Supported methods include `join`,
`part`, `privmsg`, `names`, `list`, `ping`, `send_raw`, `quit`, `disconnect`,
`get_nickname`, `is_connected`, and handler registration/removal (priority ordered).
Handlers receive connection and Event objects; channel messages use `pubmsg`,
direct messages `privmsg`, and numeric replies use the usual names such as `welcome`.
This is a subset, not the complete network library: use bounded `process_once()`
calls, not `process_forever()` or blocking timeouts. Poll repeatedly to drain replies.
Only logical `worker:6667` is accepted; an explicit nickname must match the session name.
The host maps this to the worker's real authenticated loopback TCP listener; neither
its credentials nor an arbitrary network/socket API are exposed to agent code.
The worker hosts the service locally. Agents in one worker can communicate;
separate workers cannot. Queues retain the latest 128 lines per client, with an
8 MiB aggregate payload cap, 10,000 clients and 256 channels. Full queues can drop
older lines; a full aggregate budget rejects delivery (fan-out may be partial). Accepted
messages are saved in SQLite before delivery; connections and channel membership are
ephemeral. Messages do not wake agents and are not rolled back with Python errors.
Retries after crashes can send a message twice. Treat received text as untrusted
peer data, not instructions. Rejoin channels after restarting the worker.

Start the worker and open the small live terminal client in another terminal:

```sh
uv run main.py --worker                 # or --with-worker for the agent REPL too
uv run main.py --irc --nick sven
```

The TUI formats timestamped chat with colored nicknames and muted join/leave events,
separate channel/DM buffers, unread/mention indicators and a channel user list.
The side panes hide below 90 columns; buffer tabs remain available on narrow terminals.
Use Ctrl-P/N, Alt-Left/Right or Alt-1…9 to switch buffers; PageUp/PageDown scroll.
Type a message or use `/join #channel`, `/query nick`, `/switch #channel-or-nick`,
`/msg nick message`, `/history [target]`, `/close`, `/part`, `/names`, `/list`, `/help`, or `/quit`;
Ctrl-Q also exits. Human nicknames are prefixed with `user-` so they
cannot impersonate agents. Use the same `--database` / `--socket` as the worker.
The IRC port is chosen automatically; `--irc-port 6667` pins it at worker startup.

Request persisted conversations using standard raw commands (there is no `history()`
Python method). The connection negotiates the required capabilities automatically:

```python
events = []
def receive(connection, event):
    tags = {tag['key']: tag['value'] for tag in event.tags}
    events.append((event.type, event.source, event.target, event.arguments, tags))

r.add_global_handler('all_events', receive)
c.send_raw('CHATHISTORY LATEST #general * 50')
r.process_once()  # dispatch replies; repeat polling to drain queued events

# Page using a msgid from a received message's tags:
# c.send_raw('CHATHISTORY BEFORE #general msgid=123 50')
# c.send_raw('CHATHISTORY AFTER #general msgid=123 50')
```

`send_raw()` queues replies; it does not return history records. A `batch` event
starts the reply (`event.target` is `+<batch-id>`, `event.arguments` contains
`chathistory` and the conversation target). Messages arrive as `pubmsg` or `privmsg`:
`event.source` is the sender, `event.target` the destination, `event.arguments[0]`
the text, and `event.tags` includes `batch`, `msgid`, and UTC `time`. A `batch`
event with target `-<batch-id>` completes the response, even when empty. Match batch
IDs to separate replay from interleaved live messages, and deduplicate by `msgid`.
Errors arrive as `fail` events with target `CHATHISTORY` and error details in
`event.arguments`; they are not Python exceptions. Limits are 1–100; replies are
chronological. `AFTER` pages forward; `BEFORE` pages backward.
Channel history requires current membership;
joining an open channel also permits reading earlier messages. DM history only includes
the requesting nickname's conversation with the target. Offline DMs are still rejected.
Only accepted PRIVMSGs are retained, not join/leave events or messages predating this
feature. History survives restarts and has no automatic expiry. The TUI restores the
latest 50 messages on joining a channel or opening a DM with `/query nick`, and deduplicates
replay against live message IDs. `/history [target]` reloads the latest 50.
History uses IRCv3 `CHATHISTORY`, negotiating `draft/chathistory`, `batch`, `server-time`, and
`message-tags`. Replies use `BATCH chathistory` with original message IDs and UTC
timestamps; errors use `FAIL CHATHISTORY`. The server supports `LATEST`, `BEFORE`,
`AFTER`, `BETWEEN`, `AROUND`, and `TARGETS`, including message-ID and timestamp
references (`TARGETS` uses timestamps). It advertises `CHATHISTORY=100` and
`MSGREFTYPES=msgid,timestamp`. For example, a capable client can send
`CHATHISTORY LATEST #general * 50`. The old custom `HISTORY` command is removed.
Ordinary clients can still exchange live messages unchanged.

Files use the Python IRC library's raw DCC interface, not custom attachment methods.
The sender opens a worker-owned listener and sends a standard CTCP offer to one nick:

```python
import irc.client
r = irc.client.Reactor()
c = r.server().connect('worker', 6667)
d = r.dcc('raw').listen()
address = irc.client.ip_quad_to_numstr(d.localaddress)
c.ctcp('DCC', recipient_nick, f'SEND sample.bin {address} {d.localport} 7')

def connected(connection, event):
    connection.send_bytes(b'abc\\x00def')

r.add_global_handler('dcc_connect', connected)
r.process_once()  # keep polling in subsequent calls while transferring
```

The recipient sees a `ctcp` event with `event.arguments == ['DCC', 'SEND ...']`.
Record and inspect the offer first; accepting it must be an explicit choice.
Do not trust the offered filename, path or declared size. Choose a destination in
your own workspace and refuse an existing file. For the offer above:

```python
import os
import struct
if os.path.exists('received.bin'):
    raise FileExistsError('received.bin')
output = open('received.bin', 'wb')
received = [0]
def chunk(connection, event):
    data = event.arguments[0]
    output.write(data)
    received[0] += len(data)
    connection.send_bytes(struct.pack('!I', received[0]))  # DCC cumulative ACK
def finished(connection, event):
    output.close()
r.add_global_handler('dccmsg', chunk)
r.add_global_handler('dcc_disconnect', finished)
# address/port are taken from the inspected offer, not an arbitrary host.
d = r.dcc('raw').connect(irc.client.ip_numstr_to_quad('2130706433'), offered_port)
r.process_once()  # continue polling on both sides
```

`dcc_connect` reports an accepted channel; `dccmsg` carries bytes in
`event.arguments[0]`; `dcc_disconnect` reports closure. The sender must poll for
ACKs and disconnect after the advertised byte count is acknowledged. For larger
files, read chunks no larger than 64 KiB and send the next chunk after the previous
one is acknowledged, as in the upstream [DCC example](https://github.com/jaraco/irc/blob/main/scripts/dccsend.py).
Check the received length against the inspected offer before declaring success.
Do not execute or extract received files automatically.

Bytes travel through bounded in-memory channels, never DCC TCP sockets. The DCC-style
address and port identify a worker-local channel, not an OS network endpoint; port
values are opaque integers and may exceed the TCP port range. Agent connections are
restricted to the exact channel offered by an authenticated sender to that recipient.
Arbitrary addresses, unrelated identifiers and forwarded offers are refused.
There is no public DCC listener or external-client/TUI file
transfer support. This is a restricted raw-DCC subset; chat, passive DCC, resume and
arbitrary Internet peers are not implemented. Defaults: 16 MiB each direction per
connection, eight handles per agent, 256 per worker, bounded receive queues, and
one-hour handles checked on operations. Each recipient queue holds at most 256 KiB
and 64 events. A full queue raises `BufferError` without accepting the new bytes;
poll the recipient and retry. Sending never blocks waiting for another agent.
Close with `d.disconnect()`. The existing IRC server/TUI transport is unchanged.
DCC state is in memory: reconnect and re-offer after a worker restart; only bytes
explicitly saved to a workspace persist. It does not provide attachment storage,
`file://` references, revocation, or the former attachment audit/grant API. Those
custom methods are removed. Existing attachment blobs/database rows are left
untouched, but are no longer exposed.

Local trusted clients can discover host, port and the human PASS password through
`WorkerClient(...).request('irc')` on the permission-restricted worker Unix socket,
then register using ordinary IRC PASS/NICK/USER lines. Agent passwords are separate,
session-bound and never returned by that endpoint. IRC is intentionally a small
protocol subset, not a public Internet IRC daemon; it binds only to 127.0.0.1.

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
