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

## Crash recovery

There is no event log or replay layer. SQLite stores only:

- `sessions`: the latest interpreter checkpoint, working directory and conversation.
- `files`: current paths and immutable storage keys, not file contents.
- `executions`: pending commands and their results, allowing safe retries.

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

## Tests

```sh
uv run python -m unittest test_agents test_concurrency test_templates -q
```

Tests cover wrappers, streaming, recovery after abrupt process death, cancellation,
session isolation, starter files and 1,000 agents sharing bounded resources.
Provider tests use mocked HTTP, not a live model.
