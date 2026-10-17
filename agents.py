"""Async Monty agents with durable SQLite checkpoints and resumable pending work.

Files are immutable storage objects referenced by SQLite; workspaces are disposable
copies. Only completed model messages and successful interpreter changes commit.
The database and interpreter snapshots must be trusted. Native-library wrappers
operate on virtual bytes/plain data outside Monty's resource accounting.
"""
from __future__ import annotations

import fnmatch
import asyncio
import hashlib
import json
import os
import posixpath
import shutil
import sqlite3
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from functools import partial, wraps
from weakref import WeakValueDictionary
from pathlib import Path as HostPath
from pathlib import PurePosixPath as Path
from typing import Protocol
from contextlib import contextmanager, asynccontextmanager, AsyncExitStack
from dataclasses import dataclass

import httpx
from jiter import from_json
from openai import AsyncOpenAI, pydantic_function_tool, LengthFinishReasonError, ContentFilterFinishReasonError
from pydantic import BaseModel, ConfigDict
from pydantic_monty import CallbackFile, AsyncMonty, MontyError, MontyCrashedError, MontyClassProxy, OSAccess
from dotenv import load_dotenv
from names import num_to_name
from wrappers import tools as library_tools
from wrappers.bridge import dispatch as library_dispatch
from wrappers.imports import BOOTSTRAP, CHECK, prepare as prepare_libraries
from wrappers.registry import help_text as library_help

load_dotenv(override=True)


def code_prefix(arguments):
    """Jiter keeps incomplete strings/escapes out of the streamed code preview."""
    try:
        return from_json(arguments.encode(), partial_mode='trailing-strings').get('code', '')
    except ValueError:
        return ''


class PythonCode(BaseModel):
    """Run Python in the durable Monty virtual Unix environment."""
    model_config = ConfigDict(extra='forbid')
    code: str

HELP = """You are a code-mode assistant. Execute Python with the python tool to
inspect and modify a persistent virtual Unix filesystem. No host shell/network.
Globals persist after successful calls. Failed calls roll back globals and files.
Use print for observations. Batch related work; avoid dumping large files.
Paths are POSIX; cwd starts at /home/<your-name>/. pathlib.Path and open use
the same virtual filesystem as the tools. /tmp and /dev/null are available.
HOME, USER, LOGNAME, PWD and TMPDIR describe your isolated virtual environment.
cd() changes tool paths immediately and native open/Path cwd on the NEXT call.
Available host functions (call directly, no imports):
  pwd(); cd(path); ls(path='.'); mkdir(path, parents=True)
  read(path, offset=1, limit=200) -> numbered lines (1-based)
  write(path, content: str|bytes); edit(path, old, new, all=False)
  glob(pattern='*', path='/', limit=200) -> recursive matching file paths
  grep(pattern, path='/', glob='*', limit=100, offset=0, literal=False)
    -> {matches: [{path,line,text}], truncated}; literal searches avoid regex
  mv(source,target); rm(path, recursive=False)
Library compatibility: write familiar imports; the agent translates them for Monty.
Only installed host libraries are exposed, as listed below.
These are supported subsets, not full native modules. Objects persist across calls
and restarts. All library file paths use ONLY the virtual filesystem (no URLs).
No wildcard imports, pickle, arbitrary callbacks/apply/query/eval, or unsupported APIs.
Use explicit methods and assignments; context managers/custom special methods are
not supported. Arrays are capped at 1,000,000 elements; images at 4,000,000 pixels.
Finish with a concise answer based on actual execution results.
"""


class Unix(OSAccess):
    """Monty's in-memory Unix tree plus bounded code-agent conveniences."""

    def __init__(self, entries=(), cwd='/', environ=None):
        # Like Bashkit, the runtime and tools share a VFS, including virtual
        # devices. This callback never opens a host path.
        super().__init__(files=[CallbackFile('/dev/null', read=lambda path: b'',
                                             write=lambda path, data: None)], environ=environ)
        self.cwd = cwd
        for path, content in sorted(entries, key=lambda row: (row[0].count('/'), row[0])):
            if path == '/':
                continue
            p = Path(path)
            if content is None:
                self.path_mkdir(p, True, True)
            else:
                self.path_mkdir(p.parent, True, True)
                self.path_write_bytes(p, bytes(content))

    def path(self, path):
        return Path(posixpath.normpath('/' + posixpath.join(self.cwd, str(path)).lstrip('/')))

    def walk(self, path='/'):
        p = self.path(path)
        yield p
        if self.path_is_dir(p):
            for child in sorted(self.path_iterdir(p)):
                yield from self.walk(str(child))

    def entries(self):
        return [(str(p), self.path_read_bytes(p) if self.path_is_file(p) else None)
                for p in self.walk()]

    def pwd(self):
        return self.cwd

    def cd(self, path):
        p = self.path(path)
        self.path_iterdir(p)  # validates directory
        self.cwd = str(p)
        self.environ['PWD'] = self.cwd
        return self.cwd

    def ls(self, path='.'):
        return [str(p) + ('/' if self.path_is_dir(p) else '')
                for p in sorted(self.path_iterdir(self.path(path)))][:1000]

    def mkdir(self, path, parents=True):
        self.path_mkdir(self.path(path), parents, True)

    def read(self, path, offset=1, limit=200):
        if offset < 1 or not 1 <= limit <= 2000:
            raise ValueError('offset >= 1 and 1 <= limit <= 2000 required')
        lines = self.path_read_text(self.path(path)).splitlines()
        return '\n'.join(f'{i + 1}: {line[:4000]}' for i, line in
                         enumerate(lines[offset-1:offset-1+limit], offset-1))

    def write(self, path, content):
        if isinstance(content, str):
            content = content.encode()
        if len(content) > 32 * 1024 * 1024:
            raise ValueError('file exceeds 32 MiB')
        return self.path_write_bytes(self.path(path), content)

    def edit(self, path, old, new, all=False):
        text = self.path_read_text(self.path(path))
        count = text.count(old)
        if not old or not count or (count != 1 and not all):
            raise ValueError('old must be nonempty and match once, or use all=True')
        self.write(path, text.replace(old, new, -1 if all else 1))
        return count

    def glob(self, pattern='*', path='/', limit=200):
        if not 1 <= limit <= 2000:
            raise ValueError('limit must be 1..2000')
        root = self.path(path)
        result = []
        for p in self.walk(str(root)):
            if self.path_is_file(p) and (fnmatch.fnmatch(str(p.relative_to(root)), pattern)
                                         or fnmatch.fnmatch(p.name, pattern)):
                result.append(str(p))
                if len(result) == limit:
                    break
        return result

    def grep(self, pattern, path='/', glob='*', limit=100, offset=0, literal=False):
        # Host regex needs its own timeout: Monty's budget does not cover callbacks.
        import regex
        if not 1 <= limit <= 1000 or offset < 0:
            raise ValueError('limit must be 1..1000; offset >= 0')
        matcher = None if literal else regex.compile(pattern)
        result, skipped = [], 0
        for p in self.walk(path):
            if not self.path_is_file(p) or not fnmatch.fnmatch(p.name, glob):
                continue
            data = self.path_read_bytes(p)
            if b'\x00' in data:
                continue
            for number, line in enumerate(data.decode('utf-8', 'replace').splitlines(), 1):
                found = pattern in line if literal else matcher.search(line, timeout=0.05)
                if found:
                    if skipped < offset:
                        skipped += 1
                    elif len(result) == limit:
                        return {'matches': result, 'truncated': True}
                    else:
                        result.append({'path': str(p), 'line': number, 'text': line[:4000]})
        return {'matches': result, 'truncated': False}

    def mv(self, source, target):
        src, dst = self.path(source), self.path(target)
        if src == Path('/') or src == dst or src in dst.parents:
            raise ValueError('invalid move')
        self.path_rename(src, dst)
        if self.cwd == str(src) or src in Path(self.cwd).parents:
            self.cwd = str(dst / Path(self.cwd).relative_to(src))

    def path_rename(self, path, target):
        # Monty 1.1 OSAccess moves a file's tree entry but leaves its name stale.
        # Keep its metadata aligned so subsequent unlink/rename uses the new name.
        super().path_rename(path, target)
        if self.path_is_file(target):
            self._get_file(target).path = Path(target)
            self._get_file(target).name = Path(target).name

    def rm(self, path, recursive=False):
        p = self.path(path)
        if p == Path('/') or p == Path(self.cwd) or p in Path(self.cwd).parents:
            raise ValueError('cannot remove root or cwd ancestor')
        if self.path_is_dir(p):
            if recursive:
                for child in self.path_iterdir(p):
                    self.rm(str(child), True)
            self.path_rmdir(p)
        else:
            self.path_unlink(p)


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class Storage(Protocol):
    """put must durably store immutable bytes; checkout is a rebuildable view."""
    def create(self, name: str) -> None: ...
    def put(self, name: str, content: bytes) -> str: ...
    def read(self, key: str) -> bytes: ...
    def checkout(self, name: str, files: list[dict]) -> HostPath | str: ...


class DirectoryStorage:
    """Durable objects plus disposable workspace directories; replace for blobs."""

    def __init__(self, root):
        self.root = HostPath(root)
        self._mkdir(self.root.absolute())
        self.root = self.root.resolve()

    def _mkdir(self, path):
        if not path.exists():
            self._mkdir(path.parent)
            path.mkdir(exist_ok=True)
            sync_directory(path.parent)

    def _path(self, name):
        if name in ('', '.', '..') or Path(name).name != name:
            raise ValueError('invalid workspace name')
        path = self.root / name
        if path.is_symlink():
            raise PermissionError('workspace symlinks are not supported')
        return path

    def create(self, name):
        if self._path(name).exists():
            raise FileExistsError(name)

    def put(self, name, content):
        self._path(name)
        key = f'.objects/{name}/{hashlib.sha256(content).hexdigest()}'
        target = self.root / key
        self._mkdir(target.parent)
        if target.exists():
            self.read(key)
        else:
            temporary = target.with_suffix('.' + uuid.uuid4().hex)
            with temporary.open('xb') as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            temporary.replace(target)
            sync_directory(target.parent)
        return key

    def read(self, key):
        parts = Path(key).parts
        if (len(parts) != 3 or parts[0] != '.objects' or parts[1] in ('.', '..')
                or len(parts[2]) != 64 or any(c not in '0123456789abcdef' for c in parts[2])):
            raise PermissionError('invalid storage key')
        host = self.root.joinpath(*parts)
        if host.resolve() != host:
            raise PermissionError('storage symlinks are not supported')
        content = host.read_bytes()
        if hashlib.sha256(content).hexdigest() != parts[2]:
            raise OSError('stored object failed its content hash check')
        return content

    def checkout(self, name, files):
        root = self._path(name)
        if root.exists():
            shutil.rmtree(root)
        root.mkdir()
        for file in files:
            if '..' in Path(file['path']).parts:
                raise ValueError('invalid virtual path')
            target = root / file['path'].lstrip('/')
            if file['kind'] == 'directory':
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(self.read(file['storage_key']))
        return root


@dataclass
class Session:
    id: int
    name: str
    cwd: str
    monty: bytes | None
    messages: list[dict]
    pending: dict | None


class ExecutionError(RuntimeError):
    """A recorded code failure; infrastructure failures remain retryable."""


async def settled(future):
    """Drain blocking work before cancellation releases its session/DB lock."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(future)
            break
        except asyncio.CancelledError:
            if future.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


def serialized(method):
    @wraps(method)
    async def call(self, *args, **kwargs):
        async with self.runtime.operation(), self._lock:
            if self._closed:
                raise RuntimeError('agent is closed')
            return await method(self, *args, **kwargs)
    return call


class Runtime:
    """One shared runtime per database, in one process. All host IO runs off-loop."""

    def __init__(self, database='agent.db', *, workspaces=None, storage=None, model=None,
                 base_url=None, api_key=None, workers=4, io_workers=8, model_concurrency=128,
                 client=None, template=None):
        if min(workers, io_workers, model_concurrency) < 1:
            raise ValueError('concurrency limits must be positive')
        if storage is not None and workspaces is not None:
            raise ValueError('pass storage or workspaces, not both')
        self.database = HostPath(database).absolute()
        self._storage = storage
        self.db = None
        self.workspaces = workspaces or self.database.parent / 'workspaces'
        self.template = (HostPath(template) if template is not None else
                         HostPath(__file__).with_name('template')).absolute()
        self.model = model or os.environ.get('LLM_MODEL', 'gpt-6-luna')
        self.base_url = base_url or os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')
        self.api_key = api_key if api_key is not None else os.environ.get('LLM_API_KEY', '')
        self.client, self.workers, self.io_workers = client, workers, io_workers
        self.model_concurrency = model_concurrency
        self._model_slots, self._sandbox_slots = asyncio.Semaphore(model_concurrency), asyncio.Semaphore(workers)
        self._creation_slots = asyncio.Semaphore(io_workers)
        self._db_slots, self._io_slots = asyncio.Semaphore(1), asyncio.Semaphore(io_workers)
        self._locks = WeakValueDictionary()
        self._tasks = set()
        self._drained = asyncio.Event()
        self._drained.set()
        self._closing = False
        self._stack = None
        self._close_task = None

    def _connect(self):
        self.db = sqlite3.connect(self.database, timeout=30, isolation_level=None, check_same_thread=False)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        columns = [r[1] for r in self.db.execute('PRAGMA table_info(sessions)')]
        if columns and columns != ['id', 'name', 'cwd', 'monty', 'messages']:
            raise RuntimeError('unsupported database schema; choose a new database and workspace root')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, cwd TEXT NOT NULL,
                monty BLOB, messages TEXT NOT NULL DEFAULT '[]');
            CREATE TABLE IF NOT EXISTS files (
                session_id INTEGER NOT NULL, path TEXT NOT NULL, kind TEXT NOT NULL,
                storage_key TEXT, size INTEGER NOT NULL, PRIMARY KEY(session_id,path));
            CREATE TABLE IF NOT EXISTS executions (
                session_id INTEGER NOT NULL, id TEXT NOT NULL, code TEXT NOT NULL,
                tool_id TEXT, output TEXT, error INTEGER NOT NULL DEFAULT 0,
                PRIMARY KEY(session_id,id));
            CREATE UNIQUE INDEX IF NOT EXISTS one_pending_execution
                ON executions(session_id) WHERE output IS NULL;
        ''')

    async def __aenter__(self):
        if self._stack is not None or self._closing:
            raise RuntimeError('runtime cannot be entered twice')
        self._stack = AsyncExitStack()
        self._loop, self._loop_thread = asyncio.get_running_loop(), threading.get_ident()
        self._database_executor = ThreadPoolExecutor(1, thread_name_prefix='swarm-db')
        self._io_executor = ThreadPoolExecutor(self.io_workers, thread_name_prefix='swarm-io')
        self._stack.push_async_callback(asyncio.to_thread, self._database_executor.shutdown, wait=True)
        self._stack.push_async_callback(asyncio.to_thread, self._io_executor.shutdown, wait=True)
        try:
            self.storage = await self.offload(lambda: self._storage if self._storage is not None
                                             else DirectoryStorage(self.workspaces))
            await self.offload(self.database.parent.mkdir, parents=True, exist_ok=True)
            self._stack.push_async_callback(self.database_call, lambda: self.db.close() if self.db is not None else None)
            await self.database_call(self._connect)
            self._pool = await settled(asyncio.create_task(self._stack.enter_async_context(
                AsyncMonty(min_processes=1, max_processes=self.workers, request_timeout=20))))
            self.tool = await self.offload(pydantic_function_tool, PythonCode, name='python')
            if self.client is None:
                def make_client():
                    self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, max_retries=0,
                        http_client=httpx.AsyncClient(timeout=120, limits=httpx.Limits(
                            max_connections=self.model_concurrency, max_keepalive_connections=32)))
                self._stack.push_async_callback(self._close_client)
                await self.offload(make_client)
            await self.offload(lambda: self.client.chat.completions)  # Warm lazy SDK imports off-loop.
            return self
        except BaseException:
            await self.close()
            raise

    async def _close_client(self):
        if self.client is not None:
            await self.client.close()

    async def __aexit__(self, *args):
        await self.close()

    @asynccontextmanager
    async def operation(self):
        if self._stack is None or self._closing:
            raise RuntimeError('runtime is not running')
        task = asyncio.current_task()
        self._tasks.add(task)
        self._drained.clear()
        try:
            yield
        finally:
            self._tasks.remove(task)
            if not self._tasks:
                self._drained.set()

    async def database_call(self, function, *args, **kwargs):
        async with self._db_slots:
            return await settled(self._loop.run_in_executor(
                self._database_executor, partial(function, *args, **kwargs)))

    async def offload(self, function, *args, **kwargs):
        async with self._io_slots:
            return await settled(self._loop.run_in_executor(
                self._io_executor, partial(function, *args, **kwargs)))

    def _template_entries(self):
        """Read trusted starter files off-loop; never import host symlinks/devices."""
        if self.template.is_symlink():
            raise ValueError('session template must not be a symlink')
        if not self.template.is_dir():
            raise NotADirectoryError(self.template)
        entries = []
        for path in sorted(self.template.rglob('*')):
            if path.is_symlink():
                raise ValueError(f'session template contains a symlink: {path}')
            if path.is_dir():
                content = None
            elif path.is_file():
                with path.open('rb') as source:
                    content = source.read(32 * 1024**2 + 1)
                if len(content) > 32 * 1024**2:
                    raise ValueError(f'template file exceeds 32 MiB: {path}')
            else:
                raise ValueError(f'unsupported template entry: {path}')
            entries.append((path.relative_to(self.template).as_posix(), content))
        return entries

    async def agent(self, *, session_id=None, model=None, on_output=None):
        async with self.operation():
            async with self._creation_slots:
                entries = await self.offload(self._template_entries) if session_id is None else ()
                agent = CodeAgent(self, model=model, on_output=on_output)
                await self.database_call(agent._initialize, session_id, entries)
                del entries
            lock = self._locks.get(agent.id)
            if lock is None:
                lock = self._locks[agent.id] = asyncio.Lock()
            agent._lock = lock
            async with lock:
                await agent._checkout()
            return agent

    async def close(self):
        if asyncio.current_task() in self._tasks:
            raise RuntimeError('cannot close a runtime inside an active operation')
        if self._close_task is None:
            self._closing = True
            self._close_task = asyncio.create_task(self._close())
        await settled(self._close_task)

    async def _close(self):
        await self._drained.wait()
        if self._stack is not None:
            await self._stack.aclose()


class CodeAgent:
    """Persist pending code, then commit its checkpoint/files/result atomically."""

    def __init__(self, runtime, *, model=None, on_output=None):
        self.runtime, self.db, self.storage = runtime, runtime.db, runtime.storage
        self.model, self.on_output = model or runtime.model, on_output
        self._streamed_calls = set()
        self._closed = self._owns_runtime = False

    @classmethod
    async def open(cls, database='agent.db', *, session_id=None, model=None, on_output=None, **options):
        """Convenience for one agent; share Runtime explicitly for many agents."""
        runtime = await Runtime(database, model=model, **options).__aenter__()
        try:
            agent = await runtime.agent(session_id=session_id, on_output=on_output)
            agent._owns_runtime = True
            return agent
        except BaseException:
            await runtime.close()
            raise

    @contextmanager
    def _transaction(self):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            yield
            self.db.execute('COMMIT')
        except BaseException:
            if self.db.in_transaction:
                self.db.execute('ROLLBACK')
            raise

    def _initialize(self, session_id, entries=()):
        with self._transaction():
            if session_id is None:
                self.id = self.db.execute('SELECT COALESCE(MAX(id)+1,0) FROM sessions').fetchone()[0]
                if self.id >= 65536:
                    raise ValueError('names.py capacity exhausted')
                self.name = num_to_name(self.id)
                self.storage.create(self.name)
                fs = Unix(cwd=self.home)
                fs.mkdir(self.home)
                fs.mkdir('/tmp')
                for relative, content in entries:
                    path = str(Path(self.home) / relative)
                    fs.mkdir(path if content is None else str(Path(path).parent))
                    if content is not None:
                        fs.write(path, content)
                self.db.execute('INSERT INTO sessions(id,name,cwd) VALUES (?,?,?)', (self.id, self.name, self.home))
                self._replace_files(self._publish(fs.entries()))
            else:
                self.id = session_id
                self.name = self._state.name

    @property
    def home(self):
        return f'/home/{self.name}'

    @property
    def _state(self):
        row = self.db.execute('SELECT id,name,cwd,monty,messages FROM sessions WHERE id=?', (self.id,)).fetchone()
        if row is None:
            raise ValueError(f'unknown session: {self.id}')
        return Session(*row[:4], json.loads(row[4]), self._pending())

    def _pending(self):
        row = self.db.execute('SELECT id,code,tool_id FROM executions WHERE session_id=? AND output IS NULL',
                              (self.id,)).fetchone()
        return dict(zip(('execution_id', 'code', 'tool_id'), row)) if row else None

    def _message(self, message):
        self.db.execute("UPDATE sessions SET messages=json_insert(messages,'$[#]',json(?)) WHERE id=?",
                        (json.dumps(message), self.id))

    def _append_message(self, message):
        with self._transaction():
            if self._pending():
                raise RuntimeError('an execution is pending; resume it before appending messages')
            self._message(message)

    def _publish(self, entries):
        return [{'path': path, 'kind': 'directory' if content is None else 'file',
                 'storage_key': None if content is None else self.storage.put(self.name, content),
                 'size': 0 if content is None else len(content)} for path, content in entries]

    def _replace_files(self, files):
        self.db.execute('DELETE FROM files WHERE session_id=?', (self.id,))
        self.db.executemany('INSERT INTO files VALUES (?,?,?,?,?)',
            [(self.id, f['path'], f['kind'], f['storage_key'], f['size']) for f in files])

    def _files(self):
        return [dict(zip(('path', 'kind', 'storage_key', 'size'), row)) for row in self.db.execute(
            'SELECT path,kind,storage_key,size FROM files WHERE session_id=? ORDER BY path', (self.id,))]

    async def _checkout(self):
        files = await self.runtime.database_call(self._files)
        self.workspace = await self.runtime.offload(self.storage.checkout, self.name, files)

    @serialized
    async def state(self):
        return await self.runtime.database_call(lambda: self._state)

    @serialized
    async def files(self):
        return await self.runtime.database_call(self._files)

    @serialized
    async def sessions(self):
        return await self.runtime.database_call(lambda: [dict(zip(('id', 'name', 'workspace'), row))
            for row in self.db.execute('SELECT id,name,name FROM sessions ORDER BY id')])

    @serialized
    async def append_message(self, message):
        await self.runtime.database_call(self._append_message, message)

    @serialized
    async def read_file(self, path):
        path = posixpath.normpath('/' + str(path).lstrip('/'))
        row = await self.runtime.database_call(lambda: self.db.execute(
            'SELECT kind,storage_key FROM files WHERE session_id=? AND path=?', (self.id, path)).fetchone())
        if row is None:
            raise FileNotFoundError(path)
        if row[0] != 'file':
            raise IsADirectoryError(path)
        return await self.runtime.offload(self.storage.read, row[1])

    async def close(self):
        async with self._lock:
            self._closed = True
        if self._owns_runtime:
            await self.runtime.close()

    def _emit(self, kind, text):
        if self.on_output is not None and (text or kind == 'result'):
            if threading.get_ident() == self.runtime._loop_thread:
                self.on_output(kind, text)
            else:
                self.runtime._loop.call_soon_threadsafe(self.on_output, kind, text)

    def _prepare(self, code, tool_id, execution_id):
        with self._transaction():
            pending = self._pending()
            execution_id = execution_id or ('tool:' + tool_id if tool_id is not None else
                                            pending['execution_id'] if pending else uuid.uuid4().hex)
            row = self.db.execute('SELECT code,tool_id,output,error FROM executions WHERE session_id=? AND id=?',
                                   (self.id, execution_id)).fetchone()
            if row:
                if code is not None and (row[0] != code or row[1] != tool_id):
                    raise ValueError('execution ID was already used for another command')
                if row[2] is not None:
                    if row[3]:
                        raise ExecutionError(row[2])
                    return row[2]
            elif pending:
                raise RuntimeError('an execution is pending; call execute() or run() to resume')
            else:
                if not isinstance(code, str):
                    raise ValueError('code must be a string')
                self.db.execute('INSERT INTO executions(session_id,id,code,tool_id) VALUES (?,?,?,?)',
                                (self.id, execution_id, code, tool_id))
            return self._state, self._files()

    def _commit(self, request, output, error, files=None, cwd=None, snapshot=None):
        with self._transaction():
            if not error:
                self.db.execute('UPDATE sessions SET cwd=?,monty=? WHERE id=?', (cwd, snapshot, self.id))
                self._replace_files(files)
            self.db.execute('UPDATE executions SET output=?,error=? WHERE session_id=? AND id=?',
                            (output, error, self.id, request['execution_id']))
            if request['tool_id'] is not None:
                self._message({'role': 'tool', 'tool_call_id': request['tool_id'], 'content': output})

    @serialized
    async def execute(self, code=None, *, tool_id=None, execution_id=None):
        return await self._execute(code, tool_id, execution_id)

    async def _execute(self, code=None, tool_id=None, execution_id=None):
        async with self.runtime._sandbox_slots:
            prepared = await self.runtime.database_call(self._prepare, code, tool_id, execution_id)
            if isinstance(prepared, str):
                await self._checkout()
                self._emit('result', prepared)
                return prepared
            state, files = prepared
            request = state.pending
            if request['tool_id'] not in self._streamed_calls:
                self._emit('python', request['code'])
            try:
                fs, snapshot, output, stdout = await self._evaluate(request['code'], state, files)
            except MontyCrashedError:
                raise  # Infrastructure failures/cancellation leave the request pending.
            except (MontyError, SyntaxError, ImportError) as exc:
                output = f'{type(exc).__name__}: {exc}'[:24000]
                await self.runtime.database_call(self._commit, request, output, True)
                raise ExecutionError(output) from exc
            files = await self.runtime.offload(lambda: self._publish(fs.entries()))
            await self.runtime.database_call(self._commit, request, output, False, files, fs.cwd, snapshot)
            await self._checkout()
            self._emit('result', output[len(stdout):])
            return output

    async def _evaluate(self, code, state, files):
        entries = await self.runtime.offload(lambda: [
            (f['path'], None if f['kind'] == 'directory' else self.storage.read(f['storage_key']))
            for f in files])
        fs = await self.runtime.offload(Unix, entries, state.cwd, environ={
            'HOME': self.home, 'USER': self.name, 'LOGNAME': self.name,
            'PWD': state.cwd, 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8',
        })
        fs.path_iterdir(Path(state.cwd))
        functions = {'pwd': fs.pwd, 'cd': fs.cd, 'ls': fs.ls, 'mkdir': fs.mkdir,
                     'read': fs.read, 'write': fs.write, 'edit': fs.edit,
                     'glob': fs.glob, 'grep': fs.grep, 'mv': fs.mv, 'rm': fs.rm,
                     **library_tools(fs), '_library_rpc': partial(library_dispatch, fs)}
        # AsyncMonty runs synchronous OS/library callbacks on its worker threads.
        # Keep these synchronous so sandbox callers need not write `await numpy(...)`.
        async with self.runtime._pool.checkout(limits={'max_memory': 64 * 1024**2,
                                       'max_feed_duration_secs': 10, 'max_suspensions': 1000}) as session:
            active = False
            loaded = None
            if state.monty is not None:
                await session.load_session(state.monty)
                loaded = await session.feed_run(CHECK, external_lookup=functions, print_callback=lambda *args: None)
                active = loaded is not None
            code, libraries = await self.runtime.offload(prepare_libraries, code, active, loaded or ())
            if libraries and not active:
                await session.feed_run(BOOTSTRAP, external_lookup=functions, print_callback=lambda *args: None)
            printed, byte_count = [], 0
            def capture(stream, text):
                nonlocal byte_count
                byte_count += len(text.encode())
                if byte_count > 16000:
                    raise MemoryError('printed output exceeds 16000 bytes')
                printed.append(text)
                self._emit('stdout', text)
            value = await session.feed_run(code, external_lookup=functions, os=fs,
                                     cwd=fs.cwd, print_callback=capture)
            stdout = ''.join(printed)
            if isinstance(value, MontyClassProxy) and value.name == '_LibraryRender':
                rendered = value.attributes['text']
            else:
                rendered = repr(value) if value is not None else ''
            output = (stdout + rendered)[:24000]
            return fs, await session.dump(), output, stdout

    async def _tool(self, call):
        try:
            if call['function']['name'] != 'python':
                raise ValueError('unknown tool')
            code = PythonCode.model_validate_json(call['function']['arguments']).code
        except (ValueError, KeyError, TypeError) as exc:
            await self.runtime.database_call(self._append_message, {'role': 'tool', 'tool_call_id': call['id'], 'content': str(exc)})
            self._emit('error', str(exc))
            return
        try:
            await self._execute(code, tool_id=call['id'])
        except ExecutionError as exc:
            self._emit('error', str(exc))  # The failure is already recorded.

    async def _response(self, client, messages):
        """The SDK handles SSE decoding, typed events and tool-call assembly."""
        shown = {}
        try:
            async with client.chat.completions.stream(
                model=self.model, messages=[{'role': 'system', 'content':
                    f'Your session ID is {self.id}; your name is {self.name}.\n' + HELP + '\n' + library_help()}, *messages],
                tools=[self.runtime.tool], parallel_tool_calls=False,
            ) as stream:
                async for event in stream:
                    if event.type in ('content.delta', 'refusal.delta'):
                        self._emit('assistant', event.delta)
                    elif event.type == 'tool_calls.function.arguments.delta':
                        code = code_prefix(event.arguments)
                        self._emit('python', code[len(shown.get(event.index, '')):])
                        shown[event.index] = code
                choice = (await stream.get_final_completion()).choices[0]
        except (LengthFinishReasonError, ContentFilterFinishReasonError) as exc:
            raise RuntimeError('Incomplete model response; resume to retry') from exc
        if (choice.finish_reason not in ('stop', 'tool_calls') or
                (choice.finish_reason == 'tool_calls' and not choice.message.tool_calls)):
            raise RuntimeError('Incomplete model response; resume to retry')
        message = choice.message.model_dump(include={'role', 'content', 'tool_calls'}, exclude_none=True)
        message['content'] = choice.message.content or choice.message.refusal or ''
        for call in message.get('tool_calls', []):
            call['function'].pop('parsed_arguments', None)
        self._streamed_calls = {call['id'] for call in message.get('tool_calls', [])}
        return message

    @serialized
    async def run(self, prompt=None, *, max_steps=20):
        """Resume pending work automatically before accepting a new user turn."""
        return await self._run(prompt, max_steps=max_steps)

    def _turn(self):
        return json.loads(self.db.execute('SELECT messages FROM sessions WHERE id=?', (self.id,)).fetchone()[0])

    async def _run(self, prompt=None, *, max_steps=20):
        pending = await self.runtime.database_call(self._pending)
        if pending:
            try:
                await self._execute()
            except ExecutionError as exc:
                self._emit('error', str(exc))
        if prompt is not None:
            async with self.runtime._model_slots:
                messages = await self.runtime.database_call(self._turn)
                unfinished = bool(messages and (messages[-1]['role'] in ('user', 'tool') or
                                                messages[-1].get('tool_calls')))
                repeated = bool(messages and messages[-1]['role'] == 'user' and messages[-1]['content'] == prompt)
                del messages
            if unfinished:
                answer = await self._run(max_steps=max_steps)
                if repeated:
                    return answer
            await self.runtime.database_call(self._append_message, {'role': 'user', 'content': prompt})
        for _ in range(max_steps):
            # Acquire before loading history: thousands of waiting agents do not
            # retain thousands of histories or interpreter checkpoints in RAM.
            async with self.runtime._model_slots:
                messages = await self.runtime.database_call(self._turn)
                if not messages:
                    return ''
                assistant = next((m for m in reversed(messages) if m['role'] == 'assistant'), {})
                answered = {m['tool_call_id'] for m in messages if m['role'] == 'tool'}
                call = next((c for c in assistant.get('tool_calls', []) if c['id'] not in answered), None)
                if call is None:
                    if messages[-1]['role'] == 'assistant':
                        return messages[-1].get('content') or ''
                    message = await self._response(self.runtime.client, messages)
                    await self.runtime.database_call(self._append_message, message)
                    if not message.get('tool_calls'):
                        return message.get('content') or ''
                del messages, assistant, answered
            if call is not None:
                await self._tool(call)
        raise RuntimeError('Step budget reached; call run() to continue the durable turn')
