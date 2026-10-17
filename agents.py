"""Durable code-mode agent. All sandbox capabilities live in this module.

The SQLite database is trusted local state (Monty dumps are not safe to load
from strangers). Each code call atomically commits files, interpreter globals,
and its tool response; failed calls roll back all three. Native library wrappers
accept plain data, never host paths or arbitrary Python. They are deliberately
small APIs, not drop-in implementations of those libraries. Host library work
is outside Monty's CPU/memory accounting; use process isolation for hostile users.

The files table stores session-scoped virtual paths and opaque storage keys, not
contents or absolute host paths. DirectoryStorage backs those keys with files in
workspaces/. Supply CodeAgent(storage=...) with a Storage implementation to use
another backend without changing the agent, tools, or database file references.

events is the append-only source of truth. sessions has explicit runtime columns;
sessions/files and the visible workspace are rebuildable projections. Historical
file objects under workspaces/.objects are immutable and retained for replay.
Write through agent tools so changes are recorded; direct edits of the visible
workspace are not authoritative. rebuild() restores projections without running
code. execute() resumes a pending command; execution_id makes retries idempotent.
This schema intentionally has no migration from the earlier key/value design.
"""
from __future__ import annotations

import fnmatch
import hashlib
import io
import json
import os
import posixpath
import shutil
import sqlite3
import uuid
from pathlib import Path as HostPath
from pathlib import PurePosixPath as Path
from typing import Iterable, Protocol
from contextlib import contextmanager
from dataclasses import dataclass

import httpx
import numpy as np
import pandas as pd
from PIL import Image
from docx import Document
from openpyxl import Workbook, load_workbook
from pypdf import PdfReader, PdfWriter
from pydantic_monty import CallbackFile, CollectString, Monty, MontyError, MontyCrashedError, OSAccess
from dotenv import load_dotenv
from names import num_to_name

load_dotenv(override=True)

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
  numpy(op, data, other=None, axis=None): array, sum, mean, min, max, std,
    median, transpose, reshape(other=shape), dot(other=array), unique
    Arrays cross the boundary as lists; scalars as Python numbers.
  pandas(op, data=None, path=None, columns=None): read_csv/read_excel -> records;
    to_csv/to_excel write records to path; describe/head (columns=count, default 5),
    select(columns=list), sort(columns=list), drop_duplicates -> plain data
  PIL(op, path=None, output=None, size=None, mode='RGB', color='white'):
    info, new, resize, convert. Writing requires output; format from extension.
  openpyxl(op, path, rows=None, sheet=None): read -> {sheet: rows},
    write -> new workbook, append -> existing workbook (sheet defaults active)
  pypdf(op, path, output=None, pages=None): read -> page texts,
    select -> write selected zero-based pages (default all)
  docx(op, path, paragraphs=None): read -> paragraphs, write -> new document,
    append -> existing document. All paths refer ONLY to virtual files.
These are functional wrappers, not importable native modules or native objects.
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


class Libraries:
    def __init__(self, fs):
        self.fs = fs

    def source(self, path):
        return io.BytesIO(self.fs.path_read_bytes(self.fs.path(path)))

    def numpy(self, op, data, other=None, axis=None):
        a = np.asarray(data)
        operations = {'array': lambda: a, 'sum': lambda: a.sum(axis=axis),
                      'mean': lambda: a.mean(axis=axis), 'min': lambda: a.min(axis=axis),
                      'max': lambda: a.max(axis=axis), 'std': lambda: a.std(axis=axis),
                      'median': lambda: np.median(a, axis=axis),
                      'transpose': lambda: a.T, 'reshape': lambda: a.reshape(other),
                      'dot': lambda: a.dot(other), 'unique': lambda: np.unique(a)}
        value = operations[op]()
        return value.tolist()

    def pandas(self, op, data=None, path=None, columns=None):
        if op in ('read_csv', 'read_excel'):
            frame = (pd.read_csv(self.source(path)) if op == 'read_csv'
                     else pd.read_excel(self.source(path)))
        else:
            frame = pd.DataFrame(data)
            if op in ('to_csv', 'to_excel'):
                out = io.BytesIO()
                if op == 'to_csv':
                    frame.to_csv(out, index=False)
                else:
                    frame.to_excel(out, index=False)
                return self.fs.write(path, out.getvalue())
            operations = {'head': lambda: frame.head(5 if columns is None else columns),
                          'describe': frame.describe, 'select': lambda: frame[columns],
                          'sort': lambda: frame.sort_values(columns),
                          'drop_duplicates': frame.drop_duplicates}
            frame = operations[op]()
        return json.loads(frame.to_json(orient='records'))

    def PIL(self, op, path=None, output=None, size=None, mode='RGB', color='white'):
        if op not in ('info', 'new', 'resize', 'convert'):
            raise ValueError('unknown image operation')
        image = Image.new(mode, tuple(size), color) if op == 'new' else Image.open(self.source(path))
        try:
            if op == 'info':
                return {'size': list(image.size), 'mode': image.mode, 'format': image.format}
            if op == 'resize':
                image = image.resize(tuple(size))
            elif op == 'convert':
                image = image.convert(mode)
            out = io.BytesIO()
            fmt = Image.registered_extensions()[Path(output).suffix.lower()]
            image.save(out, format=fmt)
            return self.fs.write(output, out.getvalue())
        finally:
            image.close()

    def openpyxl(self, op, path, rows=None, sheet=None):
        if op not in ('read', 'write', 'append'):
            raise ValueError('unknown workbook operation')
        book = Workbook() if op == 'write' else load_workbook(self.source(path))
        try:
            if op == 'read':
                return {s.title: [list(row) for row in s.values] for s in book.worksheets}
            target = book[sheet] if sheet in book.sheetnames else book.active
            if sheet and sheet not in book.sheetnames:
                target = book.create_sheet(sheet)
            for row in rows or []:
                target.append(row)
            out = io.BytesIO()
            book.save(out)
            return self.fs.write(path, out.getvalue())
        finally:
            book.close()

    def pypdf(self, op, path, output=None, pages=None):
        reader = PdfReader(self.source(path))
        indices = range(len(reader.pages)) if pages is None else pages
        if op == 'read':
            return [reader.pages[i].extract_text() for i in indices]
        if op != 'select':
            raise ValueError('unknown PDF operation')
        writer = PdfWriter()
        for i in indices:
            writer.add_page(reader.pages[i])
        out = io.BytesIO()
        writer.write(out)
        return self.fs.write(output, out.getvalue())

    def docx(self, op, path, paragraphs=None):
        if op not in ('read', 'write', 'append'):
            raise ValueError('unknown document operation')
        document = Document() if op == 'write' else Document(self.source(path))
        if op == 'read':
            return [p.text for p in document.paragraphs]
        for text in paragraphs or []:
            document.add_paragraph(text)
        out = io.BytesIO()
        document.save(out)
        return self.fs.write(path, out.getvalue())


def sync_directory(path):
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class WorkspaceStore(Protocol):
    """One session's storage boundary. Keys are opaque, backend-relative IDs.

    publish must durably store immutable file versions before an event commits.
    Keys must remain readable for the lifetime of the event log, even after a
    later publication or deletion. recover repairs the live workspace view;
    restore rebuilds that view from an event's file manifest. Blob backends can
    use content-addressed or generation-prefixed objects. No code is replayed.
    """

    @property
    def location(self) -> HostPath | str: ...
    def initialize(self) -> None: ...
    def recover(self, generation: str | None) -> None: ...
    def publish(self, entries: Iterable[tuple[str, bytes | None]], generation: str) -> None: ...
    def key(self, path: str, generation: str) -> str: ...
    def read(self, key: str) -> bytes: ...
    def restore(self, files: list[dict], generation: str) -> None: ...


class Storage(Protocol):
    def open(self, namespace: str) -> WorkspaceStore: ...


class DirectoryStorage:
    """Local implementation; replace this object to use another storage backend."""

    def __init__(self, root):
        self.root = HostPath(root).absolute()
        missing = []
        directory = self.root
        while not directory.exists():
            missing.append(directory)
            directory = directory.parent
        self.root.mkdir(parents=True, exist_ok=True)
        for directory in reversed(missing):
            sync_directory(directory.parent)
        self.root = self.root.resolve()

    def open(self, namespace):
        if namespace in ('', '.', '..') or Path(namespace).name != namespace:
            raise ValueError('storage namespace must be a single path component')
        return Workspace(self.root / namespace)


class Workspace:
    """Disk backing for the virtual tree; never follows host symlinks.

    The caller holds SQLite's write lock during recovery and publication. A
    journal plus the session's committed generation reconciles filesystem renames
    with the SQLite transaction after a process or machine interruption.
    """

    def __init__(self, root):
        self.root = HostPath(root)
        self.stage = self.root.with_name('.' + self.root.name + '.stage')
        self.backup = self.root.with_name('.' + self.root.name + '.backup')
        self.journal = self.root.with_name('.' + self.root.name + '.journal')
        self.archive = self.root.parent / '.objects' / self.root.name
        self._keys = {}

    @property
    def location(self):
        return self.root

    def initialize(self):
        # Recover a creation interrupted before its session record committed.
        self.recover(None)
        if self.root.is_symlink() or (self.root.exists() and any(self.root.iterdir())):
            raise FileExistsError(f'workspace already exists: {self.root}')
        self.root.mkdir(exist_ok=True)
        sync_directory(self.root.parent)

    def key(self, path, generation):
        return self._keys[path]

    def read(self, key):
        parts = Path(key).parts
        if (len(parts) != 3 or parts[:2] != ('.objects', self.root.name)
                or len(parts[2]) != 64 or any(c not in '0123456789abcdef' for c in parts[2])):
            raise PermissionError('key is outside the session namespace')
        host = self.root.parent
        for part in parts:
            if host.is_symlink():
                raise PermissionError('storage symlinks are not supported')
            host = host / part
        if host.is_symlink():
            raise PermissionError('storage symlinks are not supported')
        content = host.read_bytes()
        if hashlib.sha256(content).hexdigest() != parts[2]:
            raise OSError('stored object failed its content hash check')
        return content

    def restore(self, files, generation):
        self.recover(generation)
        self.root.mkdir(exist_ok=True)
        self.publish([(f['path'], None if f['kind'] == 'directory' else self.read(f['storage_key']))
                      for f in files], generation)
        self.recover(generation)

    def recover(self, generation):
        if not self.journal.exists():
            # A crash while preparing files has no published effects.
            if self.stage.exists():
                shutil.rmtree(self.stage)
            return
        committed = self.journal.read_text() == generation
        if self.backup.exists():
            if committed:
                shutil.rmtree(self.backup)
            else:
                if self.root.exists():
                    shutil.rmtree(self.root)
                self.backup.rename(self.root)
        if self.stage.exists():
            shutil.rmtree(self.stage)
        sync_directory(self.root.parent)
        self.journal.unlink()
        sync_directory(self.root.parent)

    def publish(self, entries, generation):
        entries = list(entries)
        self.archive.mkdir(parents=True, exist_ok=True)
        sync_directory(self.archive.parent)
        sync_directory(self.root.parent)
        self._keys = {}
        for path, content in entries:
            if content is None:
                continue
            digest = hashlib.sha256(content).hexdigest()
            self._keys[path] = f'.objects/{self.root.name}/{digest}'
            target = self.archive / digest
            if not target.exists():
                temporary = self.archive / (uuid.uuid4().hex + '.tmp')
                with temporary.open('xb') as file:
                    file.write(content)
                    file.flush()
                    os.fsync(file.fileno())
                temporary.replace(target)
            else:
                self.read(self._keys[path])  # detect damaged or replaced objects
        sync_directory(self.archive)
        self.stage.mkdir()
        for path, content in entries:
            target = self.stage / path.lstrip('/')
            if content is None:
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open('wb') as file:
                    file.write(content)
                    file.flush()
                    os.fsync(file.fileno())
        for directory, _, _ in os.walk(self.stage, topdown=False):
            sync_directory(directory)
        with self.journal.open('x') as file:
            file.write(generation)
            file.flush()
            os.fsync(file.fileno())
        sync_directory(self.root.parent)
        self.root.rename(self.backup)
        sync_directory(self.root.parent)
        self.stage.rename(self.root)
        sync_directory(self.root.parent)


@dataclass(frozen=True)
class Session:
    id: int
    name: str
    workspace: str
    cwd: str
    monty: bytes | None
    messages: list[dict]
    generation: str
    pending: dict | None
    last_event_id: int


class ExecutionError(RuntimeError):
    """A recorded sandbox failure; infrastructure failures remain retryable."""


class CodeAgent:
    """Append-only events are authoritative; sessions/files are projections.

    Requests commit before code runs. File objects are immutable and fsynced
    before the completion event, Monty checkpoint, projections and tool response
    commit together. Resume retries pending requests and returns recorded results
    for completed execution IDs. Model requests can repeat after a crash; local
    committed effects do not. SQLite serializes writers, including across agents.
    """

    def __init__(self, database='agent.db', *, session_id=None, workspaces=None, storage: Storage | None = None,
                 model=None, base_url=None, api_key=None):
        if storage is not None and workspaces is not None:
            raise ValueError('pass storage or workspaces, not both')
        self.storage = storage if storage is not None else DirectoryStorage(
            workspaces if workspaces is not None else HostPath(database).absolute().parent / 'workspaces')
        self.db = sqlite3.connect(database, timeout=30, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        columns = [r[1] for r in self.db.execute('PRAGMA table_info(sessions)')]
        if columns and 'last_event_id' not in columns:
            self.db.close()
            raise RuntimeError('unsupported database schema; use a fresh database and workspace root')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS events (
                id INTEGER PRIMARY KEY AUTOINCREMENT, session_id INTEGER NOT NULL,
                kind TEXT NOT NULL, execution_id TEXT, payload TEXT NOT NULL,
                monty BLOB, created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(session_id,execution_id,kind));
            CREATE INDEX IF NOT EXISTS events_session ON events(session_id,id);
            CREATE TRIGGER IF NOT EXISTS events_no_update BEFORE UPDATE ON events
                BEGIN SELECT RAISE(ABORT,'events are append-only'); END;
            CREATE TRIGGER IF NOT EXISTS events_no_delete BEFORE DELETE ON events
                BEGIN SELECT RAISE(ABORT,'events are append-only'); END;
            CREATE TABLE IF NOT EXISTS sessions (
                id INTEGER PRIMARY KEY, name TEXT UNIQUE NOT NULL, workspace TEXT NOT NULL,
                cwd TEXT NOT NULL, monty BLOB, messages TEXT NOT NULL,
                generation TEXT NOT NULL, pending TEXT, last_event_id INTEGER NOT NULL);
            CREATE TABLE IF NOT EXISTS files (
                session_id INTEGER NOT NULL, path TEXT NOT NULL,
                kind TEXT NOT NULL CHECK(kind IN ('file','directory')),
                storage_key TEXT, size INTEGER NOT NULL, generation TEXT NOT NULL,
                PRIMARY KEY (session_id,path));
        ''')
        try:
            with self._transaction():
                if session_id is None:
                    self.id = self.db.execute('SELECT COALESCE(MAX(session_id)+1,0) FROM events').fetchone()[0]
                    if not 0 <= self.id < 65536:
                        raise ValueError('names.py capacity exhausted')
                    self.name = num_to_name(self.id)
                    self._workspace = self.storage.open(self.name)
                    self._workspace.initialize()
                    fs = Unix(cwd=self.home)
                    fs.mkdir(self.home)
                    fs.mkdir('/tmp')
                    payload = self._publish(fs.entries())
                    self._append('session_created', {**payload, 'name': self.name, 'cwd': self.home})
                else:
                    self.id = session_id
                    self._replay()
                    self.name = self.state.name
                    self._workspace = self.storage.open(self.name)
                    self._workspace.recover(self.state.generation)
                    self._workspace.restore(self.files(), self.state.generation)
                self.workspace = self._workspace.location
        except BaseException:
            self.db.close()
            raise
        self.model = model or os.environ.get('LLM_MODEL', 'gpt-6-luna')
        self.base_url = (base_url or os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')).rstrip('/')
        self.api_key = api_key if api_key is not None else os.environ.get('LLM_API_KEY', '')

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

    @property
    def home(self):
        return f'/home/{self.name}'

    @property
    def state(self):
        row = self.db.execute('SELECT id,name,workspace,cwd,monty,messages,generation,pending,last_event_id '
                              'FROM sessions WHERE id=?', (self.id,)).fetchone()
        if row is None:
            raise ValueError(f'unknown session: {self.id}')
        return Session(*row[:5], json.loads(row[5]), row[6],
                       None if row[7] is None else json.loads(row[7]), row[8])

    def _append(self, kind, payload, monty=None):
        if not self.db.in_transaction:
            raise RuntimeError('events require a transaction')
        cursor = self.db.execute('INSERT INTO events(session_id,kind,execution_id,payload,monty) VALUES (?,?,?,?,?)',
                                 (self.id, kind, payload.get('execution_id'), json.dumps(payload), monty))
        self._project(cursor.lastrowid, kind, payload, monty)

    def _project(self, event_id, kind, payload, monty):
        if kind == 'session_created':
            self.db.execute('INSERT INTO sessions VALUES (?,?,?,?,?,?,?,?,?)',
                            (self.id, payload['name'], payload['name'], payload['cwd'], None,
                             '[]', payload['generation'], None, event_id))
        elif kind == 'message_appended':
            messages = self.state.messages + [payload['message']]
            self.db.execute('UPDATE sessions SET messages=?,last_event_id=? WHERE id=?',
                            (json.dumps(messages), event_id, self.id))
        elif kind == 'execution_requested':
            self.db.execute('UPDATE sessions SET pending=?,last_event_id=? WHERE id=?',
                            (json.dumps(payload), event_id, self.id))
        elif kind in ('execution_committed', 'execution_failed'):
            messages = self.state.messages
            if payload['tool_id'] is not None:
                messages.append({'role': 'tool', 'tool_call_id': payload['tool_id'], 'content': payload['output']})
            self.db.execute('UPDATE sessions SET messages=?,pending=NULL,last_event_id=? WHERE id=?',
                            (json.dumps(messages), event_id, self.id))
            if kind == 'execution_committed':
                self.db.execute('UPDATE sessions SET cwd=?,monty=?,generation=? WHERE id=?',
                                (payload['cwd'], monty, payload['generation'], self.id))
        else:
            raise ValueError(f'unknown event kind: {kind}')
        if kind in ('session_created', 'execution_committed'):
            self.db.execute('DELETE FROM files WHERE session_id=?', (self.id,))
            self.db.executemany('INSERT INTO files VALUES (?,?,?,?,?,?)',
                [(self.id, f['path'], f['kind'], f['storage_key'], f['size'], payload['generation'])
                 for f in payload['files']])

    def _replay(self):
        row = self.db.execute('SELECT last_event_id FROM sessions WHERE id=?', (self.id,)).fetchone()
        after = row[0] if row else 0
        for event_id, kind, payload, monty in self.db.execute(
                'SELECT id,kind,payload,monty FROM events WHERE session_id=? AND id>? ORDER BY id', (self.id, after)).fetchall():
            self._project(event_id, kind, json.loads(payload), monty)

    def rebuild(self):
        """Rebuild projections and live files from events, without executing code."""
        with self._transaction():
            self.db.execute('DELETE FROM files WHERE session_id=?', (self.id,))
            self.db.execute('DELETE FROM sessions WHERE id=?', (self.id,))
            self._replay()
            self._workspace.restore(self.files(), self.state.generation)

    def events(self):
        return [{'id': row[0], 'kind': row[1], 'payload': json.loads(row[2]), 'created_at': row[3]}
                for row in self.db.execute('SELECT id,kind,payload,created_at FROM events WHERE session_id=? ORDER BY id',
                                           (self.id,))]

    def append_message(self, message, *, expected_event=None):
        with self._transaction():
            self._replay()
            if self.state.pending is not None:
                raise RuntimeError('an execution is pending; resume it before appending messages')
            if expected_event is not None and self.state.last_event_id != expected_event:
                raise RuntimeError('session changed during the model request; resume the current turn')
            self._append('message_appended', {'message': message})

    def sessions(self):
        return [{'id': row[0], 'name': row[1], 'workspace': str(self.storage.open(row[1]).location)}
                for row in self.db.execute('SELECT id,name FROM sessions ORDER BY id')]

    def _publish(self, entries):
        entries, generation = list(entries), uuid.uuid4().hex
        self._workspace.publish(entries, generation)
        return {'generation': generation, 'files': [
            {'path': path, 'kind': 'directory' if content is None else 'file',
             'storage_key': None if content is None else self._workspace.key(path, generation),
             'size': 0 if content is None else len(content)} for path, content in entries]}

    def files(self):
        return [dict(zip(('path', 'kind', 'storage_key', 'size', 'generation'), row)) for row in
                self.db.execute('SELECT path,kind,storage_key,size,generation FROM files '
                                'WHERE session_id=? ORDER BY path', (self.id,))]

    def read_file(self, path):
        path = '/' + posixpath.normpath('/' + str(path).lstrip('/')).lstrip('/')
        row = self.db.execute('SELECT kind,storage_key FROM files WHERE session_id=? AND path=?',
                              (self.id, path)).fetchone()
        if row is None:
            raise FileNotFoundError(path)
        if row[0] != 'file':
            raise IsADirectoryError(path)
        return self._workspace.read(row[1])

    def close(self):
        self.db.close()

    def _result(self, execution_id):
        row = self.db.execute("SELECT kind,payload FROM events WHERE session_id=? AND execution_id=? "
                              "AND kind IN ('execution_committed','execution_failed')", (self.id, execution_id)).fetchone()
        if row is None:
            return None
        payload = json.loads(row[1])
        if row[0] == 'execution_failed':
            raise ExecutionError(payload['output'])
        return payload['output']

    def execute(self, code=None, *, tool_id=None, execution_id=None):
        """Resume with execute(); reuse execution_id to retrieve a committed result."""
        with self._transaction():
            self._replay()
            pending = self.state.pending
            execution_id = execution_id or ('tool:' + tool_id if tool_id is not None else
                                            pending['execution_id'] if pending else uuid.uuid4().hex)
            requested = self.db.execute("SELECT payload FROM events WHERE session_id=? AND execution_id=? "
                                        "AND kind='execution_requested'", (self.id, execution_id)).fetchone()
            if requested:
                request = json.loads(requested[0])
                if code is not None and (request['code'] != code or request['tool_id'] != tool_id):
                    raise ValueError('execution ID was already used for another command')
                result = self._result(execution_id)
                if result is not None:
                    return result
            if pending:
                if execution_id != pending['execution_id'] or (code is not None and code != pending['code']):
                    raise RuntimeError('an execution is pending; call execute() or run() to resume')
            else:
                if not isinstance(code, str):
                    raise ValueError('code must be a string')
                self._append('execution_requested', {'execution_id': execution_id, 'code': code, 'tool_id': tool_id})
        # The request is now durable, independent of the transaction below.
        failed = False
        try:
            with self._transaction():
                self._replay()
                result = self._result(execution_id)
                if result is not None:
                    return result
                request = self.state.pending
                self._workspace.recover(self.state.generation)
                try:
                    fs, snapshot, output = self._evaluate(request['code'])
                except MontyCrashedError:
                    raise  # worker/infrastructure failures keep the request pending
                except MontyError as exc:
                    output = f'{type(exc).__name__}: {exc}'[:24000]
                    self._append('execution_failed', {**request, 'output': output})
                    failed = True
                else:
                    payload = self._publish(fs.entries())
                    self._append('execution_committed', {**request, **payload, 'cwd': fs.cwd, 'output': output}, snapshot)
        except BaseException:
            with self._transaction():
                self._workspace.recover(self.state.generation)
            raise
        if failed:
            raise ExecutionError(output)
        return output

    def _evaluate(self, code):
        state = self.state
        entries = [(f['path'], None if f['kind'] == 'directory' else self._workspace.read(f['storage_key']))
                   for f in self.files()]
        fs = Unix(entries, state.cwd, environ={
            'HOME': self.home, 'USER': self.name, 'LOGNAME': self.name,
            'PWD': state.cwd, 'TMPDIR': '/tmp', 'LANG': 'C.UTF-8',
        })
        fs.path_iterdir(Path(state.cwd))
        libs = Libraries(fs)
        functions = {'pwd': fs.pwd, 'cd': fs.cd, 'ls': fs.ls, 'mkdir': fs.mkdir,
                     'read': fs.read, 'write': fs.write, 'edit': fs.edit,
                     'glob': fs.glob, 'grep': fs.grep, 'mv': fs.mv, 'rm': fs.rm,
                     'numpy': libs.numpy, 'pandas': libs.pandas, 'PIL': libs.PIL,
                     'openpyxl': libs.openpyxl, 'pypdf': libs.pypdf, 'docx': libs.docx}
        with Monty(max_processes=1, request_timeout=20) as pool:
            with pool.checkout(limits={'max_memory': 64 * 1024**2,
                                       'max_feed_duration_secs': 10, 'max_suspensions': 1000}) as session:
                if state.monty is not None:
                    session.load_session(state.monty)
                printed = CollectString(max_bytes=16000)
                value = session.feed_run(code, external_lookup=functions, os=fs,
                                         cwd=fs.cwd, print_callback=printed)
                output = (printed.output + (repr(value) if value is not None else ''))[:24000]
                return fs, session.dump(), output

    def run(self, prompt=None, *, max_steps=20):
        """Resume pending work automatically before accepting a new user turn."""
        if self.state.pending:
            try:
                self.execute()
            except ExecutionError:
                pass
        messages = self.state.messages
        if prompt is not None:
            if messages and (messages[-1]['role'] in ('user', 'tool') or messages[-1].get('tool_calls')):
                repeated = messages[-1]['role'] == 'user' and messages[-1]['content'] == prompt
                answer = self.run(max_steps=max_steps)
                if repeated:
                    return answer
            self.append_message({'role': 'user', 'content': prompt})
        if not self.state.messages:
            return ''
        schema = {'type': 'function', 'function': {'name': 'python',
                  'description': 'Run Python in the durable Monty virtual Unix environment.',
                  'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                                 'required': ['code'], 'additionalProperties': False}}}
        with httpx.Client(timeout=120) as client:
            for _ in range(max_steps):
                state = self.state
                messages = state.messages
                if messages[-1].get('tool_calls') or messages[-1]['role'] == 'tool':
                    index = len(messages) - 1
                    while messages[index]['role'] == 'tool':
                        index -= 1
                    done = {m['tool_call_id'] for m in messages[index+1:]}
                    pending = [c for c in messages[index]['tool_calls'] if c['id'] not in done]
                    if pending:
                        call = pending[0]
                        try:
                            if call['function']['name'] != 'python':
                                raise ValueError('unknown tool')
                            code = json.loads(call['function']['arguments'])['code']
                            if not isinstance(code, str):
                                raise ValueError('code must be a string')
                        except (ValueError, KeyError, TypeError) as exc:
                            self.append_message({'role': 'tool', 'tool_call_id': call['id'],
                                                 'content': f'{type(exc).__name__}: {exc}'[:24000]},
                                                expected_event=state.last_event_id)
                            continue
                        try:
                            self.execute(code, tool_id=call['id'])
                        except ExecutionError:
                            pass  # failure and tool response were committed together
                        continue
                if messages[-1]['role'] == 'assistant':
                    return messages[-1].get('content') or ''
                response = client.post(self.base_url + '/chat/completions',
                                       headers={'Authorization': f'Bearer {self.api_key}'},
                                       json={'model': self.model, 'messages': [
                                           {'role': 'system', 'content':
                                            f'Your session ID is {self.id}; your name is {self.name}.\n' + HELP}, *messages],
                                           'tools': [schema], 'parallel_tool_calls': False})
                response.raise_for_status()
                message = response.json()['choices'][0]['message']
                message = {k: v for k, v in message.items()
                           if k in ('role', 'content', 'tool_calls') and v is not None}
                self.append_message(message, expected_event=state.last_event_id)
                if message['role'] == 'assistant' and not message.get('tool_calls'):
                    return message.get('content') or ''
        raise RuntimeError('Step budget reached; call run() to continue the durable turn')
