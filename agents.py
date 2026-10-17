"""Durable code-mode agent. All sandbox capabilities live in this module.

The SQLite database is trusted local state (Monty dumps are not safe to load
from strangers). Each code call atomically commits files, interpreter globals,
and its tool response; failed calls roll back all three. Native library wrappers
accept plain data, never host paths or arbitrary Python. They are deliberately
small APIs, not drop-in implementations of those libraries. Host library work
is outside Monty's CPU/memory accounting; use process isolation for hostile users.
"""
from __future__ import annotations

import fnmatch
import io
import json
import os
import posixpath
import sqlite3
from pathlib import PurePosixPath as Path

import httpx
import numpy as np
import pandas as pd
from PIL import Image
from docx import Document
from openpyxl import Workbook, load_workbook
from pypdf import PdfReader, PdfWriter
from pydantic_monty import CollectString, Monty, OSAccess
from dotenv import load_dotenv

load_dotenv(override=True)

HELP = """You are a code-mode assistant. Execute Python with the python tool to
inspect and modify a persistent virtual Unix filesystem. No host shell/network.
Globals persist after successful calls. Failed calls roll back globals and files.
Use print for observations. Batch related work; avoid dumping large files.
Paths are POSIX; cwd starts at /. pathlib.Path and open use the virtual filesystem.
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

    def __init__(self, entries=(), cwd='/'):
        super().__init__()
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


class CodeAgent:
    """One durable agent per database; SQLite serializes concurrent code calls.

    A process interrupted during a code call replays the uncommitted call on
    resume. All exposed effects are transactional virtual-file operations.
    Model requests may repeat after a crash; committed code effects do not.
    Use one REPL/model driver per database.
    """

    def __init__(self, database='agent.db', *, model=None, base_url=None, api_key=None):
        self.db = sqlite3.connect(database, timeout=30, isolation_level=None)
        self.db.execute('PRAGMA journal_mode=WAL')
        self.db.execute('PRAGMA synchronous=FULL')
        self.db.executescript('''
            CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value BLOB);
            CREATE TABLE IF NOT EXISTS files (path TEXT PRIMARY KEY, content BLOB);
            INSERT OR IGNORE INTO files VALUES ('/', NULL);
        ''')
        self.model = model or os.environ.get('LLM_MODEL', 'gpt-6-luna')
        self.base_url = (base_url or os.environ.get('LLM_BASE_URL', 'https://api.openai.com/v1')).rstrip('/')
        self.api_key = api_key if api_key is not None else os.environ.get('LLM_API_KEY', '')

    def close(self):
        self.db.close()

    def get(self, key, default=None):
        row = self.db.execute('SELECT value FROM state WHERE key=?', (key,)).fetchone()
        return row[0] if row else default

    def put(self, key, value):
        self.db.execute('INSERT OR REPLACE INTO state VALUES (?,?)', (key, value))

    def execute(self, code, *, tool_id=None, messages=None):
        self.db.execute('BEGIN IMMEDIATE')
        try:
            fs = Unix(self.db.execute('SELECT path,content FROM files'), self.get('cwd', '/'))
            libs = Libraries(fs)
            functions = {'pwd': fs.pwd, 'cd': fs.cd, 'ls': fs.ls, 'mkdir': fs.mkdir,
                         'read': fs.read, 'write': fs.write, 'edit': fs.edit,
                         'glob': fs.glob, 'grep': fs.grep, 'mv': fs.mv, 'rm': fs.rm,
                         'numpy': libs.numpy, 'pandas': libs.pandas, 'PIL': libs.PIL,
                         'openpyxl': libs.openpyxl, 'pypdf': libs.pypdf, 'docx': libs.docx}
            with Monty(max_processes=1, request_timeout=20) as pool:
                with pool.checkout(limits={'max_memory': 64 * 1024**2,
                                           'max_feed_duration_secs': 10,
                                           'max_suspensions': 1000}) as session:
                    saved = self.get('monty')
                    if saved:
                        session.load_session(saved)
                    printed = CollectString(max_bytes=16000)
                    value = session.feed_run(code, external_lookup=functions, os=fs,
                                             cwd=fs.cwd, print_callback=printed)
                    output = (printed.output + (repr(value) if value is not None else ''))[:24000]
                    self.put('monty', session.dump())
            self.db.execute('DELETE FROM files')
            self.db.executemany('INSERT INTO files VALUES (?,?)', fs.entries())
            self.put('cwd', fs.cwd)
            if tool_id is not None:
                history = messages + [{'role': 'tool', 'tool_call_id': tool_id, 'content': output}]
                self.put('messages', json.dumps(history))
            self.db.execute('COMMIT')
            return output
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def run(self, prompt=None, *, max_steps=20):
        """Resume pending work automatically before accepting a new user turn."""
        messages = json.loads(self.get('messages', '[]'))
        if prompt is not None:
            if messages and (messages[-1]['role'] in ('user', 'tool') or messages[-1].get('tool_calls')):
                repeated = messages[-1]['role'] == 'user' and messages[-1]['content'] == prompt
                answer = self.run(max_steps=max_steps)
                if repeated:
                    return answer
                messages = json.loads(self.get('messages', '[]'))
            messages.append({'role': 'user', 'content': prompt})
            self.put('messages', json.dumps(messages))
        if not messages:
            return ''
        schema = {'type': 'function', 'function': {'name': 'python',
                  'description': 'Run Python in the durable Monty virtual Unix environment.',
                  'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                                 'required': ['code'], 'additionalProperties': False}}}
        with httpx.Client(timeout=120) as client:
            for _ in range(max_steps):
                # Resume a partially completed batch without rerunning committed calls.
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
                            self.execute(json.loads(call['function']['arguments'])['code'],
                                         tool_id=call['id'], messages=messages)
                            messages = json.loads(self.get('messages'))
                        except Exception as exc:
                            messages.append({'role': 'tool', 'tool_call_id': call['id'],
                                             'content': f'{type(exc).__name__}: {exc}'[:24000]})
                            self.put('messages', json.dumps(messages))
                        continue
                if messages[-1]['role'] == 'assistant':
                    return messages[-1].get('content') or ''
                response = client.post(self.base_url + '/chat/completions',
                                       headers={'Authorization': f'Bearer {self.api_key}'},
                                       json={'model': self.model, 'messages': [
                                           {'role': 'system', 'content': HELP}, *messages],
                                           'tools': [schema], 'parallel_tool_calls': False})
                response.raise_for_status()
                message = response.json()['choices'][0]['message']
                messages.append({k: v for k, v in message.items()
                                 if k in ('role', 'content', 'tool_calls') and v is not None})
                self.put('messages', json.dumps(messages))
        raise RuntimeError('Step budget reached; call run() to continue the durable turn')
