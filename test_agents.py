import io
import json
import hashlib
import os
import sqlite3
import subprocess
import sys
import shutil
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

import httpx
import main as repl
from pypdf import PdfWriter

from agents import CodeAgent, DirectoryStorage, ExecutionError, Unix, code_prefix
from main import ReplOutput
from rich.console import Console
from names import num_to_name


def sse(delta, finish=None):
    return ('data: ' + json.dumps({'choices': [{'index': 0, 'delta': delta,
                                               'finish_reason': finish}]}) + '\n\n').encode()


def stream_response(message):
    delta = {k: v for k, v in message.items() if k != 'role'}
    if delta.get('tool_calls'):
        delta['tool_calls'] = [dict(call, index=i) for i, call in enumerate(delta['tool_calls'])]
    return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                          content=sse(delta, 'tool_calls' if delta.get('tool_calls') else 'stop') +
                          b'data: [DONE]\n\n')


class ByteStream(httpx.SyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    def __iter__(self):
        yield from self.chunks


class AgentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name + '/agent.db'
        self.agent = CodeAgent(self.path)
        self.session_id = self.agent.id

    def tearDown(self):
        self.agent.close()
        self.tmp.cleanup()

    def test_restart_and_atomic_rollback(self):
        self.agent.execute("mkdir('/w'); cd('/w'); write('a', 'original'); x=42")
        with self.assertRaises(Exception):
            self.agent.execute("write('a','broken'); x=0; raise ValueError('abort')")
        self.agent.close()
        self.agent = CodeAgent(self.path, session_id=self.session_id)
        self.assertEqual(self.agent.execute("(x, read('a'), pwd())"), "(42, '1: original', '/w')")
        self.assertEqual(self.agent.execute("from pathlib import Path; Path('a').read_text()"), "'original'")

    def test_native_fs_and_tools(self):
        self.agent.execute("from pathlib import Path; Path('/w').mkdir(); Path('/w/a.py').write_text('a\\nb\\na'); Path('/empty').mkdir()")
        out = self.agent.execute("grep('a', '/w', limit=1)")
        self.assertIn("'truncated': True", out)
        self.assertIn("'line': 3", self.agent.execute("grep('a', '/w', offset=1)"))
        self.agent.execute("mv('/w','/new'); edit('/new/a.py', 'b', 'c')")
        self.agent.execute("mv('/new/a.py','/new/b.py'); rm('/new/b.py'); write('/new/a.py','c')")
        self.assertIn('/new/a.py', self.agent.execute("glob('*.py')"))
        self.assertEqual(self.agent.execute("Path('/empty').is_dir()"), 'True')
        self.assertEqual(self.agent.execute("Path('/etc/passwd').exists()"), 'False')
        self.agent.execute("rm('/new', recursive=True)")
        self.assertEqual(self.agent.execute("ls('/')"), "['/dev/', '/empty/', '/home/', '/tmp/']")

    def test_library_roundtrips(self):
        self.assertEqual(self.agent.execute("numpy('dot', [1,2], other=[3,4])"), '11')
        self.agent.execute("pandas('to_csv', [{'a': 2}], path='/a.csv')")
        self.assertEqual(self.agent.execute("pandas('read_csv', path='/a.csv')"), "[{'a': 2}]")
        self.agent.execute("pandas('to_excel', [{'a': 2}], path='/a.xlsx')")
        self.assertEqual(self.agent.execute("pandas('read_excel', path='/a.xlsx')"), "[{'a': 2}]")
        self.agent.execute("openpyxl('append','/a.xlsx',rows=[[3]])")
        self.assertIn('[3]', self.agent.execute("openpyxl('read','/a.xlsx')"))
        self.agent.execute("PIL('new',output='/a.png',size=[8,8]); PIL('resize',path='/a.png',output='/b.png',size=[2,2])")
        self.assertIn('[2, 2]', self.agent.execute("PIL('info',path='/b.png')"))
        self.agent.execute("docx('write','/a.docx',paragraphs=['hello']); docx('append','/a.docx',paragraphs=['world'])")
        self.assertEqual(self.agent.execute("docx('read','/a.docx')"), "['hello', 'world']")
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        out = io.BytesIO()
        writer.write(out)
        self.agent.execute(f"write('/a.pdf', {out.getvalue()!r})")
        self.agent.execute("pypdf('select','/a.pdf',output='/b.pdf',pages=[0])")
        self.assertEqual(self.agent.execute("pypdf('read','/b.pdf')"), "['']")

    def test_resume_partial_batch_without_repeating_effects(self):
        calls = [{'id': str(i), 'type': 'function', 'function': {'name': 'python',
                  'arguments': json.dumps({'code': code})}} for i, code in enumerate([
                      "write('/a', 'once')", "edit('/a', 'once', 'twice')"])]
        messages = [{'role': 'user', 'content': 'edit a'},
                    {'role': 'assistant', 'tool_calls': calls}]
        for message in messages:
            self.agent.append_message(message)
        self.agent.execute("write('/a', 'once')", tool_id='0')
        self.agent.close()
        self.agent = CodeAgent(self.path, session_id=self.session_id)
        def respond(request):
            history = json.loads(request.content)['messages']
            self.assertEqual([m['tool_call_id'] for m in history if m['role'] == 'tool'], ['0', '1'])
            return stream_response({'role': 'assistant', 'content': 'done'})
        client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch('agents.httpx.Client', return_value=client):
            self.assertEqual(self.agent.run(), 'done')
        self.assertEqual(self.agent.execute("read('/a')"), "'1: twice'")

    def test_retry_pending_user_prompt(self):
        self.agent.append_message({'role': 'user', 'content': 'hi'})
        requests = []
        def respond(request):
            requests.append(json.loads(request.content)['messages'])
            return stream_response({'role': 'assistant', 'content': 'hello'})
        client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch('agents.httpx.Client', return_value=client):
            self.assertEqual(self.agent.run('hi'), 'hello')
        self.assertEqual(len(requests), 1)
        history = self.agent.state.messages
        self.assertEqual([m['role'] for m in history], ['user', 'assistant'])

    def test_new_prompt_after_interrupted_turn(self):
        self.agent.append_message({'role': 'user', 'content': 'old'})
        requests = []
        def respond(request):
            history = json.loads(request.content)['messages']
            requests.append(history)
            return stream_response({'role': 'assistant', 'content': 'answer ' + history[-1]['content']})
        clients = [httpx.Client(transport=httpx.MockTransport(respond)) for _ in range(2)]
        with patch('agents.httpx.Client', side_effect=clients):
            self.assertEqual(self.agent.run('new'), 'answer new')
        self.assertEqual(len(requests), 2)
        self.assertEqual([m['content'] for m in self.agent.state.messages],
                         ['old', 'answer old', 'new', 'answer new'])

    def test_streams_text_decoded_code_and_results_before_completion(self):
        observed, requests = [], []
        code = 'print("hé😀")\n21 * 2'
        arguments = json.dumps({'code': code}, ensure_ascii=True)
        def record(kind, text):
            observed.append((kind, text))
            if kind == 'stdout':
                self.assertIsNotNone(self.agent.state.pending)
            if kind == 'result':
                self.assertIsNone(self.agent.state.pending)
        self.agent.on_output = record

        def first_response():
            packet = sse({'content': 'Let me calculate.'})
            yield b': heartbeat\n\n' + packet[:9]
            yield packet[9:]
            self.assertIn(('assistant', 'Let me calculate.'), observed)
            yield sse({'tool_calls': [{'index': 0, 'id': 'calc', 'type': 'function',
                                      'function': {'name': 'python', 'arguments': ''}}]})
            for character in arguments:
                yield sse({'tool_calls': [{'index': 0, 'function': {'arguments': character}}]})
            self.assertEqual(''.join(t for k, t in observed if k == 'python'), code)
            self.assertEqual([m['role'] for m in self.agent.state.messages], ['user'])
            yield sse({}, 'tool_calls') + b'data: [DONE]\n\n'

        def final_response():
            yield sse({'content': 'Done'})
            self.assertIn(('assistant', 'Done'), observed)
            yield sse({'content': '.'}, 'stop') + b'data: [DONE]\n\n'

        def respond(request):
            body = json.loads(request.content)
            self.assertTrue(body['stream'])
            requests.append(body)
            if len(requests) == 2:
                self.assertEqual(body['messages'][-1]['content'], 'hé😀\n42')
                self.assertIn(('result', '42'), observed)
            return httpx.Response(200, stream=ByteStream(
                first_response() if len(requests) == 1 else final_response()))

        client = httpx.Client(transport=httpx.MockTransport(respond))
        with patch('agents.httpx.Client', return_value=client):
            self.assertEqual(self.agent.run('calculate'), 'Done.')
        self.assertEqual(len(requests), 2)
        self.assertEqual(''.join(t for k, t in observed if k == 'python'), code)
        self.assertEqual(''.join(t for k, t in observed if k == 'stdout'), 'hé😀\n')
        self.assertEqual(''.join(t for k, t in observed if k == 'result'), '42')
        self.assertEqual(''.join(t for k, t in observed if k == 'assistant'),
                         'Let me calculate.Done.')

    def test_incomplete_model_stream_never_records_or_executes_partial_code(self):
        observed = []
        self.agent.on_output = lambda kind, text: observed.append((kind, text))
        code = "write('unsafe', 'partial')"
        for finish in (None, 'length'):
            with self.subTest(finish=finish):
                content = sse({'content': 'Starting.', 'tool_calls': [{
                    'index': 0, 'id': 'partial', 'function': {'name': 'python',
                    'arguments': json.dumps({'code': code})}}]}, finish)
                if finish:
                    content += b'data: [DONE]\n\n'
                client = httpx.Client(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, stream=ByteStream([content]))))
                with patch('agents.httpx.Client', return_value=client):
                    with self.assertRaisesRegex(RuntimeError, 'Incomplete model response'):
                        self.agent.run('try it')
                self.assertEqual(self.agent.state.messages, [{'role': 'user', 'content': 'try it'}])
                self.assertIsNone(self.agent.state.pending)
                self.assertNotIn('execution_requested', [e['kind'] for e in self.agent.events()])
                self.assertFalse(any(f['path'].endswith('/unsafe') for f in self.agent.files()))
        self.assertTrue(any(kind == 'python' for kind, _ in observed))
        self.agent.close()
        self.agent = CodeAgent(self.path, session_id=self.session_id)
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: stream_response({'role': 'assistant', 'content': 'Recovered.'})))
        with patch('agents.httpx.Client', return_value=client):
            self.assertEqual(self.agent.run(), 'Recovered.')

    def test_partial_json_code_decoding_handles_split_escapes(self):
        code = 'print("hé😀")\nvalue = "\\n"\t# tab'
        encoded = json.dumps({'code': code}, ensure_ascii=True)
        previous = ''
        for end in range(len(encoded) + 1):
            current = code_prefix(encoded[:end])
            self.assertTrue(current.startswith(previous))
            self.assertTrue(code.startswith(current))
            previous = current
        self.assertEqual(previous, code)

    def test_live_output_is_provisional_and_bounded(self):
        self.agent.execute("write('/a','before')")
        observed = []
        self.agent.on_output = lambda kind, text: observed.append((kind, text))
        with self.assertRaises(ExecutionError):
            self.agent.execute("print('preview'); write('/a','after'); raise ValueError('rollback')")
        self.assertEqual(observed[-1], ('stdout', 'preview\n'))
        self.assertFalse(any(kind == 'result' for kind, _ in observed))
        self.assertEqual(self.agent.read_file('/a'), b'before')
        observed.clear()
        with self.assertRaises(ExecutionError):
            self.agent.execute("print('x' * 16001)")
        self.assertLessEqual(sum(len(text.encode()) for kind, text in observed if kind == 'stdout'), 16000)
        self.assertIsNone(self.agent.state.pending)

    def test_repl_displays_streamed_trace_without_duplicate_final_answer(self):
        code = "print('live value')\n6 * 7"
        messages = iter([
            {'role': 'assistant', 'content': 'Calculating.', 'tool_calls': [{
                'id': 'repl-calc', 'type': 'function', 'function': {
                    'name': 'python', 'arguments': json.dumps({'code': code})}}]},
            {'role': 'assistant', 'content': 'Final answer: 42'},
        ])
        client = httpx.Client(transport=httpx.MockTransport(
            lambda request: stream_response(next(messages))))
        output = io.StringIO()
        with patch('sys.argv', ['main.py', '--database', self.tmp.name + '/repl.db',
                                '--workspaces', self.tmp.name + '/repl-workspaces']), \
             patch('builtins.input', side_effect=['calculate', '/quit']), \
             patch('agents.httpx.Client', return_value=client), redirect_stdout(output):
            repl.main()
        transcript = output.getvalue()
        self.assertIn('Assistant', transcript)
        self.assertIn('Calculating.', transcript)
        self.assertEqual(transcript.count("print('live value')"), 1)
        self.assertEqual(transcript.count('6 * 7'), 1)
        self.assertIn('Output · pending commit', transcript)
        self.assertIn('│ live value', transcript)
        self.assertIn('Result', transcript)
        self.assertNotIn('[assistant]', transcript)
        self.assertNotIn('[python]', transcript)
        self.assertNotIn('[result]', transcript)
        self.assertEqual(transcript.count('Final answer: 42'), 1)

    def test_rich_panels_keep_streamed_blocks_and_dark_backgrounds(self):
        buffer = io.StringIO()
        console = Console(file=buffer, force_terminal=True, color_system='truecolor', no_color=False, width=60)
        output = ReplOutput(console)
        output('python', 'value = ')
        output('python', '42')
        self.assertEqual(output.text, 'value = 42')
        output('result', '42')
        self.assertIn('value', buffer.getvalue())
        self.assertIn('48;2;17;22;29', buffer.getvalue())
        output.finish()
        # Repeated errors after a closed block must also start a fresh panel.
        output('error', 'first error')
        output.finish()
        output('error', 'second error')
        output.finish()
        self.assertIn('second error', buffer.getvalue())

    def test_isolated_sessions_and_disk_files(self):
        self.agent.execute("write('/a','first'); x=1")
        self.agent.append_message({'role': 'user', 'content': 'first'})
        second = CodeAgent(self.path)
        try:
            self.assertNotEqual(second.id, self.agent.id)
            self.assertEqual(second.name, num_to_name(second.id))
            self.assertEqual(second.workspace.name, second.name)
            self.assertEqual(second.execute("ls()"), '[]')
            self.assertEqual(second.state.messages, [])
            second.execute("write('/a','second'); x=2")
            self.assertEqual((second.workspace / 'a').read_text(), 'second')
            self.assertEqual((self.agent.workspace / 'a').read_text(), 'first')
            self.assertEqual(self.agent.execute('x'), '1')
            self.assertEqual(second.execute('x'), '2')
            self.assertEqual(len(second.sessions()), 2)
            self.assertTrue(second.db.execute("SELECT 1 FROM sqlite_master WHERE name='files'").fetchone())
            self.assertEqual(second.read_file('/a'), b'second')
            self.assertEqual(self.agent.read_file('/a'), b'first')
        finally:
            second.close()

    def test_symlink_cannot_escape_workspace(self):
        (self.agent.workspace / 'escape').symlink_to('/etc/passwd')
        with self.assertRaises(ExecutionError):
            self.agent.execute("read('/escape')")

    def test_uncommitted_objects_do_not_change_session_files(self):
        self.agent.execute("write('/a','before'); x=1")
        self.agent.db.execute('BEGIN IMMEDIATE')
        self.agent._publish(Unix([('/a', b'uncommitted')]).entries())
        # Closing the connection simulates loss of the process before DB commit.
        self.agent.close()
        self.agent = CodeAgent(self.path, session_id=self.session_id)
        self.assertEqual((self.agent.workspace / 'a').read_text(), 'before')
        self.assertEqual(self.agent.execute('x'), '1')

    def test_failure_before_event_commit_preserves_files_and_globals(self):
        self.agent.execute("write('/a','before'); x=1")
        before = self.agent.files()
        append = self.agent._append
        snapshot = self.agent.state.monty
        def fail_commit(kind, payload, monty=None):
            if kind == 'execution_committed':
                raise OSError('simulated state write failure')
            append(kind, payload, monty)
        with patch.object(self.agent, '_append', side_effect=fail_commit):
            with self.assertRaises(OSError):
                self.agent.execute("write('/a','after'); x=2")
        self.assertEqual((self.agent.workspace / 'a').read_text(), 'before')
        self.assertEqual(self.agent.files(), before)
        self.assertEqual(self.agent.state.monty, snapshot)
        self.assertIsNotNone(self.agent.state.pending)
        self.agent.execute()
        self.assertEqual(self.agent.execute('x'), '2')

    def test_home_environment_and_native_io_share_virtual_filesystem(self):
        home = self.agent.home
        self.assertEqual(self.agent.execute('pwd()'), repr(home))
        result = self.agent.execute("import os; from pathlib import Path; (str(Path.cwd()), os.getenv('HOME'), os.getenv('USER'), os.getenv('TMPDIR'), os.getenv('PWD'))")
        self.assertEqual(result, repr((home, home, self.agent.name, '/tmp', home)))
        self.agent.execute("write('relative.txt','tools'); Path('relative.txt').write_text('native')")
        self.assertEqual(self.agent.execute("read('relative.txt')"), "'1: native'")
        self.assertEqual((self.agent.workspace / home.lstrip('/') / 'relative.txt').read_text(), 'native')
        self.agent.execute("with open('/dev/null','wb') as f:\n    f.write(b'discard me')")
        self.assertEqual(self.agent.execute("Path('/dev/null').read_bytes()"), "b''")
        self.agent.execute("cd('/tmp'); write('file','temp')")
        self.agent.close()
        self.agent = CodeAgent(self.path, session_id=self.session_id)
        self.assertEqual(self.agent.execute("(pwd(), str(Path.cwd()), os.getenv('PWD'), read('file'))"),
                         "('/tmp', '/tmp', '/tmp', '1: temp')")
        self.assertEqual(self.agent.execute("os.getenv('LLM_API_KEY')"), '')

    def test_database_links_follow_file_changes(self):
        self.agent.execute("write('a.txt','hé'); mkdir('empty')")
        path = self.agent.home + '/a.txt'
        record = next(f for f in self.agent.files() if f['path'] == path)
        self.assertEqual(record['storage_key'], f'.objects/{self.agent.name}/' + hashlib.sha256('hé'.encode()).hexdigest())
        self.assertEqual(record['size'], len('hé'.encode()))
        self.assertEqual(self.agent.read_file(path), 'hé'.encode())
        self.assertNotIn('content', [r[1] for r in self.agent.db.execute('PRAGMA table_info(files)')])
        self.agent.execute("mv('a.txt','b.txt')")
        with self.assertRaises(FileNotFoundError):
            self.agent.read_file(path)
        self.assertEqual(self.agent.read_file(self.agent.home + '/b.txt'), 'hé'.encode())
        self.agent.execute("rm('b.txt'); rm('empty')")
        self.assertNotIn(path, [f['path'] for f in self.agent.files()])
        with self.assertRaises(IsADirectoryError):
            self.agent.read_file(self.agent.home)
        with self.assertRaises(PermissionError):
            self.agent.storage.read('../other/file')

    def test_storage_root_can_move_without_rewriting_database_keys(self):
        self.agent.execute("write('a','portable'); x=7")
        records = self.agent.files()
        old_root = self.agent.storage.root
        self.agent.close()
        new_root = old_root.with_name('relocated')
        shutil.move(old_root, new_root)
        self.agent = CodeAgent(self.path, session_id=self.session_id, storage=DirectoryStorage(new_root))
        self.assertEqual(self.agent.files(), records)
        self.assertEqual(self.agent.read_file(self.agent.home + '/a'), b'portable')
        self.assertEqual(self.agent.execute("(x, read('a'))"), "(7, '1: portable')")

    def test_replaceable_object_storage_backend(self):
        storage = ObjectStorage()
        path = self.tmp.name + '/objects.db'
        agent = CodeAgent(path, storage=storage)
        try:
            agent.execute("write('a','object'); x=9")
            record = next(f for f in agent.files() if f['path'] == agent.home + '/a')
            self.assertTrue(record['storage_key'].startswith(agent.name + '/'))
            self.assertEqual(agent.read_file(agent.home + '/a'), b'object')
            session_id = agent.id
        finally:
            agent.close()
        agent = CodeAgent(path, session_id=session_id, storage=storage)
        try:
            self.assertEqual(agent.execute("(x, read('a'))"), "(9, '1: object')")
            before = agent.files()
            append = agent._append
            def fail_commit(kind, payload, monty=None):
                if kind == 'execution_committed':
                    raise OSError('failed after object upload')
                append(kind, payload, monty)
            with patch.object(agent, '_append', side_effect=fail_commit):
                with self.assertRaises(OSError):
                    agent.execute("write('a','uncommitted')")
            self.assertEqual(agent.files(), before)
            self.assertEqual(agent.read_file(agent.home + '/a'), b'object')
            agent.execute()
            self.assertEqual(agent.read_file(agent.home + '/a'), b'uncommitted')
            with self.assertRaises(Exception):
                agent.execute("write('a','bad'); raise ValueError('rollback')")
            self.assertEqual(agent.read_file(agent.home + '/a'), b'uncommitted')
        finally:
            agent.close()

    def test_event_log_rebuilds_every_projection_without_running_code(self):
        self.agent.append_message({'role': 'user', 'content': 'a durable turn'})
        self.agent.execute("write('version.txt','old'); n=1", execution_id='version-one')
        old = next(f for f in self.agent.files() if f['path'].endswith('/version.txt'))
        self.agent.execute("write('version.txt','new'); mkdir('empty'); n=2")
        self.agent.append_message({'role': 'assistant', 'content': 'done'})
        state, files, events = self.agent.state, self.agent.files(), self.agent.events()
        # Simulate lost/corrupt materialized tables and a deleted live workspace.
        self.agent.db.execute('DELETE FROM files WHERE session_id=?', (self.agent.id,))
        self.agent.db.execute('DELETE FROM sessions WHERE id=?', (self.agent.id,))
        shutil.rmtree(self.agent.workspace)
        with patch.object(self.agent, '_evaluate', side_effect=AssertionError('replay executed code')):
            self.agent.rebuild()
        self.assertEqual(self.agent.state, state)
        self.assertEqual(self.agent.files(), files)
        self.assertEqual(self.agent.events(), events)
        self.assertEqual(self.agent.storage.read(old['storage_key']), b'old')
        self.assertEqual(self.agent.read_file(self.agent.home + '/version.txt'), b'new')
        self.assertEqual((self.agent.workspace / self.agent.home.lstrip('/') / 'version.txt').read_text(), 'new')
        self.assertEqual(self.agent.execute('n'), '2')
        columns = [r[1] for r in self.agent.db.execute('PRAGMA table_info(sessions)')]
        for name in ('cwd', 'monty', 'messages', 'generation', 'pending', 'last_event_id'):
            self.assertIn(name, columns)
        self.assertIsNone(self.agent.db.execute("SELECT 1 FROM sqlite_master WHERE name='session_state'").fetchone())

    def test_event_log_is_append_only_and_execution_ids_are_idempotent(self):
        self.agent.execute('n=0')
        self.assertEqual(self.agent.execute('n += 1; n', execution_id='increment'), '1')
        self.assertEqual(self.agent.execute('n += 1; n', execution_id='increment'), '1')
        with self.assertRaises(ValueError):
            self.agent.execute('n += 2; n', execution_id='increment')
        self.assertEqual(self.agent.execute('n'), '1')
        with self.assertRaises(sqlite3.IntegrityError):
            self.agent.db.execute("UPDATE events SET kind='changed'")
        with self.assertRaises(sqlite3.IntegrityError):
            self.agent.db.execute('DELETE FROM events')

    def test_stale_model_response_cannot_overwrite_another_turn(self):
        revision = self.agent.state.last_event_id
        self.agent.append_message({'role': 'user', 'content': 'newer message'})
        with self.assertRaises(RuntimeError):
            self.agent.append_message({'role': 'assistant', 'content': 'stale'}, expected_event=revision)
        self.assertEqual(self.agent.state.messages, [{'role': 'user', 'content': 'newer message'}])

    def test_recorded_failure_is_not_executed_again(self):
        self.agent.execute("n=0; write('a','before')")
        command = "n=1; write('a','bad'); raise ValueError('failure')"
        with self.assertRaises(ExecutionError):
            self.agent.execute(command, execution_id='failed-command')
        with patch.object(self.agent, '_evaluate', side_effect=AssertionError('retried failed code')):
            with self.assertRaises(ExecutionError):
                self.agent.execute(command, execution_id='failed-command')
            self.agent.rebuild()
        self.assertIsNone(self.agent.state.pending)
        self.assertEqual(self.agent.read_file(self.agent.home + '/a'), b'before')
        self.assertEqual(self.agent.execute('n'), '0')
        events = [e['kind'] for e in self.agent.events() if e['payload'].get('execution_id') == 'failed-command']
        self.assertEqual(events, ['execution_requested', 'execution_failed'])

    def test_abrupt_process_exit_during_session_creation(self):
        script = '''
import os, sys
from agents import CodeAgent
append = CodeAgent._append
def crash_append(agent, kind, payload, monty=None):
    append(agent, kind, payload, monty)
    if kind == 'session_created':
        os._exit(92)
CodeAgent._append = crash_append
CodeAgent(sys.argv[1], workspaces=sys.argv[2])
'''
        path, root = self.tmp.name + '/creation.db', self.tmp.name + '/creation-workspaces'
        result = subprocess.run([sys.executable, '-c', script, path, root],
                                cwd=os.path.dirname(__file__), timeout=30,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertEqual(result.returncode, 92)
        agent = CodeAgent(path, workspaces=root)
        try:
            self.assertEqual(agent.id, 0)
            self.assertEqual(agent.execute('pwd()'), repr(agent.home))
            self.assertEqual([e['kind'] for e in agent.events()].count('session_created'), 1)
        finally:
            agent.close()

    def test_abrupt_process_exit_at_commit_boundaries(self):
        script = '''
import os, sys
from agents import CodeAgent
agent = CodeAgent(sys.argv[1], session_id=int(sys.argv[2]))
phase = sys.argv[3]
if phase == 'requested':
    agent._evaluate = lambda code: os._exit(91)
elif phase == 'published':
    publish = agent._publish
    def crash_publish(entries):
        publish(entries)
        os._exit(91)
    agent._publish = crash_publish
elif phase == 'checkout':
    checkout = agent.storage.checkout
    def crash_checkout(name, files):
        checkout(name, files[:2])
        os._exit(91)
    agent.storage.checkout = crash_checkout
elif phase == 'projected':
    project = agent._project
    def crash_project(event_id, kind, payload, monty):
        project(event_id, kind, payload, monty)
        if kind == 'execution_committed':
            os._exit(91)
    agent._project = crash_project
agent.execute("n += 1; write('counter.txt', str(n)); n", tool_id='counter-tool', execution_id='crash-test')
os._exit(91)
'''
        for phase in ('requested', 'published', 'projected', 'checkout', 'committed'):
            with self.subTest(phase=phase):
                agent = CodeAgent(self.path)
                agent.execute("n=0; write('counter.txt','0')")
                agent.append_message({'role': 'user', 'content': 'increment the counter'})
                agent.append_message({'role': 'assistant', 'tool_calls': [{
                    'id': 'counter-tool', 'type': 'function', 'function': {'name': 'python',
                    'arguments': json.dumps({'code': "n += 1; write('counter.txt', str(n)); n"})}}]})
                session_id = agent.id
                agent.close()
                result = subprocess.run([sys.executable, '-c', script, self.path, str(session_id), phase],
                                        cwd=os.path.dirname(__file__), timeout=30,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.assertEqual(result.returncode, 91)
                agent = CodeAgent(self.path, session_id=session_id)
                try:
                    self.assertEqual(agent.execute(execution_id='crash-test'), '1')
                    self.assertEqual(agent.execute(execution_id='crash-test'), '1')
                    self.assertIsNone(agent.state.pending)
                    self.assertEqual([m['content'] for m in agent.state.messages if m['role'] == 'tool'], ['1'])
                    self.assertEqual(agent.read_file(agent.home + '/counter.txt'), b'1')
                    events = [e for e in agent.events() if e['payload'].get('execution_id') == 'crash-test']
                    self.assertEqual([e['kind'] for e in events], ['execution_requested', 'execution_committed'])
                    self.assertEqual(agent.execute('n'), '1')
                finally:
                    agent.close()
class ObjectStorage:
    """A blob-like backend with no local filesystem operations."""

    def __init__(self):
        self.objects = {}
        self.workspaces = {}

    def create(self, name):
        if name in self.workspaces:
            raise FileExistsError(name)

    def put(self, name, content):
        key = name + '/' + hashlib.sha256(content).hexdigest()
        self.objects[key] = content
        return key

    def read(self, key):
        return self.objects[key]

    def checkout(self, name, files):
        self.workspaces[name] = files
        return 'memory://' + name


if __name__ == '__main__':
    unittest.main()
