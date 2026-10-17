import io
import asyncio
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
from openai import AsyncOpenAI
import main as repl
from pypdf import PdfWriter

from agents import CodeAgent, DirectoryStorage, ExecutionError, Unix, code_prefix
from main import ReplOutput
from rich.console import Console
from names import num_to_name


def sse(delta, finish=None):
    return ('data: ' + json.dumps({'id': 'chat-test', 'object': 'chat.completion.chunk',
                                   'created': 0, 'model': 'test', 'choices': [{'index': 0, 'delta': delta,
                                               'finish_reason': finish}]}) + '\n\n').encode()


def stream_response(message):
    delta = dict(message)
    if delta.get('tool_calls'):
        delta['tool_calls'] = [dict(call, index=i) for i, call in enumerate(delta['tool_calls'])]
    return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                          content=sse(delta, 'tool_calls' if delta.get('tool_calls') else 'stop') +
                          b'data: [DONE]\n\n')


def model_client(client):
    return AsyncOpenAI(api_key='test', base_url='http://test/v1', http_client=client, max_retries=0)


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, chunks):
        self.chunks = chunks

    async def __aiter__(self):
        for chunk in self.chunks:
            yield chunk


class AgentTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name + '/agent.db'
        self.agent = await CodeAgent.open(self.path)
        self.session_id = self.agent.id

    async def asyncTearDown(self):
        await self.agent.close()
        self.tmp.cleanup()

    async def test_restart_and_atomic_rollback(self):
        await self.agent.execute("mkdir('/w'); cd('/w'); write('a', 'original'); x=42")
        with self.assertRaises(Exception):
            await self.agent.execute("write('a','broken'); x=0; raise ValueError('abort')")
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute("(x, read('a'), pwd())"), "(42, '1: original', '/w')")
        self.assertEqual(await self.agent.execute("from pathlib import Path; Path('a').read_text()"), "'original'")

    async def test_native_fs_and_tools(self):
        await self.agent.execute("from pathlib import Path; Path('/w').mkdir(); Path('/w/a.py').write_text('a\\nb\\na'); Path('/empty').mkdir()")
        out = await self.agent.execute("grep('a', '/w', limit=1)")
        self.assertIn("'truncated': True", out)
        self.assertIn("'line': 3", await self.agent.execute("grep('a', '/w', offset=1)"))
        await self.agent.execute("mv('/w','/new'); edit('/new/a.py', 'b', 'c')")
        await self.agent.execute("mv('/new/a.py','/new/b.py'); rm('/new/b.py'); write('/new/a.py','c')")
        self.assertIn('/new/a.py', await self.agent.execute("glob('*.py')"))
        self.assertEqual(await self.agent.execute("Path('/empty').is_dir()"), 'True')
        self.assertEqual(await self.agent.execute("Path('/etc/passwd').exists()"), 'False')
        await self.agent.execute("rm('/new', recursive=True)")
        self.assertEqual(await self.agent.execute("ls('/')"), "['/dev/', '/empty/', '/home/', '/tmp/']")

    async def test_library_roundtrips(self):
        self.assertEqual(await self.agent.execute("numpy('dot', [1,2], other=[3,4])"), '11')
        await self.agent.execute("pandas('to_csv', [{'a': 2}], path='/a.csv')")
        self.assertEqual(await self.agent.execute("pandas('read_csv', path='/a.csv')"), "[{'a': 2}]")
        await self.agent.execute("pandas('to_excel', [{'a': 2}], path='/a.xlsx')")
        self.assertEqual(await self.agent.execute("pandas('read_excel', path='/a.xlsx')"), "[{'a': 2}]")
        await self.agent.execute("openpyxl('append','/a.xlsx',rows=[[3]])")
        self.assertIn('[3]', await self.agent.execute("openpyxl('read','/a.xlsx')"))
        await self.agent.execute("PIL('new',output='/a.png',size=[8,8]); PIL('resize',path='/a.png',output='/b.png',size=[2,2])")
        self.assertIn('[2, 2]', await self.agent.execute("PIL('info',path='/b.png')"))
        await self.agent.execute("docx('write','/a.docx',paragraphs=['hello']); docx('append','/a.docx',paragraphs=['world'])")
        self.assertEqual(await self.agent.execute("docx('read','/a.docx')"), "['hello', 'world']")
        writer = PdfWriter()
        writer.add_blank_page(100, 100)
        out = io.BytesIO()
        writer.write(out)
        await self.agent.execute(f"write('/a.pdf', {out.getvalue()!r})")
        await self.agent.execute("pypdf('select','/a.pdf',output='/b.pdf',pages=[0])")
        self.assertEqual(await self.agent.execute("pypdf('read','/b.pdf')"), "['']")

    async def test_resume_partial_batch_without_repeating_effects(self):
        calls = [{'id': str(i), 'type': 'function', 'function': {'name': 'python',
                  'arguments': json.dumps({'code': code})}} for i, code in enumerate([
                      "write('/a', 'once')", "edit('/a', 'once', 'twice')"])]
        messages = [{'role': 'user', 'content': 'edit a'},
                    {'role': 'assistant', 'tool_calls': calls}]
        for message in messages:
            await self.agent.append_message(message)
        await self.agent.execute("write('/a', 'once')", tool_id='0')
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        def respond(request):
            history = json.loads(request.content)['messages']
            self.assertEqual([m['tool_call_id'] for m in history if m['role'] == 'tool'], ['0', '1'])
            return stream_response({'role': 'assistant', 'content': 'done'})
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(self.agent.runtime, 'client', model_client(client)):
            self.assertEqual(await self.agent.run(), 'done')
        self.assertEqual(await self.agent.execute("read('/a')"), "'1: twice'")

    async def test_retry_pending_user_prompt(self):
        await self.agent.append_message({'role': 'user', 'content': 'hi'})
        requests = []
        def respond(request):
            requests.append(json.loads(request.content)['messages'])
            return stream_response({'role': 'assistant', 'content': 'hello'})
        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(self.agent.runtime, 'client', model_client(client)):
            self.assertEqual(await self.agent.run('hi'), 'hello')
        self.assertEqual(len(requests), 1)
        history = (await self.agent.state()).messages
        self.assertEqual([m['role'] for m in history], ['user', 'assistant'])

    async def test_new_prompt_after_interrupted_turn(self):
        await self.agent.append_message({'role': 'user', 'content': 'old'})
        requests = []
        def respond(request):
            history = json.loads(request.content)['messages']
            requests.append(history)
            return stream_response({'role': 'assistant', 'content': 'answer ' + history[-1]['content']})
        clients = [httpx.AsyncClient(transport=httpx.MockTransport(respond)) for _ in range(2)]
        with patch.object(self.agent.runtime, 'client', model_client(clients[0])):
            self.assertEqual(await self.agent.run('new'), 'answer new')
        self.assertEqual(len(requests), 2)
        self.assertEqual([m['content'] for m in (await self.agent.state()).messages],
                         ['old', 'answer old', 'new', 'answer new'])

    async def test_streams_text_decoded_code_and_results_before_completion(self):
        observed, requests = [], []
        code = 'print("hé😀")\n21 * 2'
        arguments = json.dumps({'code': code}, ensure_ascii=True)
        def record(kind, text):
            observed.append((kind, text))
            if kind == 'stdout':
                self.assertIsNotNone(self.agent._state.pending)
            if kind == 'result':
                self.assertIsNone(self.agent._state.pending)
        self.agent.on_output = record

        def first_response():
            packet = sse({'role': 'assistant', 'content': 'Let me calculate.'})
            yield b': heartbeat\n\n' + packet[:9]
            yield packet[9:]
            self.assertIn(('assistant', 'Let me calculate.'), observed)
            yield sse({'tool_calls': [{'index': 0, 'id': 'calc', 'type': 'function',
                                      'function': {'name': 'python', 'arguments': ''}}]})
            for character in arguments:
                yield sse({'tool_calls': [{'index': 0, 'function': {'arguments': character}}]})
            self.assertEqual(''.join(t for k, t in observed if k == 'python'), code)
            self.assertEqual([m['role'] for m in self.agent._state.messages], ['user'])
            yield sse({}, 'tool_calls') + b'data: [DONE]\n\n'

        def final_response():
            yield sse({'role': 'assistant', 'content': 'Done'})
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

        client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        with patch.object(self.agent.runtime, 'client', model_client(client)):
            self.assertEqual(await self.agent.run('calculate'), 'Done.')
        self.assertEqual(len(requests), 2)
        self.assertEqual(''.join(t for k, t in observed if k == 'python'), code)
        self.assertEqual(''.join(t for k, t in observed if k == 'stdout'), 'hé😀\n')
        self.assertEqual(''.join(t for k, t in observed if k == 'result'), '42')
        self.assertEqual(''.join(t for k, t in observed if k == 'assistant'),
                         'Let me calculate.Done.')

    async def test_incomplete_model_stream_never_records_or_executes_partial_code(self):
        observed = []
        self.agent.on_output = lambda kind, text: observed.append((kind, text))
        code = "write('unsafe', 'partial')"
        for finish in (None, 'length'):
            with self.subTest(finish=finish):
                content = sse({'role': 'assistant', 'content': 'Starting.', 'tool_calls': [{
                    'index': 0, 'id': 'partial', 'type': 'function', 'function': {'name': 'python',
                    'arguments': json.dumps({'code': code})}}]}, finish)
                if finish:
                    content += b'data: [DONE]\n\n'
                client = httpx.AsyncClient(transport=httpx.MockTransport(
                    lambda request: httpx.Response(200, stream=ByteStream([content]))))
                with patch.object(self.agent.runtime, 'client', model_client(client)):
                    with self.assertRaisesRegex(RuntimeError, 'Incomplete model response'):
                        await self.agent.run('try it')
                self.assertEqual((await self.agent.state()).messages, [{'role': 'user', 'content': 'try it'}])
                self.assertIsNone((await self.agent.state()).pending)
                self.assertEqual(self.agent.db.execute('SELECT COUNT(*) FROM executions').fetchone()[0], 0)
                self.assertFalse(any(f['path'].endswith('/unsafe') for f in await self.agent.files()))
        self.assertTrue(any(kind == 'python' for kind, _ in observed))
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: stream_response({'role': 'assistant', 'content': 'Recovered.'})))
        with patch.object(self.agent.runtime, 'client', model_client(client)):
            self.assertEqual(await self.agent.run(), 'Recovered.')

    async def test_partial_json_code_decoding_handles_split_escapes(self):
        code = 'print("hé😀")\nvalue = "\\n"\t# tab'
        encoded = json.dumps({'code': code}, ensure_ascii=True)
        previous = ''
        for end in range(len(encoded) + 1):
            current = code_prefix(encoded[:end])
            self.assertTrue(current.startswith(previous))
            self.assertTrue(code.startswith(current))
            previous = current
        self.assertEqual(previous, code)

    async def test_live_output_is_provisional_and_bounded(self):
        await self.agent.execute("write('/a','before')")
        observed = []
        self.agent.on_output = lambda kind, text: observed.append((kind, text))
        with self.assertRaises(ExecutionError):
            await self.agent.execute("print('preview'); write('/a','after'); raise ValueError('rollback')")
        self.assertEqual(observed[-1], ('stdout', 'preview\n'))
        self.assertFalse(any(kind == 'result' for kind, _ in observed))
        self.assertEqual(await self.agent.read_file('/a'), b'before')
        observed.clear()
        with self.assertRaises(ExecutionError):
            await self.agent.execute("print('x' * 16001)")
        self.assertLessEqual(sum(len(text.encode()) for kind, text in observed if kind == 'stdout'), 16000)
        self.assertIsNone((await self.agent.state()).pending)

    async def test_repl_displays_streamed_trace_without_duplicate_final_answer(self):
        code = "print('live value')\n6 * 7"
        messages = iter([
            {'role': 'assistant', 'content': 'Calculating.', 'tool_calls': [{
                'id': 'repl-calc', 'type': 'function', 'function': {
                    'name': 'python', 'arguments': json.dumps({'code': code})}}]},
            {'role': 'assistant', 'content': 'Final answer: 42'},
        ])
        client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: stream_response(next(messages))))
        output = io.StringIO()
        with patch('sys.argv', ['main.py', '--database', self.tmp.name + '/repl.db',
                                '--workspaces', self.tmp.name + '/repl-workspaces']), \
             patch('builtins.input', side_effect=['calculate', '/quit']), \
             patch('agents.AsyncOpenAI', return_value=model_client(client)), redirect_stdout(output):
            await repl.main()
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

    async def test_rich_panels_keep_streamed_blocks_and_dark_backgrounds(self):
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

    async def test_isolated_sessions_and_disk_files(self):
        await self.agent.execute("write('/a','first'); x=1")
        await self.agent.append_message({'role': 'user', 'content': 'first'})
        second = await CodeAgent.open(self.path)
        try:
            self.assertNotEqual(second.id, self.agent.id)
            self.assertEqual(second.name, num_to_name(second.id))
            self.assertEqual(second.workspace.name, second.name)
            self.assertEqual(await second.execute("ls()"), repr([second.home + '/README.md']))
            self.assertEqual((await second.state()).messages, [])
            await second.execute("write('/a','second'); x=2")
            self.assertEqual((second.workspace / 'a').read_text(), 'second')
            self.assertEqual((self.agent.workspace / 'a').read_text(), 'first')
            self.assertEqual(await self.agent.execute('x'), '1')
            self.assertEqual(await second.execute('x'), '2')
            self.assertEqual(len(await second.sessions()), 2)
            self.assertTrue(second.db.execute("SELECT 1 FROM sqlite_master WHERE name='files'").fetchone())
            self.assertEqual(await second.read_file('/a'), b'second')
            self.assertEqual(await self.agent.read_file('/a'), b'first')
        finally:
            await second.close()

    async def test_symlink_cannot_escape_workspace(self):
        (self.agent.workspace / 'escape').symlink_to('/etc/passwd')
        with self.assertRaises(ExecutionError):
            await self.agent.execute("read('/escape')")

    async def test_uncommitted_objects_do_not_change_session_files(self):
        await self.agent.execute("write('/a','before'); x=1")
        self.agent.db.execute('BEGIN IMMEDIATE')
        self.agent._publish(Unix([('/a', b'uncommitted')]).entries())
        # Closing the connection simulates loss of the process before DB commit.
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual((self.agent.workspace / 'a').read_text(), 'before')
        self.assertEqual(await self.agent.execute('x'), '1')

    async def test_failed_checkpoint_commit_preserves_files_and_globals(self):
        await self.agent.execute("write('/a','before'); x=1")
        before, snapshot = await self.agent.files(), (await self.agent.state()).monty
        replace = self.agent._replace_files
        def fail_commit(files):
            replace(files)
            raise OSError('simulated checkpoint write failure')
        with patch.object(self.agent, '_replace_files', side_effect=fail_commit):
            with self.assertRaises(OSError):
                await self.agent.execute("write('/a','after'); x=2")
        self.assertEqual(await self.agent.files(), before)
        self.assertEqual((await self.agent.state()).monty, snapshot)
        self.assertEqual((self.agent.workspace / 'a').read_text(), 'before')
        self.assertIsNotNone((await self.agent.state()).pending)
        await self.agent.execute()
        self.assertEqual(await self.agent.execute('x'), '2')

    async def test_home_environment_and_native_io_share_virtual_filesystem(self):
        home = self.agent.home
        self.assertEqual(await self.agent.execute('pwd()'), repr(home))
        result = await self.agent.execute("import os; from pathlib import Path; (str(Path.cwd()), os.getenv('HOME'), os.getenv('USER'), os.getenv('TMPDIR'), os.getenv('PWD'))")
        self.assertEqual(result, repr((home, home, self.agent.name, '/tmp', home)))
        await self.agent.execute("write('relative.txt','tools'); Path('relative.txt').write_text('native')")
        self.assertEqual(await self.agent.execute("read('relative.txt')"), "'1: native'")
        self.assertEqual((self.agent.workspace / home.lstrip('/') / 'relative.txt').read_text(), 'native')
        await self.agent.execute("with open('/dev/null','wb') as f:\n    f.write(b'discard me')")
        self.assertEqual(await self.agent.execute("Path('/dev/null').read_bytes()"), "b''")
        await self.agent.execute("cd('/tmp'); write('file','temp')")
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute("(pwd(), str(Path.cwd()), os.getenv('PWD'), read('file'))"),
                         "('/tmp', '/tmp', '/tmp', '1: temp')")
        self.assertEqual(await self.agent.execute("os.getenv('LLM_API_KEY')"), '')

    async def test_database_links_follow_file_changes(self):
        await self.agent.execute("write('a.txt','hé'); mkdir('empty')")
        path = self.agent.home + '/a.txt'
        record = next(f for f in await self.agent.files() if f['path'] == path)
        self.assertEqual(record['storage_key'], f'.objects/{self.agent.name}/' + hashlib.sha256('hé'.encode()).hexdigest())
        self.assertEqual(record['size'], len('hé'.encode()))
        self.assertEqual(await self.agent.read_file(path), 'hé'.encode())
        self.assertNotIn('content', [r[1] for r in self.agent.db.execute('PRAGMA table_info(files)')])
        await self.agent.execute("mv('a.txt','b.txt')")
        with self.assertRaises(FileNotFoundError):
            await self.agent.read_file(path)
        self.assertEqual(await self.agent.read_file(self.agent.home + '/b.txt'), 'hé'.encode())
        await self.agent.execute("rm('b.txt'); rm('empty')")
        self.assertNotIn(path, [f['path'] for f in await self.agent.files()])
        with self.assertRaises(IsADirectoryError):
            await self.agent.read_file(self.agent.home)
        with self.assertRaises(PermissionError):
            self.agent.storage.read('../other/file')

    async def test_storage_root_can_move_without_rewriting_database_keys(self):
        await self.agent.execute("write('a','portable'); x=7")
        records = await self.agent.files()
        old_root = self.agent.storage.root
        await self.agent.close()
        new_root = old_root.with_name('relocated')
        shutil.move(old_root, new_root)
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id, storage=DirectoryStorage(new_root))
        self.assertEqual(await self.agent.files(), records)
        self.assertEqual(await self.agent.read_file(self.agent.home + '/a'), b'portable')
        self.assertEqual(await self.agent.execute("(x, read('a'))"), "(7, '1: portable')")

    async def test_replaceable_object_storage_backend(self):
        storage = ObjectStorage()
        path = self.tmp.name + '/objects.db'
        agent = await CodeAgent.open(path, storage=storage)
        try:
            await agent.execute("write('a','object'); x=9")
            record = next(f for f in await agent.files() if f['path'] == agent.home + '/a')
            self.assertTrue(record['storage_key'].startswith(agent.name + '/'))
            self.assertEqual(await agent.read_file(agent.home + '/a'), b'object')
            session_id = agent.id
        finally:
            await agent.close()
        agent = await CodeAgent.open(path, session_id=session_id, storage=storage)
        try:
            self.assertEqual(await agent.execute("(x, read('a'))"), "(9, '1: object')")
            before = await agent.files()
            with patch.object(agent, '_commit', side_effect=OSError('failed after object upload')):
                with self.assertRaises(OSError):
                    await agent.execute("write('a','uncommitted')")
            self.assertEqual(await agent.files(), before)
            self.assertEqual(await agent.read_file(agent.home + '/a'), b'object')
            await agent.execute()
            self.assertEqual(await agent.read_file(agent.home + '/a'), b'uncommitted')
            with self.assertRaises(Exception):
                await agent.execute("write('a','bad'); raise ValueError('rollback')")
            self.assertEqual(await agent.read_file(agent.home + '/a'), b'uncommitted')
        finally:
            await agent.close()

    async def test_restart_repairs_workspace_without_reexecuting_code(self):
        await self.agent.execute("write('version.txt','new'); mkdir('empty'); n=2")
        state, files, workspace = await self.agent.state(), await self.agent.files(), self.agent.workspace
        await self.agent.close()
        shutil.rmtree(workspace)
        with patch.object(CodeAgent, '_evaluate', side_effect=AssertionError('recovery executed code')):
            self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.state(), state)
        self.assertEqual(await self.agent.files(), files)
        self.assertEqual((workspace / self.agent.home.lstrip('/') / 'version.txt').read_text(), 'new')
        self.assertEqual(await self.agent.execute('n'), '2')
        tables = {r[0] for r in self.agent.db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        self.assertEqual(tables, {'sessions', 'files', 'executions'})

    async def test_execution_ids_are_idempotent(self):
        await self.agent.execute('n=0')
        self.assertEqual(await self.agent.execute('n += 1; n', execution_id='increment'), '1')
        self.assertEqual(await self.agent.execute('n += 1; n', execution_id='increment'), '1')
        with self.assertRaises(ValueError):
            await self.agent.execute('n += 2; n', execution_id='increment')
        self.assertEqual(await self.agent.execute('n'), '1')

    async def test_cannot_append_messages_while_execution_is_pending(self):
        with patch.object(self.agent, '_evaluate', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                await self.agent.execute('n=1')
        with self.assertRaises(RuntimeError):
            await self.agent.append_message({'role': 'user', 'content': 'too early'})
        self.assertEqual((await self.agent.state()).messages, [])
        await self.agent.execute()
        self.assertEqual(await self.agent.execute('n'), '1')

    async def test_repl_reopens_and_automatically_continues_a_pending_turn(self):
        await self.agent.append_message({'role': 'user', 'content': 'increment'})
        code = 'n=1; n'
        await self.agent.append_message({'role': 'assistant', 'tool_calls': [{
            'id': 'resume-tool', 'type': 'function', 'function': {'name': 'python',
            'arguments': json.dumps({'code': code})}}]})
        with patch.object(self.agent, '_evaluate', side_effect=OSError('interrupted')):
            with self.assertRaises(OSError):
                await self.agent.execute(code, tool_id='resume-tool')
        await self.agent.close()
        client = httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
            stream_response({'role': 'assistant', 'content': 'Continued automatically.'})))
        output = io.StringIO()
        with patch('sys.argv', ['main.py', '--database', self.path, '--session', str(self.session_id)]), \
             patch('builtins.input', side_effect=['/quit']), \
             patch('agents.AsyncOpenAI', return_value=model_client(client)), redirect_stdout(output):
            await repl.main()
        self.assertIn('Continued automatically.', output.getvalue())
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertIsNone((await self.agent.state()).pending)
        self.assertEqual(await self.agent.execute('n'), '1')
        self.assertEqual([m['content'] for m in (await self.agent.state()).messages if m['role'] == 'tool'], ['1'])

    async def test_old_schema_is_rejected_without_deleting_saved_data(self):
        path = self.tmp.name + '/old.db'
        with sqlite3.connect(path) as db:
            db.execute('CREATE TABLE sessions(id INTEGER PRIMARY KEY,name TEXT,last_event_id INTEGER)')
            db.execute("INSERT INTO sessions VALUES (7,'saved',42)")
        with self.assertRaisesRegex(RuntimeError, 'unsupported database schema'):
            await CodeAgent.open(path, workspaces=self.tmp.name + '/old-workspaces')
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute('SELECT * FROM sessions').fetchall(), [(7, 'saved', 42)])

    async def test_recorded_failure_is_not_executed_again(self):
        await self.agent.execute("n=0; write('a','before')")
        command = "n=1; write('a','bad'); raise ValueError('failure')"
        with self.assertRaises(ExecutionError):
            await self.agent.execute(command, execution_id='failed-command')
        with patch.object(self.agent, '_evaluate', side_effect=AssertionError('retried failed code')):
            with self.assertRaises(ExecutionError):
                await self.agent.execute(command, execution_id='failed-command')
        self.assertIsNone((await self.agent.state()).pending)
        self.assertEqual(await self.agent.read_file(self.agent.home + '/a'), b'before')
        self.assertEqual(await self.agent.execute('n'), '0')
        self.assertEqual(self.agent.db.execute(
            "SELECT error FROM executions WHERE id='failed-command'").fetchone()[0], 1)

    async def test_abrupt_process_exit_during_session_creation(self):
        script = '''
import os, sys, asyncio
from agents import CodeAgent
replace = CodeAgent._replace_files
def crash_replace(agent, files):
    replace(agent, files)
    os._exit(92)
CodeAgent._replace_files = crash_replace
asyncio.run(CodeAgent.open(sys.argv[1], workspaces=sys.argv[2]))
'''
        path, root = self.tmp.name + '/creation.db', self.tmp.name + '/creation-workspaces'
        result = await asyncio.to_thread(subprocess.run, [sys.executable, '-c', script, path, root],
                                cwd=os.path.dirname(__file__), timeout=30,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.assertEqual(result.returncode, 92)
        agent = await CodeAgent.open(path, workspaces=root)
        try:
            self.assertEqual(agent.id, 0)
            self.assertEqual(await agent.execute('pwd()'), repr(agent.home))
            self.assertEqual(agent.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0], 1)
        finally:
            await agent.close()

    async def test_abrupt_process_exit_at_commit_boundaries(self):
        script = '''
import os, sys, asyncio
from agents import CodeAgent
async def main():
    agent = await CodeAgent.open(sys.argv[1], session_id=int(sys.argv[2]))
    phase = sys.argv[3]
    if phase == 'requested':
        agent._evaluate = lambda *args: os._exit(91)
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
        replace = agent._replace_files
        def crash_replace(files):
            replace(files)
            os._exit(91)
        agent._replace_files = crash_replace
    await agent.execute("n += 1; write('counter.txt', str(n)); n", tool_id='counter-tool', execution_id='crash-test')
    os._exit(91)
asyncio.run(main())
'''
        for phase in ('requested', 'published', 'projected', 'checkout', 'committed'):
            with self.subTest(phase=phase):
                agent = await CodeAgent.open(self.path)
                await agent.execute("n=0; write('counter.txt','0')")
                await agent.append_message({'role': 'user', 'content': 'increment the counter'})
                await agent.append_message({'role': 'assistant', 'tool_calls': [{
                    'id': 'counter-tool', 'type': 'function', 'function': {'name': 'python',
                    'arguments': json.dumps({'code': "n += 1; write('counter.txt', str(n)); n"})}}]})
                session_id = agent.id
                await agent.close()
                result = await asyncio.to_thread(subprocess.run, [sys.executable, '-c', script, self.path, str(session_id), phase],
                                        cwd=os.path.dirname(__file__), timeout=30,
                                        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                self.assertEqual(result.returncode, 91)
                agent = await CodeAgent.open(self.path, session_id=session_id)
                try:
                    self.assertEqual(await agent.execute(execution_id='crash-test'), '1')
                    self.assertEqual(await agent.execute(execution_id='crash-test'), '1')
                    self.assertIsNone((await agent.state()).pending)
                    self.assertEqual([m['content'] for m in (await agent.state()).messages if m['role'] == 'tool'], ['1'])
                    self.assertEqual(await agent.read_file(agent.home + '/counter.txt'), b'1')
                    self.assertEqual(agent.db.execute(
                        "SELECT output,error FROM executions WHERE session_id=? AND id='crash-test'",
                        (session_id,)).fetchone(), ('1', 0))
                    self.assertEqual(await agent.execute('n'), '1')
                finally:
                    await agent.close()
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
