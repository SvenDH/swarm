"""Whole-block deadlines include host calls and preserve durable checkpoints."""
import asyncio
import json
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import httpx
from pydantic_monty import AsyncMonty

from agents import ExecutionBudget, ExecutionError, Runtime, Worker, WorkerClient
from test_agents import model_client, stream_response
from wrappers.protocol import Bridge


class ExecutionTimeoutTests(unittest.IsolatedAsyncioTestCase):
    async def test_repeated_cancellation_also_drains_preparation_io(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder + '/agents.db') as runtime:
                a, b = await runtime.agent(), await runtime.agent()
                entered, release = threading.Event(), threading.Event()
                original = a._evaluate
                def blocked():
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('test did not release preparation')
                async def prepare(*args):
                    await runtime.offload(blocked)
                    return await original(*args)
                with patch.object(a, '_evaluate', prepare):
                    running = asyncio.create_task(a.execute('42'))
                    queued = None
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                        running.cancel()
                        await asyncio.sleep(.03)
                        running.cancel()
                        queued = asyncio.create_task(b.execute('42'))
                        await asyncio.sleep(.03)
                        self.assertFalse(running.done())
                        self.assertFalse(queued.done())
                    finally:
                        release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(running, 5)
                    self.assertEqual(await queued, '42')

    async def test_parent_watchdog_timeout_is_not_treated_as_retryable_crash(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder + '/agents.db', execution_timeout=5) as runtime:
                a = await runtime.agent()
                async with AsyncMonty(min_processes=1, max_processes=1, request_timeout=.1) as pool:
                    original = runtime._pool
                    runtime._pool = pool
                    try:
                        with self.assertRaisesRegex(ExecutionError, 'TimeoutError'):
                            await a.execute('while True: pass')
                        self.assertIsNone((await a.state()).pending)
                    finally:
                        runtime._pool = original
                self.assertEqual(await a.execute('42'), '42')

    async def test_overdue_completion_cannot_commit_when_event_loop_timer_was_delayed(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder + '/agents.db', execution_timeout=.2) as runtime:
                a = await runtime.agent()
                original = a._evaluate
                async def delayed(*args):
                    result = await original(*args)
                    time.sleep(.3)  # Simulate a native conversion delaying timer delivery.
                    return result
                with patch.object(a, '_evaluate', delayed):
                    with self.assertRaisesRegex(ExecutionError, 'TimeoutError'):
                        await a.execute('value=99')
                self.assertIsNone((await a.state()).pending)
                self.assertEqual(await a.execute("'value' in locals()"), 'False')

    async def test_worker_records_timeout_as_tool_result_without_replaying_code(self):
        with tempfile.TemporaryDirectory() as folder:
            responses = []
            def respond(request):
                history = json.loads(request.content)['messages']
                responses.append(history)
                if len(responses) == 1:
                    return stream_response({'role': 'assistant', 'tool_calls': [{
                        'id': 'slow', 'type': 'function', 'function': {'name': 'python',
                        'arguments': json.dumps({'code': "while True: _library_rpc('import','os')"})}}]})
                self.assertEqual(history[-1]['role'], 'tool')
                self.assertIn('TimeoutError', history[-1]['content'])
                return stream_response({'role': 'assistant', 'content': 'Use smaller blocks.'})
            client = model_client(httpx.AsyncClient(transport=httpx.MockTransport(respond)))
            original = Bridge.__call__
            def slow(bridge, *args, **kwargs):
                time.sleep(.04)
                return original(bridge, *args, **kwargs)
            async with Worker(folder + '/agents.db', client=client, execution_timeout=.25, retry_delay=.01) as worker:
                remote = WorkerClient(folder + '/agents.db')
                info = await remote.request('agent')
                with patch.object(Bridge, '__call__', slow):
                    self.assertEqual(await asyncio.wait_for(remote.request(
                        'run', session_id=info['id'], prompt='work'), 5), 'Use smaller blocks.')
                self.assertEqual(len(responses), 2)
                rows = await worker.runtime.database_call(lambda: worker.runtime.db.execute(
                    'SELECT output,error FROM executions').fetchall())
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0][1], 1)
                self.assertIn('TimeoutError', rows[0][0])

    async def test_pure_python_timeout_discards_worker_and_allows_next_block(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder + '/agents.db', execution_timeout=.2) as runtime:
                a = await runtime.agent()
                with self.assertRaisesRegex(ExecutionError, 'TimeoutError'):
                    await asyncio.wait_for(a.execute('while True: pass'), 5)
                self.assertIsNone((await a.state()).pending)
                self.assertEqual(await a.execute('42'), '42')

    async def test_repeated_host_calls_timeout_once_and_rollback_after_restart(self):
        with tempfile.TemporaryDirectory() as folder:
            database = folder + '/agents.db'
            async with Runtime(database) as runtime:
                agent = await runtime.agent()
                await agent.execute("value=7; open('saved','w').write('before')")
                runtime.execution_timeout = .25
                observed, calls = [], []
                agent.on_output = lambda *event: observed.append(event)
                original = Bridge.__call__
                def slow(bridge, *args, **kwargs):
                    calls.append(args)
                    time.sleep(.04)
                    return original(bridge, *args, **kwargs)
                with patch.object(Bridge, '__call__', slow), patch.object(ExecutionBudget, 'progress_interval', .03):
                    with self.assertRaisesRegex(ExecutionError, 'TimeoutError'):
                        await asyncio.wait_for(agent.execute('''
value=99
open('saved','w').write('after')
for i in range(100):
    _library_rpc('import','os')
''', execution_id='slow'), 5)
                self.assertGreater(len(calls), 0)
                self.assertLess(len(calls), 100)
                self.assertTrue(any(kind == 'status' and 'Python running' in text for kind, text in observed))
                self.assertIsNone((await agent.state()).pending)
                session_id = agent.id
            async with Runtime(database) as runtime:
                agent = await runtime.agent(session_id=session_id)
                with patch.object(Bridge, '__call__', side_effect=AssertionError('must not replay')):
                    with self.assertRaisesRegex(ExecutionError, 'TimeoutError'):
                        await agent.execute(execution_id='slow')
                self.assertEqual(await agent.execute("(value,open('saved').read())"), "(7, 'before')")

    async def test_native_callback_drains_before_next_agent_runs(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder + '/agents.db', execution_timeout=.2) as runtime:
                a, b = await runtime.agent(), await runtime.agent()
                entered, release, finished = threading.Event(), threading.Event(), threading.Event()
                original = Bridge.__call__
                def blocked(bridge, *args, **kwargs):
                    entered.set()
                    try:
                        if not release.wait(5):
                            raise RuntimeError('test did not release callback')
                        return original(bridge, *args, **kwargs)
                    finally:
                        finished.set()
                observed = []
                b.on_output = lambda *event: observed.append(event)
                with patch.object(Bridge, '__call__', blocked):
                    running = asyncio.create_task(a.execute("_library_rpc('import','os')"))
                    queued = None
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                        queued = asyncio.create_task(b.execute('42'))
                        await asyncio.sleep(.3)
                        self.assertFalse(running.done())
                        self.assertFalse(queued.done())
                        self.assertFalse(finished.is_set())
                    finally:
                        release.set()
                    with self.assertRaisesRegex(ExecutionError, 'TimeoutError'):
                        await asyncio.wait_for(running, 5)
                    self.assertEqual(await asyncio.wait_for(queued, 5), '42')
                    self.assertTrue(finished.is_set())
                self.assertTrue(any(kind == 'status' and 'Waiting' in text for kind, text in observed))

    async def test_cancellation_still_leaves_resumable_work(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder + '/agents.db') as runtime:
                a, b = await runtime.agent(), await runtime.agent()
                entered, release = threading.Event(), threading.Event()
                original = Bridge.__call__
                def blocked(bridge, *args, **kwargs):
                    entered.set()
                    if not release.wait(5):
                        raise RuntimeError('test did not release callback')
                    return original(bridge, *args, **kwargs)
                with patch.object(Bridge, '__call__', blocked):
                    running = asyncio.create_task(a.execute("_library_rpc('import','os')", execution_id='paused'))
                    queued = None
                    try:
                        self.assertTrue(await asyncio.to_thread(entered.wait, 3))
                        running.cancel()
                        await asyncio.sleep(.05)
                        running.cancel()  # Repeated cancellation must not bypass draining.
                        queued = asyncio.create_task(b.execute('42'))
                        await asyncio.sleep(.05)
                        self.assertFalse(running.done())
                        self.assertFalse(queued.done())
                    finally:
                        release.set()
                    with self.assertRaises(asyncio.CancelledError):
                        await asyncio.wait_for(running, 5)
                    self.assertEqual(await queued, '42')
                self.assertIsNotNone((await a.state()).pending)
                await a.execute(execution_id='paused')
                self.assertIsNone((await a.state()).pending)

    def test_timeout_validation(self):
        for value in (0, -1, float('inf'), float('nan'), True, '60'):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'execution_timeout'):
                Runtime(execution_timeout=value)
