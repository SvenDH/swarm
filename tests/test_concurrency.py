"""Concurrency, cancellation and scale tests use real Monty and durable SQLite."""
import asyncio
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
from openai import AsyncOpenAI

from agents import Runtime
from test_agents import stream_response


class ConcurrencyTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.runtime = await Runtime(self.tmp.name + '/agent.db', workers=2).__aenter__()

    async def asyncTearDown(self):
        await self.runtime.close()
        self.tmp.cleanup()

    async def test_slow_library_does_not_block_other_agents_or_event_loop(self):
        a, b = await asyncio.gather(self.runtime.agent(), self.runtime.agent())
        started, release = threading.Event(), threading.Event()
        loop_thread = threading.get_ident()
        observed = []
        a.on_output = lambda kind, text: observed.append(threading.get_ident())

        def slow_numpy(fs, *args, **kwargs):
            self.assertNotEqual(threading.get_ident(), loop_thread)
            started.set()
            if not release.wait(5):
                raise TimeoutError('test did not release callback')
            return 7

        with patch('agents.library_tools', lambda fs: {'numpy': lambda *args: slow_numpy(fs, *args)}):
            running = asyncio.create_task(a.execute("print('start'); numpy('sum', [1])"))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                self.assertEqual(await asyncio.wait_for(b.execute('21 * 2'), 2), '42')
                self.assertFalse(running.done())
            finally:
                release.set()
            self.assertEqual(await running, 'start\n7')
        self.assertTrue(observed)
        self.assertEqual(set(observed), {loop_thread})

    async def test_duplicate_handles_serialize_and_execution_ids_deduplicate(self):
        a = await self.runtime.agent()
        await a.execute('n=0')
        b = await self.runtime.agent(session_id=a.id)
        self.assertIs(a._lock, b._lock)
        results = await asyncio.gather(*[(a if i % 2 else b).execute('n += 1; n') for i in range(30)])
        self.assertEqual(set(results), {str(i) for i in range(1, 31)})
        results = await asyncio.gather(*[a.execute('n += 1; n', execution_id='once') for _ in range(20)])
        self.assertEqual(set(results), {'31'})
        self.assertEqual(await b.execute('n'), '31')

    async def test_cancel_during_publication_drains_worker_and_leaves_resumable_request(self):
        a = await self.runtime.agent()
        await a.execute("n=0; open('a','w').write('before')")
        started, release = threading.Event(), threading.Event()
        put = self.runtime.storage.put

        def slow_put(*args):
            started.set()
            if not release.wait(5):
                raise TimeoutError('test did not release storage')
            return put(*args)

        with patch.object(self.runtime.storage, 'put', slow_put):
            running = asyncio.create_task(a.execute("n += 1; open('a','w').write(str(n)); n", execution_id='cancelled'))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                running.cancel()
                await asyncio.sleep(.02)
                running.cancel()  # Repeated cancellation must not abandon the worker.
                self.assertFalse(running.done())
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await running
        self.assertIsNotNone((await a.state()).pending)
        self.assertEqual(await a.read_file(a.home + '/a'), b'before')
        self.assertEqual(await a.execute(execution_id='cancelled'), '1')
        self.assertEqual(await a.execute(execution_id='cancelled'), '1')
        self.assertEqual(await a.execute('n'), '1')

    async def test_cancel_during_sandbox_execution_is_recoverable(self):
        a = await self.runtime.agent()
        await a.execute('n=0')
        started, release = threading.Event(), threading.Event()

        def slow_call(*args):
            started.set()
            if not release.wait(5):
                raise TimeoutError('test did not release callback')
            return 1

        with patch('agents.library_tools', lambda fs: {'numpy': slow_call}):
            running = asyncio.create_task(a.execute("n += 1; numpy('sum', [1]); n", execution_id='cancel-sandbox'))
            try:
                self.assertTrue(await asyncio.to_thread(started.wait, 5))
                running.cancel()
            finally:
                release.set()
            with self.assertRaises(asyncio.CancelledError):
                await running
        self.assertIsNotNone((await a.state()).pending)
        self.assertEqual(await a.execute(execution_id='cancel-sandbox'), '1')
        self.assertEqual(await a.execute('n'), '1')

    async def test_model_requests_overlap_but_are_bounded(self):
        active, peak = 0, 0

        async def respond(request):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(.02)
            active -= 1
            return stream_response({'role': 'assistant', 'content': 'hello'})

        async with AsyncOpenAI(api_key='test', base_url='http://test/v1', max_retries=0,
                http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond))) as client:
            await self.runtime.client.close()
            self.runtime.client = client
            self.runtime._model_slots = asyncio.Semaphore(3)
            agents = await asyncio.gather(*[self.runtime.agent() for _ in range(20)])
            self.assertEqual(await asyncio.gather(*[a.run('hi') for a in agents]), ['hello'] * 20)
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, 3)

    async def test_runtime_shutdown_drains_operations_and_rejects_new_work(self):
        a = await self.runtime.agent()
        entered, release = asyncio.Event(), asyncio.Event()
        evaluate = a._evaluate

        async def paused(*args):
            entered.set()
            await release.wait()
            return await evaluate(*args)

        with patch.object(a, '_evaluate', paused):
            running = asyncio.create_task(a.execute('42'))
            await entered.wait()
            closing = asyncio.create_task(self.runtime.close())
            await asyncio.sleep(.02)
            self.assertFalse(closing.done())
            with self.assertRaises(RuntimeError):
                await self.runtime.agent()
            release.set()
            self.assertEqual(await running, '42')
            await closing
        await self.runtime.close()

    async def test_thousand_agents_share_resources_and_bounded_execution(self):
        # Real sessions and sandbox evaluations, not just a thousand coroutines.
        agents = await asyncio.gather(*[self.runtime.agent() for _ in range(1000)])
        self.assertEqual(len({a.id for a in agents}), 1000)
        self.assertEqual(len({id(a.db) for a in agents}), 1)
        self.assertEqual(len({id(a.runtime.client) for a in agents}), 1)
        self.assertTrue(all('_evaluation' not in a.__dict__ for a in agents))
        active, peak = 0, 0
        evaluate = agents[0].__class__._evaluate

        async def counted(agent, *args):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            try:
                return await evaluate(agent, *args)
            finally:
                active -= 1

        with patch('agents.CodeAgent._evaluate', counted):
            results = await asyncio.gather(*[a.execute(f"n={a.id}; open('id','w').write(str(n)); n") for a in agents])
        self.assertEqual(results, [str(a.id) for a in agents])
        self.assertGreater(peak, 1)
        self.assertLessEqual(peak, self.runtime.workers)
        self.assertTrue(all('_evaluation' not in a.__dict__ for a in agents))
        for a in agents[::100]:
            reopened = await self.runtime.agent(session_id=a.id)
            self.assertEqual(await reopened.execute('n'), str(a.id))
            self.assertEqual(await reopened.read_file(a.home + '/id'), str(a.id).encode())

