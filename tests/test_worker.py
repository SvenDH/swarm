"""Worker ownership, durable inbox recovery and detachable local clients."""
import asyncio
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agents import CodeAgent, Runtime, Worker, WorkerClient
from test_agents import model_client, stream_response


class WorkerTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name + '/agents.db'
        self.requests = []

    async def asyncTearDown(self):
        self.tmp.cleanup()

    def client(self, respond=None):
        def default(request):
            history = json.loads(request.content)['messages']
            self.requests.append(history)
            return stream_response({'role': 'assistant', 'reasoning_content': 'Checking the request.',
                                    'content': 'done: ' + history[-1].get('content', '')})
        return model_client(httpx.AsyncClient(transport=httpx.MockTransport(respond or default)))

    async def result(self, worker, job_id):
        async def wait():
            while True:
                row = await worker.runtime.database_call(lambda: worker.runtime.db.execute(
                    'SELECT result FROM prompts WHERE id=?', (job_id,)).fetchone())
                if row and row[0] is not None:
                    return row[0]
                await asyncio.sleep(0.01)
        return await asyncio.wait_for(wait(), 10)

    async def test_socket_client_and_multiple_queued_prompts(self):
        observed = []
        async with Worker(self.path, client=self.client()) as worker:
            client = WorkerClient(self.path, on_output=lambda *event: observed.append(event))
            info = await client.request('agent')
            self.assertEqual(await client.request('sessions'), [info])
            first = await worker.submit(info['id'], 'first')
            second = await worker.submit(info['id'], 'second')
            self.assertEqual(await self.result(worker, first), 'done: first')
            self.assertEqual(await self.result(worker, second), 'done: second')
            self.assertEqual(await client.request('run', session_id=info['id'], prompt='third'), 'done: third')
            self.assertIn(('assistant', 'done: third'), observed)
            self.assertIn(('reasoning', 'Checking the request.'), observed)
            agent = await worker.runtime.agent(session_id=info['id'])
            self.assertEqual([m['content'] for m in (await agent.state()).messages if m['role'] == 'user'],
                             ['first', 'second', 'third'])
            await agent.close()

    async def test_disconnecting_client_does_not_cancel_agent(self):
        entered, release = asyncio.Event(), asyncio.Event()
        async def respond(request):
            entered.set()
            await release.wait()
            return stream_response({'role': 'assistant', 'content': 'finished independently'})
        async with Worker(self.path, client=self.client(respond)) as worker:
            client = WorkerClient(self.path)
            info = await client.request('agent')
            request = asyncio.create_task(client.request('run', session_id=info['id'], prompt='work'))
            await asyncio.wait_for(entered.wait(), 5)
            # Reading metadata must not wait for the session's execution lock.
            self.assertEqual(await asyncio.wait_for(client.request('agent', session_id=info['id']), 1), info)
            request.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await request
            job = await worker.runtime.database_call(worker._next, info['id'])
            release.set()
            self.assertEqual(await self.result(worker, job[0]), 'finished independently')

    async def test_all_interrupted_sessions_resume_without_a_repl(self):
        async with Runtime(self.path) as runtime:
            one, two = await runtime.agent(), await runtime.agent()
            await one.append_message({'role': 'user', 'content': 'first interrupted'})
            await two.append_message({'role': 'user', 'content': 'second interrupted'})
            await two.append_message({'role': 'assistant', 'tool_calls': [{'id': 'pending', 'type': 'function',
                'function': {'name': 'python', 'arguments': json.dumps({'code': 'n=7; n'})}}]})
            with patch.object(two, '_evaluate', side_effect=OSError('crash')):
                with self.assertRaises(OSError):
                    await two.execute('n=7; n', tool_id='pending')
            ids = one.id, two.id
            await one.close()
            await two.close()
        async with Worker(self.path, client=self.client()) as worker:
            for session_id in ids:
                async def wait_for_job():
                    while True:
                        row = await worker.runtime.database_call(lambda: worker.runtime.db.execute(
                            'SELECT id FROM prompts WHERE session_id=?', (session_id,)).fetchone())
                        if row:
                            return row[0]
                        await asyncio.sleep(0.01)
                await self.result(worker, await asyncio.wait_for(wait_for_job(), 5))
            agent = await worker.runtime.agent(session_id=ids[1])
            self.assertEqual(await agent.execute('n'), '7')
            self.assertEqual(len([m for m in (await agent.state()).messages if m['role'] == 'tool']), 1)
            await agent.close()
        self.assertEqual(len(self.requests), 2)

    async def test_running_prompt_recovers_after_worker_restart(self):
        entered = asyncio.Event()
        async def interrupted(request):
            entered.set()
            await asyncio.Event().wait()
        async with Worker(self.path, client=self.client(interrupted)) as worker:
            info = await WorkerClient(self.path).request('agent')
            job = await worker.submit(info['id'], 'only once')
            await asyncio.wait_for(entered.wait(), 5)
        async with Worker(self.path, client=self.client()) as worker:
            self.assertEqual(await self.result(worker, job), 'done: only once')
            agent = await worker.runtime.agent(session_id=info['id'])
            self.assertEqual([m['content'] for m in (await agent.state()).messages if m['role'] == 'user'], ['only once'])
            await agent.close()

    async def test_completed_answer_before_inbox_commit_is_not_submitted_again(self):
        agent = await CodeAgent.open(self.path)
        await agent.append_message({'role': 'user', 'content': 'once'})
        await agent.append_message({'role': 'assistant', 'content': 'already complete'})
        job = await agent.runtime.database_call(lambda: agent.db.execute(
            'INSERT INTO prompts(session_id,prompt,started) VALUES (?,?,1)', (agent.id, 'once')).lastrowid)
        await agent.close()
        async with Worker(self.path, client=self.client()) as worker:
            self.assertEqual(await self.result(worker, job), 'already complete')
        self.assertEqual(self.requests, [])

    async def test_one_failing_session_does_not_block_others_and_retries(self):
        attempts = 0
        def respond(request):
            nonlocal attempts
            prompt = json.loads(request.content)['messages'][-1]['content']
            if prompt == 'retry':
                attempts += 1
                if attempts == 1:
                    return httpx.Response(503, json={'error': {'message': 'temporary'}})
            return stream_response({'role': 'assistant', 'content': 'ok'})
        async with Worker(self.path, client=self.client(respond), retry_delay=0.01) as worker:
            client = WorkerClient(self.path)
            one, two = await client.request('agent'), await client.request('agent')
            first, second = await worker.submit(one['id'], 'retry'), await worker.submit(two['id'], 'other')
            self.assertEqual(await self.result(worker, second), 'ok')
            self.assertEqual(await self.result(worker, first), 'ok')
        self.assertEqual(attempts, 2)

    async def test_second_worker_cannot_take_database_or_remove_socket(self):
        async with Worker(self.path, client=self.client()):
            with self.assertRaisesRegex(RuntimeError, 'already owns'):
                async with Worker(self.path):
                    self.fail('second worker started')
            self.assertEqual(await WorkerClient(self.path).request('sessions'), [])

    async def test_sigkill_leaves_durable_prompt_and_stale_socket_recoverable(self):
        script = '''
import asyncio, sys, httpx
from openai import AsyncOpenAI
from agents import Worker
async def respond(request):
    print('READY', flush=True)
    await asyncio.Event().wait()
async def main():
    client = AsyncOpenAI(api_key='test', base_url='http://test/v1', max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)))
    async with Worker(sys.argv[1], client=client) as worker:
        agent = await worker.runtime.agent()
        await worker.submit(agent.id, 'survive process death')
        await worker.serve()
asyncio.run(main())
'''
        process = await asyncio.create_subprocess_exec(sys.executable, '-u', '-c', script, self.path,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            self.assertEqual(await asyncio.wait_for(process.stdout.readline(), 15), b'READY\n')
            process.kill()
            await asyncio.wait_for(process.wait(), 10)
            async with Worker(self.path, client=self.client()) as worker:
                self.assertEqual(await self.result(worker, 1), 'done: survive process death')
                agent = await worker.runtime.agent(session_id=0)
                self.assertEqual(len([m for m in (await agent.state()).messages if m['role'] == 'user']), 1)
                await agent.close()
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def test_step_budget_is_a_yield_not_an_abandoned_turn(self):
        responses = iter([
            {'role': 'assistant', 'tool_calls': [{'id': 'increment', 'type': 'function',
             'function': {'name': 'python', 'arguments': '{"code":"n=1; n"}'}}]},
            {'role': 'assistant', 'content': 'finished'},
        ])
        original = CodeAgent.run
        async def one_step(agent):
            return await original(agent, max_steps=1)
        async with Worker(self.path, client=self.client(lambda request: stream_response(next(responses)))) as worker:
            with patch.object(CodeAgent, 'run', one_step):
                info = await WorkerClient(self.path).request('agent')
                job = await worker.submit(info['id'], 'work')
                self.assertEqual(await self.result(worker, job), 'finished')

    async def test_dispatch_does_not_rescan_inbox_or_load_history_twice(self):
        async with Worker(self.path, client=self.client()) as worker:
            info = await WorkerClient(self.path).request('agent')
            queries = []
            await worker.runtime.database_call(worker.runtime.db.set_trace_callback, queries.append)
            for index in range(10):
                await self.result(worker, await worker.submit(info['id'], str(index)))
            self.assertFalse(any('SELECT DISTINCT session_id' in sql for sql in queries))
            self.assertEqual(sum('SELECT messages FROM sessions' in sql for sql in queries), 10)

    async def test_prompt_submitted_as_driver_exits_is_not_lost(self):
        idle, release = asyncio.Event(), asyncio.Event()
        async with Worker(self.path, client=self.client()) as worker:
            info = await WorkerClient(self.path).request('agent')
            original = worker.runtime.database_call
            async def database_call(function, *args, **kwargs):
                result = await original(function, *args, **kwargs)
                if function == worker._next and result is None and not idle.is_set():
                    idle.set()
                    await release.wait()
                return result
            with patch.object(worker.runtime, 'database_call', database_call):
                await self.result(worker, await worker.submit(info['id'], 'first'))
                await asyncio.wait_for(idle.wait(), 5)
                second = await worker.submit(info['id'], 'racing prompt')
                release.set()
                self.assertEqual(await self.result(worker, second), 'done: racing prompt')
