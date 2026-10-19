"""Automatic compaction through real SDK streams with a mocked provider."""
import asyncio
import json
import tempfile
import unittest
from unittest.mock import patch

import httpx

from agents import CodeAgent
from test_agents import model_client, stream_response


class CompactionTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name + '/agent.db'
        self.agent = await CodeAgent.open(self.path, context_chars=4096)
        self.session_id = self.agent.id
        self.requests = []

    async def asyncTearDown(self):
        await self.agent.close()
        self.tmp.cleanup()

    async def seed(self):
        await self.agent.execute("counter=1; open('saved.txt','w').write('saved')")
        await self.agent.append_message({'role': 'user', 'content': 'old task ' + 'x' * 6000})
        await self.agent.append_message({'role': 'assistant', 'content': 'Old task complete.'})

    def respond(self, request):
        body = json.loads(request.content)
        self.requests.append(body)
        return stream_response({'role': 'assistant', 'content':
            'Old task complete. counter=1; saved.txt contains saved.' if 'tools' not in body else 'continued'})

    async def test_automatic_compaction_is_durable_and_keeps_interpreter_and_files(self):
        await self.seed()
        observed = []
        self.agent.on_output = lambda kind, text: observed.append((kind, text))
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run('next task'), 'continued')
        self.assertEqual(len(self.requests), 2)
        self.assertNotIn('tools', self.requests[0])
        self.assertIn('max_completion_tokens', self.requests[0])
        self.assertEqual(self.requests[1]['messages'][-1]['content'], 'next task')
        self.assertTrue(any(kind == 'compaction' for kind, _ in observed))
        history = (await self.agent.state()).messages
        self.assertEqual(len(history), 3)
        self.assertIn('Session continuation memory', history[0]['content'])
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id, context_chars=4096)
        self.assertEqual((await self.agent.state()).messages, history)
        self.assertEqual(await self.agent.execute("(counter,open('saved.txt').read())"), "(1, 'saved')")

    async def test_pending_tools_finish_before_compacting_without_splitting_batch(self):
        await self.seed()
        await self.agent.append_message({'role': 'user', 'content': 'increment twice'})
        calls = [{'id': str(i), 'type': 'function', 'function': {
            'name': 'python', 'arguments': json.dumps({'code': 'counter += 1'})}} for i in range(2)]
        await self.agent.append_message({'role': 'assistant', 'tool_calls': calls})
        await self.agent.execute('counter += 1', tool_id='0')
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run(), 'continued')
        sent = self.requests[-1]['messages']
        tool_calls = [c['id'] for m in sent for c in m.get('tool_calls', [])]
        results = [m['tool_call_id'] for m in sent if m['role'] == 'tool']
        self.assertEqual(tool_calls, ['0', '1'])
        self.assertEqual(results, tool_calls)
        self.assertEqual(await self.agent.execute('counter'), '3')

    async def test_failed_or_cancelled_summary_keeps_original_history_and_retries(self):
        for cancel in (False, True):
            with self.subTest(cancel=cancel):
                await self.seed()
                await self.agent.append_message({'role': 'user', 'content': 'continue'})
                before = (await self.agent.state()).messages
                async def fail(request):
                    if cancel:
                        raise asyncio.CancelledError()
                    return httpx.Response(503, json={'error': {'message': 'unavailable'}})
                async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
                    with patch.object(self.agent.runtime, 'client', model_client(client)):
                        with self.assertRaises(asyncio.CancelledError if cancel else Exception):
                            await self.agent.run()
                self.assertEqual((await self.agent.state()).messages, before)
                await self.agent.close()
                self.agent = await CodeAgent.open(self.path, session_id=self.session_id, context_chars=4096)
                async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
                    with patch.object(self.agent.runtime, 'client', model_client(client)):
                        self.assertEqual(await self.agent.run(), 'continued')

    async def test_small_history_and_disabled_compaction_make_no_summary_requests(self):
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                await self.agent.run('hi')
                self.agent.runtime.context_chars = None
                await self.agent.run('x' * 6000)
        self.assertEqual(len(self.requests), 2)
        self.assertTrue(all('tools' in body for body in self.requests))

    async def test_restart_after_summary_commit_does_not_summarize_again(self):
        await self.seed()
        async def respond(request):
            if 'tools' in json.loads(request.content):
                return httpx.Response(503, json={'error': {'message': 'interrupted after commit'}})
            return self.respond(request)
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                with self.assertRaises(Exception):
                    await self.agent.run('continue')
        saved = (await self.agent.state()).messages
        self.assertEqual(len(saved), 2)
        self.assertIn('Session continuation memory', saved[0]['content'])
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id, context_chars=4096)
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run(), 'continued')
        self.assertEqual(len(self.requests), 2)
        self.assertIn('tools', self.requests[-1])

    async def test_repeated_compaction_folds_old_memory_and_is_session_local(self):
        await self.seed()
        async with httpx.AsyncClient(transport=httpx.MockTransport(self.respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                await self.agent.run('first')
                await self.agent.run('y' * 6000)
                await self.agent.run('third')
        summaries = [body for body in self.requests if 'tools' not in body]
        self.assertEqual(len(summaries), 2)
        self.assertIn('Session continuation memory', summaries[-1]['messages'][-1]['content'])
        self.assertEqual((await self.agent.state()).messages[-2]['content'], 'third')
        other = await self.agent.runtime.agent()
        try:
            self.assertEqual((await other.state()).messages, [])
        finally:
            await other.close()

    async def test_invalid_summaries_do_not_replace_history(self):
        await self.seed()
        await self.agent.append_message({'role': 'user', 'content': 'continue'})
        before = (await self.agent.state()).messages
        for message in ({'role': 'assistant', 'content': ''},
                        {'role': 'assistant', 'content': 's' * 1000},
                        {'role': 'assistant', 'refusal': 'No summary'},
                        {'role': 'assistant', 'tool_calls': [{'id': 'blocked', 'type': 'function',
                          'function': {'name': 'python', 'arguments': '{"code":"counter=0"}'}}]}):
            async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request:
                    stream_response(message))) as client:
                with patch.object(self.agent.runtime, 'client', model_client(client)):
                    with self.assertRaisesRegex(RuntimeError, 'summary'):
                        await self.agent.run()
            self.assertEqual((await self.agent.state()).messages, before)
        with self.assertRaisesRegex(RuntimeError, 'did not reduce'):
            await self.agent.runtime.database_call(self.agent._save_compaction, before, before)

