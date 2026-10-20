import asyncio
import json
import unittest
from unittest.mock import patch

import httpx
from openai import BadRequestError

import test_agents
from test_agents import model_client, stream_response, sse, ByteStream
from agents import CodeAgent, Worker, WorkerClient


class ReasoningTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_restart_preserves_exact_reasoning_for_tool_continuation(self):
        def respond(request):
            return httpx.Response(200, stream=ByteStream([
                sse({'role': 'assistant', 'reasoning_content': 'Use the '}),
                sse({'reasoning_content': 'saved value.'}),
                sse({'tool_calls': [{'index': 0, 'id': 'calc', 'type': 'function',
                    'function': {'name': 'python', 'arguments': json.dumps({'code': 'x=42; x'})}}]}, 'tool_calls'),
                b'data: [DONE]\n\n']))
        await self.agent.append_message({'role': 'user', 'content': 'calculate'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            message = await self.agent._response(model_client(client), (await self.agent.state()).messages)
        await self.agent.append_message(message)
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        def continuation(request):
            history = json.loads(request.content)['messages']
            self.assertEqual(history[-2]['reasoning_content'], 'Use the saved value.')
            self.assertEqual(history[-1]['role'], 'tool')
            return stream_response({'role': 'assistant', 'content': '42', 'reasoning_content': ''})
        async with httpx.AsyncClient(transport=httpx.MockTransport(continuation)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run(), '42')
        self.assertEqual((await self.agent.state()).messages[-1]['reasoning_content'], '')

    async def test_invalid_old_history_is_not_repaired_or_replayed(self):
        await self.agent.execute("x=42")
        await self.agent.append_message({'role':'user','content':'continue'})
        before=(await self.agent.state()).messages
        requests=[]
        def respond(request):
            requests.append(request)
            return httpx.Response(400,json={'error':{'message':'The reasoning_content must be passed back'}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with patch.object(self.agent.runtime,'client',model_client(client)):
                with self.assertRaises(BadRequestError):
                    await self.agent.run()
        self.assertEqual(len(requests),1)
        self.assertEqual((await self.agent.state()).messages,before)
        self.assertEqual(await self.agent.execute('x'),'42')

    async def test_worker_does_not_retry_other_bad_requests(self):
        calls = []
        def respond(request):
            calls.append(request)
            return httpx.Response(400, json={'error': {'message': 'Unsupported model option'}})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as http:
            async with Worker(self.tmp.name + '/worker/agent.db', client=model_client(http), retry_delay=0.01) as worker:
                client = WorkerClient(self.tmp.name + '/worker/agent.db')
                info = await client.request('agent')
                with self.assertRaisesRegex(RuntimeError, 'request rejected'):
                    await client.request('run', session_id=info['id'], prompt='hello')
                await asyncio.sleep(0.05)
                self.assertEqual(len(calls), 1)
                self.assertIsNotNone(await worker.runtime.database_call(worker._next, info['id']))
