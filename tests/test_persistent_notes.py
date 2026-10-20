import json
import unittest
from unittest.mock import patch

import httpx

from agents import CodeAgent, ExecutionError
import test_agents
from test_agents import model_client, stream_response


class PersistentNotesTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown
    async def test_notes_reload_partial_failure_isolation_and_restart(self):
        self.assertEqual(await self.agent._memory(), [])
        await self.agent.execute("open('MEMORY.md','w').write('Prefer concise answers.')")
        with self.assertRaises(ExecutionError):
            await self.agent.execute("open('MEMORY.md','w').write('Prefer concise answers. Partial edit.'); raise ValueError('abort')")
        other = await self.agent.runtime.agent()
        self.assertEqual(await other._memory(), [])
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        requests = []

        def respond(request):
            requests.append(json.loads(request.content))
            return stream_response({'role': 'assistant', 'content': 'ok'})

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                await self.agent.run('hello')
                await self.agent.execute("open('MEMORY.md','w').write('Use metric units.')")
                await self.agent.run('again')
        self.assertIn('Partial edit.', requests[0]['messages'][1]['content'])
        self.assertIn('Use metric units.', requests[1]['messages'][1]['content'])
        self.assertNotIn('Prefer concise answers.', json.dumps(requests[1]))
        self.assertEqual(requests[0]['messages'][1]['role'], 'user')
        self.assertNotIn('Persistent notes', json.dumps((await self.agent.state()).messages))

    async def test_preview_bounds_and_deletion(self):
        await self.agent.execute("open('MEMORY.md','w').write('a'*9000)")
        self.assertIn('truncated', (await self.agent._memory())[0]['content'])
        await self.agent.execute("open('MEMORY.md','w').write('a'*40000)")
        with patch.object(self.agent.storage, 'read', side_effect=AssertionError('large read')):
            self.assertIn('exceeds automatic preview', (await self.agent._memory())[0]['content'])
        await self.agent.execute("from pathlib import Path; Path('MEMORY.md').unlink()")
        self.assertEqual(await self.agent._memory(), [])
