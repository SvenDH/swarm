import base64
import io
import json
import unittest
from unittest.mock import patch

import httpx

import test_agents
from test_agents import model_client, stream_response
from agents import CodeAgent, ExecutionError
from wrappers.registry import available


@unittest.skipUnless('PIL' in available(), 'Pillow is not installed')
class DisplayTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_paths_objects_and_restart(self):
        from PIL import Image
        await self.agent.execute("from PIL import Image; im=Image.new('RGB',(1600,900),'red'); im.save('scene.png'); display(im)")
        await self.agent.execute("display('scene.png')")
        history = (await self.agent.state()).messages
        self.assertEqual(len(history), 2)
        self.assertNotIn('base64', json.dumps(history))
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        await self.agent.execute("from pathlib import Path; display(Path('scene.png'))")
        with Image.open(io.BytesIO(await self.agent.read_file(self.agent.home + '/scene.png'))) as original:
            self.assertEqual(original.size, (1600, 900))
        requests = []
        def respond(request):
            requests.append(json.loads(request.content))
            return stream_response({'role': 'assistant', 'content': 'A red image.'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run(), 'A red image.')
        parts = [p for m in requests[0]['messages'] if isinstance(m.get('content'), list)
                 for p in m['content'] if p['type'] == 'image_url']
        self.assertEqual(len(parts), 3)
        with Image.open(io.BytesIO(base64.b64decode(parts[0]['image_url']['url'].split(',')[1]))) as preview:
            self.assertEqual(preview.size, (1024, 576))
        self.assertNotIn('_images', json.dumps(requests[0]))

    async def test_display_in_tool_call_reaches_next_model_request(self):
        requests = []
        def respond(request):
            body = json.loads(request.content)
            requests.append(body)
            if len(requests) == 1:
                return stream_response({'role': 'assistant', 'tool_calls': [{'id': 'draw', 'type': 'function',
                    'function': {'name': 'python', 'arguments': json.dumps({'code':
                        "from PIL import Image; display(Image.new('RGB',(8,8),'blue'))"})}}]})
            self.assertEqual(body['messages'][-2]['role'], 'tool')
            self.assertEqual(body['messages'][-1]['content'][1]['type'], 'image_url')
            return stream_response({'role': 'assistant', 'content': 'Blue.'})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with patch.object(self.agent.runtime, 'client', model_client(client)):
                self.assertEqual(await self.agent.run('draw'), 'Blue.')

    async def test_invalid_inputs_do_not_attach_images_but_partial_display_does(self):
        await self.agent.execute("from PIL import Image; im=Image.new('RGB',(8,8))")
        for code in ("display('/etc/passwd')", "display('https://example.com/a.png')", "display(123)",
                     "display({'__library_value__': {}, 'path': [], 'auth': None})",
                     "display(im); raise ValueError('abort')"):
            with self.subTest(code=code), self.assertRaises(ExecutionError):
                await self.agent.execute(code)
        messages = (await self.agent.state()).messages
        self.assertEqual(len(messages), 1)
        self.assertEqual(len(messages[0]['_images']), 1)

    async def test_batch_images_follow_all_tool_results(self):
        await self.agent.execute("from PIL import Image; display(Image.new('RGB',(8,8)))")
        key = (await self.agent.state()).messages[0]['_images'][0]
        messages = [{'role': 'assistant', 'tool_calls': [{'id': 'a'}, {'id': 'b'}]},
                    {'role': 'tool', 'tool_call_id': 'a', 'content': '', '_images': [key]},
                    {'role': 'tool', 'tool_call_id': 'b', 'content': ''}]
        expanded = self.agent._vision_messages(messages)
        self.assertEqual([m['role'] for m in expanded], ['assistant', 'tool', 'tool', 'user'])
