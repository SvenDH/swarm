"""Pillow pixel access stays attached to its durable image, never a host handle."""
import unittest

import test_agents
from agents import CodeAgent, ExecutionError
from wrappers.registry import available


@unittest.skipUnless('PIL' in available(), 'Pillow not installed')
class PillowLoadTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_in_memory_png_round_trip(self):
        self.assertEqual(await self.agent.execute('''
from PIL import Image
from io import BytesIO
image=Image.new('RGB',(3,2),(1,2,3))
buffer=BytesIO()
image.save(buffer,format='PNG')
buffer.seek(0)
reopened=Image.open(buffer)
(len(buffer.getvalue())>0,reopened.load()[0,0],buffer.tell()>0)
'''), '(True, (1, 2, 3), True)')

    async def test_pixels_alias_restart_and_file_round_trip(self):
        self.assertEqual(await self.agent.execute('''
from PIL import Image
image=Image.new('RGB',(3,2),(10,20,30))
pixels=image.load()
alias=image.load()
pixels[1,0]=(40,50,60)
(image.getpixel((1,0)),alias[1,0],pixels[-1,-1])
'''), '((40, 50, 60), (40, 50, 60), (10, 20, 30))')
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute('''
alias[0,1]=(70,80,90)
image.putpixel((2,1),(1,2,3))
image.save('pixels.png')
reopened=Image.open('pixels.png')
(pixels[0,1],pixels[2,1],reopened.load()[0,1])
'''), '((70, 80, 90), (1, 2, 3), (70, 80, 90))')

    async def test_scalar_modes_errors_and_rollback(self):
        self.assertEqual(await self.agent.execute('''
from PIL import Image
gray=Image.new('L',(2,2),7)
pixels=gray.load()
pixels[0,0]=123
(pixels[0,0],gray.getpixel((0,0)))
'''), '(123, 123)')
        for code in ("pixels[2,0]", "pixels[0,2]=1", "pixels[0,0]=(1,2,3)",
                     "pixels[0,0]=42; raise ValueError('abort')",
                     'pixels.image', 'pixels.__class__'):
            with self.subTest(code=code), self.assertRaises(ExecutionError):
                await self.agent.execute(code)
        self.assertEqual(await self.agent.execute('(pixels[0,0],gray.getpixel((0,0)))'), '(123, 123)')
