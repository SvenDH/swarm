import json
import unittest

import test_agents
from agents import CodeAgent, ExecutionError
from wrappers.registry import available


@unittest.skipUnless(all(name in available() for name in ('PIL', 'numpy')), 'optional packages missing')
class ExecTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_file_imports_use_same_translation_and_persist(self):
        source = "from PIL import Image\nimport numpy as np\na=np.arange(6).reshape(2,3)\nImage.new('RGB',(3,2),'red').save('out.png')\nanswer=a.sum()"
        await self.agent.execute(f"open('draw.py','w').write({source!r})")
        await self.agent.execute("exec(open('draw.py').read())")
        self.assertEqual(await self.agent.execute('(answer,a.tolist(),Image.open("out.png").size)'),
                         '(15, [[0, 1, 2], [3, 4, 5]], (3, 2))')
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute("exec('a += 1'); a.tolist()"), '[[1, 2, 3], [4, 5, 6]]')

    async def test_nested_exec_and_variables_in_caller_scope(self):
        inner = "import numpy as np; x=np.arange(3)"
        nested = f'exec({inner!r})'
        self.assertEqual(await self.agent.execute(f"exec({nested!r}); x.tolist()"), '[0, 1, 2]')
        self.assertEqual(await self.agent.execute("value=7; exec('value += 1'); value"), '8')

    async def test_exec_rolls_back_files_and_globals(self):
        await self.agent.execute("x=1; open('saved','w').write('original')")
        source = "import numpy as np; x=99; open('saved','w').write('bad'); raise ValueError('abort')"
        with self.assertRaisesRegex(ExecutionError, '^ValueError: abort$'):
            await self.agent.execute(f'exec({source!r})')
        self.assertEqual(await self.agent.execute("(x,open('saved').read())"), "(1, 'original')")

    async def test_dynamic_source_cannot_import_host_modules_or_read_host_paths(self):
        for source in ("import os; os.system('true')", "import subprocess", "import numpy as np; np.load('/etc/passwd')",
                       "from PIL import Image; Image.open('/etc/passwd')", "import numpy as np; np.__dict__"):
            with self.subTest(source=source), self.assertRaises(ExecutionError):
                await self.agent.execute(f'exec({source!r})')

    async def test_versions_and_normal_error_names(self):
        from importlib.metadata import version
        self.assertEqual(await self.agent.execute('import numpy as np; import PIL; (np.__version__, PIL.__version__)'),
                         repr((version('numpy'), version('Pillow'))))
        for source, expected in (("raise ValueError('bad')", 'ValueError'), ('if', 'SyntaxError'),
                                 ('np.missing_api()', 'AttributeError'), ('np.__dict__', 'AttributeError')):
            with self.subTest(source=source), self.assertRaises(ExecutionError) as caught:
                await self.agent.execute(source)
            message = str(caught.exception)
            self.assertTrue(message.startswith(expected + ':'), message)
            for detail in ('Monty', 'wrappers', 'Private library'):
                self.assertNotIn(detail, message)

    async def test_display_inside_exec(self):
        await self.agent.execute("exec(\"from PIL import Image; display(Image.new('RGB',(8,8)))\")")
        self.assertIn('_images', json.dumps((await self.agent.state()).messages))
