"""Normal exceptions checkpoint partial state; infrastructure failures do not."""
import unittest
from unittest.mock import patch

import test_agents
from agents import CodeAgent, ExecutionError
from wrappers.registry import available


class PartialErrorTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_user_raised_timeout_is_an_ordinary_exception(self):
        with self.assertRaisesRegex(ExecutionError, 'TimeoutError: user error'):
            await self.agent.execute("value=42; raise TimeoutError('user error')")
        self.assertEqual(await self.agent.execute('value'), '42')

    async def test_definitions_deletions_and_files_survive_error_and_restart(self):
        await self.agent.execute('removed=42')
        with self.assertRaisesRegex(ExecutionError, 'ZeroDivisionError'):
            await self.agent.execute('''
import os
os.mkdir('work')
open('work/saved','w').write('partial')
del removed
def answer(): return 42
1/0
''')
        self.assertIsNone((await self.agent.state()).pending)
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute("(answer(),open('work/saved').read(),'removed' in globals())"),
                         repr((42, 'partial', False)))

    async def test_failed_partial_commit_remains_pending_and_replays_from_checkpoint(self):
        await self.agent.execute("count=0; open('saved','w').write('old')")
        with patch.object(self.agent, '_commit', side_effect=OSError('storage unavailable')):
            with self.assertRaises(OSError):
                await self.agent.execute("count+=1; open('saved','w').write('new'); raise ValueError('abort')")
        self.assertIsNotNone((await self.agent.state()).pending)
        self.assertEqual(await self.agent.read_file(self.agent.home + '/saved'), b'old')
        with self.assertRaisesRegex(ExecutionError, 'abort'):
            await self.agent.execute()
        self.assertEqual(await self.agent.execute("(count,open('saved').read())"), "(1, 'new')")

    @unittest.skipUnless('numpy' in available(), 'NumPy missing')
    async def test_syntax_error_and_cached_error_do_not_invalidate_native_objects(self):
        await self.agent.execute('import numpy as np; a=np.arange(3)')
        with self.assertRaisesRegex(ExecutionError, 'SyntaxError'):
            await self.agent.execute('if')
        self.assertEqual(await self.agent.execute('a.tolist()'), '[0, 1, 2]')
        command = "a[0]+=1; raise ValueError('once')"
        for _ in range(2):
            with self.assertRaisesRegex(ExecutionError, 'once'):
                await self.agent.execute(command, execution_id='once')
        self.assertEqual(await self.agent.execute('a.tolist()'), '[1, 1, 2]')
