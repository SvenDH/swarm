import unittest
from unittest.mock import patch

import test_agents
from agents import ExecutionError
from wrappers.limits import memory_errors
from wrappers.registry import available


class MemoryErrorTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_caught_wrapper_memory_error_is_generic(self):
        with patch('wrappers.protocol.Bridge._call', side_effect=MemoryError(
                'native worker allocation exceeds session memory budget 1000 bytes')):
            result = await self.agent.execute('''
try:
    _library_rpc('import','numpy')
except MemoryError as error:
    print(str(error))
''')
        self.assertEqual(result, '\n')

    @unittest.skipUnless('numpy' in available(), 'NumPy missing')
    async def test_numpy_allocation_error_matches_numpy(self):
        import numpy as np
        from numpy._core._exceptions import _ArrayMemoryError
        self.agent.memory_budget = 2 * 1024**2
        expected = str(_ArrayMemoryError((1200, 1080, 3), np.dtype('uint8')))
        with self.assertRaises(ExecutionError) as caught:
            await self.agent.execute('import numpy as np; np.zeros((1200,1080,3),dtype=np.uint8)')
        self.assertEqual(str(caught.exception), 'MemoryError: ' + expected)
        self.assertEqual(await self.agent.execute('''
import numpy as np
try:
    np.zeros((1200,1080,3),dtype=np.uint8)
except MemoryError as error:
    print(str(error))
'''), expected + '\n')

    async def test_generic_memory_failure_has_no_message(self):
        with self.assertRaises(ExecutionError) as caught:
            await self.agent.execute("raise MemoryError('worker budget details')")
        self.assertEqual(str(caught.exception), 'MemoryError')

    async def test_ordinary_errors_keep_their_messages(self):
        with patch('wrappers.protocol.Bridge._call', side_effect=ValueError('Invalid library state signature')):
            with self.assertRaisesRegex(ExecutionError, 'Invalid library state signature'):
                await self.agent.execute("_library_rpc('import','numpy')")
        with self.assertRaisesRegex(ExecutionError, 'ValueError: bad shape'):
            await self.agent.execute("raise ValueError('bad shape')")

    def test_only_memory_errors_are_intercepted(self):
        for error in (TypeError('specific type error'), FileNotFoundError('/tmp/missing.txt')):
            @memory_errors
            def fail():
                raise error
            with self.assertRaises(type(error)) as caught:
                fail()
            self.assertIs(caught.exception, error)
