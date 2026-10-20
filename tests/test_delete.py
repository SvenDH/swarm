import unittest

import test_agents
from agents import CodeAgent, ExecutionError
from wrappers.registry import available


class DeleteTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_names_restart_rebinding_and_missing_names(self):
        await self.agent.execute('x=[1,2,3]; alias=x; del x')
        with self.assertRaisesRegex(ExecutionError, "NameError: name 'x' is not defined"):
            await self.agent.execute('x')
        self.assertEqual(await self.agent.execute('alias'), '[1, 2, 3]')
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        with self.assertRaisesRegex(ExecutionError, 'NameError'):
            await self.agent.execute('x')
        with self.assertRaisesRegex(ExecutionError, 'NameError'):
            await self.agent.execute('del missing')
        self.assertEqual(await self.agent.execute('x=7; x'), '7')

    async def test_containers_slices_order_and_tuples(self):
        self.assertEqual(await self.agent.execute("a=list(range(8)); d={'x':1,'y':2}; del a[1:6:2], d['x']; (a,d)"),
                         "([0, 2, 4, 6, 7], {'y': 2})")
        self.assertEqual(await self.agent.execute('del a[::-2]; a'), '[2, 6]')
        await self.agent.execute('x=1; y=2; del (x,y)')
        for name in ('x', 'y'):
            with self.assertRaisesRegex(ExecutionError, 'NameError'):
                await self.agent.execute(name)

    async def test_exec_and_function_scopes(self):
        await self.agent.execute("exec('x=42; del x')")
        with self.assertRaisesRegex(ExecutionError, 'NameError'):
            await self.agent.execute('x')
        self.assertEqual(await self.agent.execute('''
def f():
    x=[1,2,3]
    del x
    try:
        return x
    except NameError:
        return 'deleted'
f()
'''), "'deleted'")

    async def test_failed_delete_rolls_back(self):
        await self.agent.execute("x=42; a=[1,2,3]")
        with self.assertRaisesRegex(ExecutionError, 'ValueError'):
            await self.agent.execute("del x, a[0]; raise ValueError('abort')")
        self.assertEqual(await self.agent.execute('(x,a)'), '(42, [1, 2, 3])')

    @unittest.skipUnless('pandas' in available(), 'pandas missing')
    async def test_dataframe_column_deletion_and_readonly_namespaces(self):
        self.assertEqual(await self.agent.execute("import pandas as pd; df=pd.DataFrame({'x':[1],'y':[2]}); del df['x']; df.to_dict('records')"),
                         "[{'y': 2}]")
        with self.assertRaises(ExecutionError):
            await self.agent.execute("del pd['DataFrame']")

    @unittest.skipUnless('numpy' in available(), 'numpy missing')
    async def test_array_payload_is_not_in_checkpoints(self):
        await self.agent.execute('import numpy as np; a=np.zeros(200000)')
        before = len((await self.agent.state()).monty)
        await self.agent.execute('del a')
        after = len((await self.agent.state()).monty)
        self.assertLess(before, 250000)
        self.assertLess(after, 250000)
        await self.agent.execute('b=np.zeros(200000)')
        self.assertEqual(await self.agent.execute('b.shape'), '(200000,)')
