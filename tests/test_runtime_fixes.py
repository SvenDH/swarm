import asyncio
import unittest
from unittest.mock import patch

import test_agents
from agents import CodeAgent, ExecutionError, Runtime
from wrappers.registry import available


@unittest.skipUnless(all(name in available() for name in ('numpy', 'PIL')), 'optional libraries missing')
class RuntimeFixTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_explicit_exec_namespaces(self):
        self.assertEqual(await self.agent.execute('''
ns={}
exec('import numpy as np; a=np.arange(4); answer=a.sum()',ns)
ns['answer']
'''), '6')
        self.assertEqual(await self.agent.execute("exec('answer += 1',ns); ns['answer']"), '7')
        self.assertEqual(await self.agent.execute("scope={}; local={}; exec('import numpy as np; answer=np.arange(3).sum()',scope,local); local['answer']"), '3')
        self.assertEqual(await self.agent.execute("x=10; ns=globals().copy(); exec('answer=x+1',ns); ns['answer']"), '11')
        self.assertEqual(await self.agent.execute("exec('answer=x+2',globals())"), '')
        self.assertEqual(await self.agent.execute("import numpy as np; a=np.arange(4); copied=globals().copy(); exec('answer=a.sum()',copied); copied['answer']"), '6')
        with self.assertRaises(ExecutionError):
            await self.agent.execute("exec(\"import os; os.system('true')\",{})")
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute("ns['answer']"), '11')

    async def test_random_legacy_out_and_explicit_reseeding(self):
        self.assertEqual(await self.agent.execute('''
import numpy as np
np.random.seed(123)
a=np.random.rand(2,3)
r=np.random.RandomState(123)
(a.tolist()==r.rand(2,3).tolist(),np.random.randn(3).shape)
'''), '(True, (3,))')
        self.assertEqual(await self.agent.execute('''
rng=np.random.default_rng(123)
out=np.zeros(5)
result=rng.random(out=out)
expected=np.random.default_rng(123)
(result is out,out.tolist()==expected.random(5).tolist())
'''), '(True, True)')
        self.assertEqual(await self.agent.execute('''
out32=np.zeros(5,dtype=np.float32)
result=rng.standard_normal(dtype=np.float32,out=out32)
(result is out32,out32.tolist()!=[0,0,0,0,0])
'''), '(True, True)')
        await self.agent.execute('np.random.seed(42)')
        with self.assertRaises(ExecutionError):
            await self.agent.execute("np.random.rand(5); raise ValueError('abort')")
        self.assertEqual(await self.agent.execute('np.random.seed(42); np.random.rand(3).tolist()==np.random.RandomState(42).rand(3).tolist()'), 'True')
        await self.agent.execute('np.random.seed(7)')
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute('np.random.seed(7); np.random.rand(3).tolist()==np.random.RandomState(7).rand(3).tolist()'), 'True')

    async def test_point_callbacks_and_import(self):
        self.assertEqual(await self.agent.execute('''
from PIL import Image
image=Image.new('RGB',(4,4),(10,20,30))
image.point(lambda value: round(value*0.5)).getpixel((0,0))
'''), '(5, 10, 15)')
        self.assertEqual(await self.agent.execute("__import__('numpy').arange(3).tolist()"), '[0, 1, 2]')
        with self.assertRaises(ExecutionError):
            await self.agent.execute("__import__('subprocess')")

    async def test_exception_formatting_without_host_tracebacks(self):
        self.assertEqual(await self.agent.execute('''
from traceback import format_exception_only
try:
    raise ValueError('bad input')
except Exception as error:
    formatted=format_exception_only(error)
formatted
'''), repr(['ValueError: bad input\n']))

    async def test_broadcasting_and_partial_file_commit(self):
        self.assertEqual(await self.agent.execute('''
import numpy as np
H,W=40,60
img=np.zeros((H,W,3),dtype=np.float32)
m=np.ones((H,W),dtype=np.float32)
c=np.array([1,2,3],dtype=np.float32)
(img+m.reshape(H,W,1)*(c.reshape(1,1,3)*0.5)).shape
'''), '(40, 60, 3)')
        self.assertEqual(await self.agent.execute("open('saved','w').write('hello'); open('saved').read()"), "'hello'")
        with self.assertRaises(ExecutionError):
            await self.agent.execute("open('saved','w').write('bad'); print('wrote'); raise ValueError('abort')")
        self.assertEqual(await self.agent.execute("open('saved').read()"), "'bad'")

    async def test_shared_budget_serializes_execution_not_agents(self):
        async with Runtime(self.tmp.name + '/shared.db', workspaces=self.tmp.name + '/shared-files', workers=4, memory_budget=8*1024**2) as runtime:
            agents = [await runtime.agent(), await runtime.agent()]
            active = peak = 0
            original = CodeAgent._evaluate
            async def counted(agent, *args):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                try:
                    await asyncio.sleep(0.01)
                    return await original(agent, *args)
                finally:
                    active -= 1
            with patch.object(CodeAgent, '_evaluate', counted):
                results = await asyncio.gather(*(agent.execute("data=b'x'*1500000; len(data)") for agent in agents))
            self.assertEqual(results, ['1500000', '1500000'])
            self.assertEqual(peak, 1)
            with self.assertRaises(TypeError):
                await runtime.agent(memory_budget=16*1024**2)

    async def test_checkpoint_headroom_keeps_cleanup_possible(self):
        async with Runtime(self.tmp.name + '/headroom.db', workspaces=self.tmp.name + '/headroom-files', memory_budget=4*1024**2) as runtime:
            agent = await runtime.agent()
            await agent.execute("data=b'x'*1000000")
            with self.assertRaises(ExecutionError):
                await agent.execute("data=b'x'*3400000")
            self.assertEqual(await agent.execute('data=None; 42'), '42')

    async def test_minimal_cleanup_preserves_deleted_name_semantics(self):
        await self.agent.execute('value=7')
        await self.agent.execute('del value')
        with self.assertRaises(ExecutionError):
            await self.agent.execute('value')
        await self.agent.execute('value=8')
        with self.assertRaises(ExecutionError):
            await self.agent.execute('del value, value')
        self.assertEqual(await self.agent.execute('value'), '8')

    async def test_cleanup_of_legacy_checkpoint_without_helper_headroom(self):
        async with Runtime(self.tmp.name + '/legacy.db', workspaces=self.tmp.name + '/legacy-files', memory_budget=4*1024**2) as runtime:
            agent = await runtime.agent()
            # Trusted legacy state predating the checkpoint-headroom guard.
            async with runtime._pool.checkout(limits={'max_memory': 4*1024**2}) as session:
                await session.feed_run("large=b'x'*1800000")
                snapshot = await session.dump()
            await runtime.database_call(lambda: runtime.db.execute('UPDATE sessions SET monty=? WHERE id=?', (snapshot, agent.id)))
            with self.assertRaises(ExecutionError):
                await agent.execute('42')
            await agent.execute('del large')
            with self.assertRaises(ExecutionError):
                await agent.execute('large')
            self.assertEqual(await agent.execute('42'), '42')
