"""RAM-only handles: no implicit payload serialization or replay after restart."""
import tempfile
import unittest
from unittest.mock import patch

from agents import Runtime, ExecutionError
from wrappers.protocol import Bridge
from wrappers.registry import available


@unittest.skipUnless('numpy' in available(), 'NumPy not installed')
class NativeMemoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_arrays_never_cross_bridge_or_enter_checkpoint(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder + '/agents.db') as runtime:
                agent = await runtime.agent()
                sizes = []
                original = Bridge.__call__
                def measure(value):
                    if isinstance(value, bytes): return len(value)
                    if isinstance(value, (list,tuple)): return sum(map(measure,value))
                    if isinstance(value, dict): return sum(map(measure,value.values()))
                    return 0
                def counted(bridge,*args,**kwargs):
                    sizes.append(measure(args)+measure(kwargs))
                    result=original(bridge,*args,**kwargs)
                    sizes.append(measure(result))
                    return result
                with patch('wrappers.numpy.pack',side_effect=AssertionError('implicit serialization')), \
                     patch('wrappers.numpy.restore',side_effect=AssertionError('implicit restore')), \
                     patch.object(Bridge,'__call__',counted):
                    self.assertEqual(await agent.execute('''
import numpy as np
array=np.ones((1024,1024),dtype=np.float32)
for i in range(8):
    array=np.sin(array)+0.1
array.shape
'''), '(1024, 1024)')
                    self.assertEqual(await agent.execute('array.shape'), '(1024, 1024)')
                self.assertEqual(max(sizes),0)
                self.assertLess(len((await agent.state()).monty),250_000)
                self.assertGreater(runtime._native.size(),4*1024**2)

    async def test_restart_preserves_files_and_python_but_drops_native_objects(self):
        with tempfile.TemporaryDirectory() as folder:
            database=folder+'/agents.db'
            async with Runtime(database) as runtime:
                a=await runtime.agent()
                await a.execute("import numpy as np; count=42; x=np.arange(4); np.save('saved.npy',x)")
                await a.execute('def calculate(): return np.arange(4).sum()')
                session_id=a.id
                reopened=await runtime.agent(session_id=session_id)
                self.assertEqual(await reopened.execute('x.tolist()'),'[0, 1, 2, 3]')
            async with Runtime(database) as runtime:
                a=await runtime.agent(session_id=session_id)
                self.assertEqual(runtime._native.size(),0)
                self.assertEqual(await a.execute('(count,calculate())'),'(42, 6)')
                with self.assertRaisesRegex(ExecutionError,'recompute'):
                    await a.execute('x.tolist()')
                self.assertEqual(await a.execute("x=np.load('saved.npy'); x.tolist()"),'[0, 1, 2, 3]')

    async def test_native_failure_invalidates_only_its_session(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder+'/agents.db') as runtime:
                a,b=await runtime.agent(),await runtime.agent()
                await a.execute("import numpy as np; count=1; x=np.arange(3); open('saved','w').write('old')")
                await b.execute('import numpy as np; x=np.arange(3)+10')
                with self.assertRaisesRegex(ExecutionError,'abort'):
                    await a.execute("count=99; x[0]=99; open('saved','w').write('bad'); raise ValueError('abort')")
                self.assertEqual(await a.execute("(count,open('saved').read())"),"(1, 'old')")
                with self.assertRaisesRegex(ExecutionError,'recompute'):
                    await a.execute('x.tolist()')
                self.assertEqual(await b.execute('x.tolist()'),'[10, 11, 12]')

    async def test_shared_lru_budget_evicts_old_objects_instead_of_serializing(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder+'/agents.db',memory_budget=8*1024**2) as runtime:
                a,b=await runtime.agent(),await runtime.agent()
                await a.execute('import numpy as np; x=np.zeros(600000)')
                await b.execute('import numpy as np; x=np.ones(600000)')
                self.assertLessEqual(runtime._native.size(),runtime.memory_budget)
                with self.assertRaisesRegex(ExecutionError,'recompute'):
                    await a.execute('x.sum()')
                self.assertEqual(await b.execute('float(x.sum())'),'600000.0')

    async def test_views_out_arrays_iteration_and_authentication(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Runtime(folder+'/agents.db') as runtime:
                a,b=await runtime.agent(),await runtime.agent()
                self.assertEqual(await a.execute('''
import numpy as np
x=np.arange(6).reshape(2,3)
band=x[0]
band+=2
out=np.zeros(3,dtype=np.int64)
same=np.add(band,1,out=out)
(x.tolist(),same is out,list(zip(band,out)),next(iter(out)))
'''), '([[2, 3, 4], [3, 4, 5]], True, [(2, 3), (3, 4), (4, 5)], 3)')
                import ast
                token=ast.literal_eval(await a.execute('getattr(x,"root")'))
                self.assertNotIn('data',token)
                with self.assertRaisesRegex(ExecutionError,'signature'):
                    await b.execute(f"_library_rpc('repr',{token!r})")
                with self.assertRaisesRegex(ExecutionError,'signature'):
                    await a.execute(f"spec={token!r}; spec['identity']='forged'; _library_rpc('repr',spec)")
