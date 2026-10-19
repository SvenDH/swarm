"""Common NumPy factories and ufuncs through real Monty, not native-only mocks."""
import unittest
import tempfile
from unittest.mock import patch

from agents import Runtime, ExecutionError
from wrappers.registry import available


@unittest.skipUnless('numpy' in available(), 'NumPy not installed')
class NumpyCommonTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = self.tmp.name + '/agent.db'
        self.runtime = await Runtime(self.path, workers=2).__aenter__()
        self.agent = await self.runtime.agent()

    async def asyncTearDown(self):
        await self.runtime.close()
        self.tmp.cleanup()

    async def test_grid_indexing_matches_numpy_and_returns_mutable_values(self):
        import numpy as np
        await self.agent.execute('import numpy as np; step=0.5; points=3j')
        for expression in ('np.mgrid[0:4]', 'np.mgrid[3:0:-1]', 'np.mgrid[-1:1:5j]',
                           'np.mgrid[0:1:step]', 'np.mgrid[0:1:points]',
                           'np.mgrid[0:2,0:3]', 'np.mgrid[0:1:3j,0:2:2j]',
                           'np.mgrid[0:0,0:3]', 'np.ogrid[0:2,0:3]'):
            value = eval(expression, {'np': np, 'step': 0.5, 'points': 3j})
            expected = [item.tolist() for item in value] if isinstance(value, (tuple, list)) else value.tolist()
            # Recent NumPy returns tuples for sparse multi-axis grids.
            if expression.startswith('np.ogrid'):
                expected = [item.tolist() for item in value]
                code = '[item.tolist() for item in ' + expression + ']'
            else:
                code = expression + '.tolist()'
            with self.subTest(expression=expression):
                self.assertEqual(await self.agent.execute('import numpy as np; ' + code), repr(expected))
        self.assertEqual(await self.agent.execute('grid=np.mgrid[0:2,0:3]; grid[0,0,0]=8; grid.tolist()'),
                         '[[[8, 0, 0], [1, 1, 1]], [[0, 1, 2], [0, 1, 2]]]')
        self.assertEqual(await self.agent.execute('(np.indices((2,2)).tolist(),np.broadcast_to([1,2],(2,2)).tolist())'),
                         '([[[0, 0], [1, 1]], [[0, 1], [0, 1]]], [[1, 2], [1, 2]])')
        self.assertEqual(await self.agent.execute('y,x=np.mgrid[0:2,0:3]; (y.tolist(),x.tolist())'),
                         '([[0, 0, 0], [1, 1, 1]], [[0, 1, 2], [0, 1, 2]])')

    async def test_bincount_weights_empty_inputs_and_limits(self):
        self.assertEqual(await self.agent.execute('''
import numpy as np
(np.bincount([0,2,2],minlength=4).tolist(),
 np.bincount([0,1,1],weights=[0.5,1.,2.]).tolist(), np.bincount([]).tolist())
'''), '([1, 0, 2, 0], [0.5, 3.0], [])')
        from wrappers.numpy import bincount
        with patch('wrappers.numpy.np.bincount', side_effect=AssertionError('native allocation reached')):
            for values, minlength in (([1000000000],0), ([1],1000000000), ([-1],0), ([1.5],0)):
                with self.subTest(values=values, minlength=minlength), self.assertRaises((ValueError, MemoryError)):
                    bincount(values, minlength=minlength)

    async def test_ufunc_calls_methods_indexed_updates_and_restart(self):
        from wrappers.numpy import UFUNCS
        for name, function in UFUNCS.items():
            arguments = [[1,2]] * function.nin
            if name in ('arccos', 'arcsin', 'arctanh'):
                arguments = [[0,0]] * function.nin
            native = function(*arguments)
            expected = tuple(part.tolist() for part in native) if isinstance(native, tuple) else native.tolist()
            with self.subTest(ufunc=name):
                expression = 'np.' + name + '(' + ','.join(map(repr, arguments)) + ')'
                expression = ('tuple([part.tolist() for part in ' + expression + '])'
                              if isinstance(native, tuple) else expression + '.tolist()')
                code = 'import numpy as np; ' + expression
                self.assertEqual(await self.agent.execute(code), repr(expected))
        self.assertEqual(await self.agent.execute('''
import numpy as np
from numpy import add, multiply
a=np.array([1,2,3])
add.at(a,[0,0,2],1)
(a.tolist(),add([1,2],10).tolist(),np.subtract([3,4],1).tolist(),
 add.reduce(a),multiply.accumulate([1,2,3]).tolist(),add.outer([1,2],[3,4]).tolist())
'''), '([3, 2, 4], [11, 12], [2, 3], 9, [1, 2, 6], [[4, 5], [5, 6]])')
        session_id = self.agent.id
        await self.runtime.close()
        self.runtime = await Runtime(self.path, workers=2).__aenter__()
        self.agent = await self.runtime.agent(session_id=session_id)
        self.assertEqual(await self.agent.execute('view=a[1:]; add.at(view,[0],5); a.tolist()'), '[3, 7, 4]')
        self.assertEqual(await self.agent.execute('(add.reduce(a,0),add.reduceat(a,[0,2]).tolist(),np.logical_and.reduce([True,False]))'), '(14, [10, 4], False)')
        for code in ('add(a,1,out=a)', 'add(a,1,a)', 'add.reduce(a,out=a)', 'add(a,1,where=[False,True,True])',
                     'add.outer(np.ones(10001),np.ones(10001))',
                     'add(np.ones((10001,1)),1,where=np.ones((1,10001),dtype=bool))',
                     'add.at(np.zeros((2,2)),(np.zeros((10001,1),dtype=int),np.zeros((1,10001),dtype=int)),1)'):
            with self.subTest(code=code), self.assertRaises(ExecutionError):
                await self.agent.execute(code)
        self.assertEqual(await self.agent.execute('a.tolist()'), '[3, 7, 4]')

    async def test_grids_preflight_before_native_allocations(self):
        from wrappers.numpy import Grid
        with patch('wrappers.numpy.np.mgrid') as native:
            native.__getitem__.side_effect = AssertionError('native allocation reached')
            for key in (slice(0,1000000000), (slice(0,10001),slice(0,10001)), slice(0,1,1000000000j)):
                with self.subTest(key=key), self.assertRaises(MemoryError):
                    Grid()[key]

