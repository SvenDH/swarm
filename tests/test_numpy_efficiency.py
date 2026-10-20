import unittest
from unittest.mock import patch

import test_agents
from agents import CodeAgent, ExecutionError
from wrappers.registry import available
from wrappers.protocol import Bridge


@unittest.skipUnless('numpy' in available(), 'NumPy not installed')
class NumpyEfficiencyTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    def test_array_buffer_serialization_layouts(self):
        import numpy as np
        from wrappers.numpy import pack, restore
        base = np.arange(24).reshape(4, 6)
        for value in (base, base.T, base[:, ::2], np.zeros((0, 2)), np.array(3)):
            with self.subTest(shape=value.shape, strides=value.strides):
                np.testing.assert_array_equal(restore('array', pack(value)['data']), value)

    async def test_callable_scalar_types_dtype_arguments_and_restart(self):
        result = await self.agent.execute('''
import numpy as np
dtype=np.float32
a=np.array([1,2],dtype=dtype)
(dtype(1.25),np.uint8(7),a.astype(np.int32).tolist(),np.zeros(2,np.float32).dtype.name)
''')
        self.assertEqual(result, "(1.25, 7, [1, 2], 'float32')")
        self.agent = await test_agents.reopen_live(self.agent)
        self.assertEqual(await self.agent.execute('dtype(2.5)'), '2.5')

    async def test_empty_is_initialized_and_bounded(self):
        self.assertEqual(await self.agent.execute('import numpy as np; a=np.empty((2,3),np.uint8); (a.tolist(),np.empty_like(a).tolist())'),
                         '([[0, 0, 0], [0, 0, 0]], [[0, 0, 0], [0, 0, 0]])')
        with self.assertRaises(ExecutionError):
            await self.agent.execute('np.empty(1000000000)')
        with self.assertRaises(ExecutionError):
            await self.agent.execute('np.empty_like(a,shape=(1000000000,))')

    async def test_out_aliases_views_masking_and_rollback(self):
        self.assertEqual(await self.agent.execute('''
import numpy as np
a=np.array([-2,1,9]); alias=a
r=np.clip(a,0,4,out=a)
b=np.zeros(2,dtype=np.int64)
np.add(a[1:],2,out=b)
(a.tolist(),alias.tolist(),b.tolist(),r is a)
'''), '([0, 1, 4], [0, 1, 4], [3, 6], True)')
        self.assertEqual(await self.agent.execute('v=a[1:]; np.add(v,1,out=v,where=[True,False]); a.tolist()'), '[0, 2, 4]')
        with self.assertRaises(ExecutionError):
            await self.agent.execute("np.clip(a,0,1,out=a); raise ValueError('abort')")
        with self.assertRaisesRegex(ExecutionError, 'recompute'):
            await self.agent.execute('a.tolist()')
        await self.agent.execute('a=np.array([0,2,4])')
        self.assertEqual(await self.agent.execute('s=np.zeros((),dtype=np.int64); np.add.reduce(a,out=s); s.item()'), '6')
        for code in ('np.add(a,1,out=[0,0,0])', "np.add(a,1,out={'__library_value__':{}})"):
            await self.agent.execute('a=np.array([0,2,4])')
            with self.assertRaises(ExecutionError):
                await self.agent.execute(code)

    async def test_direct_calls_use_one_rpc_per_method(self):
        actions = []
        original = Bridge.__call__
        def counted(bridge, action, *args, **kwargs):
            actions.append(action)
            return original(bridge, action, *args, **kwargs)
        with patch.object(Bridge, '__call__', counted):
            self.assertEqual(await self.agent.execute('import numpy as np; np.arange(4).sum()'), '6')
        self.assertEqual(actions, ['import', 'method', 'method'])

    async def test_readonly_buffer_restore_and_writable_mutation(self):
        import numpy as np
        from wrappers.numpy import pack, restore
        data = pack(np.arange(8))['data']
        with patch.object(np, 'load', side_effect=AssertionError('copying np.load path')):
            readonly = restore('array', data, copy=False)
            writable = restore('array', data)
        self.assertFalse(readonly.flags.owndata)
        self.assertFalse(readonly.flags.writeable)
        self.assertTrue(writable.flags.writeable)
        self.assertEqual(readonly.tolist(), writable.tolist())
        with self.assertRaises(ValueError):
            restore('array', data + b'extra')
        self.assertEqual(await self.agent.execute('import numpy as np; a=np.array([3,1,2]); a.sort(); a.tolist()'), '[1, 2, 3]')

    @unittest.skipUnless('PIL' in available(), 'Pillow not installed')
    async def test_drawing_loop_uses_one_bridge_call_per_line(self):
        actions = []
        original = Bridge.__call__
        def counted(bridge, action, *args, **kwargs):
            actions.append(action)
            return original(bridge, action, *args, **kwargs)
        with patch.object(Bridge, '__call__', counted):
            await self.agent.execute('''
from PIL import Image, ImageDraw
image=Image.new('RGB',(32,32))
draw=ImageDraw.Draw(image)
for i in range(150):
    draw.line((0,i%32,31,i%32),fill='white')
''')
        self.assertEqual(actions.count('method'), 152)
        self.assertEqual(await self.agent.execute('image.getpixel((0,0))'), '(255, 255, 255)')
