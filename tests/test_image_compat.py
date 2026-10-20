"""Image-generation compatibility through real Monty and authenticated wrappers."""
import tempfile
import unittest

from agents import Runtime, ExecutionError
from wrappers.registry import available


@unittest.skipUnless(all(name in available() for name in ('numpy', 'PIL', 'matplotlib')),
                     'image libraries not installed')
class ImageCompatibilityTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.database = self.tmp.name + '/agent.db'
        self.runtime = await Runtime(self.database, workers=2).__aenter__()
        self.agent = await self.runtime.agent()

    async def asyncTearDown(self):
        await self.runtime.close()
        self.tmp.cleanup()

    async def test_reviewed_type_checks_and_accurate_errors(self):
        self.assertEqual(await self.agent.execute('''
from PIL import Image
import numpy as np
image=Image.new('RGB',(2,2))
values=np.arange(3)
image_type=Image.Image
array_type=np.ndarray
(isinstance(image,image_type),isinstance(values,array_type),
 isinstance(image,array_type),isinstance(values,image_type),
 isinstance(1,array_type),isinstance(values,(image_type,array_type)),
 image_type.__name__,array_type.__name__)
'''), "(True, True, False, False, False, True, 'Image', 'ndarray')")
        session_id = self.agent.id
        await self.agent.close()
        self.agent = await self.runtime.agent(session_id=session_id)
        self.assertEqual(await self.agent.execute('(isinstance(image,image_type),isinstance(values,array_type))'), '(True, True)')
        with self.assertRaisesRegex(ExecutionError, "'Image' object has no attribute 'missing_method'"):
            await self.agent.execute('image.missing_method()')
        for code in ('image_type.cls', 'image_type.__mro__', 'array_type.__subclasses__()',
                     'isinstance(image,Image)', 'np.ndarray((10,))'):
            with self.subTest(code=code), self.assertRaises(ExecutionError):
                await self.agent.execute(code)

    async def test_image_array_operations_and_checkpoint(self):
        result = await self.agent.execute('''
import numpy as np
from PIL import Image
pixels=np.zeros((3,4,4), dtype=np.uint8)
pixels[...,0]=23
pixels[...,3]=255
image=Image.fromarray(pixels)
alias=image
image.putalpha(127)
merged=Image.merge('RGBA', image.split())
scaled=merged.resize((8,6), Image.LANCZOS)
(pixels.nbytes,len(pixels.tobytes()), image.getpixel((0,0)),
 merged.getpixel((1,1)), scaled.size, pixels[...,0].tolist())
''')
        self.assertEqual(result, '(48, 48, (23, 0, 0, 127), (23, 0, 0, 127), (8, 6), [[23, 23, 23, 23], [23, 23, 23, 23], [23, 23, 23, 23]])')
        session_id = self.agent.id
        await self.agent.close()
        self.agent = await self.runtime.agent(session_id=session_id)
        self.assertEqual(await self.agent.execute("image.putalpha(42); alias.getpixel((0,0))"), '(23, 0, 0, 42)')
        self.assertEqual(await self.agent.execute('pixels[...,1]=9; pixels[0,0].tolist()'), '[23, 9, 0, 255]')

    async def test_add_axes_and_safe_save_options(self):
        await self.agent.execute('''
import matplotlib.pyplot as plt
fig=plt.figure(figsize=(2,2))
ax=fig.add_axes([0.1,0.1,0.8,0.8], facecolor='white')
ax.plot([1,2],[3,4])
fig.savefig('chart.png', pil_kwargs={'compress_level': 1, 'optimize': True})
''')
        self.assertEqual(await self.agent.execute('len(fig.axes)'), '1')
        session_id = self.agent.id
        await self.agent.close()
        self.agent = await self.runtime.agent(session_id=session_id)
        await self.agent.execute("ax.set_title('restored'); fig.savefig('chart.png',pil_kwargs={'compress_level':9})")
        self.assertTrue((await self.agent.read_file(self.agent.home + '/chart.png')).startswith(b'\x89PNG'))
        for code in ("fig.savefig('bad.png',pil_kwargs={'pnginfo':'bad'})",
                     "fig.add_axes([0,0,1,1], projection='polar')",
                     "ax.text(0,0,'bad',usetex=True)"):
            with self.subTest(code=code), self.assertRaises(ExecutionError):
                await self.agent.execute(code)

    async def test_limits_are_configurable_without_allocating_large_images(self):
        from wrappers.numpy import checked_size as array_size
        from wrappers.pil import checked_size as image_size
        from wrappers.limits import memory_scope
        with memory_scope(100):
            image_size((4,5))
            with self.assertRaisesRegex(MemoryError, '100 bytes'):
                image_size((5,6))
            array_size((30,), 'uint8')
            array_size((100,), 'uint8')
            with self.assertRaisesRegex(MemoryError, '101 bytes'):
                array_size((101,), 'uint8')
            with self.assertRaisesRegex(MemoryError, '240 bytes'):
                array_size((30,), 'float64')
            with self.assertRaisesRegex(ValueError, 'object dtype'):
                array_size((1,), 'object')
        with memory_scope(120):
            image_size((5,6))

    async def test_numpy_scalars_and_virtual_path_sorting(self):
        self.assertEqual(await self.agent.execute('''
import numpy as np
from PIL import Image
from pathlib import Path
a=np.array([10,20], dtype=np.uint8)
im=Image.new('L',(2,2),int(a[0]))
(im.getpixel((0,0)), np.ones((2,2))[...,None].shape,
 [str(p) for p in sorted([Path('z'),Path('a')],key=str)])
'''), "(10, (2, 2, 1), ['a', 'z'])")
