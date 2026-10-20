"""Curated submodule imports, durable objects and allocation boundaries."""
import tempfile
import asyncio
import subprocess
import sys
import unittest
from unittest.mock import patch

from agents import Runtime, ExecutionError
from wrappers.registry import available


class SubmoduleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.database = self.tmp.name + '/agent.db'
        self.runtime = await Runtime(self.database, workers=2).__aenter__()
        self.agent = await self.runtime.agent()

    async def asyncTearDown(self):
        await self.runtime.close()
        self.tmp.cleanup()

    def need(self, *names):
        if not all(name in available() for name in names):
            self.skipTest('optional packages not installed')

    async def reopen(self):
        session_id = self.agent.id
        await self.agent.close()
        self.agent = await self.runtime.agent(session_id=session_id)

    async def test_numpy_fft_imports_and_checkpoint(self):
        self.need('numpy')
        self.assertEqual(await self.agent.execute('''
import numpy as np
import numpy.fft as fft
from numpy.fft import rfft, irfft
spectrum = rfft([1.,2.,3.,4.])
(np.round(irfft(spectrum),6).tolist(), fft.fftfreq(4).tolist(),
 fft.fftshift(np.arange(4)).tolist(), np.round(fft.ifft(fft.fft([1.,2.])).real,6).tolist())
'''), '([1.0, 2.0, 3.0, 4.0], [0.0, 0.25, -0.5, -0.25], [2, 3, 0, 1], [1.0, 2.0])')
        await self.reopen()
        self.assertEqual(await self.agent.execute('np.round(irfft(spectrum),6).tolist()'), '[1.0, 2.0, 3.0, 4.0]')
        from wrappers.numpy import fft_transform
        with patch('wrappers.numpy.np.fft.fft', side_effect=AssertionError('native allocation reached')) as native:
            with self.assertRaises(MemoryError):
                fft_transform(native)([1], n=100_000_000)

    async def test_scipy_sampled_integration_and_rotations(self):
        self.need('scipy', 'numpy')
        self.assertEqual(await self.agent.execute('''
import numpy as np
from scipy.integrate import trapezoid, simpson, cumulative_trapezoid, cumulative_simpson
from scipy.spatial.transform import Rotation
rotation = Rotation.from_euler('z',90,degrees=True)
(float(trapezoid([0.,1.,2.])),float(simpson([0.,1.,4.])),
 cumulative_trapezoid([0.,1.,2.],initial=0).tolist(),
 cumulative_simpson([0.,1.,4.],initial=0).shape,
 np.round(rotation.apply([1.,0.,0.]),6).tolist())
'''), '(2.0, 2.6666666666666665, [0.0, 0.5, 2.0], (3,), [0.0, 1.0, 0.0])')
        await self.reopen()
        self.assertEqual(await self.agent.execute('np.round(rotation.inv().apply([0.,1.,0.]),6).tolist()'), '[1.0, 0.0, 0.0]')
        with self.assertRaises(ExecutionError):
            await self.agent.execute('from scipy.integrate import quad')

    async def test_pillow_submodules_objects_and_virtual_save(self):
        self.need('PIL')
        self.assertEqual(await self.agent.execute('''
from PIL import Image, ImageOps, ImageEnhance
import PIL.ImageFilter as filters
image = Image.new('RGB',(4,2),(100,50,20))
contrast = ImageEnhance.Brightness(image)
blur = filters.GaussianBlur(radius=1)
processed = ImageOps.mirror(contrast.enhance(2)).filter(blur)
processed.save('processed.png')
(processed.getpixel((0,0)), ImageOps.expand(image,border=(1,2)).size,
 ImageOps.fit(image,(2,2)).size, ImageOps.pad(image,(3,3)).size,
 ImageOps.contain(image,(2,2)).size, ImageOps.grayscale(image).mode,
 image.filter(filters.MedianFilter(size=3)).size, image.filter(filters.SHARPEN).size)
'''), "((200, 100, 40), (6, 6), (2, 2), (3, 3), (2, 1), 'L', (4, 2), (4, 2))")
        self.assertTrue((await self.agent.read_file(self.agent.home + '/processed.png')).startswith(b'\x89PNG'))
        await self.reopen()
        self.assertEqual(await self.agent.execute('contrast.enhance(0.5).filter(blur).getpixel((0,0))'), '(50, 25, 10)')
        for code in ('ImageOps.expand(image,1000000)', 'ImageOps.pad(image,(100000,100000))',
                     'ImageOps.cover(Image.new("RGB",(100,1)),(1,100000))',
                     'filters.GaussianBlur(radius=100000)', 'filters.MedianFilter(size=2)',
                     'from PIL.ImageFilter import Kernel'):
            with self.subTest(code=code), self.assertRaises(ExecutionError):
                await self.agent.execute(code)

    async def test_matplotlib_color_submodule(self):
        self.need('matplotlib')
        self.assertEqual(await self.agent.execute('''
from matplotlib.colors import to_rgb, to_rgba, to_hex, is_color_like, rgb_to_hsv, hsv_to_rgb
(to_rgb('red'),to_rgba('blue',0.5),to_hex((0.,1.,0.)),is_color_like('bad color'),
 rgb_to_hsv([1.,0.,0.]).tolist(),hsv_to_rgb([0.,1.,1.]).tolist())
'''), "((1.0, 0.0, 0.0), (0.0, 0.0, 1.0, 0.5), '#00ff00', False, [0.0, 1.0, 1.0], [1.0, 0.0, 0.0])")

    async def test_matplotlib_use_agg_does_not_change_host_backend(self):
        self.need('matplotlib')
        with patch('matplotlib.use', side_effect=AssertionError('host backend changed')):
            self.assertEqual(await self.agent.execute('''
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
matplotlib.use('agg', force=False)
plt.plot([1,2],[3,4])
plt.savefig('agg.png')
(matplotlib.get_backend(), plt.is_interactive())
'''), "('Agg', False)")
            for backend in ('TkAgg', 'QtAgg', 'module://os', 'svg', None):
                with self.subTest(backend=backend), self.assertRaises(ExecutionError):
                    await self.agent.execute(f'matplotlib.use({backend!r}, force=False)')
        self.assertTrue((await self.agent.read_file(self.agent.home + '/agg.png')).startswith(b'\x89PNG'))
        await self.reopen()
        self.assertEqual(await self.agent.execute("matplotlib.use('Agg'); matplotlib.get_backend()"), "'Agg'")

    async def test_image_draw_mutates_shared_image_and_survives_restart(self):
        self.need('PIL')
        self.assertEqual(await self.agent.execute('''
from PIL import Image, ImageDraw
image=Image.new('RGB',(64,32),'white')
draw=ImageDraw.Draw(image)
draw.rectangle((0,0,10,10),fill='red')
draw.line([(20,0),(20,20)],fill='blue',width=2)
draw.ellipse((30,0,40,10),fill='green')
draw.text((0,15),'hello',fill='black',font_size=10)
image.save('drawing.png')
(image.getpixel((1,1)),image.getpixel((20,1)),image.getpixel((35,5)),draw.textbbox((0,0),'hello')[2]>0)
'''), '((255, 0, 0), (0, 0, 255), (0, 128, 0), True)')
        await self.reopen()
        self.assertEqual(await self.agent.execute("draw.point((1,1),fill='blue'); image.getpixel((1,1))"), '(0, 0, 255)')
        # Multiple drawing views and image writes retain one authenticated root.
        self.assertEqual(await self.agent.execute("other=ImageDraw.Draw(image); image.putpixel((2,2),(1,2,3)); other.point((3,3),fill='black'); (image.getpixel((2,2)),image.getpixel((3,3)))"), '((1, 2, 3), (0, 0, 0))')
        with self.assertRaises(ExecutionError):
            await self.agent.execute("draw.point((1,1),fill='red'); raise ValueError('rollback')")
        self.assertEqual(await self.agent.execute('image.getpixel((1,1))'), '(255, 0, 0)')
        for code in ("draw.text((0,0),'bad',font='/etc/passwd')",
                     "draw.text((0,0),'bad','black','/etc/passwd')",
                     "draw.text((0,0),'bad',font_size=1000000)",
                     "draw.line([(0,0),(1,1)],'black',1000000)",
                     "draw.regular_polygon((10,10,5),1000000000)",
                     "draw.text((0,0),'x'*1000,font_size=512)", 'draw._image'):
            await self.agent.execute("image=Image.new('RGB',(64,32)); draw=ImageDraw.Draw(image)")
            with self.subTest(code=code), self.assertRaises(ExecutionError):
                await self.agent.execute(code)

    async def test_fft_and_cross_library_arrays_reopen_in_fresh_process(self):
        self.need('numpy', 'matplotlib')
        await self.agent.execute('import numpy.fft as fft; from matplotlib.colors import rgb_to_hsv')
        session_id = self.agent.id
        await self.runtime.close()
        program = '''
import asyncio, sys
from agents import Runtime
async def main():
    async with Runtime(sys.argv[1],workers=1) as runtime:
        agent=await runtime.agent(session_id=int(sys.argv[2]))
        print(await agent.execute('rgb_to_hsv([1.,0.,0.]).tolist()'))
        print(await agent.execute('fft.fft([1.,0.]).real.tolist()'))
asyncio.run(main())
'''
        result = await asyncio.to_thread(subprocess.run,
            [sys.executable, '-X', 'faulthandler', '-c', program, self.database, str(session_id)],
            capture_output=True, text=True, timeout=30)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.splitlines(), ['[0.0, 1.0, 1.0]', '[1.0, 1.0]'])

