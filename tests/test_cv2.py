"""OpenCV through real Monty: array interoperability, virtual IO and denied APIs."""
import unittest
from unittest.mock import patch

import test_agents
from agents import ExecutionError
from wrappers import registry


class Cv2DiscoveryTests(unittest.TestCase):
    def test_cv2_requires_host_package_and_dependencies(self):
        for missing in ('cv2', 'numpy', 'PIL'):
            registry.available.cache_clear()
            try:
                with patch.object(registry, 'find_spec', side_effect=lambda name: None if name == missing else object()):
                    self.assertNotIn('cv2', registry.available())
            finally:
                registry.available.cache_clear()


@unittest.skipUnless('cv2' in registry.available(), 'OpenCV not installed')
class Cv2Tests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_writes_never_touch_host_paths(self):
        from pathlib import Path
        target = self.tmp.name + '/not-a-host-image.png'
        await self.agent.execute(f'''
import cv2, numpy as np, os
os.makedirs({self.tmp.name!r},exist_ok=True)
cv2.imwrite({target!r},np.zeros((3,3,3),dtype=np.uint8))
''')
        self.assertFalse(Path(target).exists())
        self.assertTrue((await self.agent.read_file(target)).startswith(b'\x89PNG'))

    async def test_image_roundtrip_and_numpy_interoperability(self):
        import cv2
        self.assertEqual(await self.agent.execute('''
import cv2
import numpy as np
image=np.zeros((12,16,3),dtype=np.uint8)
cv2.rectangle(image,(2,2),(8,8),(0,0,255),-1)
written=cv2.imwrite('sample.png',image)
loaded=cv2.imread('sample.png')
gray=cv2.cvtColor(loaded,cv2.COLOR_BGR2GRAY)
small=cv2.resize(loaded,(8,6))
(written,loaded.shape,small.shape,int(gray[3,3]),loaded[3,3].tolist(),cv2.__version__)
'''), repr((True,(12,16,3),(6,8,3),76,[0,0,255],cv2.__version__)))
        self.assertTrue((await self.agent.read_file(self.agent.home+'/sample.png')).startswith(b'\x89PNG'))
        self.assertEqual(await self.agent.execute("cv2.imread('missing.png') is None"), 'True')

    async def test_encoding_filters_contours_and_scaling(self):
        self.assertEqual(await self.agent.execute('''
import cv2, numpy as np
image=np.zeros((16,16),dtype=np.uint8)
cv2.rectangle(image,(3,3),(11,11),255,-1)
ok,data=cv2.imencode('.png',image)
decoded=cv2.imdecode(data,cv2.IMREAD_GRAYSCALE)
blurred=cv2.GaussianBlur(decoded,(3,3),0)
mask=cv2.threshold(blurred,127,255,cv2.THRESH_BINARY)[1]
contours,hierarchy=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
scaled=cv2.resize(image,None,fx=0.5,fy=0.5)
(ok,np.array_equal(decoded,image),len(contours),scaled.shape,cv2.contourArea(contours[0])>0)
'''), '(True, True, 1, (8, 8), True)')

    async def test_host_paths_devices_and_callbacks_are_not_exposed(self):
        await self.agent.execute('import cv2, numpy as np; image=np.zeros((2,2,3),dtype=np.uint8)')
        for name in ('VideoCapture','VideoWriter','imshow','waitKey','FileStorage','CascadeClassifier',
                     'dnn','utils','samples','getBuildInformation','setMouseCallback','redirectError','UMat'):
            with self.subTest(name=name), self.assertRaises(ExecutionError):
                await self.agent.execute('cv2.'+name)
        self.assertEqual(await self.agent.execute("cv2.imread('/etc/passwd') is None"), 'True')
        self.assertEqual(await self.agent.execute("cv2.imread('https://example.com/image.png') is None"), 'True')
        with self.assertRaises(ExecutionError):
            await self.agent.execute('cv2.imdecode(np.zeros(10,dtype=np.uint8),cv2.IMREAD_LOAD_GDAL)')

    async def test_known_large_outputs_are_checked_before_native_allocation(self):
        await self.agent.execute('import cv2, numpy as np; image=np.zeros((2,2,3),dtype=np.uint8)')
        for code in ('cv2.resize(image,(100000,100000))',
                     'cv2.getStructuringElement(cv2.MORPH_RECT,(100000,100000))',
                     'cv2.calcHist([image],[0,1],None,[100000,100000],[0,256,0,256])',
                     'cv2.GaussianBlur(image,(100000,100000),0)'):
            with self.subTest(code=code), self.assertRaises(ExecutionError):
                await self.agent.execute(code)
