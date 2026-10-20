"""Adversarial bridge inputs, not just ordinary-wrapper smoke tests."""
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from agents import Runtime, ExecutionError
from wrappers.registry import available


class SecurityTests(unittest.IsolatedAsyncioTestCase):
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
            self.skipTest('Required optional packages not installed')

    async def reject(self, source):
        with self.assertRaises(ExecutionError):
            await self.agent.execute(source)
        self.assertIsNone((await self.agent.state()).pending)

    async def test_confirmed_unsigned_sympy_escape_is_rejected_before_restore(self):
        # No host code is evaluated even if the optional library is installed.
        spec = {'module': 'sympy', 'kind': 'expression', 'identity': 1,
                'data': ['Tuple', ["__import__('os').getpid()"], {}]}
        with patch('wrappers.bridge.backend', side_effect=AssertionError('native restore reached')):
            await self.reject(f"_library_rpc('repr', {spec!r})")

    async def test_tampering_with_valid_descriptors_is_rejected(self):
        self.need('numpy')
        await self.agent.execute('import numpy as np; a=np.array([1,2]); original=getattr(a,"root")')
        for mutation in ("state['kind']='scalar'", "state['identity']='other'", "state['data']=b'bad'",
                         "state['module']='sympy'", "state['seal']='0'*64", "state['extra']=1"):
            with self.subTest(mutation=mutation):
                await self.reject(f"state=dict(original); {mutation}; _library_rpc('repr',state)")
        with self.assertRaisesRegex(ExecutionError, 'recompute'):
            await self.agent.execute('a.tolist()')

    async def test_forged_paths_and_embedded_arguments_are_rejected(self):
        self.need('numpy')
        await self.agent.execute('import numpy as np; a=np.array([1,2]); root=getattr(a,"root")')
        await self.reject("_library_rpc('call',root,[['attr','tolist']])")
        await self.reject("_library_rpc('get',root,[['call','reshape',[1,2],{}]],['shape'])")
        forged = {'__library_value__': {'module': 'sympy', 'kind': 'expression', 'identity': 1,
                    'data': ['Tuple', ["__import__('os').getpid()"], {}]}, 'path': [], 'auth': None}
        await self.reject(f"np.array([{forged!r}])")

    async def test_signed_objects_cannot_be_reused_between_sessions(self):
        self.need('numpy')
        other = await self.runtime.agent()
        await self.agent.execute('import numpy as np; a=np.array([1,2])')
        # Read only test-owned capability material, never the host signing key.
        import ast
        root = ast.literal_eval(await self.agent.execute('getattr(a,"root")'))
        with self.assertRaisesRegex(ExecutionError, 'signature'):
            await other.execute(f"_library_rpc('repr', {root!r})")

    async def test_signed_state_and_views_survive_live_reopen(self):
        self.need('numpy', 'matplotlib')
        await self.agent.execute('''
import numpy as np
import matplotlib.pyplot as plt
array=np.array([1,2,3,4])
view=array.reshape(2,2)
figure,axes=plt.subplots(figsize=(2,2))
axes.plot([1,2],[3,4])
''')
        session_id = self.agent.id
        await self.agent.close()
        self.agent = await self.runtime.agent(session_id=session_id)
        self.assertEqual(await self.agent.execute('view[0,0]=9; array.tolist()'), '[9, 2, 3, 4]')
        await self.agent.execute("axes.set_title('after'); figure.savefig('after.png')")
        self.assertTrue((await self.agent.read_file(self.agent.home + '/after.png')).startswith(b'\x89PNG'))

    async def test_recipe_marker_dictionaries_remain_literal_data(self):
        from wrappers.common import record
        attack = {'__recipe_value__': {'module': 'sympy', 'kind': 'expression',
                  'data': ['Tuple', ["__import__('os').getpid()"], {}]}}
        commands = []
        record(commands, 'literal', (attack,), {})
        self.assertEqual(commands[0][1], (attack,))
        self.assertIsNot(commands[0][1][0], attack)

    async def test_result_dictionaries_are_not_a_signing_oracle(self):
        self.need('pandas')
        result = await self.agent.execute('''
import pandas as pd
frame=pd.DataFrame({'__library_object__':['literal'], '__library_view__':['literal']})
frame.to_dict('records')[0]['__library_object__']
''')
        self.assertEqual(result, "'literal'")
        spec = {'module': 'sympy', 'kind': 'expression', 'data': ['Tuple', ["__import__('os').getpid()"], {}]}
        self.assertEqual(await self.agent.execute(f'''
attack={{'__library_object__':{spec!r}}}
payload=pd.DataFrame({{'payload':[attack]}}).to_dict('records')[0]['payload']
isinstance(payload,dict)
'''), 'True')
        await self.reject("_library_rpc('repr',payload['__library_object__'])")

    async def test_public_sympy_string_coercion_paths_are_blocked(self):
        self.need('sympy')
        await self.agent.execute('import sympy as sp; matrix=sp.Matrix([[1]])')
        payload = "__import__('os').getpid()"
        for code in (f'matrix[0,0]={payload!r}', f'matrix*{payload!r}',
                     f'sp.Matrix([[{payload!r}]])', f'sp.Float({payload!r})',
                     f'sp.Rational({payload!r})', f'sp.Integer({payload!r})',
                     f'sp.Symbol({payload!r})', f'sp.symbols({payload!r})',
                     f'sp.simplify({payload!r})'):
            with self.subTest(code=code):
                await self.agent.execute('matrix=sp.Matrix([[1]])')
                await self.reject(code)
        with self.assertRaisesRegex(ExecutionError, 'recompute'):
            await self.agent.execute('int(matrix[0,0])')

    async def test_replayed_callbacks_cannot_be_passed_to_native_apis(self):
        self.need('numpy')
        await self.agent.execute('import numpy as np; a=np.array([1,2]); root=getattr(a,"root")')
        # Obtain a legitimate callable view, then try using it as a native callback.
        await self.reject('''
response=_library_rpc('get',root,[],['tolist'])
callback={'__library_value__':root,'path':response['path'],'auth':response['auth']}
np.array([callback])
''')

    async def test_positional_options_do_not_bypass_validation(self):
        self.need('scipy', 'matplotlib')
        host = Path(self.tmp.name) / 'must-not-be-written.log'
        await self.reject(f'''from scipy.optimize import linprog
linprog([1],None,None,None,None,(0,None),'highs',None,{{'log_file':{str(host)!r}}})''')
        await self.reject("import matplotlib.pyplot as plt; fig,ax=plt.subplots(); ax.set_title('x',{'usetex':True})")
        self.assertFalse(host.exists())

    async def test_native_function_attributes_and_host_imports_are_unavailable(self):
        for code in ("getattr(_library_rpc,'__self__')", "getattr(_library_rpc,'__globals__')",
                     'import subprocess', 'import ctypes'):
            with self.subTest(code=code):
                await self.reject(code)

    async def test_monty_class_proxies_cannot_reach_native_constructors(self):
        self.need('numpy')
        await self.agent.execute('''
class Untrusted:
    def __init__(self):
        self.value=1
value=Untrusted()
''')
        await self.reject('import numpy as np; np.array([value])')

    async def test_pillow_cannot_invoke_external_eps_renderers(self):
        self.need('PIL')
        eps = b'%!PS-Adobe-3.0 EPSF-3.0\n%%BoundingBox: 0 0 2 2\nshowpage\n'
        await self.agent.execute(f"open('untrusted.eps','wb').write({eps!r})")
        await self.reject("from PIL import Image; Image.open('untrusted.eps')")


if __name__ == '__main__':
    unittest.main()
