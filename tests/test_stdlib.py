import tempfile
import unittest
from pathlib import Path

from agents import CodeAgent, ExecutionError
import test_agents


class StdlibTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_readonly_globals_snapshot_and_restart(self):
        self.assertEqual(await self.agent.execute('''
x=7
def inspect_globals():
    local_only=99
    return globals()['x'], 'local_only' in globals()
snapshot=globals()
x=8
(snapshot['x'],inspect_globals(),snapshot.get('missing',42))
'''), '(7, (8, False), 42)')
        self.assertEqual(await self.agent.execute("('x' in list(globals()),len(globals())>0,'x' in globals().keys())"), '(True, True, True)')
        self.assertEqual(await self.agent.execute("any(name.startswith(('_library_', '_Library')) for name in globals().keys())"), 'False')
        for source in ("globals()['x']=9", "del globals()['x']", "globals().clear()"):
            with self.subTest(source=source), self.assertRaises(ExecutionError):
                await self.agent.execute(source)
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute("del x; ('x' in globals(),snapshot['x'])"), '(False, 7)')

    async def test_file_operations_stat_and_restart(self):
        result = await self.agent.execute('''
import os, stat, shutil, tempfile
os.makedirs('docs',exist_ok=True)
open('docs/a.txt','w').write('hello')
shutil.copyfile('docs/a.txt','docs/b.txt')
shutil.copytree('docs','copied')
shutil.move('copied/b.txt','copied/c.txt')
temp=tempfile.mkdtemp(prefix='test-')
info=os.stat('docs/a.txt')
(info.st_size,stat.S_ISREG(info.st_mode),os.path.join('a','b'),os.path.exists('copied/c.txt'),os.getenv('HOME'))
''')
        self.assertEqual(result, repr((5, True, 'a/b', True, self.agent.home)))
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute("(info.st_size,os.path.isdir(temp),open('copied/c.txt').read())"), "(5, True, 'hello')")
        await self.agent.execute("shutil.rmtree('copied'); os.remove('docs/b.txt')")
        self.assertEqual(await self.agent.execute("os.listdir('docs')"), "['a.txt']")

    async def test_discovery_dynamic_import_and_introspection(self):
        result = await self.agent.execute('''
import importlib
from importlib import import_module
mod=import_module('os.path')
os=importlib.import_module('os')
math=importlib.import_module('math')
(importlib.util.find_spec('os').name,importlib.util.find_spec('not_installed_123'),
 mod.basename('/a/b'),math.sqrt(9),hasattr(os,'stat'),hasattr(os,'system'),callable(os.stat),callable(os),
 'path' in dir(os),getattr(os,'missing',42))
''')
        self.assertEqual(result, "('os', None, 'b', 3.0, True, False, True, False, True, 42)")

    async def test_resources_do_not_change_host_and_host_modules_are_blocked(self):
        self.assertEqual(await self.agent.execute('import resource; resource.getrlimit(resource.RLIMIT_AS)'),
                         repr((self.agent.memory_budget, self.agent.memory_budget)))
        for code in ("resource.setrlimit(resource.RLIMIT_AS,(1,1))", 'resource.getrusage(resource.RUSAGE_SELF)',
                     "import importlib; importlib.import_module('subprocess')", "import os; os.system('true')",
                     "import shutil; shutil.rmtree('/')"):
            with self.subTest(code=code), self.assertRaises(ExecutionError):
                await self.agent.execute(code)
        secret = Path(self.tmp.name) / 'host-secret'
        secret.write_text('secret')
        self.assertEqual(await self.agent.execute(f'import os; os.path.exists({str(secret)!r})'), 'False')
        with self.assertRaises(ExecutionError):
            await self.agent.execute(f"import shutil; shutil.copyfile({str(secret)!r}, 'leak')")
        self.assertEqual(await self.agent.execute("os.path.exists('leak')"), 'False')

    async def test_file_mutations_rollback(self):
        await self.agent.execute("import shutil; open('a','w').write('original')")
        with self.assertRaises(ExecutionError):
            await self.agent.execute("shutil.copyfile('a','b'); raise ValueError('abort')")
        self.assertEqual(await self.agent.execute("from pathlib import Path; Path('b').exists()"), 'False')

    async def test_common_builtin_helpers(self):
        self.assertEqual(await self.agent.execute('''
def example():
    return 42
(callable(example),callable(len),callable(1),getattr('a','missing','fallback'),hasattr('a','missing'))
'''), "(True, True, False, 'fallback', False)")
        self.assertEqual(await self.agent.execute('''
class Example:
    def __init__(self):
        self.x=7
obj=Example()
(vars(obj),getattr(obj,'x'),hasattr(obj,'x'),'append' in dir([]),'x' in dir(obj))
'''), "({'x': 7}, 7, True, True, True)")
