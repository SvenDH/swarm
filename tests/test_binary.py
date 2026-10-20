"""Safe stdlib operations through real Monty checkpoints and rollback."""
import hashlib
import unittest
import zlib

import test_agents
from agents import CodeAgent, ExecutionError
from wrappers.binary import Adapter
from wrappers.limits import memory_scope


class BinaryTests(unittest.IsolatedAsyncioTestCase):
    asyncSetUp = test_agents.AgentTests.asyncSetUp
    asyncTearDown = test_agents.AgentTests.asyncTearDown

    async def test_native_dynamic_import_and_discovery(self):
        self.assertEqual(await self.agent.execute('''
import sys, pathlib, importlib
(__import__('sys').version==sys.version,
 importlib.import_module('json').loads('{"a":1}'),
 importlib.import_module('base64').b64encode(b'hi'),
 importlib.import_module('binascii').hexlify(b'hi'),
 importlib.util.find_spec('time').name,
 'version' in dir(sys),'capitalize' in dir(sys),'Path' in dir(pathlib),
 vars(sys)['version']==sys.version)
'''), "(True, {'a': 1}, b'aGk=', b'6869', 'time', True, False, True, True)")
        for source in ("__import__('subprocess')", "__import__('json.dumps')",
                       "importlib.import_module('os.system')", "__import__('sys').modules"):
            with self.subTest(source=source), self.assertRaises(ExecutionError):
                await self.agent.execute(source)

    async def test_binary_round_trips_and_checkpoint(self):
        self.assertEqual(await self.agent.execute('''
import struct, zlib, hashlib, array, io
fmt=struct.Struct('>Ih')
data=fmt.pack(123, -4)
h=hashlib.sha256(data)
digest=h.hexdigest()
buffer=io.BytesIO(data)
text=io.StringIO('hello\\nworld')
numbers=array.array('i',[1,2,3])
numbers.append(4)
(fmt.unpack(zlib.decompress(zlib.compress(data))), buffer.read(4),text.readline(),numbers.tolist())
'''), "((123, -4), b'\\x00\\x00\\x00{', 'hello\\n', [1, 2, 3, 4])")
        await self.agent.close()
        self.agent = await CodeAgent.open(self.path, session_id=self.session_id)
        self.assertEqual(await self.agent.execute('''
buffer.seek(0,2)
buffer.write(b'!')
h.update(b'!')
copy=h.copy()
numbers[1]=9
numbers.extend([5,6])
(h.hexdigest()==copy.hexdigest(), h.hexdigest()!=digest,
 text.read(),numbers.tolist(),buffer.getvalue()[-1])
'''), "(True, True, 'world', [1, 9, 3, 4, 5, 6], 33)")
        with self.assertRaises(ExecutionError):
            await self.agent.execute("h.update(b'bad'); buffer.write(b'bad'); numbers.append(7); raise ValueError('abort')")
        self.assertEqual(await self.agent.execute('(h.hexdigest()==copy.hexdigest(),len(numbers),buffer.getvalue()[-1])'), '(True, 6, 33)')
        await self.agent.execute('buffer.close(); text.close()')
        self.assertEqual(await self.agent.execute('(buffer.closed,text.closed)'), '(True, True)')
        with self.assertRaises(ExecutionError):
            await self.agent.execute('buffer.getvalue()')

    async def test_hash_and_array_apis_and_denied_host_access(self):
        self.assertEqual(await self.agent.execute("import hashlib; hashlib.sha256(b'abc').hexdigest()"),
                         repr(hashlib.sha256(b'abc').hexdigest()))
        self.assertEqual(await self.agent.execute('''
import array, struct, io
a=array.array('h',[1,2]); b=array.array('h'); b.frombytes(a.tobytes())
b.insert(1,3); b.reverse(); b.fromlist([4]); text=io.StringIO()
text.writelines(['a','b']); text.seek(0)
(b.tolist(),(a*2).tolist(),a[1:].tolist(),text.read(),struct.unpack_from('>H',b'xx\\x00\\x07',2))
'''), "([2, 3, 1, 4], [1, 2, 1, 2], [2], 'ab', (7,))")
        for source in ("io.open('/etc/passwd')", 'io.FileIO(0)', 'io.BytesIO().fileno()',
                       "a.fromfile('secret',1)", "hashlib.file_digest('secret','sha256')",
                       "hashlib.new('unsupported')", 'a.value', 'hashlib.sha256().data'):
            with self.subTest(source=source), self.assertRaises(ExecutionError):
                await self.agent.execute(source)
        self.assertEqual(await self.agent.execute('''
alias=a
a*=2
a+=array.array('h',[3])
(alias.tolist(),a==alias,(a+array.array('h',[4])).tolist())
'''), '([1, 2, 1, 2, 3], True, [1, 2, 1, 2, 3, 4])')

    async def test_text_newline_policy_survives_calls(self):
        await self.agent.execute("import io; text=io.StringIO('a\\nb',newline='\\r\\n')")
        self.assertEqual(await self.agent.execute('text.getvalue()'), repr('a\r\nb'))
        await self.agent.execute("text.seek(0,2); text.write('\\nc')")
        self.assertEqual(await self.agent.execute('text.getvalue()'), repr('a\r\nb\r\nc'))


class BinaryBudgetTests(unittest.TestCase):
    def test_allocations_and_compressed_bombs_are_bounded(self):
        compressed = zlib.compress(b'x' * 4096)
        with memory_scope(128):
            io_api = Adapter('io').exports(None)
            buffer = io_api['BytesIO']()
            buffer.stream.seek(128)
            h = Adapter('hashlib').exports(None)['sha256'](b'a'*120)
            a = Adapter('array').exports(None)['array']('b', [1]*120)
            for operation in (lambda: buffer.write(b'x'), lambda: h.update(b'x'*9),
                              lambda: a.grow('extend', [1]*9), lambda: a*2,
                              lambda: Adapter('struct').exports(None)['pack']('129s', b''),
                              lambda: Adapter('zlib').exports(None)['decompress'](compressed),
                              lambda: Adapter('io').members(buffer,None)['read'](129),
                              lambda: Adapter('hashlib').exports(None)['shake_128']().digest(129)):
                with self.subTest(operation=operation), self.assertRaises(MemoryError):
                    operation()

