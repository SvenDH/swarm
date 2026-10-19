import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agents import Runtime, CodeAgent


class TemplateTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.template = self.root / 'template'
        self.template.mkdir()
        (self.template / 'notes').mkdir()
        (self.template / 'empty').mkdir()
        (self.template / 'notes' / 'hello.txt').write_text('hello', encoding='utf-8')
        (self.template / '.settings').write_bytes(b'\x00\xff')
        self.runtime = await Runtime(self.root / 'agent.db', template=self.template, workers=1).__aenter__()

    async def asyncTearDown(self):
        await self.runtime.close()
        self.tmp.cleanup()

    async def test_nested_binary_and_empty_directories_are_durable_and_private(self):
        a, b = await asyncio.gather(self.runtime.agent(), self.runtime.agent())
        for agent in (a, b):
            self.assertEqual(await agent.execute("open('notes/hello.txt').read()"), "'hello'")
            self.assertEqual(await agent.read_file(agent.home + '/.settings'), b'\x00\xff')
            self.assertEqual(await agent.execute("from pathlib import Path; Path('empty').is_dir()"), 'True')
            self.assertEqual((agent.workspace / agent.home.lstrip('/') / '.settings').read_bytes(), b'\x00\xff')
        await a.execute("open('notes/hello.txt','w').write('changed')")
        self.assertEqual(await b.read_file(b.home + '/notes/hello.txt'), b'hello')
        self.assertEqual((self.template / 'notes' / 'hello.txt').read_text(), 'hello')
        entry = next(f for f in await b.files() if f['path'].endswith('/notes/hello.txt'))
        self.assertEqual(self.runtime.storage.read(entry['storage_key']), b'hello')

    async def test_template_updates_only_affect_new_sessions_and_recovery_needs_no_source(self):
        first = await self.runtime.agent()
        (self.template / 'notes' / 'hello.txt').write_text('updated')
        second = await self.runtime.agent()
        self.assertEqual(await first.read_file(first.home + '/notes/hello.txt'), b'hello')
        self.assertEqual(await second.read_file(second.home + '/notes/hello.txt'), b'updated')
        await first.execute("open('notes/hello.txt','w').write('my edit')")
        shutil.rmtree(self.template)
        reopened = await self.runtime.agent(session_id=first.id)
        self.assertEqual(await reopened.read_file(first.home + '/notes/hello.txt'), b'my edit')
        shutil.rmtree(reopened.workspace)
        with patch.object(CodeAgent, '_evaluate', side_effect=AssertionError('recovery ran code')):
            reopened = await self.runtime.agent(session_id=first.id)
        self.assertEqual(await reopened.read_file(first.home + '/notes/hello.txt'), b'my edit')
        with self.assertRaises(NotADirectoryError):
            await self.runtime.agent()

    async def test_invalid_template_does_not_create_partial_session(self):
        outside = self.root / 'secret'
        outside.write_text('not a starter file')
        (self.template / 'link').symlink_to(outside)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            await self.runtime.agent()
        count = await self.runtime.database_call(lambda: self.runtime.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0])
        self.assertEqual(count, 0)
        (self.template / 'link').unlink()
        agent = await self.runtime.agent()
        self.assertEqual(agent.id, 0)

    async def test_oversized_file_is_rejected_before_creation(self):
        self.runtime.memory_budget = 1024**2
        with (self.template / 'huge').open('wb') as file:
            file.truncate(1024**2 + 1)
        with self.assertRaisesRegex(MemoryError, 'memory budget'):
            await self.runtime.agent()
        count = await self.runtime.database_call(lambda: self.runtime.db.execute('SELECT COUNT(*) FROM sessions').fetchone()[0])
        self.assertEqual(count, 0)
