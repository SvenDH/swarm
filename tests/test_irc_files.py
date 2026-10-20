"""Durable attachment snapshots, explicit receiving and workspace boundaries."""
import ast
import tempfile
import unittest
from unittest.mock import patch

from agents import Worker, ExecutionError


SETUP = "from irc.client import Reactor; r=Reactor(); c=r.server().connect('worker',6667)"


class IRCFileTests(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_announcement_explicit_receive_and_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            async with Worker(directory + '/agent.db') as worker:
                alice, bob = await worker.runtime.agent(), await worker.runtime.agent()
                await alice.execute(SETUP + "; c.join('#files')")
                await bob.execute(SETUP + "; c.join('#files')")
                reference = ast.literal_eval(await alice.execute(
                    "open('source.bin','wb').write(b'\\x00original\\xff'); ref=c.send_file('#files','source.bin'); ref"))
                self.assertTrue(reference.startswith('irc-file:'))
                events = await bob.execute("seen=[]; r.add_global_handler('pubmsg',lambda c,e: seen.append(e.arguments)); r.process_once(); seen")
                self.assertIn('[file] ' + reference, events)
                self.assertNotIn('source.bin', [row['path'].split('/')[-1] for row in await bob.files()])
                info = ast.literal_eval(await bob.execute(f"c.file_info({reference!r})"))
                self.assertEqual((info['owner'], info['filename'], info['size']), (alice.name, 'source.bin', 10))
                self.assertNotIn('storage_key', info)
                await alice.execute("open('source.bin','wb').write(b'changed')")
                await alice.execute("import os; os.remove('source.bin')")
                self.assertIn('/received.bin', await bob.execute(f"c.receive_file({reference!r},'received.bin')"))
                self.assertEqual(await bob.execute("open('received.bin','rb').read()"), "b'\\x00original\\xff'")
                await bob.execute(f"ns={{'c':c}}; exec(\"c.receive_file({reference!r},'from-exec.bin')\",ns)")
                self.assertEqual(await bob.execute("open('from-exec.bin','rb').read()"), "b'\\x00original\\xff'")
                with self.assertRaisesRegex(ExecutionError, 'FileExistsError'):
                    await bob.execute(f"c.receive_file({reference!r},'received.bin')")
                await bob.execute("open('received.bin','wb').write(b'local')")
                await bob.execute(f"c.receive_file({reference!r},'received.bin',overwrite=True)")
                self.assertEqual(await bob.execute("open('received.bin','rb').read()"), "b'\\x00original\\xff'")

    async def test_restart_deduplication_and_worker_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/a/agent.db'
            async with Worker(path) as worker:
                alice, bob = await worker.runtime.agent(), await worker.runtime.agent()
                ids = alice.id, bob.id
                await alice.execute(SETUP)
                reference = ast.literal_eval(await alice.execute(
                    "open('data.txt','w').write('persisted'); ref=c.send_file('nobody','data.txt'); ref"))
                self.assertEqual(ast.literal_eval(await alice.execute("c.send_file('nobody','data.txt')")), reference)
                self.assertEqual((await worker.runtime.query('SELECT count(*) AS n FROM irc_attachments'))[0]['n'], 1)
            async with Worker(path) as worker, Worker(directory + '/b/agent.db') as other:
                bob = await worker.runtime.agent(session_id=ids[1])
                await bob.execute(SETUP)
                await bob.execute(f"c.receive_file({reference!r},'restored.txt')")
                self.assertEqual(await bob.execute("open('restored.txt').read()"), "'persisted'")
                isolated = await other.runtime.agent()
                await isolated.execute(SETUP)
                with self.assertRaisesRegex(ExecutionError, 'Unknown attachment'):
                    await isolated.execute(f"c.receive_file({reference!r},'no.txt')")

    async def test_invalid_references_host_paths_and_failed_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            async with Worker(directory + '/agent.db') as worker:
                agent = await worker.runtime.agent()
                await agent.execute(SETUP)
                for reference in ('/etc/passwd', 'irc-file:../private', 'irc-file:' + '0'*64):
                    with self.subTest(reference=reference), self.assertRaises(ExecutionError):
                        await agent.execute(f"c.receive_file({reference!r},'no.txt')")
                with self.assertRaises(ExecutionError):
                    await agent.execute("c.send_file('nobody','/etc/passwd')")
                await agent.execute("open('ok.txt','w').write('safe')")
                original = worker.runtime.storage.put
                failures = [True]
                def fail_once(name, data):
                    if failures:
                        failures.pop()
                        raise OSError('storage offline')
                    return original(name, data)
                with patch.object(worker.runtime.storage, 'put', side_effect=fail_once):
                    with self.assertRaisesRegex(ExecutionError, 'storage offline'):
                        await agent.execute("c.send_file('nobody','ok.txt')")
                self.assertEqual((await worker.runtime.query('SELECT count(*) AS n FROM irc_attachments'))[0]['n'], 0)
