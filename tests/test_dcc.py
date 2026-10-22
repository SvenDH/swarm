"""Library-shaped DCC over recipient-bound in-memory channels."""
import ast
import asyncio
import tempfile
import unittest
from unittest.mock import patch

from agents import Worker, ExecutionError
from wrappers.dcc import DCC


SETUP = "import irc.client; r=irc.client.Reactor(); c=r.server().connect('worker',6667)"


class DCCTests(unittest.IsolatedAsyncioTestCase):
    async def test_no_network_and_atomic_backpressure(self):
        broker = DCC()
        with patch('socket.socket', side_effect=AssertionError('DCC opened a socket')), \
             patch('asyncio.start_server', side_effect=AssertionError('DCC opened a listener')), \
             patch('asyncio.open_connection', side_effect=AssertionError('DCC opened TCP')):
            sender = await broker.call('alice', 'listen')
            broker.offer('alice', 'bob', f"\x01DCC SEND x 2130706433 {sender['localport']} 262148\x01")
            receiver = await broker.call('bob', 'connect', None, ('127.0.0.1', sender['localport']))
            for _ in range(4):
                await broker.call('alice', 'send', sender['id'], b'x' * 65536)
            with self.assertRaises(BufferError):
                await broker.call('alice', 'send', sender['id'], b'tail')
            self.assertEqual(broker.peers[sender['id']]['sent'], 262144)
            events = await broker.call('bob', 'poll', receiver['id'])
            self.assertEqual(sum(len(data) for kind, data in events if kind == 'dccmsg'), 262144)
            await broker.call('alice', 'send', sender['id'], b'tail')
            await broker.call('alice', 'close', sender['id'])
            self.assertEqual(await broker.call('bob', 'poll', receiver['id']),
                             [('dccmsg', b'tail'), ('dcc_disconnect', None)])
            await broker.close()
            self.assertEqual(broker.peers, {})

    async def test_streaming_quotas_and_cross_worker_isolation(self):
        with tempfile.TemporaryDirectory() as directory:
            async with Worker(directory + '/one/agent.db') as worker, Worker(directory + '/two/agent.db') as other:
                alice, bob = await worker.runtime.agent(), await worker.runtime.agent()
                isolated = await other.runtime.agent()
                for agent in (alice, bob, isolated):
                    await agent.execute(SETUP)
                await alice.execute("d=r.dcc('raw').listen()")
                port = int(await alice.execute('d.localport'))
                await alice.execute(f"c.ctcp('DCC',{bob.name!r},'SEND data 2130706433 {port} 131072')")
                with self.assertRaisesRegex(ExecutionError, 'No DCC offer'):
                    await isolated.execute(f"r.dcc('raw').connect('127.0.0.1',{port})")
                await bob.execute(f"d=r.dcc('raw').connect('127.0.0.1',{port}); chunks=[]; r.add_global_handler('dccmsg',lambda c,e: chunks.append(e.arguments[0]))")
                await alice.execute('r.process_once()')
                with self.assertRaisesRegex(ExecutionError, 'byte quota'):
                    await alice.execute("d.send_bytes(b'X'*65537)")
                await alice.execute("d.send_bytes(b'Z'*65536); d.send_bytes(b'Z'*65536)")
                for _ in range(20):
                    await bob.execute('r.process_once()')
                    if await bob.execute("len(b''.join(chunks))") == '131072':
                        break
                self.assertEqual(await bob.execute("b''.join(chunks) == b'Z'*131072"), 'True')
                await alice.execute('r.disconnect_all()')
                self.assertEqual(await alice.execute('len(r.connections)'), '1')  # IRC handle remains.
                for _ in range(8):
                    await worker.runtime._irc.dcc.call('quota-session', 'listen')
                with self.assertRaisesRegex(ValueError, 'quota'):
                    await worker.runtime._irc.dcc.call('quota-session', 'listen')

    async def test_explicit_offer_binary_transfer_and_ack(self):
        with tempfile.TemporaryDirectory() as directory:
            async with Worker(directory + '/agent.db') as worker:
                alice, bob, eve = [await worker.runtime.agent() for _ in range(3)]
                for agent in (alice, bob, eve):
                    await agent.execute(SETUP)
                await alice.execute("d=r.dcc('raw').listen(); events=[]; r.add_global_handler('all_events',lambda c,e: events.append((e.type,e.arguments)))")
                self.assertEqual(await alice.execute('irc.client.ip_quad_to_numstr(d.localaddress)'), "'2130706433'")
                port = int(await alice.execute('d.localport'))
                with self.assertRaisesRegex(ExecutionError, 'No DCC offer'):
                    await eve.execute(f"r.dcc('raw').connect('127.0.0.1',{port})")
                await alice.execute(f"c.ctcp('DCC',{bob.name!r},'SEND sample.bin 2130706433 '+str(d.localport)+' 7')")
                await bob.execute("offers=[]; r.add_global_handler('ctcp',lambda c,e: offers.append(e.arguments)); r.process_once()")
                self.assertIn('SEND sample.bin', await bob.execute('offers'))
                with self.assertRaisesRegex(ExecutionError, 'No DCC offer'):
                    await eve.execute(f"r.dcc('raw').connect('127.0.0.1',{port})")
                await bob.execute(f"d=r.dcc('raw').connect(irc.client.ip_numstr_to_quad('2130706433'),{port}); chunks=[]; r.add_global_handler('dccmsg',lambda c,e: chunks.append(e.arguments[0]))")
                # No destination file is created until the recipient chooses one.
                self.assertFalse(any(row['path'].endswith('sample.bin') for row in await bob.files()))
                await alice.execute("r.process_once(); d.send_bytes(b'abc\\x00def')")
                for _ in range(10):
                    await bob.execute('r.process_once()')
                    if ast.literal_eval(await bob.execute("b''.join(chunks)")) == b'abc\x00def':
                        break
                    await asyncio.sleep(0)
                self.assertEqual(ast.literal_eval(await bob.execute("b''.join(chunks)")), b'abc\x00def')
                await bob.execute("import struct; open('chosen.bin','wb').write(b''.join(chunks)); d.send_bytes(struct.pack('!I',7))")
                await alice.execute('r.process_once()')
                self.assertIn('dccmsg', await alice.execute('events'))
                await alice.execute('d.disconnect()')
                for _ in range(10):
                    await bob.execute('r.process_once()')
                    if await bob.execute('d.is_connected()') == 'False':
                        break
                self.assertEqual(await bob.execute('d.is_connected()'), 'False')
                self.assertEqual(await bob.execute("open('chosen.bin','rb').read()"), "b'abc\\x00def'")
                for method in ('send_file', 'receive_file', 'file_info', 'revoke_file'):
                    with self.assertRaisesRegex(ExecutionError, 'AttributeError'):
                        await bob.execute('c.' + method)

    async def test_network_restrictions_spoofing_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/agent.db'
            async with Worker(path) as worker:
                agent = await worker.runtime.agent()
                identity = agent.id
                await agent.execute(SETUP + "; d=r.dcc('raw').listen()")
                port = int(await agent.execute('d.localport'))
                # DCC identifiers do not name OS network endpoints.
                self.assertGreater(port, 65535)
                for address, destination in (('example.org',443), ('127.0.0.1',22), ('169.254.169.254',80)):
                    with self.assertRaises(ExecutionError):
                        await agent.execute(f"r.dcc('raw').connect({address!r},{destination})")
                await agent.execute("bad=[]; r.add_global_handler('fail',lambda c,e: bad.append(e.arguments)); c.ctcp('DCC',c.nickname,'SEND x 2130706433 22 1'); r.process_once()")
                self.assertIn('unused listener', await agent.execute('bad'))
                self.assertLessEqual(len(worker.runtime._irc.dcc.peers), 1)
            async with Worker(path) as worker:
                agent = await worker.runtime.agent(session_id=identity)
                with self.assertRaisesRegex(ExecutionError, 'unavailable'):
                    await agent.execute('d.process_once()')
