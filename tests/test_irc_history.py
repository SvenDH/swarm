"""Durable IRC conversations, access boundaries, paging and replay."""
import ast
import asyncio
import tempfile
import unittest
from unittest.mock import patch

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agents import Worker, ExecutionError
from irc_tui import IRCView
from wrappers.irc import IRCTransport


SETUP = "from irc.client import Reactor; r=Reactor(); c=r.server().connect('worker',6667)"


class IRCHistoryTests(unittest.IsolatedAsyncioTestCase):
    async def test_standard_wire_commands_and_cap_negotiation(self):
        with tempfile.TemporaryDirectory() as directory:
            async with Worker(directory + '/agent.db') as worker:
                server = worker.runtime._irc
                endpoint = server.endpoint()
                reader, writer = await asyncio.open_connection(endpoint['host'], endpoint['port'])
                try:
                    writer.write(b'CAP LS 302\r\n')
                    await writer.drain()
                    self.assertIn(b'batch draft/chathistory', await asyncio.wait_for(reader.readline(), 2))
                    writer.write(('CAP REQ :batch draft/chathistory message-tags server-time\r\n'
                                  f'PASS {endpoint["password"]}\r\nNICK user-wire\r\nUSER user-wire 0 * :test\r\n').encode())
                    await writer.drain()
                    self.assertIn(b'CAP * ACK', await asyncio.wait_for(reader.readline(), 2))
                    # Registration remains suspended until the CAP handshake ends.
                    writer.write(b'CAP END\r\n')
                    await writer.drain()
                    self.assertIn(b'001 user-wire', await asyncio.wait_for(reader.readline(), 2))
                    self.assertIn(b'MSGREFTYPES=msgid,timestamp', await asyncio.wait_for(reader.readline(), 2))
                finally:
                    writer.close()
                    await writer.wait_closed()
                peer = IRCTransport()
                await peer.connect(endpoint['host'], endpoint['port'], 'user-client', endpoint['password'], tags=True)
                try:
                    await peer.exchange('JOIN #wire')
                    for text in ('one', 'two', 'three'):
                        await peer.exchange('PRIVMSG #wire :' + text)
                    latest = await peer.exchange('CHATHISTORY LATEST #wire * 2')
                    self.assertIn(' chathistory #wire', latest[0])
                    self.assertTrue(latest[-1].startswith(':worker BATCH -'))
                    self.assertEqual([line.rsplit(' :', 1)[-1] for line in latest[1:-1]], ['two', 'three'])
                    rows = await worker.runtime.query('SELECT id,timestamp FROM irc_messages ORDER BY id')
                    first, last = rows[0]['id'], rows[-1]['id']
                    for command, expected in (
                        (f'AFTER #wire msgid={first} 1', ['two']),
                        (f'BEFORE #wire msgid={last} 1', ['two']),
                        (f'BETWEEN #wire msgid={last} msgid={first} 10', ['two']),
                        (f'LATEST #wire msgid={first} 1', ['three']),
                        (f'AROUND #wire msgid={first+1} 3', ['one', 'two', 'three']),
                        ('AFTER #wire timestamp=' + rows[0]['timestamp'].replace('+00:00', 'Z') + ' 10', ['two', 'three']),
                        ('LATEST #empty * 1', None),
                    ):
                        with self.subTest(command=command):
                            result = await peer.exchange('CHATHISTORY ' + command)
                            if expected is None:
                                self.assertIn('FAIL CHATHISTORY INVALID_TARGET', result[0])
                            else:
                                self.assertEqual([line.rsplit(' :', 1)[-1] for line in result if ' PRIVMSG ' in line], expected)
                    result = await peer.exchange('CHATHISTORY TARGETS timestamp=2000-01-01T00:00:00.000Z timestamp=2100-01-01T00:00:00.000Z 50')
                    self.assertIn('draft/chathistory-targets', result[0])
                    self.assertIn('CHATHISTORY TARGETS #wire ', result[1])
                    await peer.exchange('JOIN #fresh')
                    result = await peer.exchange('CHATHISTORY LATEST #fresh * 50')
                    self.assertEqual(len(result), 2)  # Empty batch still has an end.
                    for request, code in (
                        ('CHATHISTORY LATEST #wire * 101', 'INVALID_PARAMS'),
                        ('CHATHISTORY BEFORE #wire timestamp=invalidZ 1', 'INVALID_PARAMS'),
                        ('CHATHISTORY BEFORE #wire unknown=1 1', 'INVALID_MSGREFTYPE'),
                        ('CHATHISTORY BOGUS #wire * 1', 'INVALID_PARAMS'),
                        ('CHATHISTORY BETWEEN #wire msgid=1 1', 'INVALID_PARAMS'),
                    ):
                        self.assertIn('FAIL CHATHISTORY ' + code, (await peer.exchange(request))[0])
                    self.assertIn('Unknown or disabled command', (await peer.exchange('HISTORY #wire 50 0 0'))[0])
                finally:
                    await peer.close()

    async def test_failed_storage_prevents_delivery_and_unicode_roundtrip(self):
        with tempfile.TemporaryDirectory() as directory:
            async with Worker(directory + '/agent.db') as worker:
                alice, bob = await worker.runtime.agent(), await worker.runtime.agent()
                for agent in (alice, bob):
                    await agent.execute(SETUP + "; c.join('#general')")
                server = worker.runtime._irc
                with patch.object(worker, '_irc_history', side_effect=OSError('storage offline')):
                    with self.assertRaises(OSError):
                        await server.dispatch(alice.name, 'PRIVMSG #general :must not arrive')
                self.assertFalse(any('must not arrive' in line for line in server.exchange(bob.name)))
                text = '雪' * 100
                await alice.execute(f"c.privmsg('#general',{text!r})")
                rows = ast.literal_eval(await bob.execute("c.history('#general')"))
                self.assertEqual([row['text'] for row in rows], [text])
                # Querying history must not discard interleaved live events.
                events = await bob.execute("seen=[]; r.add_global_handler('pubmsg',lambda c,e: seen.append(e.arguments[0])); r.process_once(); seen")
                self.assertIn(text, events)

    async def test_paging_privacy_and_restart(self):
        with tempfile.TemporaryDirectory() as directory:
            path = directory + '/agent.db'
            async with Worker(path) as worker:
                alice, bob, eve = [await worker.runtime.agent() for _ in range(3)]
                identities = alice.id, bob.id
                for agent in (alice, bob, eve):
                    await agent.execute(SETUP)
                await alice.execute("c.join('#general')")
                for text in ('first', 'second', 'third'):
                    await alice.execute(f"c.privmsg('#general', {text!r})")
                with self.assertRaisesRegex(ExecutionError, 'Join the channel'):
                    await bob.execute("c.history('#general')")
                await bob.execute("c.join('#general')")
                rows = ast.literal_eval(await bob.execute("c.history('#general')"))
                self.assertEqual([row['text'] for row in rows], ['first', 'second', 'third'])
                self.assertEqual(len({row['id'] for row in rows}), 3)
                self.assertTrue(all(row['timestamp'].endswith('+00:00') for row in rows))
                self.assertEqual(ast.literal_eval(await bob.execute("c.history('#general',limit=1)")), rows[-1:])
                self.assertEqual(ast.literal_eval(await bob.execute(
                    f"c.history('#general',after={rows[0]['id']})")), rows[1:])
                self.assertEqual(ast.literal_eval(await bob.execute(
                    f"c.history('#general',before={rows[-1]['id']})")), rows[:-1])
                await alice.execute(f"c.privmsg({bob.name!r}, 'private')")
                private = ast.literal_eval(await bob.execute(f"c.history({alice.name!r})"))
                self.assertEqual([row['text'] for row in private], ['private'])
                self.assertEqual(ast.literal_eval(await eve.execute(f"c.history({alice.name!r})")), [])
                targets = await worker.runtime._irc.agent_exchange(eve.name,
                    'CHATHISTORY TARGETS timestamp=2000-01-01T00:00:00.000Z timestamp=2100-01-01T00:00:00.000Z 50')
                self.assertFalse(any(' CHATHISTORY TARGETS ' in line for line in targets))
                targets = await worker.runtime._irc.agent_exchange(alice.name,
                    'CHATHISTORY TARGETS timestamp=2000-01-01T00:00:00.000Z timestamp=2100-01-01T00:00:00.000Z 50')
                self.assertTrue(any(' CHATHISTORY TARGETS ' + bob.name + ' ' in line for line in targets))
                await bob.execute("c.part('#general')")
                with self.assertRaisesRegex(ExecutionError, 'Join the channel'):
                    await bob.execute("c.history('#general')")
                for arguments in ('limit=101', 'limit=True', 'after=-1', 'before=2**64'):
                    with self.subTest(arguments=arguments), self.assertRaises(ExecutionError):
                        await alice.execute(f"c.history('#general',{arguments})")
                await alice.execute("c.privmsg('#absent','not accepted'); c.privmsg('nobody','offline')")
                self.assertEqual((await worker.runtime.query('SELECT count(*) AS n FROM irc_messages'))[0]['n'], 4)
            async with Worker(path) as worker:
                alice = await worker.runtime.agent(session_id=identities[0])
                await alice.execute(SETUP + "; c.join('#general')")
                self.assertEqual(ast.literal_eval(await alice.execute("c.history('#general')")), rows)
                await alice.execute("c.privmsg('#general','after restart')")
                latest = ast.literal_eval(await alice.execute("c.history('#general',limit=1)"))[0]
                self.assertGreater(latest['id'], private[-1]['id'])

    async def test_tui_replay_deduplicates_ids_not_text(self):
        with create_pipe_input() as keyboard:
            view = IRCView('user-tester', input=keyboard, output=DummyOutput())
            view.handle_line(':worker 001 user-tester :Welcome')
            view.handle_line(':user-tester JOIN #general')
            view.handle_line('@msgid=2;time=2026-10-21T10:01:00+00:00 :alice PRIVMSG #general :hello')
            view.handle_line(':worker BATCH +replay chathistory #general')
            for identity in (1, 2):
                view.handle_line(f'@batch=replay;msgid={identity};time=2026-10-21T10:0{identity-1}:00Z :alice PRIVMSG #general :hello')
            view.handle_line(':worker BATCH -replay')
            messages = [line for line in view.buffers['#general'].messages if line.identity]
            self.assertEqual([line.identity for line in messages], [1, 2])
            self.assertEqual([line.text for line in messages], ['hello', 'hello'])
            view.handle_line(':worker CAP * ACK :message-tags server-time echo-message')
            self.assertTrue(view.echo)
