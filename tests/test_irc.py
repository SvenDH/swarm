"""Worker-local IRC routing and sandbox boundary checks."""
import tempfile
import unittest

from agents import CodeAgent, Worker
from wrappers.irc import IRCServer


class IRCServerTests(unittest.TestCase):
    def test_channels_private_messages_and_identity(self):
        server = IRCServer()
        server.exchange('alice', 'JOIN #room')
        server.exchange('bob', 'JOIN #room')
        server.exchange('alice')
        server.exchange('alice', 'PRIVMSG #room :hello')
        self.assertEqual(server.exchange('bob'), [':alice PRIVMSG #room :hello'])
        self.assertEqual(server.exchange('alice'), [])
        server.exchange('bob', 'PRIVMSG alice :private')
        self.assertEqual(server.exchange('alice'), [':bob PRIVMSG alice :private'])
        self.assertIn('alice bob', server.exchange('alice', 'NAMES #room')[0])
        self.assertIn('disabled', server.exchange('alice', 'NICK bob')[0])
        self.assertIn('disabled', server.exchange('alice', 'CONNECT evil.example 6667')[0])
        self.assertIn('disabled', server.exchange('alice', ':bob PRIVMSG #room :spoof')[0])
        server.exchange('bob', 'PART #room')
        self.assertIn('Not on channel', server.exchange('bob', 'PRIVMSG #room :no')[0])
        server.exchange('bob', 'QUIT')
        self.assertNotIn('bob', server.clients)

    def test_bounds_and_no_command_injection(self):
        server = IRCServer()
        for command in ('JOIN #a\nPRIVMSG #a :bad', '\r', '\x00', 'x' * 511, 'JOIN #a,b'):
            with self.assertRaises(ValueError):
                server.exchange('alice', command)
        server.exchange('alice')
        server.exchange('bob')
        for i in range(150):
            server.exchange('alice', f'PRIVMSG bob :{i}')
        lines = [line for _ in range(8) for line in server.exchange('bob')]
        self.assertEqual(len(lines), 128)
        self.assertTrue(lines[-1].endswith(':149'))
        self.assertEqual(server.queued_bytes, 0)

    def test_aggregate_queue_budget_recovers_after_polling(self):
        server = IRCServer()
        server.exchange('sender')
        for i in range(200):
            server.exchange(f'peer-{i}')
        with self.assertRaises(BufferError):
            for i in range(200):
                for _ in range(128):
                    server.exchange('sender', f'PRIVMSG peer-{i} :' + 'x' * 480)
        self.assertLessEqual(server.queued_bytes, 8 * 1024**2)
        for i in range(200):
            for _ in range(8):
                server.exchange(f'peer-{i}')
        self.assertEqual(server.queued_bytes, 0)
        server.exchange('sender', 'PRIVMSG peer-0 :recovered')
        self.assertEqual(server.exchange('peer-0'), [':sender PRIVMSG peer-0 :recovered'])
        for _ in range(50):
            server.exchange('sender', 'PRIVMSG peer-0 :pending')
        server.exchange('peer-0', 'QUIT')
        self.assertEqual(server.queued_bytes, 0)


class IRCIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_import_forms_and_callback_order(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Worker(folder + '/agent.db') as worker:
                agent = await worker.runtime.agent()
                self.assertEqual(await agent.execute("""import irc.client as client
from irc import client as other
from irc.client import Reactor
import importlib
assert Reactor is client.Reactor
assert other.Reactor is Reactor
assert __import__('irc.client',fromlist=['Reactor']).Reactor is Reactor
assert importlib.import_module('irc.client').Reactor is Reactor
r=Reactor()
c=r.server().connect('worker',6667)
seen=[]
def last(c,e):
    seen.append('last')
def first(c,e):
    seen.append('first')
    return 'NO MORE'
r.add_global_handler('pong',last,10)
r.add_global_handler('pong',first,-10)
c.ping('one')
r.process_once()
assert seen==['first']
assert r.remove_global_handler('pong',first)==1
c.ping('two')
r.process_once()
assert seen==['first','last']
try:
    Reactor().server().connect('external',6667)
except client.ServerConnectionError:
    seen.append('blocked')
c.disconnect()
(seen,c.is_connected())
"""), "(['first', 'last', 'blocked'], False)")

    async def test_agents_share_only_their_worker_server(self):
        with tempfile.TemporaryDirectory() as folder:
            async with Worker(folder + '/a/agent.db') as worker, Worker(folder + '/b/agent.db') as other:
                alice = await worker.runtime.agent()
                bob = await worker.runtime.agent()
                isolated = await other.runtime.agent()
                setup = """import irc.client
r=irc.client.Reactor()
c=r.server().connect('worker',6667)
events=[]
def record(connection,event):
    events.append((event.type,event.source,event.target,event.arguments))
r.add_global_handler('all_events',record)
"""
                await alice.execute(setup)
                await bob.execute(setup)
                await isolated.execute(setup)
                await alice.execute("c.join('#room'); r.process_once()")
                await bob.execute("c.join('#room'); r.process_once(); events=[]")
                await alice.execute("c.privmsg('#room','hello')")
                self.assertIn(f"('pubmsg', '{alice.name}', '#room', ['hello'])", await bob.execute('r.process_once(); events'))
                self.assertNotIn('#room', await isolated.execute('c.list(); r.process_once(); events'))
                self.assertIn('pong', await alice.execute("ns={}; exec(\"from irc.client import Reactor; rr=Reactor(); cc=rr.server().connect('worker',6667); cc.ping('ok'); events=[]; rr.add_global_handler('pong',lambda c,e: events.append(e.type)); rr.process_once()\", ns); ns['events']"))
                with self.assertRaisesRegex(Exception, 'Only worker'):
                    await alice.execute("irc.client.Reactor().server().connect('example.com',6667)")
                with self.assertRaisesRegex(Exception, 'session name'):
                    await alice.execute("irc.client.Reactor().server().connect('worker',6667,'spoof')")
                with self.assertRaisesRegex(Exception, 'Only worker'):
                    await alice.execute("_irc_exchange(None,server='example.com')")
                self.assertIn(alice.name, await alice.execute('c.get_nickname()'))

    async def test_no_server_without_worker(self):
        with tempfile.TemporaryDirectory() as folder:
            agent = await CodeAgent.open(folder + '/a.db')
            try:
                with self.assertRaisesRegex(Exception, 'running worker'):
                    await agent.execute("from irc.client import Reactor; Reactor().server().connect('worker',6667)")
            finally:
                await agent.close()
