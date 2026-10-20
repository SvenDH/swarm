"""The real terminal application with pipe input and a real worker IRC socket."""
import asyncio
import tempfile
import unittest
from types import SimpleNamespace

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from agents import Worker
from irc_tui import IRCView
from wrappers.irc import IRCTransport


class IRCTUITests(unittest.IsolatedAsyncioTestCase):
    async def test_live_messages_input_and_clean_exit(self):
        with tempfile.TemporaryDirectory() as directory, create_pipe_input() as keyboard:
            async with Worker(directory + '/agent.db') as worker:
                server = worker.runtime._irc
                peer = IRCTransport()
                await peer.connect('127.0.0.1', server.port, 'user-peer', server.password)
                await peer.exchange('JOIN #general')
                view = IRCView('user-tester', input=keyboard, output=DummyOutput())
                run = asyncio.create_task(view.run(server.endpoint()))
                async def until(predicate):
                    async def wait():
                        while not predicate():
                            if run.done():
                                await run
                                self.fail('TUI exited unexpectedly')
                            await asyncio.sleep(0.01)
                    await asyncio.wait_for(wait(), 3)
                try:
                    await until(lambda: any(':user-tester JOIN #general' in line for line in view.lines))
                    await peer.exchange('PRIVMSG #general :hello terminal')
                    await until(lambda: any('hello terminal' in line for line in view.lines))
                    keyboard.send_text('/msg user-peer hello back\r')
                    await until(lambda: any('hello back' in line for line in view.lines))
                    async def delivery():
                        while True:
                            lines = await peer.exchange()
                            if any(':user-tester PRIVMSG user-peer :hello back' in line for line in lines):
                                return
                            await asyncio.sleep(0.01)
                    await asyncio.wait_for(delivery(), 3)
                    keyboard.send_text('/join #other\r')
                    await until(lambda: view.target == '#other')
                    keyboard.send_bytes(b'\x11')
                    await asyncio.wait_for(run, 3)
                finally:
                    run.cancel()
                    await asyncio.gather(run, return_exceptions=True)
                    await peer.close()

    async def test_commands_bounds_and_plain_peer_text(self):
        with create_pipe_input() as keyboard:
            view = IRCView('user-tester', input=keyboard, output=DummyOutput())
            view.accept(SimpleNamespace(text='/switch #room'))
            view.accept(SimpleNamespace(text='hello'))
            self.assertEqual(view.commands.get_nowait(), 'PRIVMSG #room :hello')
            view.accept(SimpleNamespace(text='/msg user-other private'))
            self.assertEqual(view.commands.get_nowait(), 'PRIVMSG user-other :private')
            view.accept(SimpleNamespace(text='x' * 600))
            self.assertTrue(view.commands.empty())
            view.show('\x1b[2Jmalicious\x00\x9b')
            self.assertNotIn('\x1b', view.lines[-1])
            self.assertNotIn('\x9b', view.lines[-1])
            for i in range(600):
                view.show(str(i))
            self.assertEqual(len(view.lines), 500)
