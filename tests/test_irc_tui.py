"""The real terminal application with pipe input and a real worker IRC socket."""
import asyncio
import tempfile
import unittest
from types import SimpleNamespace

from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput
from prompt_toolkit.data_structures import Size

from agents import Worker
from irc_tui import IRCView
from wrappers.irc import IRCTransport


class IRCTUITests(unittest.IsolatedAsyncioTestCase):
    async def test_wide_and_narrow_rendering_and_buffer_keys(self):
        class SizedOutput(DummyOutput):
            def __init__(self, columns):
                self.columns = columns
            def get_size(self):
                return Size(rows=24, columns=self.columns)
        for columns in (110, 60):
            with self.subTest(columns=columns), create_pipe_input() as keyboard:
                view = IRCView('user-tester', input=keyboard, output=SizedOutput(columns))
                view.handle_line(':worker 001 user-tester :Welcome')
                view.handle_line(':user-tester JOIN #general')
                view.handle_line(':red-mountain JOIN #general')
                view.handle_line(':red-mountain PRIVMSG #general :hello general')
                view.handle_line(':red-mountain PRIVMSG user-tester :private hello')
                run = asyncio.create_task(view.app.run_async())
                async def until(predicate):
                    async def wait():
                        while not predicate():
                            if run.done():
                                await run
                                self.fail('TUI exited unexpectedly')
                            await asyncio.sleep(0.01)
                    await asyncio.wait_for(wait(), 3)
                def rendered():
                    screen = view.app.renderer.last_rendered_screen
                    if screen is None:
                        return ''
                    return '\n'.join(''.join(screen.data_buffer[y][x].char for x in range(columns))
                                     for y in range(24))
                try:
                    await until(lambda: 'hello general' in rendered())
                    picture = rendered()
                    self.assertIn('swarm IRC', picture)
                    self.assertNotIn('PRIVMSG', picture)
                    self.assertNotIn('private hello', picture)
                    self.assertEqual(' Users' in picture, columns >= 90)
                    keyboard.send_bytes(b'\x0e')  # Ctrl-N: next (DM) buffer.
                    await until(lambda: view.target == 'red-mountain' and 'private hello' in rendered())
                    self.assertEqual(view.buffers['red-mountain'].unread, 0)
                    self.assertNotIn('hello general', view.history.text)
                    keyboard.send_bytes(b'\x10')  # Ctrl-P: previous channel.
                    await until(lambda: view.target == '#general')
                    for i in range(40):
                        view.show(f'line {i}')
                    keyboard.send_text('\x1b[5~')  # PageUp.
                    await until(lambda: view.history.buffer.cursor_position < len(view.history.text))
                    position = view.history.buffer.cursor_position
                    view.show('new traffic while scrolled')
                    self.assertEqual(view.history.buffer.cursor_position, position)
                    keyboard.send_bytes(b'\x11')
                    await asyncio.wait_for(run, 3)
                finally:
                    run.cancel()
                    await asyncio.gather(run, return_exceptions=True)

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
                    await until(lambda: 'user-tester joined #general' in view.history.text)
                    await until(lambda: 'user-peer' in view.buffers['#general'].members)
                    await peer.exchange('PRIVMSG #general :hello terminal')
                    await until(lambda: 'hello terminal' in view.history.text)
                    keyboard.send_text('/msg user-peer hello back\r')
                    await until(lambda: 'hello back' in view.history.text)
                    async def delivery():
                        while True:
                            lines = await peer.exchange()
                            if any(':user-tester PRIVMSG user-peer :hello back' in line for line in lines):
                                return
                            await asyncio.sleep(0.01)
                    await asyncio.wait_for(delivery(), 3)
                    keyboard.send_text('/join #other\r')
                    await until(lambda: view.target == '#other')
                    await peer.close()
                    await until(lambda: 'user-peer' not in view.buffers['#general'].members)
                    keyboard.send_bytes(b'\x11')
                    await asyncio.wait_for(run, 3)
                finally:
                    run.cancel()
                    await asyncio.gather(run, return_exceptions=True)
                    await peer.close()

    async def test_commands_bounds_and_plain_peer_text(self):
        with create_pipe_input() as keyboard:
            view = IRCView('user-tester', input=keyboard, output=DummyOutput())
            view.handle_line(':worker 001 user-tester :Welcome')
            view.accept(SimpleNamespace(text='/switch #room'))
            view.accept(SimpleNamespace(text='hello'))
            self.assertEqual(view.commands.get_nowait(), 'PRIVMSG #room :hello')
            view.accept(SimpleNamespace(text='/msg user-other private'))
            self.assertEqual(view.commands.get_nowait(), 'PRIVMSG user-other :private')
            view.accept(SimpleNamespace(text='x' * 600))
            self.assertTrue(view.commands.empty())
            view.show('\x1b[2Jmalicious\x00\x9b')
            self.assertNotIn('\x1b', view.history.text)
            self.assertNotIn('\x9b', view.history.text)
            for i in range(600):
                view.show(str(i))
            self.assertEqual(len(view.buffers[view.target].messages), 500)

    async def test_formatted_buffers_unread_members_and_errors(self):
        with create_pipe_input() as keyboard:
            view = IRCView('user-tester', input=keyboard, output=DummyOutput())
            view.handle_line(':worker 001 user-tester :Welcome')
            view.handle_line(':user-tester JOIN #general')
            view.handle_line(':worker 353 user-tester = #general :user-tester @red-mountain')
            view.handle_line(':worker 353 user-tester = #general :+blue-river')
            view.handle_line(':worker 366 user-tester #general :End of NAMES')
            self.assertEqual(view.buffers['#general'].members, {'user-tester', 'red-mountain', 'blue-river'})
            view.handle_line(':red-mountain PRIVMSG #general :hello general')
            self.assertRegex(view.history.text, r'\d\d:\d\d <red-mountain> hello general')
            self.assertNotIn('PRIVMSG', view.history.text)
            self.assertNotIn(':worker', view.history.text)
            view.handle_line(':red-mountain PRIVMSG user-tester :hello user-tester')
            self.assertNotIn('hello user-tester', view.history.text)
            self.assertEqual(view.buffers['red-mountain'].unread, 1)
            self.assertTrue(view.buffers['red-mountain'].highlighted)
            self.assertTrue(any(style == 'class:mention' for style, _ in view.buffer_fragments()))
            view.switch('red-mountain')
            self.assertIn('<red-mountain> hello user-tester', view.history.text)
            self.assertEqual(view.buffers['red-mountain'].unread, 0)
            self.assertFalse(view.buffers['red-mountain'].highlighted)
            view.switch('#general')
            self.assertNotIn('hello user-tester', view.history.text)
            view.handle_line(':red-mountain PART #general')
            self.assertNotIn('red-mountain', view.buffers['#general'].members)
            self.assertIn('red-mountain left #general', view.history.text)
            view.handle_line(':blue-river QUIT :bye')
            self.assertNotIn('blue-river', view.buffers['#general'].members)
            view.handle_line(':worker 442 user-tester #general :Not on channel')
            self.assertIn('Not on channel', view.history.text)
            self.assertNotIn('442', view.history.text)
            self.assertTrue(any(style == 'class:error' for row in view.lexer.rows for style, _ in row))
            view.accept(SimpleNamespace(text='/query green-forest'))
            self.assertEqual(view.target, 'green-forest')
            view.accept(SimpleNamespace(text='/close'))
            self.assertNotIn('green-forest', view.buffers)
            view.cycle(1)
            self.assertIn(view.target, view.buffers)
