"""IRC terminal UI; launched with main.py --irc."""
import asyncio
from collections import deque

from wrappers.irc import valid_line


class IRCView:
    """Small full-screen client for the worker's IRC service."""
    def __init__(self, nickname, channel='#general', *, input=None, output=None):
        from prompt_toolkit import Application
        from prompt_toolkit.document import Document
        from prompt_toolkit.key_binding import KeyBindings
        from prompt_toolkit.layout import HSplit, Layout
        from prompt_toolkit.styles import Style
        from prompt_toolkit.widgets import Label, TextArea
        self.nickname, self.target = nickname, channel
        self.lines, self.commands = deque(maxlen=500), asyncio.Queue(maxsize=32)
        self.history = TextArea(read_only=True, scrollbar=True, style='class:history')
        self.entry = TextArea(height=1, multiline=False, prompt='> ', accept_handler=self.accept)
        self._document = Document
        keys = KeyBindings()
        @keys.add('c-q')
        @keys.add('c-c')
        def quit(event):
            event.app.exit()
        @keys.add('pageup')
        def up(event):
            self.history.buffer.cursor_up(count=10)
        @keys.add('pagedown')
        def down(event):
            self.history.buffer.cursor_down(count=10)
        self.app = Application(layout=Layout(HSplit([
            Label(lambda: f' IRC · {self.nickname} → {self.target} · /help · Ctrl-Q quits ', style='class:bar'),
            self.history, self.entry]), focused_element=self.entry), key_bindings=keys,
            style=Style.from_dict({'history': 'bg:#11161d #b8c0cc', 'bar': 'bg:#243447 #ffffff'}),
            full_screen=True, input=input, output=output)
        self.show('Commands: /join #channel, /switch target, /msg nick text, /part, /names, /list, /quit')

    def show(self, text):
        # Peer text is plain text, never markup, ANSI, or terminal commands.
        self.lines.append(''.join(char if char.isprintable() else ' ' for char in text)[:2048])
        text = '\n'.join(self.lines)
        self.history.buffer.set_document(self._document(text, len(text)), bypass_readonly=True)
        self.app.invalidate()

    def accept(self, buffer):
        text = buffer.text.strip()
        if not text:
            return False
        try:
            if text == '/quit':
                self.app.exit()
                return False
            if text == '/help':
                self.show('/join #channel · /switch target · /msg nick text · /part [channel] · /names · /list · /quit')
                return False
            command, _, rest = text.partition(' ')
            if command == '/switch':
                valid_line(rest)
                if not rest or ' ' in rest:
                    raise ValueError('Usage: /switch #channel-or-nickname')
                self.target = rest
                return False
            if command == '/join':
                line = 'JOIN ' + rest
                if not rest.startswith('#') or ' ' in rest:
                    raise ValueError('Usage: /join #channel')
            elif command == '/part':
                line = 'PART ' + (rest or self.target)
            elif command == '/names':
                line = 'NAMES ' + (rest or self.target)
            elif command == '/list':
                line = 'LIST'
            elif command == '/msg':
                target, sep, message = rest.partition(' ')
                if not sep or not message:
                    raise ValueError('Usage: /msg nickname message')
                line = f'PRIVMSG {target} :{message}'
            elif text.startswith('/'):
                raise ValueError('Unknown command; use /help')
            else:
                line = f'PRIVMSG {self.target} :{text}'
            valid_line(line)
            self.commands.put_nowait(line)
            if command == '/join':
                self.target = rest
            if line.startswith('PRIVMSG '):
                self.show(f'{self.nickname} → {line.split(" ", 2)[1]}: {line.partition(" :")[2]}')
        except (ValueError, asyncio.QueueFull) as error:
            self.show(str(error) or 'Send queue is full')
        return False

    async def run(self, endpoint):
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(endpoint['host'], endpoint['port'], limit=512), 3)
        tasks = []
        async def receive():
            while True:
                raw = await reader.readline()
                if not raw:
                    self.show('Disconnected. Ctrl-Q to exit.')
                    return
                line = raw.decode(errors='replace').rstrip('\r\n')
                if line.startswith('PING '):
                    self.commands.put_nowait('PONG ' + line[5:])
                else:
                    self.show(line)
        async def send():
            while True:
                writer.write((await self.commands.get()).encode() + b'\r\n')
                await asyncio.wait_for(writer.drain(), 3)
        async def guarded(function):
            try:
                await function()
            except (OSError, ValueError, asyncio.TimeoutError, asyncio.QueueFull) as error:
                self.show(f'Disconnected: {error}. Ctrl-Q to exit.')
        try:
            for line in (f'PASS {endpoint["password"]}', f'NICK {self.nickname}',
                         f'USER {self.nickname} 0 * :Swarm TUI', f'JOIN {self.target}'):
                valid_line(line)
                writer.write(line.encode() + b'\r\n')
            await writer.drain()
            tasks = [asyncio.create_task(guarded(receive)), asyncio.create_task(guarded(send))]
            await self.app.run_async()
        finally:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 3)
            except (OSError, asyncio.TimeoutError):
                writer.transport.abort()
