"""Buffered IRC terminal UI; launched with main.py --irc."""
import asyncio
import hashlib
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone

from prompt_toolkit import Application
from prompt_toolkit.document import Document
from prompt_toolkit.filters import Condition
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import ConditionalContainer, HSplit, Layout, VSplit, Window
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from prompt_toolkit.widgets import Label, TextArea

from wrappers.irc import valid_line


@dataclass
class ChatLine:
    timestamp: datetime
    nick: str
    text: str
    kind: str
    identity: int = 0


@dataclass
class ChatBuffer:
    messages: deque = field(default_factory=lambda: deque(maxlen=500))
    members: set = field(default_factory=set)
    unread: int = 0
    highlighted: bool = False


class TranscriptLexer(Lexer):
    def __init__(self):
        self.rows = []

    def lex_document(self, document):
        rows = self.rows
        return lambda index: rows[index] if index < len(rows) else []


class IRCView:
    """Buffered channel/DM client. Peer content is always plain text."""
    def __init__(self, nickname, channel='#general', *, input=None, output=None):
        self.nickname, self.target = nickname.lower(), self.valid_target(channel)
        self.buffers = {'*server': ChatBuffer(), self.target: ChatBuffer()}
        self.commands = asyncio.Queue(maxsize=32)
        self.joined, self.names_pending = set(), {}
        self.connected, self.status = False, 'connecting'
        self.echo = False
        self.history_pending = {}
        self.lexer = TranscriptLexer()
        self.history = TextArea(read_only=True, lexer=self.lexer, style='class:history')
        self.entry = TextArea(height=1, multiline=False, prompt=lambda: f'{self.target}> ', accept_handler=self.accept)
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
        @keys.add('c-n')
        @keys.add('escape', 'right')
        def next_buffer(event):
            self.cycle(1)
        @keys.add('c-p')
        @keys.add('escape', 'left')
        def previous_buffer(event):
            self.cycle(-1)
        for number in range(1, 10):
            @keys.add('escape', str(number))
            def numbered_buffer(event, number=number):
                names = list(self.buffers)
                if number <= len(names):
                    self.switch(names[number - 1])
        wide = Condition(lambda: self.app.output.get_size().columns >= 90)
        sidebar = ConditionalContainer(VSplit([
            Window(FormattedTextControl(self.buffer_fragments), width=21),
            Window(width=1, char='│', style='class:muted')]), wide)
        users = ConditionalContainer(VSplit([
            Window(width=1, char='│', style='class:muted'),
            HSplit([Label(' Users', style='class:muted'),
                    Window(FormattedTextControl(self.user_fragments), width=21)])]),
            Condition(lambda: wide() and self.target.startswith('#')))
        self.app = Application(layout=Layout(HSplit([
            Label(lambda: f' swarm IRC · {self.nickname} · {self.target} · {self.status}', style='class:bar'),
            VSplit([sidebar, self.history, users]),
            Window(FormattedTextControl(lambda: self.buffer_fragments(horizontal=True)), height=1, style='class:bar'),
            self.entry]), focused_element=self.entry), key_bindings=keys,
            style=Style.from_dict({'': 'bg:#11161d #d0d7df', 'history': 'bg:#11161d #d0d7df', 'bar': 'bg:#243447 #d0d7df',
                'muted': '#718096', 'active': 'bold #ffffff', 'unread': 'bold #63b3ed',
                'mention': 'bold #f6c177', 'error': '#f28b82', 'self': '#9ca8b8'}),
            full_screen=True, input=input, output=output)
        self.show('Connecting. /help for commands · Ctrl-P/N changes buffers.', '*server')

    @staticmethod
    def valid_target(name):
        name = name.lower()
        if name != '*server' and not re.fullmatch(r'(?:#[a-z0-9_-]{1,63}|[a-z][a-z0-9_-]{0,63})', name):
            raise ValueError('Choose a channel or nickname')
        return name

    def buffer(self, name):
        name = self.valid_target(name)
        if name not in self.buffers:
            if len(self.buffers) >= 64:
                raise ValueError('Too many buffers; /close one first')
            self.buffers[name] = ChatBuffer()
        return self.buffers[name]

    def switch(self, name):
        name = self.valid_target(name)
        buffer = self.buffer(name)
        self.target = name
        buffer.unread, buffer.highlighted = 0, False
        self.refresh(follow=True)

    def cycle(self, step):
        names = list(self.buffers)
        self.switch(names[(names.index(self.target) + step) % len(names)])

    def nick_style(self, nick):
        if nick.lower() == self.nickname:
            return 'class:self'
        colors = ('#82aaff', '#c3e88d', '#c792ea', '#f78c6c', '#89ddff', '#ffcb6b')
        index = hashlib.blake2s(nick.lower().encode(), digest_size=1).digest()[0] % len(colors)
        return 'bold fg:' + colors[index]

    def buffer_fragments(self, horizontal=False):
        fragments = []
        for index, (name, buffer) in enumerate(self.buffers.items(), 1):
            style = ('class:active' if name == self.target else 'class:mention' if buffer.highlighted
                     else 'class:unread' if buffer.unread else 'class:muted')
            count = f' ({buffer.unread})' if buffer.unread else ''
            fragments.append((style, f' {index}:{name}{count}' + ('  ' if horizontal else '\n')))
        return fragments

    def user_fragments(self):
        return [(self.nick_style(nick), ' ' + nick + '\n')
                for nick in sorted(self.buffers[self.target].members)]

    def refresh(self, follow=False):
        rows = []
        for message in self.buffers[self.target].messages:
            nick, text, kind = message.nick, message.text, message.kind
            row = [('class:muted', message.timestamp.astimezone().strftime('%H:%M') + ' ')]
            if kind == 'message':
                row += [(self.nick_style(nick), '<' + nick + '> '),
                        ('class:mention' if nick != self.nickname and self.nickname in text.lower() else '', text)]
            else:
                row += [('class:error' if kind == 'error' else 'class:muted', '* ' + text)]
            rows.append(row)
        old = self.history.buffer.document
        following = follow or old.cursor_position == len(old.text)
        text = '\n'.join(''.join(value for _, value in row) for row in rows)
        self.lexer.rows = rows
        self.history.buffer.set_document(Document(text, len(text) if following else min(old.cursor_position, len(text))),
                                         bypass_readonly=True)
        self.app.invalidate()

    @staticmethod
    def chat_line(text, nick='', kind='status', identity=0, timestamp=None):
        text = ''.join(char if char.isprintable() else ' ' for char in text)[:2048]
        nick = ''.join(char if char.isprintable() else ' ' for char in nick)[:64]
        stamp = datetime.fromisoformat(timestamp.replace('Z', '+00:00')) if timestamp else datetime.now(timezone.utc)
        return ChatLine(stamp, nick, text, kind, identity)

    def show(self, text, target=None, *, nick='', kind='status', identity=0, timestamp=None):
        target = target or self.target
        try:
            buffer = self.buffer(target)
        except ValueError:
            target, buffer = '*server', self.buffers['*server']
        if identity and any(message.identity == identity for message in buffer.messages):
            return
        message = self.chat_line(text, nick, kind, identity, timestamp)
        buffer.messages.append(message)
        if target != self.target:
            buffer.unread += 1
            buffer.highlighted |= kind == 'message' and self.nickname in text.lower()
        self.refresh()

    def request_history(self, target):
        if self.connected and target != '*server':
            self.commands.put_nowait(f'CHATHISTORY LATEST {target} * 50')

    def merge_history(self, target, records):
        if target not in self.buffers:
            return
        buffer = self.buffers[target]
        known = {message.identity for message in buffer.messages if message.identity}
        messages = list(buffer.messages)
        for record in records:
            if record['id'] not in known:
                known.add(record['id'])
                messages.append(self.chat_line(record['text'], record['sender'], 'message', record['id'], record['timestamp']))
        buffer.messages = deque(sorted(messages, key=lambda message: (message.timestamp, message.identity))[-500:], maxlen=500)
        self.refresh()

    def handle_line(self, line):
        tags = {}
        if line.startswith('@'):
            raw, _, line = line.partition(' ')
            tags = dict(tag.partition('=')[::2] for tag in raw[1:].split(';'))
        head, sep, trailing = line.partition(' :')
        words = head.split()
        if not words:
            return
        source = words.pop(0)[1:] if words[0].startswith(':') else ''
        if not words:
            return
        command, params = words[0].upper(), words[1:] + ([trailing] if sep else [])
        nick = source.split('!', 1)[0].lower()
        if command == 'CAP':
            if 'ACK' in params and 'echo-message' in params[-1].split():
                self.echo = True
        elif command == 'BATCH' and params:
            if params[0].startswith('+') and len(params) == 3 and params[1] == 'chathistory':
                self.history_pending[params[0][1:]] = (params[2].lower(), [])
            elif params[0].startswith('-'):
                batch = self.history_pending.pop(params[0][1:], None)
                if batch is not None:
                    self.merge_history(*batch)
        elif command == 'FAIL' and params and params[0] == 'CHATHISTORY':
            self.show(params[-1], kind='error')
        elif command == '001' and params:
            self.nickname, self.connected, self.status = params[0].lower(), True, 'connected'
            self.show('Connected as ' + self.nickname, '*server')
        elif command in ('PRIVMSG', 'NOTICE') and len(params) >= 2:
            if tags.get('batch') in self.history_pending:
                _, records = self.history_pending[tags['batch']]
                if len(records) < 100:
                    records.append(dict(id=int(tags['msgid']), timestamp=tags['time'], sender=nick,
                                        target=params[0], text=params[-1]))
                return
            target = params[0].lower() if params[0].startswith('#') or nick == self.nickname else nick
            fresh = target not in self.buffers
            message = params[-1]
            metadata = {'identity': int(tags.get('msgid', 0)), 'timestamp': tags.get('time')}
            if message.startswith('\x01ACTION ') and message.endswith('\x01'):
                self.show(nick + ' ' + message[8:-1], target, **metadata)
            else:
                self.show(message, target, nick=nick, kind='message' if command == 'PRIVMSG' else 'status', **metadata)
            if fresh:
                self.request_history(target)
        elif command == 'JOIN' and params:
            target = params[0].lower()
            self.buffer(target).members.add(nick)
            if target in self.names_pending:
                self.names_pending[target].add(nick)
            if nick == self.nickname:
                self.joined.add(target)
                try:
                    self.commands.put_nowait('NAMES ' + target)
                    self.request_history(target)
                except asyncio.QueueFull:
                    pass
            self.show(nick + ' joined ' + target, target)
        elif command == 'PART' and params:
            target = params[0].lower()
            if target not in self.buffers:
                return
            self.buffers[target].members.discard(nick)
            self.names_pending.get(target, set()).discard(nick)
            if nick == self.nickname:
                self.joined.discard(target)
                self.buffers[target].members.clear()
            self.show(nick + ' left ' + target, target)
        elif command == 'QUIT':
            for names in self.names_pending.values():
                names.discard(nick)
            for target, buffer in self.buffers.items():
                if nick in buffer.members:
                    buffer.members.discard(nick)
                    self.names_pending.get(target, set()).discard(nick)
                    self.show(nick + ' quit' + (': ' + params[0] if params else ''), target)
        elif command == '353' and len(params) >= 4:
            target = params[2].lower()
            if target in self.buffers:
                self.names_pending.setdefault(target, set()).update(
                    name.lstrip('@+%&~').lower() for name in params[-1].split())
        elif command == '366' and len(params) >= 2:
            target = params[1].lower()
            names = self.names_pending.pop(target, set())
            if target in self.buffers:
                self.buffers[target].members = names
            self.refresh()
        elif command == '322' and len(params) >= 3:
            self.show(f'{params[1]} · {params[2]} users', '*server')
        elif command.isdigit() and int(command) >= 400:
            self.show(' · '.join(params[1:]), kind='error')
        elif command not in ('PONG', 'CAP', '323'):
            self.show(' '.join(params[1:] if command.isdigit() else params), '*server')

    def accept(self, buffer):
        text = buffer.text.strip()
        if not text:
            return False
        try:
            command, _, rest = text.partition(' ')
            if command == '/quit':
                self.app.exit()
                return False
            if command == '/help':
                self.show('/join #channel · /query nick · /msg nick text · /history [target] · /switch target · /close · /part · /names · /list · /quit')
                self.show('Alt-←/→ or Ctrl-P/N: buffers · Alt-1…9: select · PageUp/Down: scroll · Ctrl-Q: quit')
                return False
            if command in ('/switch', '/query'):
                self.switch(rest)
                if command == '/query':
                    self.request_history(self.target)
                return False
            if command == '/close':
                target = self.valid_target(rest or self.target)
                if target == '*server':
                    raise ValueError('The server buffer cannot be closed')
                if target in self.joined:
                    self.commands.put_nowait('PART ' + target)
                self.buffers.pop(target, None)
                self.joined.discard(target)
                self.names_pending.pop(target, None)
                self.switch(next(reversed(self.buffers)))
                return False
            if not self.connected:
                raise ValueError('Not connected')
            if command == '/history':
                target = self.valid_target(rest or self.target)
                self.switch(target)
                self.request_history(target)
                return False
            target = self.target
            if command == '/join':
                target = self.valid_target(rest)
                if not target.startswith('#'):
                    raise ValueError('Usage: /join #channel')
                self.buffer(target)
                line = 'JOIN ' + target
            elif command in ('/part', '/names'):
                target = self.valid_target(rest or self.target)
                if not target.startswith('#'):
                    raise ValueError('Choose a channel')
                if command == '/names':
                    self.buffer(target)
                line = ('PART ' if command == '/part' else 'NAMES ') + target
            elif command == '/list':
                line = 'LIST'
            elif command == '/msg':
                target, sep, message = rest.partition(' ')
                target = self.valid_target(target)
                if not sep or not message or target == '*server':
                    raise ValueError('Usage: /msg nickname message')
                self.buffer(target)
                line = f'PRIVMSG {target} :{message}'
            elif text.startswith('/'):
                raise ValueError('Unknown command; use /help')
            elif target == '*server':
                raise ValueError('Choose a channel or nickname with /join or /query')
            else:
                line, message = f'PRIVMSG {target} :{text}', text
            valid_line(line)
            self.commands.put_nowait(line)
            if command in ('/join', '/msg'):
                self.switch(target)
            if line.startswith('PRIVMSG ') and not self.echo:
                self.show(message, target, nick=self.nickname, kind='message')
        except (ValueError, asyncio.QueueFull) as error:
            self.show(str(error) or 'Send queue is full', kind='error')
        return False

    async def run(self, endpoint):
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(endpoint['host'], endpoint['port'], limit=4096), 3)
        tasks = []
        async def receive():
            while True:
                raw = await reader.readline()
                if not raw:
                    self.connected, self.status = False, 'disconnected'
                    self.show('Disconnected. Ctrl-Q to exit.', kind='error')
                    return
                line = raw.decode(errors='replace').rstrip('\r\n')
                if line.startswith('PING '):
                    self.commands.put_nowait('PONG ' + line[5:])
                else:
                    self.handle_line(line)
        async def send():
            while True:
                writer.write((await self.commands.get()).encode() + b'\r\n')
                await asyncio.wait_for(writer.drain(), 3)
        async def guarded(function):
            try:
                await function()
            except (OSError, ValueError, asyncio.TimeoutError, asyncio.QueueFull) as error:
                self.connected, self.status = False, 'disconnected'
                self.show(f'Disconnected: {error}. Ctrl-Q to exit.', kind='error')
        try:
            for line in ('CAP REQ :message-tags server-time echo-message batch draft/chathistory', f'PASS {endpoint["password"]}', f'NICK {self.nickname}',
                         f'USER {self.nickname} 0 * :Swarm TUI', 'CAP END', f'JOIN {self.target}'):
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
