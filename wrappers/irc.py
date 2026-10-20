"""Authenticated loopback IRC server and pinned agent socket transport."""
import asyncio
import hashlib
import hmac
import re
import secrets
import threading
from collections import deque


class IRCServer:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients = {}
        self.channels = {}
        self.queued_bytes = 0
        self.listener = None
        self.tasks = set()
        self.signals = {}
        self.agents = {}
        self.secret = secrets.token_bytes(32)
        self.password = secrets.token_hex(32)
        self.closing = False

    async def start(self, port=0):
        self.listener = await asyncio.start_server(self._connection, '127.0.0.1', port, limit=512)
        self.port = self.listener.sockets[0].getsockname()[1]

    def endpoint(self):
        return {'host': '127.0.0.1', 'port': self.port, 'password': self.password}

    def _password(self, name):
        return hmac.new(self.secret, name.encode(), hashlib.sha256).hexdigest()

    async def close(self):
        self.closing = True
        if self.listener is not None:
            self.listener.close()
            await self.listener.wait_closed()
        await asyncio.gather(*(client.close() for client in tuple(self.agents.values())))
        self.agents.clear()
        for task in tuple(self.tasks):
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)

    async def agent_exchange(self, name, command):
        valid_line(command)
        client = self.agents.get(name)
        if client is None:
            client = IRCTransport()
            try:
                await asyncio.wait_for(client.connect('127.0.0.1', self.port, name, self._password(name)), 3)
            except BaseException:
                await client.close()
                raise
            self.agents[name] = client
        try:
            return await client.exchange(command)
        except BaseException:
            self.agents.pop(name, None)
            await client.close()
            raise
        finally:
            if command.split(' ', 1)[0].upper() == 'QUIT':
                self.agents.pop(name, None)

    async def _connection(self, reader, writer):
        task = asyncio.current_task()
        self.tasks.add(task)
        name, pump = None, None
        async def write(lines):
            for line in lines:
                writer.write(line.encode() + b'\r\n')
            await asyncio.wait_for(writer.drain(), 3)
        try:
            if self.closing or len(self.tasks) > 10000:
                return
            writer.transport.set_write_buffer_limits(high=16384)
            password, nick, user = '', '', False
            async def register():
                nonlocal password, nick, user
                for _ in range(8):
                    raw = await reader.readline()
                    if not raw:
                        raise ConnectionError('IRC registration closed')
                    line = raw.decode().rstrip('\r\n')
                    valid_line(line)
                    op, _, value = line.partition(' ')
                    if op.upper() == 'PASS':
                        password = value.lstrip(':')
                    elif op.upper() == 'NICK':
                        nick = value.lstrip(':').lower()
                    elif op.upper() == 'USER':
                        user = True
                    elif op.upper() == 'CAP':
                        if value.split(' ', 1)[0].upper() == 'LS':
                            await write([':worker CAP * LS :'])
                        elif value.split(' ', 1)[0].upper() == 'REQ':
                            await write([':worker CAP * NAK :' + value.partition(' :')[2]])
                    else:
                        raise ValueError('Expected PASS, NICK and USER')
                    if password and nick and user:
                        return
                raise ValueError('Incomplete IRC registration')
            await asyncio.wait_for(register(), 5)
            if not re.fullmatch(r'[a-z][a-z0-9_-]{0,63}', nick):
                raise ValueError('Invalid nickname')
            human = nick.startswith('user-') and hmac.compare_digest(password.encode(), self.password.encode())
            agent = not nick.startswith('user-') and hmac.compare_digest(password.encode(), self._password(nick).encode())
            if not (human or agent):
                await write([':worker 464 * :Authentication failed'])
                return
            if nick in self.clients:
                await write([f':worker 433 * {nick} :Nickname is already in use'])
                return
            name = nick
            signal = self.signals[name] = asyncio.Event()
            await write(self.exchange(name))
            async def outbound():
                while True:
                    await signal.wait()
                    signal.clear()
                    while self.clients.get(name):
                        await write(self.exchange(name))
            pump = asyncio.create_task(outbound())
            pump.add_done_callback(lambda done: writer.close())
            while True:
                raw = await reader.readline()
                if not raw:
                    break
                line = raw.decode().rstrip('\r\n')
                valid_line(line)
                await write(self.exchange(name, line))
                if line.split(' ', 1)[0].upper() == 'QUIT':
                    break
        except (ValueError, UnicodeError, ConnectionError, OSError, asyncio.TimeoutError, BufferError):
            pass
        finally:
            if pump is not None:
                pump.cancel()
                await asyncio.gather(pump, return_exceptions=True)
            if name is not None:
                self.signals.pop(name, None)
                if name in self.clients:
                    self.exchange(name, 'QUIT')
            writer.close()
            try:
                await asyncio.wait_for(writer.wait_closed(), 3)
            except (ConnectionError, OSError, asyncio.TimeoutError):
                writer.transport.abort()
            self.tasks.discard(task)

    def exchange(self, name, command=''):
        valid_line(command)
        with self.lock:
            def send(target, line):
                line = line.encode()[:510].decode(errors='ignore')
                queue = self.clients[target]
                removed = len(queue[0].encode()) if len(queue) == queue.maxlen else 0
                size = len(line.encode()) - removed
                if self.queued_bytes + size > 8 * 1024**2:
                    raise BufferError('IRC message queues are full; poll replies before retrying')
                queue.append(line)
                self.queued_bytes += size
                if target in self.signals:
                    self.signals[target].set()
            inbox = self.clients.get(name)
            if inbox is None:
                if len(self.clients) >= 10000:
                    raise RuntimeError('IRC server is full')
                inbox = self.clients[name] = deque(maxlen=128)
                send(name, f':worker 001 {name} :Welcome')
            def broadcast(channel, line):
                for member in self.channels[channel]:
                    send(member, line)
            if command:
                head, _, trailing = command.partition(' :')
                words = head.split()
                op = words[0].upper() if words else ''
                target = words[1].lower() if len(words) > 1 else ''
                if op == 'PING':
                    send(name, ':worker PONG worker :' + (trailing or target))
                elif op == 'LIST':
                    for channel, members in self.channels.items():
                        send(name, f':worker 322 {name} {channel} {len(members)} :')
                    send(name, f':worker 323 {name} :End of LIST')
                elif op in ('JOIN', 'PART', 'NAMES', 'PRIVMSG'):
                    channel = target.startswith('#')
                    if channel and not re.fullmatch(r'#[a-z0-9_-]{1,63}', target):
                        raise ValueError('Invalid channel name')
                    if op == 'JOIN' and channel:
                        if target not in self.channels and len(self.channels) >= 256:
                            raise RuntimeError('Too many IRC channels')
                        self.channels.setdefault(target, set()).add(name)
                        broadcast(target, f':{name} JOIN {target}')
                    elif op == 'NAMES' and channel:
                        prefix = f':worker 353 {name} = {target} :'
                        line = prefix
                        for member in sorted(self.channels.get(target, ())):
                            if len((line + member).encode()) > 509:
                                send(name, line.rstrip())
                                line = prefix
                            line += member + ' '
                        send(name, line.rstrip())
                        send(name, f':worker 366 {name} {target} :End of NAMES')
                    elif channel and name not in self.channels.get(target, ()):
                        send(name, f':worker 442 {name} {target} :Not on channel')
                    elif op == 'PART' and channel:
                        broadcast(target, f':{name} PART {target}')
                        self.channels[target].remove(name)
                        if not self.channels[target]:
                            del self.channels[target]
                    elif op == 'PRIVMSG' and trailing:
                        if channel:
                            for member in self.channels[target] - {name}:
                                send(member, f':{name} PRIVMSG {target} :{trailing}')
                        elif target in self.clients:
                            send(target, f':{name} PRIVMSG {target} :{trailing}')
                        else:
                            send(name, f':worker 401 {name} {target} :No such nick')
                    else:
                        send(name, f':worker 461 {name} {op} :Invalid parameters')
                elif op == 'QUIT':
                    for channel in list(self.channels):
                        self.channels[channel].discard(name)
                        if not self.channels[channel]:
                            del self.channels[channel]
                    self.queued_bytes -= sum(len(line.encode()) for line in inbox)
                    inbox.clear()
                    del self.clients[name]
                else:
                    send(name, f':worker 421 {name} {op} :Unknown or disabled command')
            result = [inbox.popleft() for _ in range(min(16, len(inbox)))]
            self.queued_bytes -= sum(len(line.encode()) for line in result)
            return result


def valid_line(line):
    if not isinstance(line, str):
        raise TypeError('IRC command must be text')
    if len(line.encode()) > 510 or any(char in line for char in ('\r', '\n', '\x00')):
        raise ValueError('IRC command must be one line of at most 510 bytes')


class IRCTransport:
    """Host-only socket; endpoint and credentials never enter the interpreter."""
    def __init__(self):
        self.reader = self.writer = None
        self.lock = asyncio.Lock()

    async def connect(self, host, port, name, password):
        self.reader, self.writer = await asyncio.open_connection(host, port, limit=512)
        for line in (f'PASS {password}', f'NICK {name}', f'USER {name} 0 * :{name}'):
            valid_line(line)
            self.writer.write(line.encode() + b'\r\n')
        await self.writer.drain()

    async def exchange(self, command=''):
        valid_line(command)
        async with self.lock:
            if command:
                self.writer.write(command.encode() + b'\r\n')
            if command.split(' ', 1)[0].upper() == 'QUIT':
                await self.close()
                return []
            marker = secrets.token_hex(12)
            self.writer.write(f'PING :{marker}\r\n'.encode())
            await self.writer.drain()
            lines = deque(maxlen=128)
            async def receive():
                while True:
                    raw = await self.reader.readline()
                    if not raw:
                        raise ConnectionError('IRC connection closed; reconnect')
                    line = raw.decode().rstrip('\r\n')
                    if line == f':worker PONG worker :{marker}':
                        return list(lines)
                    lines.append(line)
            return await asyncio.wait_for(receive(), 5)

    async def close(self):
        if self.writer is not None:
            self.writer.close()
            try:
                await asyncio.wait_for(self.writer.wait_closed(), 3)
            except (ConnectionError, OSError, asyncio.TimeoutError):
                self.writer.transport.abort()
