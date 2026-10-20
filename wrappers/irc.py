"""Bounded worker-local IRC subset. No network access or caller-chosen identity."""
import re
import threading
from collections import deque


class IRCServer:
    def __init__(self):
        self.lock = threading.Lock()
        self.clients = {}
        self.channels = {}
        self.queued_bytes = 0

    def exchange(self, name, command=''):
        if not isinstance(command, str):
            raise TypeError('IRC command must be text')
        if len(command.encode()) > 510 or '\r' in command or '\n' in command or '\x00' in command:
            raise ValueError('IRC command must be one line of at most 510 bytes')
        with self.lock:
            def send(target, line):
                queue = self.clients[target]
                removed = len(queue[0].encode()) if len(queue) == queue.maxlen else 0
                size = len(line.encode()) - removed
                if self.queued_bytes + size > 8 * 1024**2:
                    raise BufferError('IRC message queues are full; poll replies before retrying')
                queue.append(line)
                self.queued_bytes += size
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
