"""Executed inside Monty: the worker-local subset of irc.client."""


_LibraryIRCError = RuntimeError


class _LibraryIRCEvent:
    def __init__(self, type, source, target, arguments=None, tags=None):
        self.type = type
        self.source = source
        self.target = target
        self.arguments = arguments or []
        self.tags = tags or []


class _LibraryIRCReactor:
    def __init__(self):
        self.connections = []
        self.handlers = []

    def server(self):
        if any(isinstance(c, _LibraryIRCConnection) for c in self.connections):
            raise _LibraryIRCError('Only one connection per reactor is supported')
        connection = _LibraryIRCConnection(self)
        self.connections.append(connection)
        return connection

    def dcc(self, dcctype='chat'):
        if dcctype != 'raw':
            raise ValueError("Use dcc('raw') for file transfers")
        connection = _LibraryDCCConnection(self)
        self.connections.append(connection)
        return connection

    def add_global_handler(self, event, handler, priority=0):
        self.handlers.append((priority, event, handler))
        self.handlers.sort(key=lambda item: item[0])

    def remove_global_handler(self, event, handler):
        for item in self.handlers[:]:
            if item[1] == event and item[2] == handler:
                self.handlers.remove(item)
                return 1
        return 0

    def process_once(self, timeout=0):
        if timeout != 0:
            raise ValueError('Use process_once(timeout=0) for nonblocking polling')
        for connection in self.connections[:]:
            connection.process_once()

    def process_forever(self, timeout=0.2):
        raise NotImplementedError('Use bounded process_once(timeout=0) calls')

    def disconnect_all(self, message=''):
        for connection in self.connections[:]:
            connection.disconnect(message)


class _LibraryIRCConnection:
    def __init__(self, reactor):
        self.reactor = reactor
        self.connected = False
        self.nickname = None
        self.pending = []

    def connect(self, server, port, nickname=None, password=None, username=None, ircname=None):
        if password is not None:
            raise _LibraryIRCError('Passwords are not supported')
        if self.connected:
            raise _LibraryIRCError('Already connected')
        try:
            reply = _irc_exchange(None, server=server, port=port, nickname=nickname)
        except Exception as error:
            raise _LibraryIRCError(str(error))
        self.nickname = reply['nickname']
        self.pending = reply['lines']
        self.connected = True
        return self

    def is_connected(self):
        return self.connected

    def get_nickname(self):
        return self.nickname

    def send_raw(self, string):
        if not self.connected:
            raise _LibraryIRCError('Not connected')
        self.pending.extend(_irc_exchange(string))

    def join(self, channel, key=''):
        if key:
            raise _LibraryIRCError('Channel keys are not supported')
        self.send_raw('JOIN ' + channel)

    def part(self, channels, message=''):
        for channel in ([channels] if isinstance(channels, str) else channels):
            self.send_raw('PART ' + channel + ' :' + message)

    def privmsg(self, target, text):
        self.send_raw('PRIVMSG ' + target + ' :' + text)

    def ctcp(self, ctcp_type, target, parameter=''):
        self.privmsg(target, '\x01' + ctcp_type + (' ' + parameter if parameter else '') + '\x01')

    def names(self, channels=None):
        for channel in ([channels] if isinstance(channels, str) else (channels or [])):
            self.send_raw('NAMES ' + channel)

    def list(self, channels=None, server=''):
        if channels or server:
            raise _LibraryIRCError('Filtered LIST is not supported')
        self.send_raw('LIST')

    def ping(self, target):
        self.send_raw('PING :' + target)

    def disconnect(self, message=''):
        if self.connected:
            self.send_raw('QUIT :' + message)
            self.connected = False

    def quit(self, message=''):
        self.disconnect(message)

    def add_global_handler(self, event, handler, priority=0):
        self.reactor.add_global_handler(event, handler, priority)

    def remove_global_handler(self, event, handler):
        return self.reactor.remove_global_handler(event, handler)

    def process_once(self, timeout=0):
        if timeout != 0:
            raise ValueError('Use process_once(timeout=0) for nonblocking polling')
        if not self.connected:
            return
        lines = self.pending + _irc_exchange()
        self.pending = []
        numeric = {'001': 'welcome', '322': 'list', '323': 'listend',
                   '353': 'namreply', '366': 'endofnames', '401': 'nosuchnick',
                   '421': 'unknowncommand', '442': 'notonchannel', '461': 'needmoreparams'}
        for line in lines:
            tags = []
            if line.startswith('@'):
                raw, _, line = line.partition(' ')
                for tag in raw[1:].split(';'):
                    key, _, value = tag.partition('=')
                    tags.append({'key': key, 'value': value})
            head, sep, trailing = line.partition(' :')
            words = head.split()
            source, command = words[0][1:], words[1]
            target = words[2] if len(words) > 2 else None
            arguments = words[3:] + ([trailing] if sep else [])
            event = _LibraryIRCEvent(numeric.get(command, command.lower()), source, target, arguments, tags)
            if command == 'PRIVMSG' and target.startswith('#'):
                event.type = 'pubmsg'
            if command == 'PRIVMSG' and trailing.startswith('\x01') and trailing.endswith('\x01'):
                ctcp_type, _, payload = trailing[1:-1].partition(' ')
                event.type, event.arguments = 'ctcp', [ctcp_type, payload]
            for priority, name, handler in self.reactor.handlers[:]:
                if name in (event.type, 'all_events'):
                    if handler(self, event) == 'NO MORE':
                        break


class _LibraryDCCConnection:
    def __init__(self, reactor):
        self.reactor = reactor
        self.identity = None
        self.connected = False
        self.localaddress = None
        self.localport = None

    def listen(self, addr=None):
        if addr not in (None, '127.0.0.1'):
            raise ValueError('Use a worker DCC listener')
        return self._open('listen', None)

    def connect(self, address, port):
        return self._open('connect', (address, port))

    def _open(self, action, value):
        if self.identity is not None:
            raise RuntimeError('DCC connection already opened')
        reply = _irc_dcc(action, None, value)
        self.identity = reply['id']
        self.localaddress, self.localport = reply['localaddress'], reply['localport']
        return self

    def is_connected(self):
        return self.connected

    def send_bytes(self, data):
        _irc_dcc('send', self.identity, data)

    def disconnect(self, message=''):
        if self.identity is not None:
            _irc_dcc('close', self.identity)
            self.identity = None
        self.connected = False
        if self in self.reactor.connections:
            self.reactor.connections.remove(self)

    def process_once(self):
        if self.identity is None:
            return
        for kind, data in _irc_dcc('poll', self.identity):
            self.connected = kind != 'dcc_disconnect'
            event = _LibraryIRCEvent(kind, self.localaddress, None, [data] if data is not None else [])
            for priority, name, handler in self.reactor.handlers[:]:
                if name in (kind, 'all_events') and handler(self, event) == 'NO MORE':
                    break
            if kind == 'dcc_disconnect':
                self.identity = None
                self.connected = False
                if self in self.reactor.connections:
                    self.reactor.connections.remove(self)


def _library_ip_numstr_to_quad(value):
    number = int(value)
    if not 0 <= number < 2**32:
        raise ValueError('Invalid IPv4 address')
    return '.'.join(str((number >> shift) & 255) for shift in (24, 16, 8, 0))


def _library_ip_quad_to_numstr(value):
    parts = value.split('.')
    if len(parts) != 4 or any(not part.isdigit() or not 0 <= int(part) <= 255 for part in parts):
        raise ValueError('Invalid IPv4 address')
    return str(sum(int(parts[i]) << (24 - 8*i) for i in range(4)))


class _LibraryIRCModule:
    def __init__(self):
        self.Reactor = _LibraryIRCReactor
        self.ServerConnection = _LibraryIRCConnection
        self.ServerConnectionError = _LibraryIRCError
        self.Event = _LibraryIRCEvent
        self.DCCConnection = _LibraryDCCConnection
        self.ip_numstr_to_quad = _library_ip_numstr_to_quad
        self.ip_quad_to_numstr = _library_ip_quad_to_numstr


class _LibraryIRCPackage:
    def __init__(self):
        self.client = _LibraryIRCModule()


_library_irc_module = _LibraryIRCPackage()


def _library_irc_import(name):
    if name == 'irc':
        return _library_irc_module
    if name == 'irc.client':
        return _library_irc_module.client
    raise ModuleNotFoundError("No module named '" + name + "'")
