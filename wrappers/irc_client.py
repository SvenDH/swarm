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
        if self.connections:
            raise _LibraryIRCError('Only one connection per reactor is supported')
        connection = _LibraryIRCConnection(self)
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
        for connection in self.connections:
            connection.process_once()

    def process_forever(self, timeout=0.2):
        raise NotImplementedError('Use bounded process_once(timeout=0) calls')

    def disconnect_all(self, message=''):
        for connection in self.connections:
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

    def send_file(self, target, path):
        if not self.connected:
            raise _LibraryIRCError('Not connected')
        if path is None:
            raise TypeError('Choose a source file')
        reference = _irc_file('share', str(path))
        self.privmsg(target, '[file] ' + reference)
        return reference

    def file_info(self, reference):
        return _irc_file('info', reference)

    def receive_file(self, reference, destination, overwrite=False):
        if destination is None:
            raise TypeError('Choose a destination file')
        return _irc_file('receive', reference, str(destination), overwrite=overwrite)

    def history(self, target, limit=50, after=0, before=0):
        if not self.connected:
            raise _LibraryIRCError('Not connected')
        if (type(limit) is not int or not 1 <= limit <= 100 or
                any(type(value) is not int or not 0 <= value < 2**63 for value in (after, before))):
            raise ValueError('History needs limit 1..100 and nonnegative integer cursors')
        if after and before:
            request = 'BETWEEN ' + target + ' msgid=' + str(after) + ' msgid=' + str(before)
        elif after or before:
            request = ('AFTER ' if after else 'BEFORE ') + target + ' msgid=' + str(after or before)
        else:
            request = 'LATEST ' + target + ' *'
        rows, batch, complete, error = [], None, False, None
        for original in _irc_exchange('CHATHISTORY ' + request + ' ' + str(limit)):
            line, tags = original, {}
            if line.startswith('@'):
                raw, _, line = line.partition(' ')
                for tag in raw[1:].split(';'):
                    key, _, value = tag.partition('=')
                    tags[key] = value
            head, _, text = line.partition(' :')
            words = head.split()
            if len(words) >= 5 and words[1] == 'BATCH' and words[2].startswith('+') and words[3] == 'chathistory' and words[4].lower() == target.lower():
                batch = words[2][1:]
            elif batch is not None and len(words) >= 3 and words[1] == 'BATCH' and words[2] == '-' + batch:
                complete = True
            elif batch is not None and tags.get('batch') == batch and len(words) >= 3 and words[1] == 'PRIVMSG':
                rows.append({'id': int(tags['msgid']), 'timestamp': tags['time'].replace('Z', '+00:00'),
                             'sender': words[0][1:].split('!')[0], 'target': words[2], 'text': text})
            elif len(words) >= 3 and words[1:3] == ['FAIL', 'CHATHISTORY']:
                error = text
            else:
                self.pending.append(original)
        if error is not None:
            raise _LibraryIRCError(error)
        if not complete:
            raise _LibraryIRCError('Incomplete history response; retry with a smaller limit')
        return rows

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
            for priority, name, handler in self.reactor.handlers[:]:
                if name in (event.type, 'all_events'):
                    if handler(self, event) == 'NO MORE':
                        break


class _LibraryIRCModule:
    def __init__(self):
        self.Reactor = _LibraryIRCReactor
        self.ServerConnection = _LibraryIRCConnection
        self.ServerConnectionError = _LibraryIRCError
        self.Event = _LibraryIRCEvent


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
