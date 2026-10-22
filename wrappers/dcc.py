"""DCC-shaped, recipient-bound byte channels held entirely in worker memory."""
import secrets
import shlex
import time


class DCC:
    def __init__(self):
        self.peers = {}
        self.next_port = secrets.randbits(64) + 65536
        self.closed = False

    async def close(self):
        self.closed = True
        self.peers.clear()

    def drop(self, peer):
        for endpoint in (peer, self.peers.get(peer['partner'])):
            if endpoint is not None and not endpoint['closed']:
                endpoint['closed'] = True
                endpoint['events'].append(('dcc_disconnect', None))

    def offer(self, sender, target, text):
        if not text.startswith('\x01DCC SEND ') or not text.endswith('\x01'):
            return
        parts = shlex.split(text[1:-1])
        if len(parts) != 6 or parts[:2] != ['DCC', 'SEND']:
            raise ValueError('Use DCC SEND filename address port size')
        _, _, _, address, port, size = parts
        if address != '2130706433' or not size.isascii() or not size.isdigit() or int(size) > 16 * 1024**2:
            raise ValueError('DCC offers must use a worker listener and at most 16 MiB')
        peer = next((p for p in self.peers.values() if p['owner'] == sender and
                     not p['closed'] and p['partner'] is None and p['port'] == int(port)), None)
        if peer is None or target.startswith('#') or peer['recipient'] is not None:
            raise PermissionError('DCC requires an unused listener and a direct recipient')
        peer['recipient'] = target

    async def call(self, owner, action, identity=None, value=None):
        if self.closed:
            raise ConnectionError('DCC worker is closed')
        for key, peer in list(self.peers.items()):
            if time.monotonic() > peer['expires']:
                self.drop(peer)
                self.peers.pop(key, None)
        if action in ('listen', 'connect'):
            if len(self.peers) >= 256 or sum(p['owner'] == owner for p in self.peers.values()) >= 8:
                raise ValueError('DCC connection quota exceeded')
            source = None
            if action == 'connect':
                address, port = value
                if address != '127.0.0.1' or type(port) is not int:
                    raise PermissionError('Only offered worker DCC listeners are available')
                source = next((p for p in self.peers.values() if p['port'] == port and
                               not p['closed'] and p['partner'] is None and p['recipient'] == owner), None)
                if source is None:
                    raise PermissionError('No DCC offer for this session')
            identity = secrets.token_hex(24)
            self.next_port += 1
            peer = dict(id=identity, owner=owner, recipient=None, port=self.next_port,
                        partner=None, closed=False, sent=0, queued=0, events=[],
                        expires=time.monotonic() + 3600)
            self.peers[identity] = peer
            if source is not None:
                source['partner'], peer['partner'] = identity, source['id']
                source['events'].append(('dcc_connect', None))
                peer['events'].append(('dcc_connect', None))
            return dict(id=identity, localaddress='127.0.0.1', localport=peer['port'])
        peer = self.peers.get(identity)
        if peer is None or peer['owner'] != owner:
            raise PermissionError('DCC connection unavailable')
        if action == 'poll':
            events, peer['events'] = peer['events'], []
            peer['queued'] = 0
            if peer['closed']:
                self.peers.pop(identity)
            return events
        if action == 'close':
            self.drop(peer)
            self.peers.pop(identity)
            return
        if action != 'send' or type(value) is not bytes:
            raise ValueError('Use send_bytes with bytes')
        other = self.peers.get(peer['partner'])
        if peer['closed'] or other is None or other['closed']:
            raise ConnectionError('DCC is not connected')
        if len(value) > 65536 or peer['sent'] + len(value) > 16 * 1024**2:
            raise ValueError('DCC byte quota exceeded')
        if other['queued'] + len(value) > 256 * 1024 or len(other['events']) >= 64:
            raise BufferError('DCC receive queue is full; poll the recipient before retrying')
        peer['sent'] += len(value)
        if value:
            other['events'].append(('dccmsg', value))
            other['queued'] += len(value)
