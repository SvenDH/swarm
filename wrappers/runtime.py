"""Monty library integration; no conversation, database, storage or agent ownership."""
import asyncio
import threading
import time
import textwrap
from pydantic_monty import MontyClassProxy
from .protocol import Bridge
from .bridge import backend as library_backend
from .imports import BOOTSTRAP, CHECK, DELETE_BOOTSTRAP, prepare as prepare_libraries, cleanup as cleanup_libraries
from .limits import memory_scope, check_memory, memory_errors


async def settled(future):
    """Drain blocking work before cancellation releases its session/DB lock."""
    cancelled = False
    while True:
        try:
            result = await asyncio.shield(future)
            break
        except asyncio.CancelledError:
            if future.cancelled():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError
    return result


class ExecutionBudget:
    """Wall-clock limit, including host calls; drain callbacks before releasing RAM."""
    progress_interval = 5

    def __init__(self, seconds, emit):
        self.seconds, self.emit = seconds, emit
        self.started = time.monotonic()
        self.condition = threading.Condition()
        self.active = self.calls = 0
        self.closed = False

    def wrap(self, function, *, library=False):
        def call(*args, **kwargs):
            with self.condition:
                if self.closed or time.monotonic() - self.started >= self.seconds:
                    raise TimeoutError('Python execution timed out; split work into smaller blocks')
                self.active += 1
                self.calls += int(library)
            try:
                result = function(*args, **kwargs)
                if self.closed or time.monotonic() - self.started >= self.seconds:
                    raise TimeoutError('Python execution timed out; split work into smaller blocks')
                return result
            finally:
                with self.condition:
                    self.active -= 1
                    self.condition.notify_all()
        return call

    async def __aenter__(self):
        async def progress():
            while True:
                await asyncio.sleep(self.progress_interval)
                elapsed = time.monotonic() - self.started
                label = 'Stopping Python safely' if elapsed >= self.seconds else 'Python running'
                self.emit('status', f'{label}: {elapsed:.0f}s; {self.calls} library calls\n')
        self.reporter = asyncio.create_task(progress())
        return self

    async def __aexit__(self, *args):
        await settled(asyncio.create_task(self._close()))

    async def _close(self):
        with self.condition:
            self.closed = True
            active = self.active
        self.reporter.cancel()
        await asyncio.gather(self.reporter, return_exceptions=True)
        if active:
            self.emit('status', 'Stopping Python; waiting for the active native operation to finish safely\n')
            def drain():
                with self.condition:
                    self.condition.wait_for(lambda: self.active == 0)
            await settled(asyncio.create_task(asyncio.to_thread(drain)))


async def evaluate(runtime, fs, owner, code, checkpoint, printed, emit, budget, *, memory_budget):
    bridge = Bridge(fs,
                     owner,
                     lambda module: runtime.offload.submit(library_backend, module).result(),
                     memory_budget=memory_budget, memory=runtime._native)
    images = []
    @memory_errors
    def display(value):
        from wrappers.display import preview
        data = preview(fs, bridge, value)
        with memory_scope(memory_budget):
            check_memory(sum(map(len, images)) + len(data))
        images.append(data)
    @memory_errors
    def prepare_exec(source, isolated=False):
        # Transform source, never execute it on the host. The builtin exec
        # stays in the caller's interpreter scope with the same IO/limits.
        if not isinstance(source, (str, bytes)):
            raise TypeError('exec() arg 1 must be a string or bytes')
        if len(source) > memory_budget:
            raise MemoryError('source exceeds the session memory budget')
        translated = runtime.offload.submit(prepare_libraries, source, True).result()[0]
        if isolated:
            bootstrap = ("try:\n    _library_import\nexcept NameError:\n" +
                         textwrap.indent(BOOTSTRAP, '    ') + '\n' + DELETE_BOOTSTRAP)
            # Populate the globals dictionary even when exec has a distinct
            # locals mapping: definitions resolve globals, not those locals.
            first, _, rest = translated.partition('\n')
            return (f'exec({bootstrap!r}, _library_exec_scope)\n' + first + '\n' +
                    "_library_exec_scope['_library_global_names'] = _library_global_names\n" + rest)
        return translated
    def object_fields(value, fields_only):
        if isinstance(value, MontyClassProxy):
            return list(value.attributes)
        if fields_only:
            raise TypeError('vars() argument must have __dict__ attribute')
        if type(value) in (list, tuple, dict, str, bytes, bool, int, float, complex, type(None)):
            return [name for name in dir(value) if not name.startswith('_')]
        raise TypeError('Object does not support attribute discovery')
    functions = {'_library_rpc': budget.wrap(bridge, library=True), 'display': budget.wrap(display),
                 '_prepare_exec': budget.wrap(prepare_exec), '_object_fields': budget.wrap(object_fields)}
    # AsyncMonty runs synchronous OS/library callbacks on its worker threads.
    # Keep these synchronous so sandbox callers need not write `await numpy(...)`.
    async with runtime._pool.checkout(limits={'max_memory': memory_budget,
                                   'max_feed_duration_secs': 10, 'max_suspensions': 1000}) as session:
        async def feed(source):
            return await session.feed_run(source, external_lookup=functions, print_callback=lambda *args: None)
        active = False
        helpers_ready = False
        loaded = None
        cleanup = await runtime.offload(cleanup_libraries, code)
        if checkpoint is not None:
            await session.load_session(checkpoint)
        if cleanup is not None:
            # Free first; do not spend the remaining heap on helper refresh.
            await feed(cleanup)
        if checkpoint is not None or cleanup is not None:
            loaded, helpers_ready = await feed(CHECK)
            active = loaded is not None
            if active and not helpers_ready:
                raise ImportError('session runtime changed; start a new session')
        if cleanup is None:
            code, libraries = await runtime.offload(prepare_libraries, code, active, loaded or ())
        else:
            code, libraries = '', True
        if libraries and not active:
            await feed(BOOTSTRAP)
        if libraries and not helpers_ready:
            await feed(DELETE_BOOTSTRAP)
        byte_count = 0
        def capture(stream, text):
            nonlocal byte_count
            byte_count += len(text.encode())
            if byte_count > 16000:
                raise MemoryError('printed output exceeds 16000 bytes')
            printed.append(text)
            emit('stdout', text)
        value = await session.feed_run(code, external_lookup=functions, os=budget.wrap(fs),
                                 cwd=fs.cwd, print_callback=budget.wrap(capture))
        stdout = ''.join(printed)
        if isinstance(value, MontyClassProxy) and value.name == '_LibraryRender':
            rendered = value.attributes['text']
        else:
            rendered = repr(value) if value is not None else ''
        output = (stdout + rendered)[:24000]
        snapshot = await session.dump()
        with runtime._native.lock:
            runtime._native.enforce(extra=len(snapshot) * 2, owner=owner)
        # Checkpoint/restore needs another serialized copy. Reserve that
        # copy plus 25% of the budget for the next feed and cleanup.
        if len(snapshot) * 2 > memory_budget * 3 // 4:
            raise MemoryError()
        return snapshot, output, stdout, images
