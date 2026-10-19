"""Session-local byte budget, including native allocation preflight checks.

Monty enforces aggregate live heap usage. Native extensions cannot be given a
hard RSS limit inside this shared host process; these checks bound known output
allocations, not all of a native library's transient/scratch memory.
"""
import os
from contextlib import contextmanager
from contextvars import ContextVar

DEFAULT_MEMORY = 50 * 1024**2
_memory = ContextVar('session_memory_budget', default=None)


def limit(name, default):
    value = int(os.environ.get('SWARM_' + name, default))
    if value <= 0:
        raise ValueError('SWARM_' + name + ' must be a positive integer')
    return value


def memory_bytes():
    current = _memory.get()
    return current if current is not None else limit('MEMORY_BYTES', DEFAULT_MEMORY)


@contextmanager
def memory_scope(budget):
    token = _memory.set(budget)
    try:
        yield
    finally:
        _memory.reset(token)


def check_memory(size):
    budget = memory_bytes()
    if size < 0 or size > budget:
        raise MemoryError(f'allocation needs {size:,} bytes; session memory budget is {budget:,} bytes. '
                          'Release unused variables with x=None, use a smaller dtype, '
                          'or process data in chunks.')
