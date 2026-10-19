"""Shared execution budget, with context-local native preflight checks.

The worker serializes active heaps; Monty bounds each active heap. Native extensions cannot be given a
hard RSS limit inside this shared host process; these checks bound known output
allocations, not all of a native library's transient/scratch memory.
"""
import os
import re
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

_NUMPY_ALLOCATION = re.compile(
    r'Unable to allocate [\d.]+ (?:bytes|KiB|MiB|GiB|TiB|PiB|EiB) '
    r'for an array with shape \([\d, ]*\) and data type [^\n]+')


def memory_message(error):
    text = str(error)
    return text if _NUMPY_ALLOCATION.fullmatch(text) else ''


def memory_errors(function):
    """Keep allocation diagnostics private without changing other exceptions."""
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except MemoryError as error:
            text = memory_message(error)
            raise (MemoryError(text) if text else MemoryError()) from None
    return call

DEFAULT_MEMORY = 1024**3
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
