"""Bind library tools to one execution's virtual filesystem."""
from functools import partial
from importlib import import_module
from .registry import available
from .limits import memory_scope


def invoke(module, fs, *args, **kwargs):
    # Called on Monty's callback workers: heavy optional imports stay off-loop
    # and hosts that never use a library do not pay its startup/RAM cost.
    from .protocol import plain
    with memory_scope(fs.memory_budget):
        return import_module('.' + module, __name__).run(fs, *plain(args), **plain(kwargs))


def tools(fs):
    return {name: partial(invoke, name.lower(), fs)
            for name in ('numpy', 'pandas', 'PIL', 'openpyxl', 'pypdf', 'docx')
            if name in available()}
