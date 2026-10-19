"""Declarative, reviewed API maps -- never fallback attribute forwarding."""
from enum import Enum


def select(namespace, names):
    """Names are developer-owned allowlists, not sandbox input.

    Omit APIs absent from the installed host version. Callers still own argument,
    result, virtual-IO and checkpoint adapters for each selected API.
    """
    return {name: namespace[name] for name in names.split() if name in namespace}


def enum_values(cls):
    """Enum constants cross as integers, not native enum classes/instances."""
    return {name: value.value for name, value in cls.__members__.items() if type(value.value) is int}


def enums(module):
    return {name: enum_values(cls) for name, cls in vars(module).items()
            if not name.startswith('_') and isinstance(cls, type) and issubclass(cls, Enum)
            and cls.__members__ and all(type(value.value) is int for value in cls.__members__.values())}
