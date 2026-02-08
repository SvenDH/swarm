from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Any

import runtime


def _user(name: str) -> str:
    return f"user:{name}"


def _group(namespace: str, layer: int | None = None) -> str:
    ns = runtime._normalize_ns(namespace)
    return f"group:{ns}:all" if layer is None else f"group:{ns}:layer:{int(layer)}"


def _grant(subject: str, target: str, *, read: bool = True, write: bool = True, temp: bool = False):
    return runtime.edge(subject, target, rel="can", read=bool(read), write=bool(write), temp=temp)


def _graph_target(namespace: str | None) -> str:
    if not namespace:
        raise ValueError("namespace is required when target is omitted")
    return f"graph:{namespace}:shared"


def terms(
    namespace: str,
    name: str,
    *,
    layer: int,
    context: str,
    memory: str,
    graph: str,
    temp: bool = False,
):
    u = _user(name)
    g_all = _group(namespace)
    g_layer = _group(namespace, layer)
    return [
        runtime.node(u, kind="user", owner=name, layer=int(layer), temp=temp),
        runtime.node(g_all, kind="group", scope="all", temp=temp),
        runtime.node(g_layer, kind="group", scope="layer", layer=int(layer), temp=temp),
        runtime.edge(name, u, rel="uses_identity", temp=temp),
        runtime.edge(u, g_all, rel="member_of", temp=temp),
        runtime.edge(u, g_layer, rel="member_of", temp=temp),
        _grant(name, context, temp=temp),
        _grant(name, memory, temp=temp),
        _grant(name, graph, temp=temp),
        _grant(u, context, temp=temp),
        _grant(u, memory, temp=temp),
        _grant(g_all, graph, temp=temp),
        _grant(g_layer, graph, temp=temp),
    ]


def check(subject: str, target: str, *, action: str = "read", namespace: str = "", engine=None) -> bool:
    if action not in {"read", "write"}:
        raise ValueError("action must be 'read' or 'write'")
    eng = _default_engine() if engine is None else engine
    ns = runtime._normalize_ns(namespace) if namespace else ""
    path = f"$.{action}"
    row = eng.db.execute(
        """
        SELECT EXISTS(
            SELECT 1
            FROM hyperedges p
            WHERE p.namespace = ?
              AND p.relation = 'can'
              AND p.node0 = ?
              AND p.node1 = ?
              AND json_extract(p.data, ?) = 1
            UNION
            SELECT 1
            FROM hyperedges m
            JOIN hyperedges p
              ON p.namespace = m.namespace
             AND p.relation = 'can'
             AND p.node0 = m.node1
             AND p.node1 = ?
             AND json_extract(p.data, ?) = 1
            WHERE m.namespace = ?
              AND m.relation = 'member_of'
              AND m.node0 = ?
        ) AS allowed
        """,
        (ns, subject, target, path, target, path, ns, subject),
    ).fetchone()
    return bool(int(row["allowed"])) if row is not None else False


@contextmanager
def _use_engine(engine):
    if engine is None:
        yield
        return
    prev = runtime._ENGINE
    runtime._ENGINE = engine
    try:
        yield
    finally:
        runtime._ENGINE = prev


def _default_engine():
    eng = getattr(runtime, "_ENGINE", None)
    if eng is None:
        eng = runtime._Engine()
        runtime._ENGINE = eng
    return eng


def _split_exec_args(args):
    cmds, seed = [], []
    for item in args:
        if isinstance(item, runtime.Command):
            cmds.append(item)
            continue
        if isinstance(item, dict):
            if not bool(item.get("temp", False)):
                raise TypeError("non-command exec args must be edge/node with temp=True")
            seed.append(item)
            continue
        raise TypeError("exec() arguments must be runtime commands or temp edge/node terms")
    if not cmds:
        raise TypeError("exec() requires at least one command")
    return cmds, seed


@dataclass(frozen=True)
class Access:
    name: str
    namespace: str | None = None
    as_user: bool = True
    engine: Any = None

    def __post_init__(self):
        ns = None if self.namespace is None else runtime._normalize_ns(self.namespace)
        object.__setattr__(self, "namespace", ns)

    @property
    def subject(self) -> str:
        return _user(self.name) if self.as_user else self.name

    def ns(self, namespace: str):
        return replace(self, namespace=runtime._normalize_ns(namespace))

    # ACL checks + mutations.
    def can(self, target: str | None = None, *, action: str = "read", namespace: str | None = None) -> bool:
        ns = self._resolve_namespace(namespace=namespace, target=target)
        t = target or _graph_target(ns)
        return check(self.subject, t, action=action, namespace=ns, engine=self.engine)

    def require(
        self,
        target: str | None = None,
        *,
        action: str = "read",
        temp: bool = False,
        namespace: str | None = None,
    ) -> None:
        if temp:
            return
        ns = self._resolve_namespace(namespace=namespace, target=target)
        t = target or _graph_target(ns)
        if not self.can(t, action=action, namespace=namespace):
            raise PermissionError(f"{self.subject} is not allowed to {action} {t}")

    def allow(
        self,
        target: str,
        *,
        subject: str | None = None,
        read: bool = True,
        write: bool = True,
        temp: bool = False,
        namespace: str | None = None,
    ):
        cmd = runtime.rewrite(to=[_grant(subject or self.subject, target, read=read, write=write)])
        return self.exec(cmd, temp=temp, target=target, namespace=namespace)

    def deny(self, target: str, *, subject: str | None = None, temp: bool = False, namespace: str | None = None):
        return self.allow(target, subject=subject, read=False, write=False, temp=temp, namespace=namespace)

    def join(self, group_id: str, *, temp: bool = False, namespace: str | None = None):
        return self.exec(
            runtime.rewrite(to=[runtime.edge(self.subject, group_id, rel="member_of")]),
            temp=temp,
            target=group_id,
            namespace=namespace,
        )

    def leave(self, group_id: str, *, temp: bool = False, namespace: str | None = None):
        return self.exec(
            runtime.match(runtime.edge(runtime.const(self.subject), runtime.const(group_id), rel="member_of")).rewrite([], limit=1),
            temp=temp,
            target=group_id,
            namespace=namespace,
        )

    def exec(self, *commands, temp=False, target: str | None = None, namespace: str | None = None):
        cmds, seed = _split_exec_args(commands)
        action = "write" if any(getattr(c, "rhs", None) is not None for c in cmds) else "read"
        ns = self._resolve_namespace(
            namespace=namespace,
            target=target,
            command_namespaces=[getattr(c, "namespace", None) for c in cmds],
        )
        self.require(target, action=action, temp=temp, namespace=ns)
        bound = [self._bind(c, namespace=ns, temp=temp) for c in cmds]
        return self._exec(*bound, mem=seed, temp=temp)

    def _bind(self, command, *, namespace: str, temp=False):
        if not hasattr(command, "namespace"):
            return command
        ns = getattr(command, "namespace", None)
        if ns and ns != namespace and not temp:
            raise PermissionError(f"cross-namespace command blocked: {ns} != {namespace}")
        if ns != namespace:
            return replace(command, namespace=namespace)
        return command

    def _resolve_namespace(
        self,
        *,
        namespace: str | None = None,
        target: str | None = None,
        command_namespaces: list[str | None] | None = None,
    ) -> str:
        if namespace:
            return runtime._normalize_ns(namespace)
        explicit = {ns for ns in (command_namespaces or []) if ns}
        if len(explicit) > 1:
            raise PermissionError(f"cross-namespace command batch blocked: {sorted(explicit)}")
        if len(explicit) == 1:
            return runtime._normalize_ns(next(iter(explicit)))
        if self.namespace:
            return self.namespace
        source = target
        if isinstance(source, str) and source.startswith("graph:"):
            parts = source.split(":")
            if len(parts) >= 3 and parts[1]:
                return runtime._normalize_ns(parts[1])
        return ""

    def _exec(self, *commands, mem=None, temp=False):
        with _use_engine(self.engine):
            eng = _default_engine()
            return eng.run(*commands, mem=mem, temp=temp)


def client(name: str, *, as_user: bool = True, engine=None) -> Access:
    return Access(name=name, as_user=as_user, engine=engine)


__all__ = ["Access", "client", "terms", "check"]
