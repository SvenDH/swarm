from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import graph as graph_rt


def _user(name: str) -> str:
    return f"user:{name}"


def _group(namespace: str, layer: int | None = None) -> str:
    ns = graph_rt.normalize_ns(namespace)
    return f"group:{ns}:all" if layer is None else f"group:{ns}:layer:{int(layer)}"


def _grant(subject: str, target: str, *, read: bool = True, write: bool = True, temp: bool = False):
    return graph_rt.edge(subject, target, rel="can", read=bool(read), write=bool(write), temp=temp)


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
    grants = [
        (u, context),
        (u, memory),
        (u, graph),
        (g_all, graph),
        (g_layer, graph),
    ]
    return [
        graph_rt.node(u, kind="user", owner=name, layer=int(layer), temp=temp),
        graph_rt.node(g_all, kind="group", scope="all", temp=temp),
        graph_rt.node(g_layer, kind="group", scope="layer", layer=int(layer), temp=temp),
        graph_rt.edge(name, u, rel="uses_identity", temp=temp),
        graph_rt.edge(u, g_all, rel="member_of", temp=temp),
        graph_rt.edge(u, g_layer, rel="member_of", temp=temp),
        *[_grant(s, t, temp=temp) for s, t in grants],
    ]


def check(subject: str, target: str, *, action: str = "read", namespace: str = "", engine=None) -> bool:
    if action not in {"read", "write"}:
        raise ValueError("action must be 'read' or 'write'")
    eng = graph_rt.get_engine() if engine is None else engine
    ns = graph_rt.normalize_ns(namespace) if namespace else ""
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


def _split_exec_args(args):
    cmds, seed = [], []
    for item in args:
        if isinstance(item, graph_rt.Command):
            cmds.append(item)
            continue
        if isinstance(item, dict):
            if not bool(item.get("temp", False)):
                raise TypeError("non-command exec args must be edge/node with temp=True")
            try:
                graph_rt.validate_term(item)
            except Exception as exc:
                raise TypeError("non-command exec args must be edge/node terms") from exc
            seed.append(item)
            continue
        raise TypeError("exec() arguments must be graph commands or temp edge/node terms")
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
        if self.namespace is not None:
            object.__setattr__(self, "namespace", graph_rt.normalize_ns(self.namespace))

    @property
    def subject(self) -> str:
        return _user(self.name) if self.as_user else self.name

    def ns(self, namespace: str):
        return replace(self, namespace=graph_rt.normalize_ns(namespace))

    def _resolve_namespace(
        self,
        *,
        namespace: str | None = None,
        target: str | None = None,
        command_namespaces: list[str | None] | None = None,
    ) -> str:
        if namespace:
            return graph_rt.normalize_ns(namespace)
        explicit = {graph_rt.normalize_ns(ns) for ns in (command_namespaces or []) if ns}
        if len(explicit) > 1:
            raise PermissionError(f"cross-namespace command batch blocked: {sorted(explicit)}")
        if explicit:
            return next(iter(explicit))
        if self.namespace:
            return self.namespace
        if isinstance(target, str) and target.startswith("graph:"):
            parts = target.split(":")
            if len(parts) >= 3 and parts[1]:
                return graph_rt.normalize_ns(parts[1])
        return ""

    def _bind_commands(self, commands: list[graph_rt.Command], namespace: str, temp: bool):
        out: list[graph_rt.Command] = []
        for cmd in commands:
            ns = getattr(cmd, "namespace", None)
            if ns and ns != namespace and not temp:
                raise PermissionError(f"cross-namespace command blocked: {ns} != {namespace}")
            out.append(cmd if ns == namespace else replace(cmd, namespace=namespace))
        return out

    def can(self, target: str | None = None, *, action: str = "read", namespace: str | None = None) -> bool:
        ns = self._resolve_namespace(namespace=namespace, target=target)
        return check(self.subject, target or _graph_target(ns), action=action, namespace=ns, engine=self.engine)

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
        if not check(self.subject, t, action=action, namespace=ns, engine=self.engine):
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
        cmd = graph_rt.rewrite(to=[_grant(subject or self.subject, target, read=read, write=write)])
        return self.exec(cmd, temp=temp, target=target, namespace=namespace)

    def deny(self, target: str, *, subject: str | None = None, temp: bool = False, namespace: str | None = None):
        return self.allow(target, subject=subject, read=False, write=False, temp=temp, namespace=namespace)

    def join(self, group_id: str, *, temp: bool = False, namespace: str | None = None):
        cmd = graph_rt.rewrite(to=[graph_rt.edge(self.subject, group_id, rel="member_of")])
        return self.exec(cmd, temp=temp, target=group_id, namespace=namespace)

    def leave(self, group_id: str, *, temp: bool = False, namespace: str | None = None):
        cmd = graph_rt.match(graph_rt.edge(graph_rt.const(self.subject), graph_rt.const(group_id), rel="member_of")).rewrite([], limit=1)
        return self.exec(cmd, temp=temp, target=group_id, namespace=namespace)

    def exec(self, *commands, temp=False, target: str | None = None, namespace: str | None = None):
        cmds, seed = _split_exec_args(commands)
        ns = self._resolve_namespace(
            namespace=namespace,
            target=target,
            command_namespaces=[c.namespace for c in cmds],
        )
        action = "write" if any(c.rhs is not None for c in cmds) else "read"
        self.require(target, action=action, temp=temp, namespace=ns)
        with graph_rt.using_engine(self.engine) as eng:
            return eng.run(*self._bind_commands(cmds, ns, temp), mem=seed, temp=temp)


def client(name: str, namespace: str | None = None, *, as_user: bool = True, engine=None) -> Access:
    return Access(name=name, namespace=namespace, as_user=as_user, engine=engine)


__all__ = ["Access", "client", "terms", "check"]
