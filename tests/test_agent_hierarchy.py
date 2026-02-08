from __future__ import annotations

import glob
import os
import tempfile
import unittest

import runtime
from hierarchy import create_agent_hierarchy, has_permission, route_messages_once, spawn_agent


class AgentHierarchyTests(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.db_path = tempfile.mkstemp(prefix="agent_hierarchy_test_", suffix=".db")
        os.close(fd)
        self.engine = runtime._Engine(self.db_path)

    def tearDown(self) -> None:
        self.engine.db.close()
        for path in [self.db_path, f"{self.db_path}-wal", f"{self.db_path}-shm"]:
            if os.path.exists(path):
                os.remove(path)
        for extra in glob.glob(f"{self.db_path}*"):
            if os.path.exists(extra):
                os.remove(extra)

    def _one_agent(self, g: runtime.Namespace, **props: object) -> str:
        a = runtime.vars("a")[0]
        predicates = [a.kind == "agent"]
        for key, value in props.items():
            predicates.append(getattr(a, key) == value)
        out = self.engine.run(g.match(g.node(a), limit=1).where(*predicates))
        self.assertEqual(len(out), 1)
        return out[0]["bindings"]["a"]

    def test_create_agent_hierarchy_basics(self) -> None:
        out = create_agent_hierarchy(2, 2, namespace="h", engine=self.engine)
        self.assertEqual(out["agents_total"], 21)
        self.assertEqual(out["users_total"], 21)
        self.assertEqual(out["groups_total"], 4)
        self.assertEqual(out["root_budget"], 4)
        self.assertEqual(out["child_budget"], 4)
        self.assertEqual(out["edges_can"], 109)
        self.assertEqual(out["edges_can_agent"], 63)
        self.assertEqual(out["edges_can_user"], 42)
        self.assertEqual(out["edges_can_group"], 4)
        self.assertEqual(out["edges_can_read"], 109)
        self.assertEqual(out["edges_can_write"], 109)
        self.assertEqual(out["edges_member_of"], 42)
        self.assertEqual(out["edges_uses_identity"], 21)

        g = runtime.ns("h")
        a, b = runtime.vars("a b")

        agents = self.engine.run(g.match(g.node(a)).where(a.kind == "agent"))
        controls = self.engine.run(g.match(g.edge(a, b, rel="controls")))
        has_context = self.engine.run(g.match(g.edge(a, b, rel="has_context")))
        has_memory = self.engine.run(g.match(g.edge(a, b, rel="has_memory_graph")))
        shares = self.engine.run(g.match(g.edge(a, b, rel="shares_graph")))
        users = self.engine.run(g.match(g.node(a)).where(a.kind == "user"))
        groups = self.engine.run(g.match(g.node(a)).where(a.kind == "group"))
        member_of = self.engine.run(g.match(g.edge(a, b, rel="member_of")))
        uses_identity = self.engine.run(g.match(g.edge(a, b, rel="uses_identity")))
        can = self.engine.run(g.match(g.edge(a, b, rel="can")))
        can_read = self.engine.run(g.match(g.edge(a, b, rel="can")).where(runtime.on(1).read == True))
        can_write = self.engine.run(g.match(g.edge(a, b, rel="can")).where(runtime.on(1).write == True))
        root_id = out["root_agent"]
        root_graph_read = self.engine.run(
            g.match(g.edge(runtime.const(root_id), runtime.const(out["graph_node"]), rel="can")).where(
                runtime.on(1).read == True
            )
        )
        root_graph_write = self.engine.run(
            g.match(g.edge(runtime.const(root_id), runtime.const(out["graph_node"]), rel="can")).where(
                runtime.on(1).write == True
            )
        )

        self.assertEqual(len(agents), 21)
        self.assertEqual(len(controls), 20)
        self.assertEqual(len(has_context), 21)
        self.assertEqual(len(has_memory), 21)
        self.assertEqual(len(shares), 21)
        self.assertEqual(len(users), 21)
        self.assertEqual(len(groups), 4)
        self.assertEqual(len(member_of), 42)
        self.assertEqual(len(uses_identity), 21)
        self.assertEqual(len(can), 109)
        self.assertEqual(len(can_read), 109)
        self.assertEqual(len(can_write), 109)
        self.assertEqual(len(root_graph_read), 1)
        self.assertEqual(len(root_graph_write), 1)

        root_user = f"user:{root_id}"
        self.assertTrue(has_permission(root_user, out["graph_node"], namespace="h", engine=self.engine))

    def test_spawn_agent_respects_parent_budget(self) -> None:
        out = create_agent_hierarchy(
            1,
            1,
            namespace="h2",
            root_budget=2,
            child_budget=1,
            engine=self.engine,
        )
        root = out["root_agent"]

        # Root starts with one layer-1 child, so one additional spawn is allowed.
        spawned = spawn_agent(
            root,
            None,
            namespace="h2",
            max_children=0,
            engine=self.engine,
        )
        self.assertEqual(spawned["parent_children_used"], 2)
        self.assertEqual(spawned["parent_children_remaining"], 0)
        self.assertTrue(spawned["child"].startswith("agent:"))

        with self.assertRaises(ValueError):
            spawn_agent(
                root,
                None,
                namespace="h2",
                max_children=0,
                engine=self.engine,
            )

    def test_has_permission_group_inheritance(self) -> None:
        out = create_agent_hierarchy(1, 1, namespace="hperm", engine=self.engine)
        g = runtime.ns("hperm")
        l1 = self._one_agent(g, layer=1, x=0, y=0)
        user = f"user:{l1}"
        group = out["group_all"]
        self.assertTrue(has_permission(user, out["graph_node"], action="read", namespace="hperm", engine=self.engine))
        self.assertTrue(has_permission(user, out["graph_node"], action="write", namespace="hperm", engine=self.engine))
        self.assertTrue(has_permission(group, out["graph_node"], action="read", namespace="hperm", engine=self.engine))

    def test_route_messages_once(self) -> None:
        create_agent_hierarchy(2, 2, namespace="h3", engine=self.engine)
        g = runtime.ns("h3")

        parent_l1 = self._one_agent(g, layer=1, x=0, y=0)
        leaf = self._one_agent(g, layer=2, parent=parent_l1, x=0, y=0)
        src_ctx = f"ctx:{leaf}"
        msg_a = "msg:answer:1"
        msg_c = "msg:command:1"

        self.engine.run(
            g.rewrite(
                to=[
                    g.edge(src_ctx, msg_a, rel="out_answer"),
                    g.edge(f"ctx:{parent_l1}", msg_c, rel="out_command"),
                ]
            )
        )

        routed = route_messages_once("h3", engine=self.engine)
        self.assertEqual(routed["answer_delivered"], 3)
        self.assertEqual(routed["command_delivered"], 4)
        self.assertEqual(routed["memory_items_written"], 7)

        a = runtime.vars("a")[0]
        answers = self.engine.run(g.match(g.edge(a, runtime.const(msg_a), rel="in_answer")))
        commands = self.engine.run(g.match(g.edge(a, runtime.const(msg_c), rel="in_command")))
        memory_items = self.engine.run(g.match(g.edge(a, runtime.const(msg_a), rel="memory_item")))

        self.assertEqual(len(answers), 3)
        self.assertEqual(len(commands), 4)
        self.assertEqual(len(memory_items), 3)


if __name__ == "__main__":
    unittest.main()
