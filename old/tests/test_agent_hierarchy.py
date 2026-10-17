from __future__ import annotations

import glob
import os
import tempfile
import unittest

import graph
from hierarchy import has_permission, route_messages_once, spawn_node


class AgentHierarchyTests(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.db_path = tempfile.mkstemp(prefix="agent_hierarchy_test_", suffix=".db")
        os.close(fd)
        self.engine = graph._Engine(self.db_path)

    def tearDown(self) -> None:
        self.engine.db.close()
        for path in [self.db_path, f"{self.db_path}-wal", f"{self.db_path}-shm"]:
            if os.path.exists(path):
                os.remove(path)
        for extra in glob.glob(f"{self.db_path}*"):
            if os.path.exists(extra):
                os.remove(extra)

    def test_spawn_agent_connects_siblings(self) -> None:
        root = spawn_node(namespace="h", engine=self.engine)["node"]
        c1 = spawn_node(root, namespace="h", engine=self.engine)["node"]
        out = spawn_node(root, namespace="h", engine=self.engine)
        c2 = out["node"]
        self.assertEqual(out["connected_siblings"], 1)

        g = graph.ns("h")
        a, b = graph.vars("a b")

        controls = self.engine.run(g.match(g.edge(a, b, rel="controls")))
        lateral_1 = self.engine.run(g.match(g.edge(graph.const(c1), graph.const(c2), rel="lateral")))
        lateral_2 = self.engine.run(g.match(g.edge(graph.const(c2), graph.const(c1), rel="lateral")))
        ctx_lateral_1 = self.engine.run(
            g.match(g.edge(graph.const(f"ctx:{c1}"), graph.const(f"ctx:{c2}"), rel="answers_lateral_ctx"))
        )
        ctx_lateral_2 = self.engine.run(
            g.match(g.edge(graph.const(f"ctx:{c2}"), graph.const(f"ctx:{c1}"), rel="answers_lateral_ctx"))
        )

        self.assertEqual(len(controls), 2)
        self.assertEqual(len(lateral_1), 1)
        self.assertEqual(len(lateral_2), 1)
        self.assertEqual(len(ctx_lateral_1), 1)
        self.assertEqual(len(ctx_lateral_2), 1)

    def test_spawn_agent_enforces_max_depth(self) -> None:
        root = spawn_node(namespace="h2", engine=self.engine)["node"]
        l1 = spawn_node(root, namespace="h2", engine=self.engine)["node"]
        l2 = spawn_node(l1, namespace="h2", engine=self.engine)["node"]
        l3 = spawn_node(l2, namespace="h2", engine=self.engine)["node"]
        self.assertTrue(l3.startswith("node:"))
        with self.assertRaises(ValueError):
            spawn_node(
                l3,
                namespace="h2",
                engine=self.engine,
            )

    def test_has_permission_group_inheritance(self) -> None:
        root = spawn_node(namespace="hperm", engine=self.engine)["node"]
        l1 = spawn_node(root, namespace="hperm", engine=self.engine)["node"]
        user = f"user:{l1}"
        graph_node = "graph:hperm:shared"
        group = "group:hperm:all"
        self.assertTrue(has_permission(user, graph_node, action="read", namespace="hperm", engine=self.engine))
        self.assertTrue(has_permission(user, graph_node, action="write", namespace="hperm", engine=self.engine))
        self.assertTrue(has_permission(group, graph_node, action="read", namespace="hperm", engine=self.engine))

    def test_route_messages_once(self) -> None:
        root = spawn_node(namespace="h3", engine=self.engine)["node"]
        parent_l1 = spawn_node(root, namespace="h3", engine=self.engine)["node"]
        sibling_l1 = spawn_node(root, namespace="h3", engine=self.engine)["node"]
        leaf = spawn_node(parent_l1, namespace="h3", engine=self.engine)["node"]
        self.assertTrue(sibling_l1.startswith("node:"))
        g = graph.ns("h3")
        src_ctx = f"ctx:{leaf}"
        msg_a = "msg:answer:1"
        msg_c = "msg:command:1"

        self.engine.run(
            g.rewrite(
                to=[
                    g.edge(src_ctx, msg_a, rel="out_answer"),
                    g.edge(f"ctx:{root}", msg_c, rel="out_command"),
                ]
            )
        )

        routed = route_messages_once("h3", engine=self.engine)
        self.assertEqual(routed["answer_delivered"], 1)
        self.assertEqual(routed["command_delivered"], 2)
        self.assertEqual(routed["memory_items_written"], 3)

        a = graph.vars("a")[0]
        answers = self.engine.run(g.match(g.edge(a, graph.const(msg_a), rel="in_answer")))
        commands = self.engine.run(g.match(g.edge(a, graph.const(msg_c), rel="in_command")))
        memory_items = self.engine.run(g.match(g.edge(a, graph.const(msg_a), rel="memory_item")))

        self.assertEqual(len(answers), 1)
        self.assertEqual(len(commands), 2)
        self.assertEqual(len(memory_items), 1)

if __name__ == "__main__":
    unittest.main()
