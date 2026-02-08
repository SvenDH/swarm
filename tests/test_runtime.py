from __future__ import annotations

import glob
import os
import tempfile
import unittest

import acl
import runtime


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        fd, self.db_path = tempfile.mkstemp(prefix="runtime_test_", suffix=".db")
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

    def test_runtime_exec_removed(self) -> None:
        self.assertFalse(hasattr(runtime, "exec"))

    def _seed_two_terms(self) -> None:
        self.engine.run(
            runtime.rewrite(
                to=[
                    runtime.edge("a", "a", "b", tag="lhs"),
                    runtime.edge("b", "c", "d", rel="r2", tag="rhs"),
                ]
            ),
        )

    def test_match_order_is_deterministic(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        self._seed_two_terms()
        cmd = runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2"))

        out1 = self.engine.run(cmd)
        out2 = self.engine.run(cmd)

        self.assertEqual(len(out1), 1)
        self.assertEqual(len(out2), 1)
        edges1 = [e["edge_id"] for e in out1[0]["hyperedges"]]
        edges2 = [e["edge_id"] for e in out2[0]["hyperedges"]]
        self.assertEqual(edges1, edges2)

    def test_rewrite_returns_rewritten_subgraph(self) -> None:
        x, y, z, u, v = runtime.vars("x y z u v")
        self._seed_two_terms()

        out = self.engine.run(
            runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")).rewrite(
                [runtime.edge(x, v, u), runtime.edge(y, v, z), runtime.edge(v, v, u)],
            ),
        )

        self.assertEqual(len(out), 1)
        step = out[0]
        self.assertEqual(len(step["hyperedges"]), 3)
        self.assertIn("v", step["bindings"])
        v_id = step["bindings"]["v"]
        node_triples = [tuple(edge["nodes"]) for edge in step["hyperedges"]]
        self.assertIn(("a", v_id, "d"), node_triples)
        self.assertIn(("b", v_id, "c"), node_triples)
        self.assertIn((v_id, v_id, "d"), node_triples)

    def test_rewrite_limit_caps_steps(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        self._seed_two_terms()
        out = self.engine.run(
            runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")).rewrite(
            [runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")],
            limit=2,
        )
        )
        self.assertEqual(len(out), 2)

        after = self.engine.run(runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")))
        self.assertEqual(len(after), 1)

    def test_rewrite_all_overlapping_matches(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        # Two candidate matches share the same first edge:
        # (a,a,b) with (b,c,d) and (a,a,b) with (b,e,f).
        self.engine.run(
            runtime.rewrite(
                to=
                [
                    runtime.edge("a", "a", "b"),
                    runtime.edge("b", "c", "d", rel="r2"),
                    runtime.edge("b", "e", "f", rel="r2"),
                ]
            ),
        )

        cmd = runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")).rewrite(
            [runtime.edge(x, z, u, rel="out")],
            limit=10,
        )
        out = self.engine.run(cmd)

        # Only one rewrite step can run because both initial matches overlap on (a,a,b).
        self.assertEqual(len(out), 1)

        remaining = self.engine.run(runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")))
        self.assertEqual(len(remaining), 0)

    def test_edge_filters_track_original_term_positions(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        self._seed_two_terms()

        cmd = runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")).where(
            runtime.Edge(1).tag == "lhs",
            runtime.Edge(2).tag == "rhs",
        )
        out = self.engine.run(cmd)
        self.assertEqual(len(out), 1)

    def test_node_property_filter(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        self._seed_two_terms()
        self.engine.run(runtime.rewrite(to=[runtime.node("a", kind="entity")]))

        cmd = runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")).where(x.kind == "entity")
        out = self.engine.run(cmd)
        self.assertEqual(len(out), 1)

    def test_random_mode_respects_limit(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        seed_terms = []
        for i in range(8):
            a, b, c, d = f"a{i}", f"b{i}", f"c{i}", f"d{i}"
            seed_terms.append(runtime.edge(a, a, b))
            seed_terms.append(runtime.edge(b, c, d, rel="r2"))
        self.engine.run(runtime.rewrite(to=seed_terms))

        out = self.engine.run(runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2"), limit=8, random=True))
        self.assertEqual(len(out), 8)

    def test_all_mode_returns_multiple_matches(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        seed_terms = []
        for i in range(4):
            a, b, c, d = f"a{i}", f"b{i}", f"c{i}", f"d{i}"
            seed_terms.append(runtime.edge(a, a, b))
            seed_terms.append(runtime.edge(b, c, d, rel="r2"))
        self.engine.run(runtime.rewrite(to=seed_terms))

        out = self.engine.run(runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2"), limit=4))
        self.assertEqual(len(out), 4)

    def test_default_limit_returns_all_matches(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        seed_terms = []
        for i in range(5):
            a, b, c, d = f"a{i}", f"b{i}", f"c{i}", f"d{i}"
            seed_terms.append(runtime.edge(a, a, b))
            seed_terms.append(runtime.edge(b, c, d, rel="r2"))
        self.engine.run(runtime.rewrite(to=seed_terms))

        out = self.engine.run(runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")))
        self.assertEqual(len(out), 5)

    def test_namespace_match_isolated(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        g1, g2 = runtime.ns("g1"), runtime.ns("g2")
        self.engine.run(
            g1.rewrite(to=[g1.edge("a", "a", "b"), g1.edge("b", "c", "d", rel="r2")]),
            g2.rewrite(to=[g2.edge("a", "a", "b"), g2.edge("b", "c", "d", rel="r2")]),
        )

        out = self.engine.run(g1.match(g1.edge(x, x, y), g1.edge(y, z, u, rel="r2"), limit=10))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"u": "d", "x": "a", "y": "b", "z": "c"})

    def test_namespace_rewrite_scopes_fresh_nodes(self) -> None:
        x, y, z, u, v = runtime.vars("x y z u v")
        g1 = runtime.ns("g1")
        self.engine.run(g1.rewrite(to=[g1.edge("a", "a", "b"), g1.edge("b", "c", "d", rel="r2")]))

        out = self.engine.run(g1.match(g1.edge(x, x, y), g1.edge(y, z, u, rel="r2")).rewrite([runtime.edge(x, v, u)]))
        self.assertEqual(len(out), 1)
        self.assertIn("v", out[0]["bindings"])
        self.assertTrue(out[0]["bindings"]["v"].startswith("n_"))

    def test_namespace_match_method(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        g1 = runtime.ns("g1")
        self.engine.run(g1.rewrite(to=[g1.edge("a", "a", "b"), g1.edge("b", "c", "d", rel="r2")]))

        out = self.engine.run(g1.match(g1.edge(x, x, y), g1.edge(y, z, u, rel="r2")))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"u": "d", "x": "a", "y": "b", "z": "c"})

    def test_namespace_handle_and_function_style_helpers(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        g1 = runtime.ns("g1")
        self.engine.run(
            g1.rewrite(
                to=[
                    g1.edge("a", "a", "b", tag="lhs"),
                    g1.edge("b", "c", "d", rel="r2", tag="rhs"),
                    g1.node("a", kind="entity"),
                ]
            )
        )

        out = self.engine.run(g1.match(g1.edge(x, x, y), g1.edge(y, z, u, rel="r2")).where(runtime.on(1).tag == "lhs", x.kind == "entity"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"u": "d", "x": "a", "y": "b", "z": "c"})

        out2 = self.engine.run(g1.match(g1.edge(runtime.const("b"), z, u, rel="r2"), limit=10))
        self.assertEqual(len(out2), 1)
        self.assertEqual(out2[0]["bindings"]["z"], "c")

    def test_namespace_validation(self) -> None:
        with self.assertRaises(ValueError):
            runtime.ns("bad:name")

    def test_friendly_api_aliases(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        self.engine.run(runtime.rewrite(to=[runtime.edge("a", "a", "b"), runtime.edge("b", "c", "d", rel="r2")]))
        self.engine.run(runtime.rewrite(to=[runtime.node("a", kind="entity")]))

        cmd = runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")).where(x.kind == "entity")
        out = self.engine.run(cmd)
        self.assertEqual(len(out), 1)

    def test_permission_options_on_edge(self) -> None:
        a, b = runtime.vars("a b")
        self.engine.run(
            runtime.rewrite(
                to=[
                    runtime.edge("agent:1", "ctx:1", rel="can", read=True, write=True),
                    runtime.edge("agent:1", "mem:1", rel="can", read=True, write=True),
                ]
            )
        )

        can_read = self.engine.run(runtime.match(runtime.edge(a, b, rel="can"), limit=10).where(runtime.on(1).read == True))
        can_write = self.engine.run(runtime.match(runtime.edge(a, b, rel="can"), limit=10).where(runtime.on(1).write == True))
        self.assertEqual(len(can_read), 2)
        self.assertEqual(len(can_write), 2)

    def test_acl_helpers_and_has_permission(self) -> None:
        terms = acl.terms(
            "gacl",
            "agent:alice",
            layer=2,
            context="ctx:agent:alice",
            memory="mem:agent:alice",
            graph="graph:gacl:shared",
        )
        ids = {n for t in terms for n in t.get("nodes", ())}
        self.assertIn("user:agent:alice", ids)
        self.assertIn("group:gacl:all", ids)
        self.assertIn("group:gacl:layer:2", ids)
        self.assertTrue(any(t["relation"] == "member_of" for t in terms))
        self.assertTrue(any(t["relation"] == "uses_identity" for t in terms))
        self.engine.run(runtime.ns("gacl").rewrite(to=terms))

        self.assertTrue(
            acl.check(
                "user:agent:alice",
                "graph:gacl:shared",
                action="read",
                namespace="gacl",
                engine=self.engine,
            )
        )

    def test_acl_user_facing_api(self) -> None:
        g = runtime.ns("aclx")
        user = "user:agent:bob"
        group = "group:aclx:layer:1"
        target = "graph:aclx:shared"

        self.assertEqual(user, "user:agent:bob")
        self.assertEqual("group:aclx:all", "group:aclx:all")
        self.assertEqual(group, "group:aclx:layer:1")

        self.engine.run(
            g.rewrite(
                to=[
                    runtime.node(user, kind="user"),
                    runtime.node(group, kind="group"),
                    runtime.edge("agent:bob", user, rel="uses_identity"),
                    runtime.edge(user, group, rel="member_of"),
                    runtime.edge(group, target, rel="can", read=True, write=False),
                ]
            )
        )

        self.assertTrue(acl.check(user, target, action="read", namespace="aclx", engine=self.engine))
        self.assertFalse(acl.check(user, target, action="write", namespace="aclx", engine=self.engine))
        bob = acl.client("agent:bob", engine=self.engine).ns("aclx")
        self.assertTrue(bob.can(action="read"))
        self.assertFalse(bob.can(action="write"))

    def test_acl_access_wrapper(self) -> None:
        g = runtime.ns("aclw")
        graph = "graph:aclw:shared"
        self.engine.run(
            g.rewrite(
                to=[
                    runtime.node(graph, kind="shared_graph"),
                    runtime.edge("user:alice", graph, rel="can", read=True, write=True),
                ]
            )
        )

        w = acl.client("alice", engine=self.engine).ns("aclw")
        x, y = runtime.vars("x y")
        out = w.exec(
            runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]),
            runtime.match(runtime.edge(x, y, rel="friend"), limit=1),
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1][0]["bindings"], {"x": "a", "y": "b"})

        w.deny(graph)
        with self.assertRaises(PermissionError):
            w.exec(runtime.rewrite(to=[runtime.edge("b", "c", rel="friend")]))

    def test_acl_access_wrapper_temp_always_allowed(self) -> None:
        g = runtime.ns("aclwt")
        graph = "graph:aclwt:shared"
        self.engine.run(g.rewrite(to=[runtime.node(graph, kind="shared_graph")]))
        w = acl.client("nobody", engine=self.engine).ns("aclwt")

        x, y = runtime.vars("x y")
        out = w.exec(
            runtime.rewrite(to=[runtime.edge("t", "u", rel="temp_friend")]),
            runtime.match(runtime.edge(x, y, rel="temp_friend"), limit=1),
            temp=True,
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1][0]["bindings"], {"x": "t", "y": "u"})

        persisted = self.engine.run(g.match(g.edge(x, y, rel="temp_friend"), limit=10))
        self.assertEqual(persisted, [])

    def test_acl_wrap_executes_runtime_commands(self) -> None:
        g = runtime.ns("aclr")
        graph = "graph:aclr:shared"
        self.engine.run(
            g.rewrite(
                to=[
                    runtime.node(graph, kind="shared_graph"),
                    runtime.edge("user:eva", graph, rel="can", read=True, write=True),
                ]
            )
        )
        r = acl.client("eva", engine=self.engine).ns("aclr")
        x, y = runtime.vars("x y")
        out = r.exec(
            runtime.rewrite(to=[runtime.edge("p", "q", rel="friend")]),
            runtime.match(runtime.edge(x, y, rel="friend"), limit=1),
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1][0]["bindings"], {"x": "p", "y": "q"})

    def test_execution_ops_arithmetic(self) -> None:
        a, b, out = runtime.vars("a b out")
        self.engine.run(runtime.rewrite(to=[runtime.edge("2", "3", "sum", rel="add")]))

        step = self.engine.run(
            runtime.match(runtime.edge(a, b, out, rel="add")).rewrite(
                [runtime.node(out, value=a + b, diff=a - b, prod=a * b, quo=b // a)],
            )
        )
        self.assertEqual(len(step), 1)
        data = step[0]["hyperedges"][0]["data"]
        self.assertEqual(data["value"], 5)
        self.assertEqual(data["diff"], -1)
        self.assertEqual(data["prod"], 6)
        self.assertEqual(data["quo"], 1)

    def test_execution_ops_strings(self) -> None:
        a, b, out = runtime.vars("a b out")
        self.engine.run(runtime.rewrite(to=[runtime.edge(" Hello ", "world", "msg", rel="concat")]))

        step = self.engine.run(
            runtime.match(runtime.edge(a, b, out, rel="concat")).rewrite(
                [runtime.node(out, text=a.strip().concat("-", b.upper()), n=a.concat(b).strlen())],
            )
        )
        self.assertEqual(len(step), 1)
        data = step[0]["hyperedges"][0]["data"]
        self.assertEqual(data["text"], "Hello-WORLD")
        self.assertEqual(data["n"], 12)

    def test_run_many_commits_all_commands(self) -> None:
        x, y = runtime.vars("x y")
        out = self.engine.run(
            runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]),
            runtime.match(runtime.edge(x, y, rel="friend")),
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[1]), 1)
        self.assertEqual(out[1][0]["bindings"], {"x": "a", "y": "b"})

    def test_run_many_rolls_back_on_failure(self) -> None:
        x, y = runtime.vars("x y")
        with self.assertRaises(ValueError):
            self.engine.run(
                runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]),
                runtime.match(runtime.edge(x, y, rel="friend")).rewrite([runtime.edge(x, y)], limit=0),
            )

        out = self.engine.run(runtime.match(runtime.edge(x, y, rel="friend")))
        self.assertEqual(out, [])

    def test_module_exec_varargs_runs_atomically(self) -> None:
        x, y = runtime.vars("x y")
        g = runtime.ns("mexec1")
        self.engine.run(
            g.rewrite(
                to=[
                    runtime.node("graph:mexec1:shared", kind="shared_graph"),
                    runtime.edge("user:runner", "graph:mexec1:shared", rel="can", read=True, write=True),
                ]
            )
        )
        w = acl.client("runner", engine=self.engine).ns("mexec1")
        out = w.exec(
            runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]),
            runtime.match(runtime.edge(x, y, rel="friend")),
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1][0]["bindings"], {"x": "a", "y": "b"})

    def test_module_exec_rejects_mem_argument(self) -> None:
        x, y = runtime.vars("x y")
        w = acl.client("runner", engine=self.engine).ns("mexec2")
        with self.assertRaises(TypeError):
            w.exec(
                runtime.match(runtime.edge(x, y, rel="friend"), limit=10),
                mem=[runtime.edge("a", "b", rel="friend")],  # type: ignore[call-arg]
            )

    def test_module_exec_accepts_inline_temp_terms(self) -> None:
        x, y = runtime.vars("x y")
        g = runtime.ns("mexec3")
        self.engine.run(
            g.rewrite(
                to=[
                    runtime.node("graph:mexec3:shared", kind="shared_graph"),
                    runtime.edge("user:runner", "graph:mexec3:shared", rel="can", read=True, write=True),
                ]
            )
        )
        w = acl.client("runner", engine=self.engine).ns("mexec3")
        out = w.exec(
            runtime.match(runtime.edge(x, y, rel="friend"), limit=10),
            runtime.edge("a", "b", rel="friend", temp=True),
            runtime.node("a", kind="person", temp=True),
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"x": "a", "y": "b"})

    def test_module_exec_rejects_non_temp_inline_terms(self) -> None:
        x, y = runtime.vars("x y")
        w = acl.client("runner", engine=self.engine).ns("mexec4")
        with self.assertRaises(TypeError):
            w.exec(runtime.match(runtime.edge(x, y, rel="friend")), runtime.edge("a", "b", rel="friend"))

    def test_result_map_and_select_helpers(self) -> None:
        x, y = runtime.vars("x y")
        self.engine.run(
            runtime.rewrite(
                to=[
                    runtime.edge("a", "b", rel="friend", weight=0.9),
                    runtime.edge("b", "c", rel="friend", weight=0.7),
                ]
            )
        )
        out = self.engine.run(runtime.match(runtime.edge(x, y, rel="friend"), limit=10))

        pairs = out.map(lambda row: (row["bindings"]["x"], row["bindings"]["y"]))
        self.assertIn(("a", "b"), pairs)
        self.assertIn(("b", "c"), pairs)

        bindings = out.bindings("x", "y")
        self.assertIn({"x": "a", "y": "b"}, bindings)
        self.assertIn({"x": "b", "y": "c"}, bindings)

        props = out.edge_data("weight")
        self.assertIn({"weight": 0.9}, props)
        self.assertIn({"weight": 0.7}, props)

        self.assertEqual(out.first()["bindings"]["x"], "a")

        bindings2 = out.bindings("x", "y")
        self.assertEqual(bindings, bindings2)

        props2 = out.edge_data("weight")
        self.assertEqual(props, props2)

    def test_where_similarity_filters_edges_and_nodes(self) -> None:
        x, y = runtime.vars("x y")
        self.engine.run(
            runtime.rewrite(
                to=[
                    runtime.node("a", kind="person", embedding=[1.0, 0.0, 0.0]),
                    runtime.node("b", kind="person", embedding=[0.0, 1.0, 0.0]),
                    runtime.node("c", kind="person", embedding=[0.8, 0.2, 0.0]),
                    runtime.edge("a", "b", rel="friend", embedding=[0.95, 0.05, 0.0]),
                    runtime.edge("c", "b", rel="friend", embedding=[0.1, 0.9, 0.0]),
                ]
            )
        )
        q = [1.0, 0.0, 0.0]
        out = self.engine.run(
            runtime.match(runtime.edge(x, y, rel="friend"), limit=10).where(
                runtime.on(1).embedding.similar(q, min_score=0.8),
                x.embedding.similar(q, min_score=0.7),
            )
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"x": "a", "y": "b"})

    def test_namespace_where_similarity(self) -> None:
        g1, g2 = runtime.ns("g1"), runtime.ns("g2")
        x, y = runtime.vars("x y")
        self.engine.run(g1.rewrite(to=[g1.node("a", embedding=[1.0, 0.0]), g1.edge("a", "b", rel="friend", embedding=[1.0, 0.0])]))
        self.engine.run(g2.rewrite(to=[g2.node("a", embedding=[0.0, 1.0]), g2.edge("a", "b", rel="friend", embedding=[0.0, 1.0])]))
        q = [1.0, 0.0]
        out = self.engine.run(
            g1.match(g1.edge(x, y, rel="friend"), limit=10).where(
                runtime.on(1).embedding.similar(q, min_score=0.8),
                x.embedding.similar(q, min_score=0.8),
            )
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"x": "a", "y": "b"})

    def test_in_memory_match_uses_virtual_subgraph(self) -> None:
        x, y = runtime.vars("x y")
        out = self.engine.run(
            runtime.match(runtime.edge(x, y, rel="friend"), limit=10).where(
                x.kind == "person", runtime.on(1).weight >= 0.8
            ),
            mem=[
                runtime.edge("a", "b", rel="friend", weight=0.9),
                runtime.node("a", kind="person"),
            ],
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"x": "a", "y": "b"})

        # Virtual terms are query-time only.
        persisted = self.engine.run(runtime.match(runtime.edge(x, y, rel="friend"), limit=10))
        self.assertEqual(persisted, [])

    def test_in_memory_rewrite_rolls_back_by_default(self) -> None:
        pc, nxt = runtime.vars("pc nxt")
        out = self.engine.run(
            runtime.match(
                runtime.edge(runtime.const("m1"), pc, rel="state"),
                runtime.edge(pc, nxt, rel="step"),
            ).rewrite(
                [runtime.edge(runtime.const("m1"), nxt, rel="state")],
            ),
            mem=[
                runtime.edge("m1", "n0", rel="state"),
                runtime.edge("n0", "n1", rel="step"),
            ],
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["hyperedges"][0]["relation"], "state")
        self.assertEqual(out[0]["hyperedges"][0]["nodes"], ["m1", "n1"])

        # Rewritten result is returned, but graph remains unchanged.
        x, y = runtime.vars("x y")
        persisted = self.engine.run(runtime.match(runtime.edge(x, y, rel="state"), limit=10))
        self.assertEqual(persisted, [])

    def test_temp_exec_persists_within_batch_only(self) -> None:
        pc, nxt, x, y = runtime.vars("pc nxt x y")
        w = acl.client("tmp1", engine=self.engine).ns("t1")
        out = w.exec(
            runtime.rewrite(
                to=[
                    runtime.edge("m1", "n0", rel="state"),
                    runtime.edge("n0", "n1", rel="step"),
                ]
            ),
            runtime.match(
                runtime.edge(runtime.const("m1"), pc, rel="state"),
                runtime.edge(pc, nxt, rel="step"),
            ).rewrite([runtime.edge(runtime.const("m1"), nxt, rel="state")]),
            runtime.match(runtime.edge(x, y, rel="state"), limit=10),
            temp=True,
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[2]), 1)
        self.assertEqual(out[2][0]["bindings"], {"x": "m1", "y": "n1"})

    def test_temp_exec_is_isolated_from_default_engine(self) -> None:
        x, y = runtime.vars("x y")
        w = acl.client("tmp2", engine=self.engine).ns("t2")
        in_state = w.exec(
            runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]),
            runtime.match(runtime.edge(x, y, rel="friend")),
            temp=True,
        )
        self.assertEqual(len(in_state[1]), 1)
        in_default = self.engine.run(runtime.ns("t2").match(runtime.edge(x, y, rel="friend")))
        self.assertEqual(in_default, [])

    def test_temp_exec_reads_default_graph_but_rolls_back_writes(self) -> None:
        x, y = runtime.vars("x y")
        g = runtime.ns("t3")
        self.engine.run(
            g.rewrite(
                to=[
                    runtime.node("graph:t3:shared", kind="shared_graph"),
                    runtime.edge("user:tmp3", "graph:t3:shared", rel="can", read=True, write=True),
                ]
            )
        )
        w = acl.client("tmp3", engine=self.engine).ns("t3")
        w.exec(runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]))
        seen = w.exec(runtime.match(runtime.edge(x, y, rel="friend")), temp=True)
        self.assertEqual(len(seen), 1)
        self.assertEqual(seen[0]["bindings"], {"x": "a", "y": "b"})

        w.exec(
            runtime.match(runtime.edge(x, y, rel="friend")).rewrite([runtime.edge(x, y, rel="friend2")]),
            temp=True,
        )
        persisted = w.exec(runtime.match(runtime.edge(x, y, rel="friend2")))
        self.assertEqual(persisted, [])

    def test_temp_exec_complex_program_tree_leaves_no_traces(self) -> None:
        pc, nxt, alt, flag, a, b, outv, av, bv, rv = runtime.vars("pc nxt alt flag a b outv av bv rv")
        w = acl.client("tmp4", engine=self.engine).ns("t4")
        out = w.exec(
            # Program + AST + constants.
            runtime.rewrite(
                to=[
                    runtime.edge("vm", "pc0", rel="state"),
                    runtime.edge("pc0", "pc1", "c2", "c3", "t1", rel="add_instr"),
                    runtime.edge("pc1", "pc_then", "pc_else", "f0", rel="branch_instr"),
                    runtime.edge("f0", "true", rel="flag"),
                    runtime.edge("pc_then", "pc_join", "t1", "c4", "t2", rel="mul_instr"),
                    runtime.edge("pc_else", "pc_join", "t1", "c1", "t2", rel="sub_instr"),
                    runtime.edge("pc_join", "pc_out", rel="jump_instr"),
                    runtime.edge("pc_out", "pc_done", "t2", "result", rel="copy_instr"),
                    # Program tree (AST-like) structure for richer control flow.
                    runtime.edge("if0", "f0", "then0", "else0", rel="ast_if"),
                    runtime.edge("then0", "t1", "c4", "t2", rel="ast_mul"),
                    runtime.edge("else0", "t1", "c1", "t2", rel="ast_sub"),
                    runtime.edge("out0", "t2", "result", rel="ast_copy"),
                    runtime.edge("c1", "1", rel="value"),
                    runtime.edge("c2", "2", rel="value"),
                    runtime.edge("c3", "3", rel="value"),
                    runtime.edge("c4", "4", rel="value"),
                ]
            ),
                # Step 1: entry arithmetic.
                runtime.match(
                    runtime.edge("vm", pc, rel="state"),
                    runtime.edge(pc, nxt, a, b, outv, rel="add_instr"),
                    runtime.edge(a, av, rel="value"),
                    runtime.edge(b, bv, rel="value"),
                ).rewrite(
                    [
                        runtime.edge("vm", nxt, rel="state"),
                        runtime.edge(outv, av + bv, rel="value"),
                    ],
                    limit=1,
                ),
                # Step 2a: branch true.
                runtime.match(
                    runtime.edge("vm", pc, rel="state"),
                    runtime.edge(pc, nxt, alt, flag, rel="branch_instr"),
                    runtime.edge(flag, runtime.const("true"), rel="flag"),
                ).rewrite(
                    [runtime.edge("vm", nxt, rel="state")],
                    limit=1,
                ),
                # Step 2b: branch false (should not fire in this seeded program).
                runtime.match(
                    runtime.edge("vm", pc, rel="state"),
                    runtime.edge(pc, alt, nxt, flag, rel="branch_instr"),
                    runtime.edge(flag, runtime.const("false"), rel="flag"),
                ).rewrite(
                    [runtime.edge("vm", nxt, rel="state")],
                    limit=1,
                ),
                # Step 3a: then-path arithmetic.
                runtime.match(
                    runtime.edge("vm", pc, rel="state"),
                    runtime.edge(pc, nxt, a, b, outv, rel="mul_instr"),
                    runtime.edge(a, av, rel="value"),
                    runtime.edge(b, bv, rel="value"),
                ).rewrite(
                    [
                        runtime.edge("vm", nxt, rel="state"),
                        runtime.edge(outv, av * bv, rel="value"),
                    ],
                    limit=1,
                ),
                # Step 3b: else-path arithmetic (should not fire).
                runtime.match(
                    runtime.edge("vm", pc, rel="state"),
                    runtime.edge(pc, nxt, a, b, outv, rel="sub_instr"),
                    runtime.edge(a, av, rel="value"),
                    runtime.edge(b, bv, rel="value"),
                ).rewrite(
                    [
                        runtime.edge("vm", nxt, rel="state"),
                        runtime.edge(outv, av - bv, rel="value"),
                    ],
                    limit=1,
                ),
                # Step 4: control-flow jump.
                runtime.match(
                    runtime.edge("vm", pc, rel="state"),
                    runtime.edge(pc, nxt, rel="jump_instr"),
                ).rewrite(
                    [runtime.edge("vm", nxt, rel="state")],
                    limit=1,
                ),
                # Step 5: output copy.
                runtime.match(
                    runtime.edge("vm", pc, rel="state"),
                    runtime.edge(pc, nxt, a, outv, rel="copy_instr"),
                    runtime.edge(a, av, rel="value"),
                ).rewrite(
                    [
                        runtime.edge("vm", nxt, rel="state"),
                        runtime.edge(outv, av, rel="value"),
                    ],
                    limit=1,
                ),
                # Read final program output.
                runtime.match(runtime.edge(runtime.const("result"), rv, rel="value"), limit=1),
                temp=True,
            )

        self.assertEqual(len(out), 9)
        self.assertEqual(len(out[1]), 1)  # add
        self.assertEqual(len(out[2]), 1)  # branch true
        self.assertEqual(len(out[3]), 0)  # branch false
        self.assertEqual(len(out[4]), 1)  # then mul
        self.assertEqual(len(out[5]), 0)  # else sub
        self.assertEqual(len(out[6]), 1)  # jump
        self.assertEqual(len(out[7]), 1)  # copy
        self.assertEqual(len(out[8]), 1)  # final read
        self.assertEqual(out[8][0]["bindings"]["rv"], "20")

        # Everything above was ephemeral: no traces in persisted DB.
        persisted_count = self.engine.db.execute("SELECT COUNT(*) FROM hyperedges").fetchone()[0]
        self.assertEqual(persisted_count, 0)

if __name__ == "__main__":
    unittest.main()
