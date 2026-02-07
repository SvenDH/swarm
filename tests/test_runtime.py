from __future__ import annotations

import glob
import os
import tempfile
import unittest

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
        cmd = runtime.match((x, x, y), ("r2", (y, z, u)))

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
            runtime.match((x, x, y), ("r2", (y, z, u))).rewrite(
                [(x, v, u), (y, v, z), (v, v, u)],
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
            runtime.match((x, x, y), ("r2", (y, z, u))).rewrite(
            [(x, x, y), ("r2", (y, z, u))],
            limit=2,
        )
        )
        self.assertEqual(len(out), 2)

        after = self.engine.run(runtime.match((x, x, y), ("r2", (y, z, u))))
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

        cmd = runtime.match((x, x, y), ("r2", (y, z, u))).rewrite(
            [runtime.edge(x, z, u, rel="out")],
            limit=10,
        )
        out = self.engine.run(cmd)

        # Only one rewrite step can run because both initial matches overlap on (a,a,b).
        self.assertEqual(len(out), 1)

        remaining = self.engine.run(runtime.match((x, x, y), ("r2", (y, z, u))))
        self.assertEqual(len(remaining), 0)

    def test_edge_filters_track_original_term_positions(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        self._seed_two_terms()

        cmd = runtime.match((x, x, y), ("r2", (y, z, u))).where(
            runtime.Edge(1).tag == "lhs",
            runtime.Edge(2).tag == "rhs",
        )
        out = self.engine.run(cmd)
        self.assertEqual(len(out), 1)

    def test_node_property_filter(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        self._seed_two_terms()
        self.engine.run(runtime.rewrite(to=[runtime.node("a", kind="entity")]))

        cmd = runtime.match((x, x, y), ("r2", (y, z, u))).where(x.kind == "entity")
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

        out = self.engine.run(runtime.match((x, x, y), ("r2", (y, z, u)), limit=8, random=True))
        self.assertEqual(len(out), 8)

    def test_all_mode_returns_multiple_matches(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        seed_terms = []
        for i in range(4):
            a, b, c, d = f"a{i}", f"b{i}", f"c{i}", f"d{i}"
            seed_terms.append(runtime.edge(a, a, b))
            seed_terms.append(runtime.edge(b, c, d, rel="r2"))
        self.engine.run(runtime.rewrite(to=seed_terms))

        out = self.engine.run(runtime.match((x, x, y), ("r2", (y, z, u)), limit=4))
        self.assertEqual(len(out), 4)

    def test_namespace_match_isolated(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        g1, g2 = runtime.ns("g1"), runtime.ns("g2")
        self.engine.run(
            [
                g1.rewrite(to=[g1.edge("a", "a", "b"), g1.edge("b", "c", "d", rel="r2")]),
                g2.rewrite(to=[g2.edge("a", "a", "b"), g2.edge("b", "c", "d", rel="r2")]),
            ]
        )

        out = self.engine.run(g1.match((x, x, y), ("r2", (y, z, u)), limit=10))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"u": "d", "x": "a", "y": "b", "z": "c"})

    def test_namespace_rewrite_scopes_fresh_nodes(self) -> None:
        x, y, z, u, v = runtime.vars("x y z u v")
        g1 = runtime.ns("g1")
        self.engine.run(g1.rewrite(to=[g1.edge("a", "a", "b"), g1.edge("b", "c", "d", rel="r2")]))

        out = self.engine.run(g1.match((x, x, y), ("r2", (y, z, u))).rewrite([(x, v, u)]))
        self.assertEqual(len(out), 1)
        self.assertIn("v", out[0]["bindings"])
        self.assertTrue(out[0]["bindings"]["v"].startswith("n_"))

    def test_namespace_match_method(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        g1 = runtime.ns("g1")
        self.engine.run(g1.rewrite(to=[g1.edge("a", "a", "b"), g1.edge("b", "c", "d", rel="r2")]))

        out = self.engine.run(g1.match((x, x, y), ("r2", (y, z, u))))
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

        out = self.engine.run(g1.match((x, x, y), ("r2", (y, z, u))).where(runtime.on(1).tag == "lhs", x.kind == "entity"))
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"u": "d", "x": "a", "y": "b", "z": "c"})

        out2 = self.engine.run(g1.match(("r2", (runtime.const("b"), z, u)), limit=10))
        self.assertEqual(len(out2), 1)
        self.assertEqual(out2[0]["bindings"]["z"], "c")

    def test_namespace_validation(self) -> None:
        with self.assertRaises(ValueError):
            runtime.ns("bad:name")

    def test_friendly_api_aliases(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        self.engine.run(runtime.rewrite(to=[runtime.edge("a", "a", "b"), runtime.edge("b", "c", "d", rel="r2")]))
        self.engine.run(runtime.rewrite(to=[runtime.node("a", kind="entity")]))

        cmd = runtime.match((x, x, y), ("r2", (y, z, u))).where(x.kind == "entity")
        out = self.engine.run(cmd)
        self.assertEqual(len(out), 1)

    def test_execution_ops_arithmetic(self) -> None:
        a, b, out = runtime.vars("a b out")
        self.engine.run(runtime.rewrite(to=[runtime.edge("2", "3", "sum", rel="add")]))

        step = self.engine.run(
            runtime.match(("add", (a, b, out))).rewrite(
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
            runtime.match(("concat", (a, b, out))).rewrite(
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
            [runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]), runtime.match(("friend", (x, y)))]
        )
        self.assertEqual(len(out), 2)
        self.assertEqual(len(out[1]), 1)
        self.assertEqual(out[1][0]["bindings"], {"x": "a", "y": "b"})

    def test_run_many_rolls_back_on_failure(self) -> None:
        x, y = runtime.vars("x y")
        with self.assertRaises(ValueError):
            self.engine.run(
                [
                    runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]),
                    runtime.match(("friend", (x, y))).rewrite([(x, y)], limit=0),
                ]
            )

        out = self.engine.run(runtime.match(("friend", (x, y))))
        self.assertEqual(out, [])

    def test_module_exec_varargs_runs_atomically(self) -> None:
        x, y = runtime.vars("x y")
        prev = runtime._ENGINE
        runtime._ENGINE = self.engine
        try:
            out = runtime.exec(
                runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]),
                runtime.match(("friend", (x, y))),
            )
        finally:
            runtime._ENGINE = prev
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1][0]["bindings"], {"x": "a", "y": "b"})

    def test_module_exec_rejects_mem_argument(self) -> None:
        x, y = runtime.vars("x y")
        with self.assertRaises(TypeError):
            runtime.exec(
                runtime.match(("friend", (x, y)), limit=10),
                mem=[runtime.edge("a", "b", rel="friend")],  # type: ignore[call-arg]
            )

    def test_module_exec_accepts_inline_temp_terms(self) -> None:
        x, y = runtime.vars("x y")
        prev = runtime._ENGINE
        runtime._ENGINE = self.engine
        try:
            out = runtime.exec(
                runtime.match(("friend", (x, y)), limit=10),
                runtime.edge("a", "b", rel="friend", temp=True),
                runtime.node("a", kind="person", temp=True),
            )
        finally:
            runtime._ENGINE = prev
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"x": "a", "y": "b"})

    def test_module_exec_rejects_non_temp_inline_terms(self) -> None:
        x, y = runtime.vars("x y")
        with self.assertRaises(TypeError):
            runtime.exec(runtime.match(("friend", (x, y))), runtime.edge("a", "b", rel="friend"))

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
        out = self.engine.run(runtime.match(("friend", (x, y)), limit=10))

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
            runtime.match(("friend", (x, y)), limit=10).where(
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
            g1.match(("friend", (x, y)), limit=10).where(
                runtime.on(1).embedding.similar(q, min_score=0.8),
                x.embedding.similar(q, min_score=0.8),
            )
        )
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["bindings"], {"x": "a", "y": "b"})

    def test_in_memory_match_uses_virtual_subgraph(self) -> None:
        x, y = runtime.vars("x y")
        out = self.engine.run(
            runtime.match(("friend", (x, y)), limit=10).where(
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
        persisted = self.engine.run(runtime.match(("friend", (x, y)), limit=10))
        self.assertEqual(persisted, [])

    def test_in_memory_rewrite_rolls_back_by_default(self) -> None:
        pc, nxt = runtime.vars("pc nxt")
        out = self.engine.run(
            runtime.match(
                ("state", (runtime.const("m1"), pc)),
                ("step", (pc, nxt)),
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
        persisted = self.engine.run(runtime.match(("state", (x, y)), limit=10))
        self.assertEqual(persisted, [])

    def test_temp_exec_persists_within_batch_only(self) -> None:
        pc, nxt, x, y = runtime.vars("pc nxt x y")
        out = runtime.exec(
            runtime.rewrite(
                to=[
                    runtime.edge("m1", "n0", rel="state"),
                    runtime.edge("n0", "n1", rel="step"),
                ]
            ),
            runtime.match(
                ("state", (runtime.const("m1"), pc)),
                ("step", (pc, nxt)),
            ).rewrite([runtime.edge(runtime.const("m1"), nxt, rel="state")]),
            runtime.match(("state", (x, y)), limit=10),
            temp=True,
        )
        self.assertEqual(len(out), 3)
        self.assertEqual(len(out[2]), 1)
        self.assertEqual(out[2][0]["bindings"], {"x": "m1", "y": "n1"})

    def test_temp_exec_is_isolated_from_default_engine(self) -> None:
        x, y = runtime.vars("x y")
        prev = runtime._ENGINE
        runtime._ENGINE = self.engine
        try:
            in_state = runtime.exec(
                runtime.rewrite(to=[runtime.edge("a", "b", rel="friend")]),
                runtime.match(("friend", (x, y))),
                temp=True,
            )
            self.assertEqual(len(in_state[1]), 1)
            in_default = runtime.exec(runtime.match(("friend", (x, y))))
            self.assertEqual(in_default, [])
        finally:
            runtime._ENGINE = prev

if __name__ == "__main__":
    unittest.main()
