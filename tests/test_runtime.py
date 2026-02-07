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
                mode="first",
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

    def test_rewrite_all_rolls_back_on_limit_error(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        self._seed_two_terms()

        before = self.engine.run(runtime.match((x, x, y), ("r2", (y, z, u))))
        self.assertEqual(len(before), 1)

        non_terminating = runtime.match((x, x, y), ("r2", (y, z, u))).rewrite(
            [(x, x, y), ("r2", (y, z, u))],
            mode="all",
            limit=2,
        )
        with self.assertRaises(ValueError):
            self.engine.run(non_terminating)

        after = self.engine.run(runtime.match((x, x, y), ("r2", (y, z, u))))
        self.assertEqual(before, after)

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
            mode="all",
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

    def test_random_mode_picks_one_match(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        seed_terms = []
        for i in range(8):
            a, b, c, d = f"a{i}", f"b{i}", f"c{i}", f"d{i}"
            seed_terms.append(runtime.edge(a, a, b))
            seed_terms.append(runtime.edge(b, c, d, rel="r2"))
        self.engine.run(runtime.rewrite(to=seed_terms))

        out = self.engine.run(runtime.match((x, x, y), ("r2", (y, z, u)), limit=8, mode="random"))
        self.assertEqual(len(out), 1)

    def test_all_mode_returns_multiple_matches(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        seed_terms = []
        for i in range(4):
            a, b, c, d = f"a{i}", f"b{i}", f"c{i}", f"d{i}"
            seed_terms.append(runtime.edge(a, a, b))
            seed_terms.append(runtime.edge(b, c, d, rel="r2"))
        self.engine.run(runtime.rewrite(to=seed_terms))

        out = self.engine.run(runtime.match((x, x, y), ("r2", (y, z, u)), limit=4, mode="all"))
        self.assertEqual(len(out), 4)

    def test_namespace_match_isolated(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        g1, g2 = runtime.ns("g1"), runtime.ns("g2")
        g1a, g1b, g1c, g1d = [g1.id(n) for n in ("a", "b", "c", "d")]
        g2a, g2b, g2c, g2d = [g2.id(n) for n in ("a", "b", "c", "d")]
        self.engine.run(
            runtime.rewrite(
                to=[
                    runtime.edge(g1a, g1a, g1b),
                    runtime.edge(g1b, g1c, g1d, rel="r2"),
                    runtime.edge(g2a, g2a, g2b),
                    runtime.edge(g2b, g2c, g2d, rel="r2"),
                ]
            )
        )

        out = self.engine.run(g1.match((x, x, y), ("r2", (y, z, u)), mode="all", limit=10))
        self.assertEqual(len(out), 1)
        self.assertTrue(all(v.startswith("g1:") for v in out[0]["bindings"].values()))

    def test_namespace_rewrite_scopes_fresh_nodes(self) -> None:
        x, y, z, u, v = runtime.vars("x y z u v")
        g1 = runtime.ns("g1")
        a, b, c, d = [g1.id(n) for n in ("a", "b", "c", "d")]
        self.engine.run(runtime.rewrite(to=[runtime.edge(a, a, b), runtime.edge(b, c, d, rel="r2")]))

        out = self.engine.run(g1.match((x, x, y), ("r2", (y, z, u))).rewrite([(x, v, u)], mode="first"))
        self.assertEqual(len(out), 1)
        self.assertIn("v", out[0]["bindings"])
        self.assertTrue(out[0]["bindings"]["v"].startswith("g1:"))

    def test_namespace_match_method(self) -> None:
        x, y, z, u = runtime.vars("x y z u")
        g1 = runtime.ns("g1")
        a, b, c, d = [g1.id(n) for n in ("a", "b", "c", "d")]
        self.engine.run(runtime.rewrite(to=[runtime.edge(a, a, b), runtime.edge(b, c, d, rel="r2")]))

        out = self.engine.run(g1.match((x, x, y), ("r2", (y, z, u))))
        self.assertEqual(len(out), 1)
        self.assertTrue(all(v.startswith("g1:") for v in out[0]["bindings"].values()))

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
        self.assertTrue(all(v.startswith("g1:") for v in out[0]["bindings"].values()))

        out2 = self.engine.run(g1.match(("r2", (runtime.const("b"), z, u)), mode="all", limit=10))
        self.assertEqual(len(out2), 1)
        self.assertEqual(out2[0]["bindings"]["z"], "g1:c")

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


if __name__ == "__main__":
    unittest.main()
