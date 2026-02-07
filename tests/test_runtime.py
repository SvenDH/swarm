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
        self.gid = "g"

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
            self.gid,
            runtime.update(
                [
                    runtime.Term("a", "a", "b", data={"tag": "lhs"}),
                    runtime.Term("b", "c", "d", rel="r2", data={"tag": "rhs"}),
                ]
            ),
        )

    def test_match_order_is_deterministic(self) -> None:
        x, y, z, u = runtime.Var.many("x y z u")
        self._seed_two_terms()
        cmd = runtime.select((x, x, y), ("r2", (y, z, u)))

        out1 = self.engine.run(self.gid, cmd)
        out2 = self.engine.run(self.gid, cmd)

        self.assertEqual(len(out1), 1)
        self.assertEqual(len(out2), 1)
        edges1 = [e["edge_id"] for e in out1[0]["hyperedges"]]
        edges2 = [e["edge_id"] for e in out2[0]["hyperedges"]]
        self.assertEqual(edges1, edges2)

    def test_rewrite_returns_rewritten_subgraph(self) -> None:
        x, y, z, u, v = runtime.Var.many("x y z u v")
        self._seed_two_terms()

        out = self.engine.run(
            self.gid,
            runtime.select((x, x, y), ("r2", (y, z, u))).update(
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
        x, y, z, u = runtime.Var.many("x y z u")
        self._seed_two_terms()

        before = self.engine.run(self.gid, runtime.select((x, x, y), ("r2", (y, z, u))))
        self.assertEqual(len(before), 1)

        non_terminating = runtime.select((x, x, y), ("r2", (y, z, u))).update(
            [(x, x, y), ("r2", (y, z, u))],
            mode="all",
            limit=2,
        )
        with self.assertRaises(ValueError):
            self.engine.run(self.gid, non_terminating)

        after = self.engine.run(self.gid, runtime.select((x, x, y), ("r2", (y, z, u))))
        self.assertEqual(before, after)

    def test_rewrite_all_overlapping_matches(self) -> None:
        x, y, z, u = runtime.Var.many("x y z u")
        # Two candidate matches share the same first edge:
        # (a,a,b) with (b,c,d) and (a,a,b) with (b,e,f).
        self.engine.run(
            self.gid,
            runtime.update(
                [
                    runtime.Term("a", "a", "b"),
                    runtime.Term("b", "c", "d", rel="r2"),
                    runtime.Term("b", "e", "f", rel="r2"),
                ]
            ),
        )

        cmd = runtime.select((x, x, y), ("r2", (y, z, u))).update(
            [runtime.Term(x, z, u, rel="out")],
            mode="all",
            limit=10,
        )
        out = self.engine.run(self.gid, cmd)

        # Only one rewrite step can run because both initial matches overlap on (a,a,b).
        self.assertEqual(len(out), 1)

        remaining = self.engine.run(self.gid, runtime.select((x, x, y), ("r2", (y, z, u))))
        self.assertEqual(len(remaining), 0)

    def test_edge_filters_track_original_term_positions(self) -> None:
        x, y, z, u = runtime.Var.many("x y z u")
        self._seed_two_terms()

        cmd = runtime.select((x, x, y), ("r2", (y, z, u))).where(
            runtime.Edge(1).tag == "lhs",
            runtime.Edge(2).tag == "rhs",
        )
        out = self.engine.run(self.gid, cmd)
        self.assertEqual(len(out), 1)

    def test_node_property_filter(self) -> None:
        x, y, z, u = runtime.Var.many("x y z u")
        self._seed_two_terms()
        self.engine.run(
            self.gid,
            runtime.update([runtime.Term("a", rel="__node__", data={"kind": "entity"})]),
        )

        cmd = runtime.select((x, x, y), ("r2", (y, z, u))).where(x.kind == "entity")
        out = self.engine.run(self.gid, cmd)
        self.assertEqual(len(out), 1)

    def test_random_mode_picks_one_match(self) -> None:
        x, y, z, u = runtime.Var.many("x y z u")
        seed_terms = []
        for i in range(8):
            a, b, c, d = f"a{i}", f"b{i}", f"c{i}", f"d{i}"
            seed_terms.append(runtime.Term(a, a, b))
            seed_terms.append(runtime.Term(b, c, d, rel="r2"))
        self.engine.run(self.gid, runtime.update(seed_terms))

        out = self.engine.run(self.gid, runtime.select((x, x, y), ("r2", (y, z, u)), limit=8, mode="random"))
        self.assertEqual(len(out), 1)


if __name__ == "__main__":
    unittest.main()
