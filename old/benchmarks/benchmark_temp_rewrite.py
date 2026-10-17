from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import graph


def build_seed_terms(pairs: int) -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    for i in range(pairs):
        a, b, c, d = f"a{i}", f"b{i}", f"c{i}", f"d{i}"
        terms.append(graph.edge(a, a, b))
        terms.append(graph.edge(b, c, d, rel="r2"))
    return terms


def build_rewrite(limit: int) -> graph.Command:
    x, y, z, u, v = graph.vars("x y z u v")
    return graph.match(
        graph.edge(x, x, y),
        graph.edge(y, z, u, rel="r2"),
    ).rewrite(
        [graph.edge(x, v, u), graph.edge(y, v, z), graph.edge(v, v, u)],
        limit=limit,
    )


def bench_temp_rewrite(pairs: int, runs: int, limit: int, warmup: int) -> dict[str, float]:
    fd, db_path = tempfile.mkstemp(prefix="runtime_temp_bench_", suffix=".db")
    os.close(fd)
    try:
        engine = graph._Engine(db_path)
        seed_start = time.perf_counter()
        engine.run(graph.rewrite(to=build_seed_terms(pairs)))
        seed_seconds = time.perf_counter() - seed_start

        cmd = build_rewrite(limit)
        for _ in range(max(0, warmup)):
            engine.run(cmd, temp=True)

        steps_total = 0
        run_seconds: list[float] = []
        for _ in range(runs):
            t0 = time.perf_counter()
            out = engine.run(cmd, temp=True)
            run_seconds.append(time.perf_counter() - t0)
            steps_total += len(out)

        total = sum(run_seconds)
        return {
            "pairs": float(pairs),
            "runs": float(runs),
            "limit": float(limit),
            "warmup": float(max(0, warmup)),
            "seed_seconds": seed_seconds,
            "total_seconds": total,
            "mean_seconds": statistics.fmean(run_seconds) if run_seconds else 0.0,
            "p95_seconds": sorted(run_seconds)[int(round((len(run_seconds) - 1) * 0.95))] if run_seconds else 0.0,
            "steps_total": float(steps_total),
            "steps_per_sec": steps_total / total if total else 0.0,
        }
    finally:
        for p in [db_path, f"{db_path}-wal", f"{db_path}-shm"]:
            if os.path.exists(p):
                os.remove(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark temp=True rewrite throughput")
    parser.add_argument("--pairs", type=int, default=500, help="Number of lhs/r2 seed pairs")
    parser.add_argument("--runs", type=int, default=40, help="Number of measured temp rewrite runs")
    parser.add_argument("--limit", type=int, default=500, help="Rewrite steps per run")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup runs before measuring")
    args = parser.parse_args()

    if args.pairs < 1:
        raise ValueError("--pairs must be >= 1")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")
    if args.limit < 1:
        raise ValueError("--limit must be >= 1")

    result = bench_temp_rewrite(args.pairs, args.runs, args.limit, args.warmup)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

