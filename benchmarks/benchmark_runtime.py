from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import runtime


def build_seed_terms(pairs: int) -> list[dict[str, object]]:
    terms: list[dict[str, object]] = []
    for i in range(pairs):
        a, b, c, d = f"a{i}", f"b{i}", f"c{i}", f"d{i}"
        terms.append(runtime.edge(a, a, b, tag="lhs"))
        terms.append(runtime.edge(b, c, d, rel="r2", tag="rhs"))
    return terms


def seed(engine: runtime._Engine, pairs: int) -> float:
    t0 = time.perf_counter()
    engine.run(runtime.rewrite(to=build_seed_terms(pairs)))
    return time.perf_counter() - t0


def bench_match(engine: runtime._Engine, pairs: int, runs: int) -> dict[str, float]:
    x, y, z, u = runtime.vars("x y z u")
    cmd = runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2"), limit=pairs)

    # Warmup
    for _ in range(3):
        engine.run(cmd)

    total = 0
    t0 = time.perf_counter()
    for _ in range(runs):
        out = engine.run(cmd)
        total += len(out)
    dt = time.perf_counter() - t0
    return {
        "runs": float(runs),
        "seconds": dt,
        "matches_total": float(total),
        "queries_per_sec": runs / dt if dt else 0.0,
        "matches_per_sec": total / dt if dt else 0.0,
    }


def bench_rewrite(engine: runtime._Engine, pairs: int, runs: int) -> dict[str, float]:
    x, y, z, u, v = runtime.vars("x y z u v")
    cmd = runtime.match(runtime.edge(x, x, y), runtime.edge(y, z, u, rel="r2")).rewrite(
        [runtime.edge(x, v, u), runtime.edge(y, v, z), runtime.edge(v, v, u)],
        limit=max(10, pairs * 4),
    )

    rewrite_count = 0
    seed_seconds = 0.0
    rewrite_seconds = 0.0
    for _ in range(runs):
        engine.db.execute("DELETE FROM hyperedges")
        engine.db.commit()
        t_seed = time.perf_counter()
        seed(engine, pairs)
        seed_seconds += time.perf_counter() - t_seed
        t_rewrite = time.perf_counter()
        out = engine.run(cmd)
        rewrite_seconds += time.perf_counter() - t_rewrite
        rewrite_count += len(out)
    total_seconds = seed_seconds + rewrite_seconds
    return {
        "runs": float(runs),
        "seconds_total": total_seconds,
        "seconds_seed": seed_seconds,
        "seconds_rewrite_only": rewrite_seconds,
        "rewrite_steps_total": float(rewrite_count),
        "rewrite_steps_per_sec": rewrite_count / rewrite_seconds if rewrite_seconds else 0.0,
    }


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    data = sorted(values)
    idx = max(0, min(len(data) - 1, int(round((len(data) - 1) * p))))
    return data[idx]


def _summary(values: list[float]) -> dict[str, float]:
    if not values:
        return {"n": 0.0, "min": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "n": float(len(values)),
        "min": min(values),
        "median": statistics.median(values),
        "p95": _percentile(values, 0.95),
        "max": max(values),
        "mean": statistics.fmean(values),
    }


def _run_trial(pairs: int, match_runs: int, rewrite_runs: int) -> dict[str, object]:
    fd, db_path = tempfile.mkstemp(prefix="runtime_bench_", suffix=".db")
    os.close(fd)
    try:
        engine = runtime._Engine(db_path)
        seed_seconds = seed(engine, pairs)
        match = bench_match(engine, pairs, match_runs)
        rewrite = bench_rewrite(engine, pairs, rewrite_runs)
        return {
            "pairs": pairs,
            "seed_seconds": seed_seconds,
            "match": match,
            "rewrite": rewrite,
        }
    finally:
        for p in [db_path, f"{db_path}-wal", f"{db_path}-shm"]:
            if os.path.exists(p):
                os.remove(p)


def main() -> None:
    parser = argparse.ArgumentParser(description="Micro-benchmark for runtime hypergraph operations")
    parser.add_argument("--pairs", type=int, default=300, help="Number of (lhs,r2) edge pairs to seed")
    parser.add_argument("--match-runs", type=int, default=80, help="How many repeated match queries to run")
    parser.add_argument("--rewrite-runs", type=int, default=12, help="How many seeded rewrite runs to execute")
    parser.add_argument("--trials", type=int, default=1, help="How many full benchmark trials to run")
    args = parser.parse_args()
    if args.trials < 1:
        raise ValueError("--trials must be >= 1")

    if args.trials == 1:
        print(json.dumps(_run_trial(args.pairs, args.match_runs, args.rewrite_runs), indent=2))
        return

    trials = [_run_trial(args.pairs, args.match_runs, args.rewrite_runs) for _ in range(args.trials)]
    seed_values = [float(t["seed_seconds"]) for t in trials]
    match_qps = [float(t["match"]["queries_per_sec"]) for t in trials]
    match_mps = [float(t["match"]["matches_per_sec"]) for t in trials]
    rewrite_sps = [float(t["rewrite"]["rewrite_steps_per_sec"]) for t in trials]
    rewrite_only = [float(t["rewrite"]["seconds_rewrite_only"]) for t in trials]

    print(
        json.dumps(
            {
                "config": {
                    "pairs": args.pairs,
                    "match_runs": args.match_runs,
                    "rewrite_runs": args.rewrite_runs,
                    "trials": args.trials,
                },
                "summary": {
                    "seed_seconds": _summary(seed_values),
                    "match_queries_per_sec": _summary(match_qps),
                    "match_matches_per_sec": _summary(match_mps),
                    "rewrite_steps_per_sec": _summary(rewrite_sps),
                    "rewrite_seconds_only": _summary(rewrite_only),
                },
                "trials": trials,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
