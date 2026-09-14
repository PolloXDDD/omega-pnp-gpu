#!/usr/bin/env python3
"""
OMEGA Hybrid Nonlocal SAT prototype
===================================

Pipeline:
    exact gadget closure
    -> signed parity components
    -> nonlocal cluster pulse
    -> exact clause verification
    -> focused clause-feedback fallback

This combines the nonlocal barrier-defeating preconditioner with the previous
focused local dynamics.  It is an experimental solver, not a worst-case
polynomial-time proof for arbitrary 3-SAT.
"""

from __future__ import annotations

import argparse
import random
import statistics
import time

from omega_nonlocal_cluster import (
    analyze_components,
    clause_sat,
    energy,
    normalize_clause,
    planted_3sat,
    strict_barrier_formula,
    cluster_pulse_assignment,
)


class FocusedFallback:
    def __init__(self, n, clauses, bits, seed):
        self.n = n
        self.clauses = clauses
        self.bits = list(bits)
        self.rng = random.Random(seed)

        self.incident = [[] for _ in range(n)]
        for ci, c in enumerate(clauses):
            for lit in c:
                self.incident[abs(lit)-1].append(ci)

        self.sat = [False] * len(clauses)
        self.unsat = set()
        self.refresh_all()

    def refresh_all(self):
        self.unsat.clear()
        for i, c in enumerate(self.clauses):
            s = clause_sat(c, self.bits)
            self.sat[i] = s
            if not s:
                self.unsat.add(i)

    def delta(self, v):
        before = sum(not self.sat[ci] for ci in self.incident[v])
        self.bits[v] ^= 1
        after = sum(
            not clause_sat(self.clauses[ci], self.bits)
            for ci in self.incident[v]
        )
        self.bits[v] ^= 1
        return after - before

    def flip(self, v):
        self.bits[v] ^= 1
        for ci in self.incident[v]:
            s = clause_sat(self.clauses[ci], self.bits)
            self.sat[ci] = s
            if s:
                self.unsat.discard(ci)
            else:
                self.unsat.add(ci)

    def run(self, max_steps, explore=0.45):
        for step in range(max_steps + 1):
            if not self.unsat:
                return True, step, self.bits

            if step == max_steps:
                break

            ci = self.rng.choice(tuple(self.unsat))
            candidates = sorted(set(abs(l)-1 for l in self.clauses[ci]))

            if self.rng.random() < explore:
                v = self.rng.choice(candidates)
            else:
                scores = [(self.delta(v), v) for v in candidates]
                best = min(d for d, _ in scores)
                pool = [v for d, v in scores if d == best]
                v = self.rng.choice(pool)

            self.flip(v)

        return False, max_steps, self.bits


def hybrid_solve(n, clauses, seed, step_factor=50, restarts=4, explore=0.45):
    analysis = analyze_components(n, clauses)

    if analysis.contradiction:
        return {
            "solved": False,
            "certified_unsat_by_component_constraints": True,
            "energy": None,
            "steps": 0,
            "cluster_only": False,
        }

    base = cluster_pulse_assignment(n, analysis, seed=seed)
    e0 = energy(clauses, base)

    if e0 == 0:
        return {
            "solved": True,
            "certified_unsat_by_component_constraints": False,
            "energy": 0,
            "steps": 0,
            "cluster_only": True,
            "assignment": base,
        }

    max_steps = step_factor * n * n
    best_e = e0
    total_steps = 0
    best_bits = base

    for r in range(restarts):
        if r == 0:
            init = list(base)
        else:
            # Keep forced component relations, randomize only unconstrained roots
            init = cluster_pulse_assignment(n, analysis, seed=seed + 10007*r)

        solver = FocusedFallback(n, clauses, init, seed + 1000003*r)
        ok, steps, bits = solver.run(max_steps, explore=explore)
        total_steps += steps
        e = energy(clauses, bits)

        if e < best_e:
            best_e = e
            best_bits = list(bits)

        if ok:
            return {
                "solved": True,
                "certified_unsat_by_component_constraints": False,
                "energy": 0,
                "steps": total_steps,
                "cluster_only": False,
                "assignment": bits,
            }

    return {
        "solved": False,
        "certified_unsat_by_component_constraints": False,
        "energy": best_e,
        "steps": total_steps,
        "cluster_only": False,
        "assignment": best_bits,
    }


def benchmark_random(sizes, trials, ratio, seed):
    print("RANDOM PLANTED 3-SAT")
    print("n,m,trials,successes,cluster_only,median_steps,median_ms")
    for n in sizes:
        successes = 0
        cluster_only = 0
        steps = []
        ms = []

        for t in range(trials):
            clauses = planted_3sat(n, ratio, seed + 100000*n + t)
            t0 = time.perf_counter()
            r = hybrid_solve(n, clauses, seed + t)
            ms.append((time.perf_counter()-t0)*1000)
            steps.append(r["steps"])
            successes += int(r["solved"])
            cluster_only += int(r.get("cluster_only", False))

        print(
            f"{n},{round(ratio*n)},{trials},{successes},"
            f"{cluster_only},{statistics.median(steps):.1f},"
            f"{statistics.median(ms):.3f}"
        )


def benchmark_barrier(sizes, seed):
    print("\nSTRICT BARRIER FAMILY")
    print("n,total_vars,clauses,solved,cluster_only,steps,ms,energy")
    for n in sizes:
        total, clauses = strict_barrier_formula(n)
        t0 = time.perf_counter()
        r = hybrid_solve(total, clauses, seed)
        elapsed = (time.perf_counter()-t0)*1000
        print(
            f"{n},{total},{len(clauses)},{int(r['solved'])},"
            f"{int(r.get('cluster_only', False))},{r['steps']},"
            f"{elapsed:.3f},{r['energy']}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", nargs="+", type=int, default=[10,20,50,100])
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--ratio", type=float, default=4.2)
    ap.add_argument("--seed", type=int, default=26032009)
    args = ap.parse_args()

    benchmark_random(args.sizes, args.trials, args.ratio, args.seed)
    benchmark_barrier(args.sizes, args.seed)


if __name__ == "__main__":
    main()
