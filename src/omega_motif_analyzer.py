#!/usr/bin/env python3
"""
OMEGA adversarial motif analyzer
================================

Exhaustively analyzes small strict 3-SAT finalists produced by the evolutionary
arena.  Intended for n <= 20.

Metrics:
  * satisfying assignment count;
  * non-satisfying one-flip local minima;
  * strict local minima;
  * energy/distance of traps;
  * positive-literal distribution;
  * variable-degree concentration;
  * exact gadget-closure/component relations.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
from collections import Counter
from pathlib import Path

from omega_nonlocal_cluster import analyze_components, parse_dimacs


def clause_sat_mask(clause, state):
    for lit in clause:
        bit = (state >> (abs(lit)-1)) & 1
        truth = bit if lit > 0 else 1-bit
        if truth:
            return True
    return False


def energy(clauses, state):
    return sum(not clause_sat_mask(c, state) for c in clauses)


def popcount(x):
    return int(x).bit_count()


def gini(values):
    vals = sorted(float(v) for v in values)
    n = len(vals)
    if n == 0 or sum(vals) == 0:
        return 0.0
    s = sum((2*i-n-1)*x for i, x in enumerate(vals, start=1))
    return s/(n*sum(vals))


def analyze(path):
    n, clauses = parse_dimacs(path)
    if n > 20:
        raise ValueError("exhaustive motif mode is limited to n <= 20")

    total = 1 << n
    E = [energy(clauses, s) for s in range(total)]
    sats = [s for s,e in enumerate(E) if e == 0]

    local = []
    strict = []
    for s,e in enumerate(E):
        if e == 0:
            continue
        neigh = [E[s ^ (1<<i)] for i in range(n)]
        if all(e <= z for z in neigh):
            local.append(s)
        if all(e < z for z in neigh):
            strict.append(s)

    def nearest_sat_distance(s):
        return min(popcount(s ^ z) for z in sats)

    local_dist = [nearest_sat_distance(s) for s in local] if local else []
    strict_dist = [nearest_sat_distance(s) for s in strict] if strict else []

    pos_hist = Counter(sum(l > 0 for l in c) for c in clauses)
    degrees = [0]*n
    pos_deg = [0]*n
    neg_deg = [0]*n
    for c in clauses:
        for lit in c:
            i = abs(lit)-1
            degrees[i] += 1
            if lit > 0:
                pos_deg[i] += 1
            else:
                neg_deg[i] += 1

    comp = analyze_components(n, clauses)

    return {
        "file": str(path),
        "n": n,
        "m": len(clauses),
        "solutions": len(sats),
        "all_ones_is_solution": int(((1<<n)-1) in sats),
        "nonzero_local_minima": len(local),
        "strict_nonzero_local_minima": len(strict),
        "max_local_min_energy": max((E[s] for s in local), default=0),
        "max_strict_min_energy": max((E[s] for s in strict), default=0),
        "max_local_min_distance_to_solution": max(local_dist, default=0),
        "mean_local_min_distance_to_solution": (
            sum(local_dist)/len(local_dist) if local_dist else 0.0
        ),
        "max_strict_min_distance_to_solution": max(strict_dist, default=0),
        "clauses_with_1_positive": pos_hist.get(1,0),
        "clauses_with_2_positive": pos_hist.get(2,0),
        "clauses_with_3_positive": pos_hist.get(3,0),
        "degree_min": min(degrees),
        "degree_max": max(degrees),
        "degree_mean": sum(degrees)/n,
        "degree_gini": gini(degrees),
        "max_abs_sign_bias": max(abs(pos_deg[i]-neg_deg[i]) for i in range(n)),
        "closure_clauses": len(comp.closure),
        "recovered_units": comp.recovered_units,
        "recovered_binaries": comp.recovered_binaries,
        "equalities": comp.equality_relations,
        "inequalities": comp.inequality_relations,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--csv", default="omega_motif_summary.csv")
    args = ap.parse_args()

    rows = [analyze(p) for p in args.files]

    for row in rows:
        print(json.dumps(row, sort_keys=True))

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"\nSaved {args.csv}")


if __name__ == "__main__":
    main()
