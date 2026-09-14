#!/usr/bin/env python3
"""
OMEGA additive-drift invariant search
=====================================

For a small strict 3-SAT instance, analyze the exact expected one-step drift of

    Phi_lambda(x) = d(x, SAT) + lambda * E(x),

under the current focused clause-feedback rule:

  * choose a violated clause uniformly;
  * with probability p, flip a uniformly random variable from that clause;
  * otherwise flip a variable having minimum exact one-flip energy change,
    breaking ties uniformly.

For every non-satisfying state x, the expected drift has the affine form

    E[Delta Phi | x] = a_x + lambda b_x.

The program asks whether there exists lambda >= 0 for which every state has
strictly negative expected drift.

A feasible lambda gives an additive-drift certificate for that particular
finite formula. It is not automatically a theorem for arbitrary 3-SAT.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from omega_nonlocal_cluster import parse_dimacs


def lit_truth(lit, state):
    bit = (state >> (abs(lit)-1)) & 1
    return bit if lit > 0 else 1-bit


def clause_sat(clause, state):
    return any(lit_truth(l, state) for l in clause)


def formula_energy(clauses, state):
    return sum(not clause_sat(c, state) for c in clauses)


def transition_distribution(n, clauses, state, energies, p):
    unsat = [ci for ci,c in enumerate(clauses) if not clause_sat(c, state)]
    if not unsat:
        return {}

    probs = {}

    clause_weight = 1.0 / len(unsat)

    for ci in unsat:
        candidates = sorted(set(abs(l)-1 for l in clauses[ci]))
        random_w = p / len(candidates)

        deltas = []
        for v in candidates:
            t = state ^ (1 << v)
            deltas.append((energies[t] - energies[state], v))
        best = min(d for d,_ in deltas)
        best_vars = [v for d,v in deltas if d == best]
        greedy_w = (1.0-p) / len(best_vars)

        for v in candidates:
            t = state ^ (1 << v)
            probs[t] = probs.get(t, 0.0) + clause_weight * random_w

        for v in best_vars:
            t = state ^ (1 << v)
            probs[t] = probs.get(t, 0.0) + clause_weight * greedy_w

    return probs


def nearest_sat_distances(n, sat_states):
    total = 1 << n
    # n <= 16 in our study; direct computation is fine.
    return [
        min((s ^ z).bit_count() for z in sat_states)
        for s in range(total)
    ]


def analyze(path, p=0.45, epsilon=1e-12):
    n, clauses = parse_dimacs(path)
    if n > 16:
        raise ValueError("exact drift mode limited to n <= 16")

    total = 1 << n
    energies = [formula_energy(clauses, s) for s in range(total)]
    sats = [s for s,e in enumerate(energies) if e == 0]
    distances = nearest_sat_distances(n, sats)

    # Constraints: a + lambda*b < 0.
    lower = 0.0
    upper = math.inf
    impossible = []
    coeffs = []

    for s in range(total):
        if energies[s] == 0:
            continue

        trans = transition_distribution(n, clauses, s, energies, p)
        exp_d = sum(prob * distances[t] for t, prob in trans.items())
        exp_e = sum(prob * energies[t] for t, prob in trans.items())

        a = exp_d - distances[s]
        b = exp_e - energies[s]
        coeffs.append((s, a, b))

        # Require a + lambda b <= -epsilon.
        if abs(b) <= epsilon:
            if a >= -epsilon:
                impossible.append((s, a, b))
        elif b < 0:
            # lambda >= (-epsilon-a)/b; division reverses inequality.
            bound = (-epsilon - a) / b
            lower = max(lower, bound)
        else:
            bound = (-epsilon - a) / b
            upper = min(upper, bound)

    feasible = (not impossible) and lower < upper and upper > 0
    lambda_star = None
    worst_drift = None
    worst_state = None

    if feasible:
        lower = max(lower, 0.0)
        if math.isinf(upper):
            lambda_star = max(lower + 1.0, 1.0)
        else:
            lambda_star = (lower + upper) / 2.0

        drifts = [(a + lambda_star*b, s) for s,a,b in coeffs]
        worst_drift, worst_state = max(drifts)

    # Also report best lambda on a logarithmic/grid scan when the simple exact
    # feasibility interval is empty, to show how close this potential gets.
    scan = [0.0]
    for k in range(-6, 7):
        for base in (1.0, 2.0, 5.0):
            scan.append(base * (10.0**k))
    scan = sorted(set(scan))

    best_grid = None
    for lam in scan:
        wd, ws = max((a + lam*b, s) for s,a,b in coeffs)
        if best_grid is None or wd < best_grid[0]:
            best_grid = (wd, lam, ws)

    return {
        "file": str(path),
        "n": n,
        "m": len(clauses),
        "solutions": len(sats),
        "p": p,
        "feasible_linear_potential": feasible,
        "lambda_lower": lower if not impossible else None,
        "lambda_upper": upper if not impossible else None,
        "lambda_star": lambda_star,
        "worst_drift_at_lambda_star": worst_drift,
        "worst_state_at_lambda_star": worst_state,
        "zero_b_impossible_states": len(impossible),
        "best_grid_lambda": best_grid[1],
        "best_grid_worst_drift": best_grid[0],
        "best_grid_worst_state": best_grid[2],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--p", type=float, default=0.45)
    ap.add_argument("--out", default="omega_drift_analysis.json")
    args = ap.parse_args()

    results = [analyze(path, args.p) for path in args.files]

    for r in results:
        print(json.dumps(r, indent=2))

    Path(args.out).write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nSaved {args.out}")


if __name__ == "__main__":
    main()
