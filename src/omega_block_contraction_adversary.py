#!/usr/bin/env python3
"""
OMEGA Block-Contraction Adversary
=================================

Searches strict planted 3-SAT formulas for which the current focused OMEGA
dynamics has the weakest probability of reaching a GLOBAL MINIMUM within a
polynomial block of L steps.

For a formula F, let Q_F be the transition matrix restricted to non-optimal
states, where the absorbing target is the set of global minima of the exact
clause energy.  Then

    s_L(x) = (Q_F^L 1)(x)

is the probability that a trajectory started at x has NOT reached a global
minimum within L updates.

Define

    epsilon_L(F) = 1 - max_x s_L(x).

If one could prove, uniformly for every 3-SAT formula, that for some
L=poly(n+m),

    epsilon_L(F) >= 1/poly(n+m),

then the expected time to reach a global minimum would be polynomial by a
geometric-block argument:

    E[T] <= L / epsilon_L.

This script searches for formulas making epsilon_L small.

The current arena uses strict 3-SAT instances planted to be satisfiable by the
all-ones assignment.  It is a falsification tool, not a proof of a uniform
bound.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import bicgstab

from omega_nonlocal_cluster import analyze_components

P_EXPLORE = 0.45


class StrictClauseUniverse:
    def __init__(self, n: int):
        self.n = n
        self.num_states = 1 << n
        self.states = np.arange(self.num_states, dtype=np.uint32)
        self.clauses = []
        self.violation = []

        # Strict 3-clauses satisfied by all-ones.
        for vars_ in itertools.combinations(range(1, n + 1), 3):
            for signs in itertools.product((-1, 1), repeat=3):
                clause = tuple(signs[i] * vars_[i] for i in range(3))
                if all(lit < 0 for lit in clause):
                    continue

                sat = np.zeros(self.num_states, dtype=bool)
                for lit in clause:
                    bit = ((self.states >> (abs(lit) - 1)) & 1).astype(bool)
                    sat |= bit if lit > 0 else ~bit

                self.clauses.append(clause)
                self.violation.append((~sat).astype(np.uint8))

        self.violation = np.stack(self.violation, axis=0)

    def transition_matrix(self, genome):
        n = self.n
        states = self.states
        selected = np.asarray(genome, dtype=np.int64)

        E = self.violation[selected].sum(axis=0, dtype=np.int16)
        Emin = int(E.min())
        target = (E == Emin)
        transient = np.flatnonzero(~target)

        if len(transient) == 0:
            return E, transient, sparse.csr_matrix((0, 0)), Emin

        deltas = np.empty((n, self.num_states), dtype=np.int16)
        for v in range(n):
            deltas[v] = E[states ^ (1 << v)] - E

        denom = E.astype(float)
        flip_prob = np.zeros((self.num_states, n), dtype=float)

        for gi in genome:
            idx = np.flatnonzero(self.violation[gi])
            if len(idx) == 0:
                continue

            clause = self.clauses[gi]
            cand = [abs(lit)-1 for lit in clause]

            rnd = P_EXPLORE / 3.0 / denom[idx]
            for v in cand:
                flip_prob[idx, v] += rnd

            dmat = np.vstack([deltas[v, idx] for v in cand]).T
            best = dmat.min(axis=1)
            ties = (dmat == best[:, None])
            counts = ties.sum(axis=1)

            for j, v in enumerate(cand):
                sel = ties[:, j]
                if np.any(sel):
                    flip_prob[idx[sel], v] += (
                        (1.0 - P_EXPLORE)
                        / denom[idx[sel]]
                        / counts[sel]
                    )

        index = np.full(self.num_states, -1, dtype=np.int64)
        index[transient] = np.arange(len(transient))

        rows = []
        cols = []
        data = []

        for v in range(n):
            pr = flip_prob[transient, v]
            nz = pr > 1e-15
            if not np.any(nz):
                continue

            source = np.flatnonzero(nz)
            dest_state = transient[nz] ^ (1 << v)
            stay = ~target[dest_state]

            if np.any(stay):
                rows.append(source[stay])
                cols.append(index[dest_state[stay]])
                data.append(pr[nz][stay])

        Q = sparse.csr_matrix(
            (
                np.concatenate(data) if data else np.array([], dtype=float),
                (
                    np.concatenate(rows) if rows else np.array([], dtype=int),
                    np.concatenate(cols) if cols else np.array([], dtype=int),
                ),
            ),
            shape=(len(transient), len(transient)),
        )

        return E, transient, Q, Emin

    def metrics(self, genome, L):
        E, transient, Q, Emin = self.transition_matrix(genome)

        s = np.ones(len(transient), dtype=float)
        for _ in range(L):
            s = Q @ s

        max_survival = float(s.max()) if len(s) else 0.0
        epsilon = max(0.0, 1.0 - max_survival)

        if len(transient):
            A = sparse.eye(len(transient), format="csr") - Q
            h, info = bicgstab(
                A,
                np.ones(len(transient)),
                rtol=1e-10,
                atol=1e-12,
                maxiter=20000,
            )
            if info != 0:
                return None
            residual = float(np.max(np.abs(A @ h - 1.0)))
            max_h = float(h.max())
            mean_h = float(h.mean())
        else:
            residual = 0.0
            max_h = 0.0
            mean_h = 0.0

        return {
            "global_min_energy": Emin,
            "global_min_count": int(np.sum(E == Emin)),
            "L": int(L),
            "max_survival_L": max_survival,
            "epsilon_L": epsilon,
            "geometric_block_bound": (
                float(L / epsilon) if epsilon > 0 else math.inf
            ),
            "max_expected_hitting": max_h,
            "mean_expected_hitting": mean_h,
            "max_residual": residual,
        }


def mutate(rng, genome, universe_size, rate):
    genes = set(genome)
    k = max(1, int(round(rate * len(genome))))

    for _ in range(k):
        genes.remove(rng.choice(tuple(genes)))
        while True:
            x = rng.randrange(universe_size)
            if x not in genes:
                genes.add(x)
                break

    return tuple(sorted(genes))


def write_dimacs(path, n, clauses):
    with open(path, "w", encoding="utf-8") as f:
        f.write("c OMEGA block-contraction adversarial finalist\n")
        f.write(f"p cnf {n} {len(clauses)}\n")
        for c in clauses:
            f.write(" ".join(map(str, c)) + " 0\n")


def evolve(n, m, population, generations, elite, seed, block_factor, prefix):
    rng = random.Random(seed)
    universe = StrictClauseUniverse(n)
    U = len(universe.clauses)
    L = block_factor * n * n

    pop = [
        tuple(sorted(rng.sample(range(U), m)))
        for _ in range(population)
    ]

    cache = {}

    def evaluate(g):
        if g in cache:
            return cache[g]

        clauses = [universe.clauses[i] for i in g]
        structural = analyze_components(n, clauses)

        if (
            structural.recovered_units
            or structural.equality_relations
            or structural.inequality_relations
        ):
            cache[g] = {
                "fitness": -1.0,
                "metrics": None,
                "structure": {
                    "units": structural.recovered_units,
                    "equalities": structural.equality_relations,
                    "inequalities": structural.inequality_relations,
                    "binaries": structural.recovered_binaries,
                },
            }
            return cache[g]

        metrics = universe.metrics(g, L)
        if metrics is None:
            fitness = -1.0
        else:
            fitness = (
                metrics["max_survival_L"]
                + 1e-9 * metrics["max_expected_hitting"]
            )

        cache[g] = {
            "fitness": fitness,
            "metrics": metrics,
            "structure": {
                "units": structural.recovered_units,
                "equalities": structural.equality_relations,
                "inequalities": structural.inequality_relations,
                "binaries": structural.recovered_binaries,
            },
        }
        return cache[g]

    history = []

    for gen in range(generations):
        ranked = sorted(
            pop,
            key=lambda g: evaluate(g)["fitness"],
            reverse=True,
        )
        best = ranked[0]
        b = evaluate(best)

        history.append({
            "generation": gen,
            "max_survival_L": (
                b["metrics"]["max_survival_L"]
                if b["metrics"] is not None else -1
            ),
            "epsilon_L": (
                b["metrics"]["epsilon_L"]
                if b["metrics"] is not None else -1
            ),
            "max_expected_hitting": (
                b["metrics"]["max_expected_hitting"]
                if b["metrics"] is not None else -1
            ),
        })

        print(
            f"n={n} gen={gen:02d} "
            f"survival={history[-1]['max_survival_L']:.8f} "
            f"eps={history[-1]['epsilon_L']:.8g} "
            f"maxH={history[-1]['max_expected_hitting']:.3f}"
        )

        parents = ranked[:elite]
        new_pop = list(parents)

        while len(new_pop) < population:
            parent = rng.choice(parents)
            child = mutate(
                rng,
                parent,
                U,
                rng.choice((0.04, 0.07, 0.10, 0.15)),
            )
            new_pop.append(child)

        pop = new_pop

    ranked = sorted(
        pop,
        key=lambda g: evaluate(g)["fitness"],
        reverse=True,
    )
    best = ranked[0]
    result = evaluate(best)
    clauses = [universe.clauses[i] for i in best]

    write_dimacs(prefix + ".cnf", n, clauses)

    with open(prefix + "_history.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        w.writeheader()
        w.writerows(history)

    report = {
        "n": n,
        "m": m,
        "block_factor": block_factor,
        "L": L,
        "population": population,
        "generations": generations,
        "fitness": result["fitness"],
        "metrics": result["metrics"],
        "structure": result["structure"],
        "cnf": prefix + ".cnf",
    }

    Path(prefix + "_report.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--m", type=int)
    ap.add_argument("--population", type=int, default=18)
    ap.add_argument("--generations", type=int, default=10)
    ap.add_argument("--elite", type=int, default=5)
    ap.add_argument("--block-factor", type=int, default=1)
    ap.add_argument("--seed", type=int, default=26032009)
    ap.add_argument("--prefix", default="omega_block_contraction_hard")
    args = ap.parse_args()

    m = args.m if args.m is not None else round(4.2 * args.n)

    report = evolve(
        n=args.n,
        m=m,
        population=args.population,
        generations=args.generations,
        elite=args.elite,
        seed=args.seed,
        block_factor=args.block_factor,
        prefix=args.prefix,
    )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
