#!/usr/bin/env python3
"""
OMEGA Nonlocal Cluster Pulse
============================

A read-only-hardware-compatible research prototype inspired by the OMEGA
Tiny Tapeout interface:

    grid / threshold bits -> Boolean DAG -> global cluster-control layer

No Tiny Tapeout repository files are modified.

The purpose of this program is to test a genuinely nonlocal update:
correlated Boolean cells are grouped into signed components and changed
simultaneously.

Exact reductions used
---------------------
The identity

    (R OR z) AND (R OR NOT z)  ==  R

is applied recursively as an exact gadget reduction. This recovers hidden
binary/unit constraints from strict 3-CNF gadgets.

From recovered binary clauses, the program detects exact pair constraints:

    (NOT x OR y) AND (x OR NOT y)  => x = y
    (x OR y) AND (NOT x OR NOT y)  => x != y

These relations are stored in a parity union-find structure. Recovered unit
clauses impose values on entire signed components. A "cluster pulse" updates
every Boolean cell in a constrained component simultaneously.

This defeats the explicit polynomial-size barrier family that traps the
previous one-cell greedy/exploratory dynamics.

It does NOT prove polynomial-time solution of arbitrary 3-SAT. Generic
3-clauses that do not reduce to these exact component constraints remain
outside the theorem proved by this prototype.
"""

from __future__ import annotations

import argparse
import itertools
import random
import time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, FrozenSet, Iterable, List, Optional, Set, Tuple


Clause = Tuple[int, ...]


def normalize_clause(clause: Iterable[int]) -> Optional[Clause]:
    """Normalize a clause; return None for a tautology."""
    lits = set(int(x) for x in clause if int(x) != 0)
    for lit in list(lits):
        if -lit in lits:
            return None
    return tuple(sorted(lits, key=lambda x: (abs(x), x < 0)))


def parse_dimacs(path: str) -> Tuple[int, List[Clause]]:
    n = 0
    clauses: List[Clause] = []
    cur: List[int] = []
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                n = int(parts[2])
                continue
            for tok in line.split():
                v = int(tok)
                if v == 0:
                    c = normalize_clause(cur)
                    if c is not None:
                        clauses.append(c)
                    cur = []
                else:
                    cur.append(v)
    if cur:
        raise ValueError("DIMACS clause missing terminal 0")
    return n, clauses


def literal_truth(lit: int, bits: List[int]) -> int:
    b = bits[abs(lit) - 1]
    return b if lit > 0 else 1 - b


def clause_sat(c: Clause, bits: List[int]) -> bool:
    return any(literal_truth(l, bits) for l in c)


def energy(clauses: List[Clause], bits: List[int]) -> int:
    return sum(not clause_sat(c, bits) for c in clauses)


def exact_gadget_closure(
    clauses: Iterable[Clause],
    max_rounds: int = 16,
) -> Set[Clause]:
    """Add exact consequences from (R∨z)&(R∨¬z) <=> R.

    Only this special equivalence reduction is used; this is not unrestricted
    resolution.
    """
    known: Set[Clause] = set()
    for c in clauses:
        nc = normalize_clause(c)
        if nc is not None:
            known.add(nc)

    for _ in range(max_rounds):
        by_signature: Dict[Tuple[Tuple[int, ...], int], Set[int]] = defaultdict(set)

        # For every literal z in clause C, key by the rest R and pivot |z|.
        for c in known:
            if len(c) < 2:
                continue
            for lit in c:
                pivot = abs(lit)
                rest = tuple(x for x in c if x != lit)
                by_signature[(rest, pivot)].add(1 if lit > 0 else -1)

        new_clauses: Set[Clause] = set()
        for (rest, _pivot), signs in by_signature.items():
            if 1 in signs and -1 in signs:
                nr = normalize_clause(rest)
                if nr is not None and nr not in known:
                    new_clauses.add(nr)

        if not new_clauses:
            break
        known.update(new_clauses)

    return known


class ParityDSU:
    """DSU with parity: value[x] XOR value[parent[x]]."""

    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.parity = [0] * n

    def find(self, x: int) -> Tuple[int, int]:
        if self.parent[x] == x:
            return x, 0
        root, p = self.find(self.parent[x])
        self.parity[x] ^= p
        self.parent[x] = root
        return self.parent[x], self.parity[x]

    def union(self, a: int, b: int, relation: int) -> bool:
        """Impose value[a] XOR value[b] = relation. Return False on conflict."""
        ra, pa = self.find(a)
        rb, pb = self.find(b)
        if ra == rb:
            return (pa ^ pb) == relation

        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
            pa, pb = pb, pa

        self.parent[rb] = ra
        # pa XOR parity_rb XOR pb = relation
        self.parity[rb] = pa ^ pb ^ relation

        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True


@dataclass
class ClusterAnalysis:
    closure: Set[Clause]
    dsu: ParityDSU
    root_values: Dict[int, int]
    contradiction: bool
    equality_relations: int
    inequality_relations: int
    recovered_units: int
    recovered_binaries: int


def clause_set_key(c: Clause) -> FrozenSet[int]:
    return frozenset(c)


def analyze_components(n: int, clauses: List[Clause]) -> ClusterAnalysis:
    closure = exact_gadget_closure(clauses)
    units = [c for c in closure if len(c) == 1]
    binaries = [c for c in closure if len(c) == 2]
    binary_set = {frozenset(c) for c in binaries}

    dsu = ParityDSU(n)
    contradiction = False
    eq_count = 0
    neq_count = 0

    # Examine unordered variable pairs.
    pairs: Set[Tuple[int, int]] = set()
    for c in binaries:
        a, b = c
        va, vb = abs(a), abs(b)
        if va != vb:
            pairs.add(tuple(sorted((va, vb))))

    for va, vb in pairs:
        # Equality: (-x or y) and (x or -y)
        eq1 = frozenset((-va, vb))
        eq2 = frozenset((va, -vb))
        # Inequality: (x or y) and (-x or -y)
        ne1 = frozenset((va, vb))
        ne2 = frozenset((-va, -vb))

        if eq1 in binary_set and eq2 in binary_set:
            if not dsu.union(va - 1, vb - 1, 0):
                contradiction = True
            eq_count += 1

        if ne1 in binary_set and ne2 in binary_set:
            if not dsu.union(va - 1, vb - 1, 1):
                contradiction = True
            neq_count += 1

    # Apply unit constraints to roots.
    root_values: Dict[int, int] = {}
    for c in units:
        lit = c[0]
        var = abs(lit) - 1
        required = 1 if lit > 0 else 0
        root, p = dsu.find(var)
        root_required = required ^ p
        if root in root_values and root_values[root] != root_required:
            contradiction = True
        root_values[root] = root_required

    return ClusterAnalysis(
        closure=closure,
        dsu=dsu,
        root_values=root_values,
        contradiction=contradiction,
        equality_relations=eq_count,
        inequality_relations=neq_count,
        recovered_units=len(units),
        recovered_binaries=len(binaries),
    )


def cluster_pulse_assignment(
    n: int,
    analysis: ClusterAnalysis,
    seed: int = 0,
) -> List[int]:
    rng = random.Random(seed)
    roots: Dict[int, int] = {}

    for i in range(n):
        r, _ = analysis.dsu.find(i)
        if r not in roots:
            roots[r] = analysis.root_values.get(r, rng.randrange(2))

    bits = [0] * n
    for i in range(n):
        r, p = analysis.dsu.find(i)
        bits[i] = roots[r] ^ p
    return bits


# ---------- Strict barrier family ----------

def two_clause_gadget(a: int, b: int, z: int) -> List[Clause]:
    return [
        normalize_clause((a, b, z)),
        normalize_clause((a, b, -z)),
    ]


def unit_gadget(x: int, u: int, v: int) -> List[Clause]:
    return [
        normalize_clause((x, u, v)),
        normalize_clause((x, u, -v)),
        normalize_clause((x, -u, v)),
        normalize_clause((x, -u, -v)),
    ]


def strict_barrier_formula(n: int) -> Tuple[int, List[Clause]]:
    if n < 2:
        raise ValueError("n must be >= 2")

    clauses: List[Clause] = []
    next_var = n + 1

    for i in range(1, n):
        w = n - i
        for _ in range(w):
            z = next_var
            next_var += 1
            clauses.extend(two_clause_gadget(-i, i + 1, z))

            z = next_var
            next_var += 1
            clauses.extend(two_clause_gadget(i, -(i + 1), z))

    u, v = next_var, next_var + 1
    next_var += 2
    clauses.extend(unit_gadget(n, u, v))

    assert all(c is not None for c in clauses)
    return next_var - 1, [c for c in clauses if c is not None]


def barrier_demo(sizes: List[int]) -> None:
    print(
        "n,total_vars,strict_clauses,closure_clauses,"
        "eq_relations,units,analysis_ms,pulse_ms,energy"
    )

    for n in sizes:
        total_vars, clauses = strict_barrier_formula(n)

        t0 = time.perf_counter()
        analysis = analyze_components(total_vars, clauses)
        t1 = time.perf_counter()

        bits = cluster_pulse_assignment(total_vars, analysis, seed=26032009)
        t2 = time.perf_counter()

        e = energy(clauses, bits)

        print(
            f"{n},{total_vars},{len(clauses)},{len(analysis.closure)},"
            f"{analysis.equality_relations},{analysis.recovered_units},"
            f"{(t1-t0)*1000:.3f},{(t2-t1)*1000:.3f},{e}"
        )

        if analysis.contradiction:
            raise AssertionError("Unexpected contradiction in satisfiable barrier")
        if e != 0:
            raise AssertionError(
                f"Cluster pulse failed barrier n={n}, energy={e}"
            )


def planted_3sat(n: int, ratio: float, seed: int) -> List[Clause]:
    rng = random.Random(seed)
    planted = [rng.randrange(2) for _ in range(n)]
    clauses: List[Clause] = []

    for _ in range(round(ratio * n)):
        vars_ = rng.sample(range(1, n + 1), 3)
        c = [v if rng.random() < 0.5 else -v for v in vars_]
        if not clause_sat(tuple(c), planted):
            v = abs(c[0])
            c[0] = v if planted[v - 1] else -v
        clauses.append(normalize_clause(c))

    return [c for c in clauses if c is not None]


def random_demo(sizes: List[int], ratio: float) -> None:
    print("\nGeneric planted 3-SAT diagnostic")
    print("n,m,recovered_binary,recovered_unit,eq,neq,pulse_energy")
    for i, n in enumerate(sizes):
        clauses = planted_3sat(n, ratio, 12345 + i)
        a = analyze_components(n, clauses)
        bits = cluster_pulse_assignment(n, a, seed=999 + i)
        print(
            f"{n},{len(clauses)},{a.recovered_binaries},"
            f"{a.recovered_units},{a.equality_relations},"
            f"{a.inequality_relations},{energy(clauses,bits)}"
        )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--barrier-demo", action="store_true")
    ap.add_argument("--random-demo", action="store_true")
    ap.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[10, 20, 50, 100, 250],
    )
    ap.add_argument("--ratio", type=float, default=4.2)
    ap.add_argument("--cnf")
    args = ap.parse_args()

    if args.barrier_demo:
        barrier_demo(args.sizes)

    if args.random_demo:
        random_demo(args.sizes, args.ratio)

    if args.cnf:
        n, clauses = parse_dimacs(args.cnf)
        a = analyze_components(n, clauses)
        print(
            {
                "n": n,
                "m": len(clauses),
                "closure": len(a.closure),
                "units": a.recovered_units,
                "binaries": a.recovered_binaries,
                "equalities": a.equality_relations,
                "inequalities": a.inequality_relations,
                "contradiction": a.contradiction,
            }
        )
        if not a.contradiction:
            bits = cluster_pulse_assignment(n, a, seed=26032009)
            print("pulse_energy:", energy(clauses, bits))


if __name__ == "__main__":
    main()
