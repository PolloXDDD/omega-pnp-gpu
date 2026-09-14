#!/usr/bin/env python3
"""
OMEGA exact 3-SAT compiler
==========================

Exact logical compiler:

    3-SAT -> sparse QUBO -> sparse Ising -> OMEGA coupling specification

The QUBO encoding is exact on binary variables.  For each 3-literal clause,
one auxiliary binary variable is introduced.  The minimum QUBO energy is zero
if and only if the original 3-SAT formula is satisfiable.

This file deliberately separates:

  (1) exact SAT semantics of the compiler, from
  (2) the dynamics used to locate the global minimum.

The compiler is O(n + m) in size for a 3-CNF formula with n variables and
m clauses.  A complete P=NP argument additionally needs a proof that the
chosen OMEGA relaxation reaches the global zero-energy state (or certifies
positive minimum) in polynomial time.

Usage
-----

Self-test:
    python omega_exact_sat_compiler.py --self-test

Compile DIMACS:
    python omega_exact_sat_compiler.py formula.cnf --out omega_formula

Outputs:
    omega_formula.qubo.csv
    omega_formula.ising.csv
    omega_formula.dag.json
    omega_formula.meta.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


Affine = Tuple[float, Dict[int, float]]


@dataclass
class QUBO:
    """Upper-triangular QUBO representation.

    E(x) = constant
         + sum_i linear[i] x_i
         + sum_{i<j} quadratic[i,j] x_i x_j

    Binary identity x_i^2 = x_i is applied automatically.
    """

    constant: float = 0.0
    linear: Dict[int, float] = field(default_factory=dict)
    quadratic: Dict[Tuple[int, int], float] = field(default_factory=dict)

    def add_constant(self, value: float) -> None:
        self.constant += value

    def add_linear(self, i: int, value: float) -> None:
        if abs(value) < 1e-15:
            return
        self.linear[i] = self.linear.get(i, 0.0) + value
        if abs(self.linear[i]) < 1e-15:
            del self.linear[i]

    def add_quadratic(self, i: int, j: int, value: float) -> None:
        if abs(value) < 1e-15:
            return
        if i == j:
            # x_i^2 = x_i for Boolean variables.
            self.add_linear(i, value)
            return
        if i > j:
            i, j = j, i
        key = (i, j)
        self.quadratic[key] = self.quadratic.get(key, 0.0) + value
        if abs(self.quadratic[key]) < 1e-15:
            del self.quadratic[key]

    def add_affine(self, a: Affine, scale: float = 1.0) -> None:
        c, terms = a
        self.add_constant(scale * c)
        for i, coeff in terms.items():
            self.add_linear(i, scale * coeff)

    def add_affine_product(
        self, a: Affine, b: Affine, scale: float = 1.0
    ) -> None:
        ca, ta = a
        cb, tb = b

        self.add_constant(scale * ca * cb)

        for i, ai in ta.items():
            self.add_linear(i, scale * ai * cb)

        for j, bj in tb.items():
            self.add_linear(j, scale * bj * ca)

        for i, ai in ta.items():
            for j, bj in tb.items():
                self.add_quadratic(i, j, scale * ai * bj)

    def energy(self, bits: List[int]) -> float:
        e = self.constant
        for i, c in self.linear.items():
            e += c * bits[i]
        for (i, j), c in self.quadratic.items():
            e += c * bits[i] * bits[j]
        return e

    @property
    def nnz_pairs(self) -> int:
        return len(self.quadratic)


@dataclass
class Ising:
    """Ising representation H(s)=constant + sum h_i s_i + sum J_ij s_i s_j."""

    constant: float
    h: Dict[int, float]
    J: Dict[Tuple[int, int], float]

    def energy(self, spins: List[int]) -> float:
        e = self.constant
        for i, c in self.h.items():
            e += c * spins[i]
        for (i, j), c in self.J.items():
            e += c * spins[i] * spins[j]
        return e


@dataclass
class ClauseInfo:
    literals: List[int]
    aux_index: int | None


@dataclass
class CompiledFormula:
    num_original_vars: int
    clauses: List[List[int]]
    qubo: QUBO
    clause_info: List[ClauseInfo]
    num_total_binary_vars: int
    penalty_M: float


def parse_dimacs(path: str | Path) -> Tuple[int, List[List[int]]]:
    num_vars = 0
    clauses: List[List[int]] = []
    current: List[int] = []

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("c"):
                continue
            if line.startswith("p"):
                parts = line.split()
                if len(parts) < 4 or parts[1].lower() != "cnf":
                    raise ValueError("Expected DIMACS header: p cnf <n> <m>")
                num_vars = int(parts[2])
                continue

            for token in line.split():
                lit = int(token)
                if lit == 0:
                    if current:
                        clauses.append(current)
                        current = []
                else:
                    current.append(lit)

    if current:
        raise ValueError("DIMACS clause missing terminating 0")

    if num_vars <= 0:
        raise ValueError("DIMACS file has no valid p cnf header")

    for clause in clauses:
        for lit in clause:
            if abs(lit) < 1 or abs(lit) > num_vars:
                raise ValueError(f"Literal {lit} outside variable range 1..{num_vars}")

    return num_vars, clauses


def false_literal_affine(lit: int) -> Affine:
    """Return the false-indicator f(lit).

    Positive literal x_i is false when 1-x_i = 1.
    Negative literal NOT x_i is false when x_i = 1.
    Internal variable indices are zero-based.
    """
    i = abs(lit) - 1
    if lit > 0:
        return (1.0, {i: -1.0})   # 1 - x_i
    return (0.0, {i: 1.0})       # x_i


def variable_affine(i: int) -> Affine:
    return (0.0, {i: 1.0})


def compile_3sat_to_qubo(
    num_vars: int,
    clauses: List[List[int]],
    penalty_M: float = 2.0,
) -> CompiledFormula:
    """Compile clauses of size 1..3 to an exact sparse QUBO.

    For a 3-clause, let a,b,c be the false-indicators of its literals and
    introduce auxiliary y.  The clause gadget is

        E_C = y c + M (ab - 2ay - 2by + 3y).

    For M >= 1,

        min_y E_C = abc,

    so the minimized clause energy is exactly 1 iff all three literals are
    false and 0 otherwise.
    """
    if penalty_M < 1.0:
        raise ValueError("penalty_M must be >= 1 for the exact clause gadget")

    qubo = QUBO()
    info: List[ClauseInfo] = []
    next_var = num_vars

    for clause in clauses:
        if not 1 <= len(clause) <= 3:
            raise ValueError(
                "This exact compiler accepts clauses of length 1..3. "
                "Convert wider CNF to 3-CNF first."
            )

        f = [false_literal_affine(lit) for lit in clause]

        if len(clause) == 1:
            # Unsatisfied iff f1 = 1.
            qubo.add_affine(f[0], 1.0)
            info.append(ClauseInfo(list(clause), None))

        elif len(clause) == 2:
            # Unsatisfied iff f1*f2 = 1.
            qubo.add_affine_product(f[0], f[1], 1.0)
            info.append(ClauseInfo(list(clause), None))

        else:
            y_index = next_var
            next_var += 1
            y = variable_affine(y_index)

            # M (ab - 2ay - 2by + 3y) + y*c
            qubo.add_affine_product(f[0], f[1], penalty_M)
            qubo.add_affine_product(f[0], y, -2.0 * penalty_M)
            qubo.add_affine_product(f[1], y, -2.0 * penalty_M)
            qubo.add_affine(y, 3.0 * penalty_M)
            qubo.add_affine_product(y, f[2], 1.0)

            info.append(ClauseInfo(list(clause), y_index))

    return CompiledFormula(
        num_original_vars=num_vars,
        clauses=[list(c) for c in clauses],
        qubo=qubo,
        clause_info=info,
        num_total_binary_vars=next_var,
        penalty_M=penalty_M,
    )


def literal_truth(lit: int, x: List[int]) -> int:
    value = x[abs(lit) - 1]
    return value if lit > 0 else 1 - value


def literal_false(lit: int, x: List[int]) -> int:
    return 1 - literal_truth(lit, x)


def unsatisfied_clause_count(clauses: List[List[int]], x: List[int]) -> int:
    return sum(
        1
        for clause in clauses
        if not any(literal_truth(lit, x) for lit in clause)
    )


def canonical_aux_assignment(
    compiled: CompiledFormula,
    original_bits: List[int],
) -> List[int]:
    """Extend x by the minimizing clause auxiliaries y=f1*f2."""
    bits = list(original_bits) + [0] * (
        compiled.num_total_binary_vars - compiled.num_original_vars
    )

    for ci in compiled.clause_info:
        if ci.aux_index is None:
            continue
        a = literal_false(ci.literals[0], original_bits)
        b = literal_false(ci.literals[1], original_bits)
        bits[ci.aux_index] = a * b

    return bits


def qubo_to_ising(qubo: QUBO, nvars: int) -> Ising:
    """Exact substitution x_i=(1+s_i)/2."""
    constant = qubo.constant
    h: Dict[int, float] = {}
    J: Dict[Tuple[int, int], float] = {}

    for i, q in qubo.linear.items():
        constant += q / 2.0
        h[i] = h.get(i, 0.0) + q / 2.0

    for (i, j), q in qubo.quadratic.items():
        constant += q / 4.0
        h[i] = h.get(i, 0.0) + q / 4.0
        h[j] = h.get(j, 0.0) + q / 4.0
        J[(i, j)] = J.get((i, j), 0.0) + q / 4.0

    h = {i: v for i, v in h.items() if abs(v) > 1e-15}
    J = {ij: v for ij, v in J.items() if abs(v) > 1e-15}

    return Ising(constant=constant, h=h, J=J)


def build_verifier_dag(
    num_vars: int,
    clauses: List[List[int]],
) -> dict:
    """Build an abstract acyclic Boolean verifier DAG.

    Input nodes 0..num_vars-1 are assignment bits.  NOT nodes are cached.
    Each clause is OR-reduced and all clause outputs are AND-reduced.
    """
    gates = []
    next_node = num_vars
    not_cache: Dict[int, int] = {}

    def negated_node(var_idx: int) -> int:
        nonlocal next_node
        if var_idx not in not_cache:
            gates.append(
                {
                    "out": next_node,
                    "op": "NOT",
                    "a": var_idx,
                    "b": None,
                }
            )
            not_cache[var_idx] = next_node
            next_node += 1
        return not_cache[var_idx]

    def literal_node(lit: int) -> int:
        idx = abs(lit) - 1
        return idx if lit > 0 else negated_node(idx)

    clause_outputs = []

    for clause in clauses:
        nodes = [literal_node(lit) for lit in clause]

        if len(nodes) == 1:
            clause_outputs.append(nodes[0])
            continue

        cur = nodes[0]
        for nxt in nodes[1:]:
            gates.append(
                {
                    "out": next_node,
                    "op": "OR",
                    "a": cur,
                    "b": nxt,
                }
            )
            cur = next_node
            next_node += 1

        clause_outputs.append(cur)

    if not clause_outputs:
        # Empty conjunction = TRUE; represent as a metadata constant.
        final_node = None
        constant_output = 1
    else:
        cur = clause_outputs[0]
        for nxt in clause_outputs[1:]:
            gates.append(
                {
                    "out": next_node,
                    "op": "AND",
                    "a": cur,
                    "b": nxt,
                }
            )
            cur = next_node
            next_node += 1
        final_node = cur
        constant_output = None

    return {
        "input_nodes": num_vars,
        "gates": gates,
        "output_node": final_node,
        "constant_output": constant_output,
        "total_nodes": next_node,
    }


def evaluate_dag(dag: dict, assignment: List[int]) -> int:
    nodes: Dict[int, int] = {i: int(v) for i, v in enumerate(assignment)}

    if dag.get("constant_output") is not None:
        return int(dag["constant_output"])

    for g in dag["gates"]:
        op = g["op"]
        a = nodes[g["a"]]
        if op == "NOT":
            value = 1 - a
        else:
            b = nodes[g["b"]]
            if op == "OR":
                value = int(bool(a or b))
            elif op == "AND":
                value = int(bool(a and b))
            else:
                raise ValueError(f"Unsupported DAG op {op}")
        nodes[g["out"]] = value

    return nodes[dag["output_node"]]


def omega_dual_rail_spec(ising: Ising) -> dict:
    """Translate Ising coefficients into an idealized passive dual-rail map.

    Logical spin:
        s_i=+1 -> (V_i+,V_i-)=(1,0)
        s_i=-1 -> (V_i+,V_i-)=(0,1)

    Pair term J_ij s_i s_j:
      J < 0: same-rail conductors, each conductance g=2|J|
      J > 0: cross-rail conductors, each conductance g=2|J|

    Bias h_i s_i:
      h > 0: connect + rail to ground with g=4h
      h < 0: connect - rail to ground with g=4|h|

    This realizes the Ising objective up to an additive constant *provided*
    every rail pair is constrained to the two Boolean states above.
    """
    couplers = []
    biases = []

    for (i, j), J in sorted(ising.J.items()):
        if J < 0:
            couplers.append(
                {
                    "i": i,
                    "j": j,
                    "mode": "same_rail",
                    "conductance": 2.0 * abs(J),
                }
            )
        elif J > 0:
            couplers.append(
                {
                    "i": i,
                    "j": j,
                    "mode": "cross_rail",
                    "conductance": 2.0 * abs(J),
                }
            )

    for i, h in sorted(ising.h.items()):
        if h > 0:
            biases.append(
                {
                    "var": i,
                    "rail": "plus",
                    "reference": 0,
                    "conductance": 4.0 * h,
                }
            )
        elif h < 0:
            biases.append(
                {
                    "var": i,
                    "rail": "minus",
                    "reference": 0,
                    "conductance": 4.0 * abs(h),
                }
            )

    return {
        "encoding": {
            "spin_plus_1": ["Vplus=1", "Vminus=0"],
            "spin_minus_1": ["Vplus=0", "Vminus=1"],
        },
        "requires_boolean_bistability": True,
        "couplers": couplers,
        "biases": biases,
    }


def write_outputs(prefix: str | Path, compiled: CompiledFormula) -> None:
    prefix = Path(prefix)
    qubo = compiled.qubo
    ising = qubo_to_ising(qubo, compiled.num_total_binary_vars)
    dag = build_verifier_dag(compiled.num_original_vars, compiled.clauses)
    omega = omega_dual_rail_spec(ising)

    qubo_path = prefix.with_suffix(".qubo.csv")
    with open(qubo_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "i", "j", "coefficient"])
        w.writerow(["constant", "", "", qubo.constant])
        for i, c in sorted(qubo.linear.items()):
            w.writerow(["linear", i, "", c])
        for (i, j), c in sorted(qubo.quadratic.items()):
            w.writerow(["quadratic", i, j, c])

    ising_path = prefix.with_suffix(".ising.csv")
    with open(ising_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["kind", "i", "j", "coefficient"])
        w.writerow(["constant", "", "", ising.constant])
        for i, c in sorted(ising.h.items()):
            w.writerow(["h", i, "", c])
        for (i, j), c in sorted(ising.J.items()):
            w.writerow(["J", i, j, c])

    dag_path = prefix.with_suffix(".dag.json")
    dag_path.write_text(json.dumps(dag, indent=2), encoding="utf-8")

    omega_path = prefix.with_suffix(".omega.json")
    omega_path.write_text(json.dumps(omega, indent=2), encoding="utf-8")

    meta = {
        "num_original_vars": compiled.num_original_vars,
        "num_clauses": len(compiled.clauses),
        "num_total_binary_vars": compiled.num_total_binary_vars,
        "num_aux_vars": compiled.num_total_binary_vars
        - compiled.num_original_vars,
        "qubo_pair_couplings": qubo.nnz_pairs,
        "penalty_M": compiled.penalty_M,
        "exact_statement":
            "min Q = 0 iff the input CNF is satisfiable",
    }
    meta_path = prefix.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote {qubo_path}")
    print(f"Wrote {ising_path}")
    print(f"Wrote {dag_path}")
    print(f"Wrote {omega_path}")
    print(f"Wrote {meta_path}")


def validate_compiler(
    num_vars: int,
    clauses: List[List[int]],
    penalty_M: float = 2.0,
    exhaustive_limit: int = 16,
) -> dict:
    compiled = compile_3sat_to_qubo(num_vars, clauses, penalty_M)
    ising = qubo_to_ising(compiled.qubo, compiled.num_total_binary_vars)
    dag = build_verifier_dag(num_vars, clauses)

    if num_vars > exhaustive_limit:
        raise ValueError(
            f"Exhaustive validation limited to n <= {exhaustive_limit}"
        )

    best_energy = math.inf
    sat_exists = False
    checked = 0

    for x_tuple in itertools.product((0, 1), repeat=num_vars):
        x = list(x_tuple)
        bits = canonical_aux_assignment(compiled, x)

        exact_unsat = unsatisfied_clause_count(clauses, x)
        q_energy = compiled.qubo.energy(bits)

        if abs(q_energy - exact_unsat) > 1e-9:
            raise AssertionError(
                f"QUBO mismatch: x={x}, Q={q_energy}, unsat={exact_unsat}"
            )

        spins = [2 * b - 1 for b in bits]
        h_energy = ising.energy(spins)
        if abs(h_energy - q_energy) > 1e-9:
            raise AssertionError(
                f"QUBO/Ising mismatch: x={x}, Q={q_energy}, H={h_energy}"
            )

        verifier = evaluate_dag(dag, x)
        if verifier != int(exact_unsat == 0):
            raise AssertionError(
                f"DAG verifier mismatch: x={x}, dag={verifier}, "
                f"unsatisfied={exact_unsat}"
            )

        best_energy = min(best_energy, q_energy)
        sat_exists |= (exact_unsat == 0)
        checked += 1

    if (best_energy == 0) != sat_exists:
        raise AssertionError("Global zero-energy iff SAT invariant failed")

    return {
        "assignments_checked": checked,
        "sat_exists": sat_exists,
        "minimum_energy": best_energy,
        "original_vars": num_vars,
        "clauses": len(clauses),
        "total_binary_vars": compiled.num_total_binary_vars,
        "aux_vars": compiled.num_total_binary_vars - num_vars,
        "qubo_pair_couplings": compiled.qubo.nnz_pairs,
        "dag_gates": len(dag["gates"]),
    }


def self_test() -> None:
    print("OMEGA exact compiler self-test")
    print("=" * 72)

    # A hand-checkable satisfiable formula.
    clauses1 = [
        [1, 2, 3],
        [-1, 2, -3],
        [1, -2, 3],
        [-1, -2, 3],
    ]
    r1 = validate_compiler(3, clauses1)
    print("Hand SAT case:", json.dumps(r1, sort_keys=True))

    # Unsatisfiable x AND NOT x.
    clauses2 = [[1], [-1]]
    r2 = validate_compiler(1, clauses2)
    print("Hand UNSAT case:", json.dumps(r2, sort_keys=True))

    # Random regression tests.
    rng = random.Random(26032009)
    cases = 50

    for case in range(cases):
        n = rng.randint(2, 7)
        m = rng.randint(1, 12)
        clauses = []

        for _ in range(m):
            clause = []
            for _ in range(3):
                v = rng.randint(1, n)
                clause.append(v if rng.random() < 0.5 else -v)
            clauses.append(clause)

        validate_compiler(n, clauses)

    print(f"Random exhaustive regression cases: {cases} / {cases} passed")
    print("ALL EXACT-COMPILER TESTS PASSED")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cnf", nargs="?")
    ap.add_argument("--out", default="omega_formula")
    ap.add_argument("--penalty", type=float, default=2.0)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument(
        "--validate",
        action="store_true",
        help="Exhaustively validate when the CNF has at most 16 variables.",
    )
    args = ap.parse_args()

    if args.self_test:
        self_test()
        return

    if not args.cnf:
        ap.error("provide a DIMACS CNF file or use --self-test")

    num_vars, clauses = parse_dimacs(args.cnf)
    compiled = compile_3sat_to_qubo(num_vars, clauses, args.penalty)

    print(
        f"Compiled n={num_vars}, m={len(clauses)} -> "
        f"{compiled.num_total_binary_vars} binary variables, "
        f"{compiled.qubo.nnz_pairs} quadratic couplings"
    )

    write_outputs(args.out, compiled)

    if args.validate:
        result = validate_compiler(
            num_vars,
            clauses,
            penalty_M=args.penalty,
        )
        print("Validation:", json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
