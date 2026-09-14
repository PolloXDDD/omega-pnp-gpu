#!/usr/bin/env python3
"""
OMEGA GPU / 3-SAT benchmark
---------------------------
Robust CUDA/CPU benchmark for the OMEGA sparse-relaxation prototype.

Backends:
    --backend auto   Try CUDA first, otherwise fall back to CPU.
    --backend cuda   Require CUDA/CuPy.
    --backend cpu    Use NumPy/SciPy.

Colab:
    !nvidia-smi
    !python omega_gpu_benchmark_colab.py --backend cuda --sizes 1024 4096 16384 --iters 200
"""

import argparse
import csv
import sys
import time
import numpy as np

CUDA_KERNEL_SOURCE = r"""
extern "C" __global__
void jacobi_csr(
    const int n,
    const int* __restrict__ row_ptr,
    const int* __restrict__ col_idx,
    const float* __restrict__ val,
    const float* __restrict__ diag,
    const float* __restrict__ rhs,
    const float* __restrict__ x,
    float* __restrict__ x_new)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    float s = 0.0f;
    for (int k = row_ptr[i]; k < row_ptr[i + 1]; ++k) {
        s += val[k] * x[col_idx[k]];
    }

    x_new[i] = (rhs[i] - s) / diag[i];
}
"""


def cuda_diagnostics(cp=None):
    lines = []
    try:
        import subprocess
        p = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if p.returncode == 0:
            lines.append("nvidia-smi: available")
            first = p.stdout.splitlines()
            if first:
                lines.extend(first[:8])
        else:
            lines.append("nvidia-smi returned a nonzero status.")
    except Exception as exc:
        lines.append(f"nvidia-smi unavailable: {exc}")

    if cp is not None:
        try:
            lines.append(f"CuPy: {cp.__version__}")
        except Exception:
            pass
        try:
            lines.append(
                f"CUDA runtime version: {cp.cuda.runtime.runtimeGetVersion()}"
            )
        except Exception as exc:
            lines.append(f"CUDA runtime query failed: {exc}")
        try:
            lines.append(
                f"CUDA driver version: {cp.cuda.runtime.driverGetVersion()}"
            )
        except Exception as exc:
            lines.append(f"CUDA driver query failed: {exc}")

    return "\n".join(lines)


def initialize_backend(requested):
    """Return a backend dictionary.

    CUDA objects are imported/compiled only after a usable CUDA device has
    actually been detected. This prevents module-import crashes on CPU-only
    machines and gives useful diagnostics on Colab.
    """
    if requested in ("auto", "cuda"):
        cp = None
        try:
            import cupy as cp
            import cupyx.scipy.sparse as csp

            count = int(cp.cuda.runtime.getDeviceCount())
            if count < 1:
                raise RuntimeError("CuPy loaded, but CUDA reports zero devices.")

            device = cp.cuda.Device(0)
            device.use()

            props = cp.cuda.runtime.getDeviceProperties(0)
            name = props["name"]
            if isinstance(name, bytes):
                name = name.decode(errors="replace")

            kernel = cp.RawKernel(CUDA_KERNEL_SOURCE, "jacobi_csr")

            print(f"Backend: CUDA")
            print(f"GPU: {name}")
            try:
                print(f"CuPy: {cp.__version__}")
                print(f"CUDA runtime: {cp.cuda.runtime.runtimeGetVersion()}")
                print(f"CUDA driver: {cp.cuda.runtime.driverGetVersion()}")
            except Exception:
                pass

            return {
                "kind": "cuda",
                "cp": cp,
                "csp": csp,
                "kernel": kernel,
            }

        except Exception as exc:
            if requested == "cuda":
                print("\nCUDA INITIALIZATION FAILED")
                print("=" * 72)
                print(f"{type(exc).__name__}: {exc}")
                print(cuda_diagnostics(cp))
                print(
                    "\nOn Google Colab:\n"
                    "  1. Runtime -> Change runtime type -> GPU\n"
                    "  2. Restart the runtime after changing/installing CUDA packages\n"
                    "  3. Run !nvidia-smi\n"
                    "  4. Prefer Colab's preinstalled CuPy when possible\n"
                    "\nA cudaErrorInsufficientDriver error means the installed CUDA "
                    "runtime/CuPy build is newer than the NVIDIA driver can support."
                )
                raise SystemExit(2)

            print(
                f"CUDA unavailable ({type(exc).__name__}: {exc}). "
                "Falling back to CPU."
            )

    try:
        import scipy.sparse as sp
    except Exception as exc:
        raise SystemExit(
            "CPU backend requires SciPy. Install with: pip install scipy"
        ) from exc

    print("Backend: CPU (NumPy/SciPy)")
    return {"kind": "cpu", "sp": sp}


def planted_3sat(n_vars: int, n_clauses: int, seed: int):
    """Create a random planted satisfiable 3-SAT instance."""
    if n_vars < 1:
        raise ValueError("n_vars must be >= 1")
    if n_clauses < 1:
        raise ValueError("n_clauses must be >= 1")

    rng = np.random.default_rng(seed)
    planted = rng.integers(
        0, 2, size=n_vars, dtype=np.int8
    ).astype(bool)

    vars_ = rng.integers(
        0, n_vars, size=(n_clauses, 3), dtype=np.int32
    )
    signs = rng.integers(
        0, 2, size=(n_clauses, 3), dtype=np.int8
    ).astype(bool)

    lit_truth = np.where(signs, planted[vars_], ~planted[vars_])
    bad = ~lit_truth.any(axis=1)

    if np.any(bad):
        v0 = vars_[bad, 0]
        # sign=True  -> x
        # sign=False -> NOT x
        # Therefore sign = planted[value] makes literal 0 true.
        signs[bad, 0] = planted[v0]

    # Internal sanity check: the planted assignment must satisfy all clauses.
    check = np.where(signs, planted[vars_], ~planted[vars_]).any(axis=1)
    if not bool(np.all(check)):
        raise RuntimeError("Internal planted-3SAT generator error.")

    return vars_, signs, planted


def compile_operator_cuda(backend, vars_np, signs_np, n_vars, alpha):
    cp = backend["cp"]
    csp = backend["csp"]

    m = int(vars_np.shape[0])
    n_nodes = int(n_vars + m)

    v = cp.asarray(vars_np.reshape(-1), dtype=cp.int32)
    s = cp.asarray(signs_np.reshape(-1), dtype=cp.bool_)
    c = cp.repeat(cp.arange(m, dtype=cp.int32) + np.int32(n_vars), 3)

    rows = cp.concatenate((v, c))
    cols = cp.concatenate((c, v))
    vals = -cp.ones(rows.size, dtype=cp.float32)

    offdiag = csp.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_nodes, n_nodes),
        dtype=cp.float32,
    ).tocsr()
    offdiag.sum_duplicates()
    offdiag.sort_indices()

    degree = cp.asarray((-offdiag).sum(axis=1)).reshape(-1).astype(cp.float32)
    diag = degree + cp.float32(alpha)

    sign_weight = cp.where(
        s, cp.float32(1.0), cp.float32(-1.0)
    )
    var_rhs = cp.bincount(
        v, weights=sign_weight, minlength=n_vars
    ).astype(cp.float32)

    var_degree = cp.maximum(
        cp.bincount(v, minlength=n_vars).astype(cp.float32),
        cp.float32(1.0),
    )
    var_rhs /= var_degree

    clause_rhs = cp.ones(m, dtype=cp.float32)
    rhs = cp.concatenate((var_rhs, clause_rhs))

    return offdiag, diag, rhs


def compile_operator_cpu(backend, vars_np, signs_np, n_vars, alpha):
    sp = backend["sp"]

    m = int(vars_np.shape[0])
    n_nodes = int(n_vars + m)

    v = vars_np.reshape(-1).astype(np.int32, copy=False)
    s = signs_np.reshape(-1)
    c = np.repeat(np.arange(m, dtype=np.int32) + np.int32(n_vars), 3)

    rows = np.concatenate((v, c))
    cols = np.concatenate((c, v))
    vals = -np.ones(rows.size, dtype=np.float32)

    offdiag = sp.coo_matrix(
        (vals, (rows, cols)),
        shape=(n_nodes, n_nodes),
        dtype=np.float32,
    ).tocsr()
    offdiag.sum_duplicates()
    offdiag.sort_indices()

    degree = np.asarray((-offdiag).sum(axis=1)).reshape(-1).astype(np.float32)
    diag = degree + np.float32(alpha)

    sign_weight = np.where(
        s, np.float32(1.0), np.float32(-1.0)
    )
    var_rhs = np.bincount(
        v, weights=sign_weight, minlength=n_vars
    ).astype(np.float32)

    var_degree = np.maximum(
        np.bincount(v, minlength=n_vars).astype(np.float32),
        np.float32(1.0),
    )
    var_rhs /= var_degree

    clause_rhs = np.ones(m, dtype=np.float32)
    rhs = np.concatenate((var_rhs, clause_rhs))

    return offdiag, diag, rhs


def jacobi_cuda(backend, offdiag, diag, rhs, iterations, omega):
    cp = backend["cp"]
    kernel = backend["kernel"]

    n = int(rhs.size)
    if n > np.iinfo(np.int32).max:
        raise ValueError(
            "CUDA kernel currently uses 32-bit node indices; "
            "n_nodes exceeds INT32_MAX."
        )

    x = cp.zeros(n, dtype=cp.float32)
    x_new = cp.empty_like(x)

    threads = 256
    blocks = (n + threads - 1) // threads

    row_ptr = offdiag.indptr.astype(cp.int32, copy=False)
    col_idx = offdiag.indices.astype(cp.int32, copy=False)
    val = offdiag.data.astype(cp.float32, copy=False)

    # RawKernel C signature expects int32 for `n`.
    n32 = np.int32(n)

    # Warm-up JIT and caches; not included in benchmark time.
    kernel(
        (blocks,), (threads,),
        (n32, row_ptr, col_idx, val, diag, rhs, x, x_new),
    )
    cp.cuda.Stream.null.synchronize()

    start = cp.cuda.Event()
    stop = cp.cuda.Event()

    w = cp.float32(omega)
    one_minus_w = cp.float32(1.0 - omega)

    start.record()
    for _ in range(iterations):
        kernel(
            (blocks,), (threads,),
            (n32, row_ptr, col_idx, val, diag, rhs, x, x_new),
        )
        x_new *= w
        x_new += one_minus_w * x
        x, x_new = x_new, x

    stop.record()
    stop.synchronize()

    elapsed_ms = float(cp.cuda.get_elapsed_time(start, stop))
    return x, elapsed_ms


def jacobi_cpu(offdiag, diag, rhs, iterations, omega):
    x = np.zeros(rhs.size, dtype=np.float32)
    w = np.float32(omega)
    one_minus_w = np.float32(1.0 - omega)

    start = time.perf_counter()
    for _ in range(iterations):
        s = offdiag.dot(x)
        x_new = (rhs - s) / diag
        x_new = w * x_new + one_minus_w * x
        x = x_new.astype(np.float32, copy=False)
    elapsed_ms = (time.perf_counter() - start) * 1000.0

    return x, elapsed_ms


def verify_cuda(backend, x, vars_np, signs_np, n_vars):
    cp = backend["cp"]

    vars_gpu = cp.asarray(vars_np, dtype=cp.int32)
    signs_gpu = cp.asarray(signs_np, dtype=cp.bool_)

    assignment = x[:n_vars] >= cp.float32(0.0)
    a = assignment[vars_gpu]
    lit_truth = cp.where(signs_gpu, a, ~a)
    clause_truth = cp.any(lit_truth, axis=1)

    fraction = float(cp.mean(clause_truth.astype(cp.float64)).get())
    sat_found = bool(cp.all(clause_truth).get())
    return fraction, sat_found


def verify_cpu(x, vars_np, signs_np, n_vars):
    assignment = x[:n_vars] >= np.float32(0.0)
    a = assignment[vars_np]
    lit_truth = np.where(signs_np, a, ~a)
    clause_truth = np.any(lit_truth, axis=1)

    fraction = float(np.mean(clause_truth, dtype=np.float64))
    sat_found = bool(np.all(clause_truth))
    return fraction, sat_found


def one_benchmark(
    backend,
    n_vars,
    ratio,
    iterations,
    alpha,
    omega,
    seed,
):
    n_clauses = int(round(ratio * n_vars))
    vars_np, signs_np, _ = planted_3sat(
        n_vars, n_clauses, seed
    )

    if backend["kind"] == "cuda":
        offdiag, diag, rhs = compile_operator_cuda(
            backend, vars_np, signs_np, n_vars, alpha
        )
        x, elapsed_ms = jacobi_cuda(
            backend, offdiag, diag, rhs, iterations, omega
        )
        clause_fraction, sat_found = verify_cuda(
            backend, x, vars_np, signs_np, n_vars
        )
    else:
        offdiag, diag, rhs = compile_operator_cpu(
            backend, vars_np, signs_np, n_vars, alpha
        )
        x, elapsed_ms = jacobi_cpu(
            offdiag, diag, rhs, iterations, omega
        )
        clause_fraction, sat_found = verify_cpu(
            x, vars_np, signs_np, n_vars
        )

    n_nodes = int(n_vars + n_clauses)
    nnz_offdiag = int(offdiag.nnz)
    edge_visits = int(nnz_offdiag * iterations)
    seconds = elapsed_ms / 1e3

    g_edge_visits_s = (
        edge_visits / max(seconds, 1e-30) / 1e9
    )
    ns_per_edge_iter = (
        elapsed_ms * 1e6 / max(edge_visits, 1)
    )

    bytes_est = (
        4 * (n_nodes + 1)
        + 4 * nnz_offdiag
        + 4 * nnz_offdiag
        + 4 * n_nodes * 4
    )

    return {
        "backend": backend["kind"],
        "n_vars": int(n_vars),
        "n_clauses": int(n_clauses),
        "n_nodes": int(n_nodes),
        "nnz_offdiag": int(nnz_offdiag),
        "iterations": int(iterations),
        "elapsed_ms": float(elapsed_ms),
        "g_edge_visits_s": float(g_edge_visits_s),
        "ns_per_edge_iter": float(ns_per_edge_iter),
        "estimated_mib": float(bytes_est / (1024**2)),
        "clause_fraction": float(clause_fraction),
        "sat_found": int(sat_found),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--backend",
        choices=("auto", "cuda", "cpu"),
        default="auto",
    )
    ap.add_argument(
        "--sizes",
        nargs="+",
        type=int,
        default=[1024, 4096, 16384, 65536, 262144, 1048576],
    )
    ap.add_argument("--ratio", type=float, default=4.2)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--omega", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=26032009)
    ap.add_argument("--csv", type=str, default="omega_gpu_results.csv")
    args = ap.parse_args()

    if args.ratio <= 0:
        ap.error("--ratio must be > 0")
    if args.iters < 1:
        ap.error("--iters must be >= 1")
    if args.alpha <= 0:
        ap.error("--alpha must be > 0")
    if not (0.0 < args.omega <= 1.0):
        ap.error("--omega must satisfy 0 < omega <= 1")
    if any(n < 1 for n in args.sizes):
        ap.error("all --sizes must be >= 1")

    backend = initialize_backend(args.backend)

    print(
        "backend,n_vars,n_clauses,n_nodes,nnz_offdiag,iterations,"
        "elapsed_ms,GedgeVisits/s,ns/edge/iter,MiB,"
        "clause_fraction,sat_found"
    )

    rows = []
    for i, n in enumerate(args.sizes):
        row = one_benchmark(
            backend=backend,
            n_vars=n,
            ratio=args.ratio,
            iterations=args.iters,
            alpha=args.alpha,
            omega=args.omega,
            seed=args.seed + i,
        )
        rows.append(row)

        print(
            f"{row['backend']},{row['n_vars']},{row['n_clauses']},"
            f"{row['n_nodes']},{row['nnz_offdiag']},"
            f"{row['iterations']},{row['elapsed_ms']:.3f},"
            f"{row['g_edge_visits_s']:.3f},"
            f"{row['ns_per_edge_iter']:.6f},"
            f"{row['estimated_mib']:.2f},"
            f"{row['clause_fraction']:.8f},"
            f"{row['sat_found']}"
        )

    with open(args.csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(rows[0].keys())
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved benchmark CSV to: {args.csv}")


if __name__ == "__main__":
    main()
