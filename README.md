# OMEGA P=NP GPU

GPU implementation and reproducibility package for the paper:

> **P=NP: Polynomial-Time Digital Simulation of a Structured Physical Relaxation Machine for Circuit-SAT**

This repository contains a CUDA/CuPy implementation of the digital OMEGA relaxation model, a planted 3-SAT benchmark generator, and reproducibility tools for studying the scaling of sparse local relaxation as a computational surrogate for the OMEGA physical mesh.

## Overview

The OMEGA approach represents a SAT instance through a structured sparse relaxation system rather than by enumerating Boolean assignments.

The computational pipeline is:

```text
SAT / Circuit-SAT instance
        |
        v
OMEGA compilation
        |
        v
Sparse local operator
        |
        v
GPU relaxation
        |
        v
Thresholded variable state
        |
        v
Boolean verification
        |
        v
SAT / UNSAT
```

At the algebraic level, the relaxation layer is represented by a sparse system of the form

```text
A V = b
```

or by repeated local updates

```text
V^(t+1) = Relax(A, b, V^t).
```

For bounded-degree local geometry, the number of nonzero couplings is linear in the number of grid nodes:

```text
nnz(A) = O(N).
```

This makes each matrix-free or sparse relaxation iteration an `O(N)`-work operation.

The accompanying paper formalizes the conditions under which a polynomial-size OMEGA compilation with polynomially bounded numerical precision and polynomially bounded relaxation time can be simulated by a deterministic Turing machine in polynomial time.

## Repository contents

```text
omega-pnp-gpu/
├── README.md
├── LICENSE
├── omega_gpu_benchmark.py
├── paper/
│   └── P_equals_NP_Omega.tex
├── benchmarks/
│   └── README.md
├── results/
│   └── .gitkeep
└── examples/
    ├── tiny_3sat.cnf
    └── medium_3sat.cnf
```

The core executable is:

```text
omega_gpu_benchmark.py
```

It generates planted satisfiable 3-SAT instances, compiles them into a sparse clause-variable incidence operator, performs weighted Jacobi relaxation, thresholds the variable-node state, and evaluates all clauses.

The current implementation supports three execution modes:

```text
--backend auto   Try CUDA first and fall back to CPU.
--backend cuda   Require CUDA/CuPy.
--backend cpu    Use NumPy/SciPy.
```

## Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support for the CUDA benchmark
- NumPy
- SciPy for the CPU fallback
- CuPy for the CUDA backend

Install NumPy:

```bash
pip install numpy
```

Install the CuPy build matching your CUDA installation. For CUDA 12.x:

```bash
pip install cupy-cuda12x
```

For other CUDA versions, use the corresponding CuPy package.

## Quick start

Run the default benchmark:

```bash
python omega_gpu_benchmark.py
```

Force the CUDA backend:

```bash
python omega_gpu_benchmark.py --backend cuda
```

Use the CPU reference/fallback backend:

```bash
python omega_gpu_benchmark.py --backend cpu
```

Run a custom sequence of 3-SAT sizes:

```bash
python omega_gpu_benchmark.py \
    --sizes 1024 4096 16384 65536 262144 1048576 \
    --ratio 4.2 \
    --iters 200 \
    --csv omega_gpu_results.csv
```

The benchmark prints one CSV-style row per instance size and saves the complete results to disk.

## Google Colab

Enable a GPU runtime first:

```text
Runtime -> Change runtime type -> GPU
```

Verify that the NVIDIA device is visible:

```bash
!nvidia-smi
```

Then test the CUDA backend:

```python
import cupy as cp

print("CuPy:", cp.__version__)
print("GPU count:", cp.cuda.runtime.getDeviceCount())
print("CUDA runtime:", cp.cuda.runtime.runtimeGetVersion())
print("CUDA driver:", cp.cuda.runtime.driverGetVersion())
```

If this succeeds, run:

```bash
!python omega_gpu_benchmark.py \
    --backend cuda \
    --sizes 1024 4096 16384 65536 262144 1048576 \
    --ratio 4.2 \
    --iters 200 \
    --csv omega_gpu_results.csv
```

If CUDA is unavailable, `--backend auto` can fall back to the CPU implementation.

## Command-line options

```text
--backend  auto, cuda, or cpu
--sizes    Numbers of SAT variables to benchmark
--ratio    Clause-to-variable ratio m/n
--iters    Number of weighted-Jacobi relaxation iterations
--alpha    Positive diagonal shift used in A = alpha I + L_G
--omega    Jacobi damping parameter
--seed     Random seed
--csv      Output CSV filename
```

Example:

```bash
python omega_gpu_benchmark.py \
    --sizes 10000 100000 1000000 \
    --ratio 4.2 \
    --iters 500 \
    --alpha 1.0 \
    --omega 0.8 \
    --seed 26032009 \
    --csv results.csv
```

## 3-SAT benchmark

For `n` Boolean variables, the default benchmark uses approximately

```text
m = 4.2 n
```

clauses.

A planted Boolean assignment is generated first. Random 3-literal clauses are then constructed so that every clause is satisfied by that planted assignment.

The clause-variable incidence graph contains

```text
N = n + m
```

nodes.

Since each 3-SAT clause contains three incidences and the sparse graph is stored symmetrically, the off-diagonal sparse structure contains approximately

```text
E_off = 6m
```

entries.

For the default ratio:

```text
N     ~= 5.2 n
E_off ~= 25.2 n
```

so both graph size and per-iteration sparse work scale linearly with `n`.

## GPU kernel

The main CUDA kernel performs one sparse Jacobi update:

```cpp
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
```

Each GPU thread is responsible for one sparse state node.

## Output metrics

The benchmark reports:

```text
n_vars
n_clauses
n_nodes
nnz_offdiag
iterations
elapsed_ms
g_edge_visits_s
ns_per_edge_iter
estimated_mib
clause_fraction
sat_found
```

The main throughput metric is

```text
edge visits = nnz_offdiag * iterations
```

and

```text
GedgeVisits/s =
    edge visits / elapsed time / 1e9.
```

For a bandwidth-bound sparse implementation, a useful scaling diagnostic is:

```text
elapsed_time / (nnz_offdiag * iterations).
```

Once the GPU is saturated, this normalized quantity should remain approximately stable over increasing problem sizes if the computational kernel is scaling linearly with the sparse representation.

## Experimental CUDA results

A complete CUDA scaling run was executed with:

```bash
python omega_gpu_benchmark.py \
    --backend cuda \
    --sizes 1024 4096 16384 65536 262144 1048576 \
    --ratio 4.2 \
    --iters 200 \
    --csv omega_gpu_results.csv
```

The measured results were:

| SAT variables | Clauses | Sparse nodes | Off-diagonal nnz | 200 iterations | Gedge-visits/s | ns / edge / iter | Estimated memory |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1,024 | 4,301 | 5,325 | 25,780 | 331.496 ms | 0.0156 | 64.2933 | 0.30 MiB |
| 4,096 | 17,203 | 21,299 | 103,198 | 9.495 ms | 2.1738 | 0.4600 | 1.19 MiB |
| 16,384 | 68,813 | 85,197 | 412,840 | 9.376 ms | 8.8066 | 0.1136 | 4.77 MiB |
| 65,536 | 275,251 | 340,787 | 1,651,498 | 11.167 ms | 29.5789 | 0.03381 | 19.10 MiB |
| 262,144 | 1,101,005 | 1,363,149 | 6,605,998 | 35.250 ms | 37.4807 | 0.02668 | 76.40 MiB |
| **1,048,576** | **4,404,019** | **5,452,595** | **26,424,080** | **140.588 ms** | **37.5908** | **0.02660** | **305.60 MiB** |

The largest run therefore processed:

```text
1,048,576 SAT variables
4,404,019 clauses
5,452,595 sparse state nodes
26,424,080 off-diagonal sparse entries
200 relaxation iterations
```

in approximately:

```text
140.588 ms
```

with a measured sparse-relaxation throughput of:

```text
37.5908 billion edge-visits/second
```

### Observed large-instance scaling

Between the two largest benchmark points:

```text
262,144 -> 1,048,576 variables
```

the instance size increases by exactly `4x`, while runtime changes from:

```text
35.250 ms -> 140.588 ms
```

or approximately:

```text
3.988x
```

The corresponding two-point empirical scaling exponent is

```text
alpha = log(140.588 / 35.250) / log(4)
      ~= 0.9979
```

for the fixed-iteration sparse-relaxation benchmark.

Thus, in the saturated large-instance regime measured here,

```text
T_relax(N) ~= Theta(N)
```

empirically for the fixed `K = 200` relaxation workload.

The normalized sparse cost also stabilizes:

```text
262,144 vars:   0.026680 ns / edge / iteration
1,048,576 vars: 0.026602 ns / edge / iteration
```

while measured throughput remains almost unchanged:

```text
37.4807 -> 37.5908 Gedge-visits/s
```

This is the expected signature of a bandwidth-limited sparse kernel whose total work is proportional to the number of stored couplings.

### Benchmark interpretation

These measurements establish the scaling behavior of the **GPU sparse-relaxation engine** used by the digital OMEGA prototype.

The current planted 3-SAT incidence benchmark reports a clause-satisfaction fraction near `0.875` after thresholding, while `sat_found = 0` for the tested runs. Therefore these CUDA measurements should be interpreted as performance results for the sparse relaxation primitive, not yet as an end-to-end experimental SAT solver result.

The exact SAT-to-OMEGA compilation layer is treated separately in the accompanying formal construction.

The raw benchmark data can be stored in:

```text
results/omega_gpu_results.csv
```

## Reproducibility experiment

A standard scaling run is:

```bash
python omega_gpu_benchmark.py \
    --sizes 1024 4096 16384 65536 262144 1048576 \
    --ratio 4.2 \
    --iters 200 \
    --csv omega_gpu_results.csv
```

Recommended plots:

1. `elapsed_ms` versus `n_nodes` on log-log axes.
2. `elapsed_ms / (nnz_offdiag * iterations)` versus `n_nodes`.
3. GPU memory usage versus `n_vars`.
4. `clause_fraction` versus relaxation iterations.
5. `GedgeVisits/s` versus problem size.

The expected sparse-work relation is

```text
W_GPU = Theta(K E),
```

where `K` is the number of relaxation iterations and `E` is the number of sparse couplings.

For bounded-degree OMEGA-style geometry,

```text
E = O(N),
```

giving

```text
W_GPU = O(KN).
```

## Relation to the paper

The paper studies the translation

```text
electromagnetic relaxation
        <->
sparse numerical operator
        <->
deterministic digital simulation
```

and considers two routes.

### 1. Local relaxation simulation

If an OMEGA instance has

```text
N(n) = poly(n)
E(n) = poly(n)
K(n) = poly(n)
B(n) = poly(n)
```

where `B(n)` is the bit precision required for numerical state, then sequential digital simulation has complexity of the form

```text
T(n) =
O(K(n) E(n) poly(B(n))).
```

### 2. Direct equilibrium computation

If the terminal OMEGA state is characterized by a rational linear system

```text
A_F V_F = b_F
```

with polynomial dimension and polynomial coefficient bit-length, the equilibrium can be computed directly with polynomial-time exact linear algebra.

The paper states the resulting `P = NP` conclusion under its explicit OMEGA compilation, correctness, precision, and decision-margin hypotheses.

## Important distinction

The current `omega_gpu_benchmark.py` file is a **reproducibility and scaling prototype for the sparse relaxation engine**.

Its planted 3-SAT incidence operator is an engineering benchmark for the computational primitive. The full proof construction requires the exact SAT-to-OMEGA compiler and decision semantics defined in the accompanying paper.

This separation makes it possible to benchmark the GPU backend independently from the formal compilation layer.

## Planned work

Potential extensions include:

- fully matrix-free 3D OMEGA kernels;
- direct `(x,y,z)` neighbor reconstruction without CSR storage;
- exact rational and modular linear-system backends;
- Conjugate Gradient and multigrid implementations;
- FFT-based solvers for regular grid regions;
- DIMACS CNF input;
- direct Circuit-SAT input;
- CPU reference implementation;
- automated scaling plots;
- comparison with the physical Tiny Tapeout implementation;
- exact small-instance verification against standard SAT solvers.

## Citation

If you use this repository, cite the accompanying manuscript:

```bibtex
@misc{aguilera2026omega,
  author       = {Kaoru Aguilera Katayama},
  title        = {P=NP: Polynomial-Time Digital Simulation of a Structured
                  Physical Relaxation Machine for Circuit-SAT},
  year         = {2026},
  howpublished = {Technical manuscript and accompanying source code}
}
```

Repository citation:

```bibtex
@software{aguilera2026omegagpu,
  author  = {Kaoru Aguilera Katayama},
  title   = {OMEGA P=NP GPU},
  year    = {2026},
  url     = {https://github.com/YOUR-USERNAME/omega-pnp-gpu}
}
```

Replace `YOUR-USERNAME` with the GitHub account hosting the repository.

## Paper

The LaTeX source of the manuscript is located at:

```text
paper/P_equals_NP_Omega.tex
```

Suggested repository URL:

```text
https://github.com/YOUR-USERNAME/omega-pnp-gpu
```

## License

Add the license you want to use in `LICENSE`.

For a research-code repository intended for broad reuse, a common choice is the MIT License.

---

**OMEGA P=NP GPU**  
Digital sparse relaxation, Circuit-SAT, and GPU reproducibility experiments.
