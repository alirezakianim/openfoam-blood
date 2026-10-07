# solver_bench: direct solvers on the block-coupled step's own matrices

Used by `notebooks/solver_speed_benchmark_colab.ipynb` on the `solver-bench` branch.
Nothing here is part of a release. The notebook checks that the solver and case it
builds hash to the P2′ release digest, and that every file below matches the sha256
it pins, before it spends the session.

| file | role |
|---|---|
| `umfdump.c` | LD_PRELOAD library: records the CSC matrix, right-hand side and solution of the listed `umfpack_di_numeric` calls, and forwards every call unchanged |
| `bench_umf.cpp` | UMFPACK, called the way the solver's kernel calls it |
| `bench_pardiso.cpp` | Intel MKL PARDISO (real unsymmetric, mtype 11) |
| `bench_cudss.cpp` | NVIDIA cuDSS on the GPU (host API only) |
| `bench_common.h` | capture loader, CSC to CSR transpose, residual and solution checks |
| `run_bench.py` | builds the programs, self-tests the capture library, sweeps 1, 2, 4 and all threads, writes `results.json` and `results.md` |
| `plugin_selftest.sh` | checks the capture library on a plugin loaded `RTLD_LAZY\|RTLD_GLOBAL`, the way OpenFOAM loads a solver module |

Pinned: `mkl==2026.1.0`, `mkl-include==2026.1.0`, `nvidia-cudss-cu12==0.8.0.10`
(installed `--no-deps`; Colab's own cuBLAS 12 serves it). cuDSS 0.8 changed
`cudssMatrixCreateCsr` to 14 arguments, so re-check `bench_cudss.cpp` against the
header before unpinning.
