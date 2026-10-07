// bench_pardiso.cpp -- Intel oneMKL PARDISO on an umfdump capture.
//
//   bench_pardiso <prefix> <nthreads> [reps=3] [refacs=reps] [iparmK=V ...] [msglvl=1]
//
// mtype 11 (real unsymmetric), zero-based indexing (iparm[34] = 1), LP64 (MKL_INT = int),
// on the CSR of A, transposed once from the capture's CSC, outside any timing. The
// defaults are pardisoinit's for mtype 11 (METIS ordering iparm[1] = 2, scaling and
// matching iparm[10] = iparm[12] = 1, pivot perturbation 1e-13, iterative refinement
// automatic), plus iparm[17] = -1 to report the factor nnz. Any iparmK=V token
// overrides iparm[K] (0-based), e.g. iparm1=3 (parallel METIS), iparm23=10.
//
// Phase 1 (reps): a fresh analysis per rep -- phase 11, 22, 33, then phase -1.
// Phase 2 (refacs): phase 22 (+ 33) only, on the last rep's analysis -- the solver's steady
//   state, where the symbolic analysis is redone only when the pattern changes.
// Threads: mkl_set_dynamic(0) + mkl_set_num_threads(nthreads); run_bench.py also sets
//   MKL_NUM_THREADS / OMP_NUM_THREADS.
// Reproducibility: every solution after the first (later fresh reps, every refactor) is
//   compared BITWISE with the first rep's x: repro_fresh_identical / repro_refac_identical
//   count the identical ones, repro_maxdiff = max|x - x0| / max|x0| over all of them.
//   PARDISO's CNR mode is iparm33=<threads> with MKL_CBWR set in the environment.
#include "bench_common.h"
#include <mkl_pardiso.h>
#include <mkl_service.h>
#include <mkl_types.h>

static_assert(sizeof(MKL_INT) == sizeof(int32_t), "bench_pardiso is LP64: MKL_INT must be 32-bit");

int main(int argc, char** argv)
{
    if (argc < 3)
    {
        std::fprintf(stderr, "usage: bench_pardiso <prefix> <nthreads> [reps=3] [refacs=reps] [iparmK=V ...] [msglvl=1]\n");
        return 2;
    }
    const std::string pre = argv[1];
    const int nthr = std::max(1, std::atoi(argv[2]));
    int reps = 3, refacs = -1, pos = 0;
    MKL_INT msglvl = 0;
    std::vector<std::pair<int, int>> overrides;
    for (int a = 3; a < argc; ++a)
    {
        const std::string t = argv[a];
        int k = -1, v = 0;
        if (std::sscanf(t.c_str(), "iparm%d=%d", &k, &v) == 2 && k >= 0 && k < 64) overrides.push_back({k, v});
        else if (std::sscanf(t.c_str(), "msglvl=%d", &v) == 1) msglvl = v;
        else if (pos == 0) { reps = std::max(1, std::atoi(t.c_str())); ++pos; }
        else if (pos == 1) { refacs = std::max(0, std::atoi(t.c_str())); ++pos; }
    }
    if (refacs < 0) refacs = reps;

    bk::Json j;
    j.str("solver", "pardiso").str("capture", pre).integer("threads", nthr).integer("reps", reps).integer("refacs", refacs);
    bk::Capture c;
    const std::string err = bk::loadCapture(pre, c);
    if (!err.empty())
    {
        std::fprintf(stderr, "bench_pardiso: %s\n", err.c_str());
        j.str("error", err);
        bk::emit(j);
        return 2;
    }
    const MKL_INT n = c.n;
    j.integer("n", n).integer("nnz", c.Ap[c.n]).boolean("csc_sorted", c.sorted);
    if (!c.sorted) std::printf("WARNING: capture has unsorted or duplicate row indices; PARDISO needs strictly increasing CSR columns\n");

    mkl_set_dynamic(0);
    mkl_set_num_threads(nthr);
    char ver[256] = {0};
    mkl_get_version_string(ver, int(sizeof ver));
    mkl_peak_mem_usage(MKL_PEAK_MEM_ENABLE);
    j.str("mkl_version", ver).integer("mkl_max_threads", mkl_get_max_threads())
        .str("MKL_NUM_THREADS", bk::env("MKL_NUM_THREADS")).str("MKL_THREADING_LAYER", bk::env("MKL_THREADING_LAYER"));
    std::printf("%s n %d nnz %d | %s | mkl_get_max_threads %d (requested %d)\n", pre.c_str(), int(n), c.Ap[c.n], ver,
                mkl_get_max_threads(), nthr);

    void* pt[64];
    MKL_INT iparm[64], iparmIn[64];
    const MKL_INT mtype = 11, maxfct = 1, mnum = 1, nrhs = 1;
    std::memset(pt, 0, sizeof pt);
    for (int k = 0; k < 64; ++k) iparm[k] = 0;
    pardisoinit(pt, &mtype, iparm);
    iparm[34] = 1;   // zero-based ia/ja
    iparm[17] = -1;  // report nnz in the factors
    for (auto& o : overrides) iparm[o.first] = o.second;
    std::memcpy(iparmIn, iparm, sizeof iparm);

    const MKL_INT* ia = reinterpret_cast<const MKL_INT*>(c.Rp.data());
    const MKL_INT* ja = reinterpret_cast<const MKL_INT*>(c.Rj.data());
    std::vector<double> bb(c.b), x(size_t(n), 0.0);
    auto phase = [&](MKL_INT ph) -> MKL_INT
    {
        MKL_INT error = 0;
        pardiso(pt, &maxfct, &mnum, &mtype, &ph, &n, c.Rx.data(), ia, ja, nullptr, &nrhs, iparm, &msglvl,
                bb.data(), x.data(), &error);
        return error;
    };

    std::vector<double> tA, tF, tS, tR, tRS;
    long long factorNnz = -1, pert = -1, pertRef = -1, ir = -1, irRef = -1;
    long long memPeakSym = -1, memPerm = -1, memNum = -1;
    MKL_INT worst = 0;
    bool live = false;
    bk::Accuracy acc{NAN, NAN, NAN}, accRef{NAN, NAN, NAN};
    std::vector<double> x0;
    int reproFreshSame = 0, reproFreshN = 0, reproRefacSame = 0, reproRefacN = 0;
    double reproMaxDiff = 0.0;
    auto reproCheck = [&](int& same, int& count)
    {
        if (x0.empty()) { x0 = x; return; }
        ++count;
        if (std::memcmp(x.data(), x0.data(), x.size()*sizeof(double)) == 0) { ++same; return; }
        double dmax = 0.0, xmax = 0.0;
        for (std::size_t i = 0; i < x.size(); ++i)
        {
            dmax = std::max(dmax, std::fabs(x[i] - x0[i]));
            xmax = std::max(xmax, std::fabs(x0[i]));
        }
        reproMaxDiff = std::max(reproMaxDiff, xmax > 0 ? dmax/xmax : dmax);
    };
    for (int r = 0; r < reps; ++r)
    {
        if (live) { phase(-1); std::memset(pt, 0, sizeof pt); live = false; }
        std::memcpy(iparm, iparmIn, sizeof iparm);
        double a = bk::now();
        MKL_INT e = phase(11);
        tA.push_back(bk::now() - a);
        live = true;
        if (e) { worst = e; break; }
        factorNnz = iparm[17]; memPeakSym = iparm[14]; memPerm = iparm[15];
        a = bk::now();
        e = phase(22);
        tF.push_back(bk::now() - a);
        if (e) { worst = e; break; }
        pert = iparm[13]; memNum = iparm[16];
        std::fill(x.begin(), x.end(), 0.0);
        a = bk::now();
        e = phase(33);
        tS.push_back(bk::now() - a);
        if (e) { worst = e; break; }
        ir = iparm[6];
        acc = bk::accuracy(c, x);
        reproCheck(reproFreshSame, reproFreshN);
    }
    for (int r = 0; r < refacs && live && worst == 0; ++r)
    {
        double a = bk::now();
        MKL_INT e = phase(22);
        tR.push_back(bk::now() - a);
        if (e) { worst = e; break; }
        pertRef = iparm[13];
        std::fill(x.begin(), x.end(), 0.0);
        a = bk::now();
        e = phase(33);
        tRS.push_back(bk::now() - a);
        if (e) { worst = e; break; }
        irRef = iparm[6];
        accRef = bk::accuracy(c, x);
        reproCheck(reproRefacSame, reproRefacN);
    }
    std::vector<double> iparmOut(iparm, iparm + 64), iparmInD(iparmIn, iparmIn + 64);
    if (live) phase(-1);
    const long long mklPeak = mkl_peak_mem_usage(MKL_PEAK_MEM);

    const bk::Stats sA = bk::stats(tA), sF = bk::stats(tF), sS = bk::stats(tS), sR = bk::stats(tR), sRS = bk::stats(tRS);
    const long long totPeakKB = std::max(memPeakSym, memPerm + memNum);
    std::printf("RESULT pardiso %s n %d thr %d | analysis(11) med %.3f [%.3f,%.3f] | factor(22) med %.3f [%.3f,%.3f] | "
                "solve(33) med %.3f | refactor(22) med %.3f [%.3f,%.3f] refac-solve med %.3f | factor nnz %lld | "
                "perturbed pivots %lld (refac %lld) | IR steps %lld (refac %lld) | peak %.3f GB (iparm) mkl %.3f GB | "
                "relres2 %.3e relresInf %.3e xdiff %.3e | refac relres2 %.3e xdiff %.3e | "
                "repro fresh %d/%d refac %d/%d maxdiff %.3e | error %d | maxrss %.3f GB\n",
                pre.c_str(), int(n), nthr, sA.med, sA.min, sA.max, sF.med, sF.min, sF.max, sS.med, sR.med, sR.min, sR.max,
                sRS.med, factorNnz, pert, pertRef, ir, irRef, totPeakKB * 1024.0 / 1e9, mklPeak / 1e9,
                acc.relres2, acc.relresInf, acc.xdiff, accRef.relres2, accRef.xdiff,
                reproFreshSame, reproFreshN, reproRefacSame, reproRefacN, reproMaxDiff, int(worst), bk::maxrssBytes() / 1e9);
    j.stats("analysis_s", sA).stats("factor_s", sF).stats("solve_s", sS).stats("refactor_s", sR).stats("refactor_solve_s", sRS)
        .num("lu_nnz", double(factorNnz)).integer("perturbed_pivots", pert).integer("perturbed_pivots_refactor", pertRef)
        .integer("ir_steps", ir).integer("ir_steps_refactor", irRef)
        .integer("mem_peak_symbolic_kb", memPeakSym).integer("mem_permanent_kb", memPerm).integer("mem_numeric_kb", memNum)
        .num("solver_peak_bytes", totPeakKB * 1024.0).integer("mkl_peak_bytes", mklPeak)
        .num("relres2", acc.relres2).num("relresInf", acc.relresInf).num("xdiff", acc.xdiff)
        .num("refactor_relres2", accRef.relres2).num("refactor_relresInf", accRef.relresInf).num("refactor_xdiff", accRef.xdiff)
        .integer("repro_fresh_identical", reproFreshSame).integer("repro_fresh_compared", reproFreshN)
        .integer("repro_refac_identical", reproRefacSame).integer("repro_refac_compared", reproRefacN)
        .num("repro_maxdiff", reproMaxDiff).str("MKL_CBWR", bk::env("MKL_CBWR"))
        .integer("status", worst).num("maxrss_bytes", bk::maxrssBytes())
        .arr("iparm_in", iparmInD).arr("iparm_out", iparmOut);
    bk::emit(j);
    return worst != 0 ? 1 : 0;
}
