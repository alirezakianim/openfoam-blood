// bench_umf.cpp -- UMFPACK exactly as the solver's kernel calls it (blockCoupledKernel.H
// L904-1029, UmfpackLU), on an umfdump capture.
//
//   bench_umf <prefix> [reps=3] [refacs=reps]
//
// Kernel call sequence reproduced here: umfpack_di_defaults(control); control[PRL] = 0;
// umfpack_di_symbolic(n, n, Ap, Ai, Ax, &Symbolic, control, info) on the CSC of A;
// umfpack_di_numeric(Ap, Ai, Ax, Symbolic, &Numeric, control, info);
// umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x, b, Numeric, control, info) with x zeroed;
// one Info array shared by all three calls, the old Numeric freed before the next one.
//
// Phase 1 (reps): fresh symbolic + numeric + solve per rep.
// Phase 2 (refacs): numeric-only refactorisations (+ solve) on the first rep's Symbolic --
//   the solver's steady state, which re-runs symbolic only when the pattern changes.
// Threads: UMFPACK's parallelism is the BLAS's; set OPENBLAS_NUM_THREADS (and
//   OMP_NUM_THREADS) in the environment. The BLAS library actually bound to dgemm_
//   is reported (dladdr), with OpenBLAS's own thread count and config.
// If <prefix>.ctl exists (umfdump writes the Control array the solver passed), it is
//   compared entry by entry with this program's control.
#include "bench_common.h"
#include <umfpack.h>
#include <dlfcn.h>

static std::string blasReport(bk::Json& j)
{
    typedef int (*gi)();
    typedef char* (*gc)();
    typedef const char* (*gcc)();
    Dl_info di;
    void* f = dlsym(RTLD_DEFAULT, "dgemm_");
    const std::string lib = (f && dladdr(f, &di) && di.dli_fname) ? di.dli_fname : "(dgemm_ not found)";
    gi g = (gi) dlsym(RTLD_DEFAULT, "openblas_get_num_threads");
    gc c = (gc) dlsym(RTLD_DEFAULT, "openblas_get_config");
    gc cn = (gc) dlsym(RTLD_DEFAULT, "openblas_get_corename");
    gi mk = (gi) dlsym(RTLD_DEFAULT, "MKL_Get_Max_Threads");
    gcc ssb = (gcc) dlsym(RTLD_DEFAULT, "SuiteSparse_BLAS_library");
    j.open("blas")
        .str("dgemm_from", lib)
        .integer("openblas_threads", g ? g() : -1)
        .str("openblas_config", c ? c() : "")
        .str("openblas_core", cn ? cn() : "")
        .integer("mkl_max_threads", mk ? mk() : -1)
        .str("suitesparse_blas_library", ssb ? ssb() : "")
        .str("OPENBLAS_NUM_THREADS", bk::env("OPENBLAS_NUM_THREADS"))
        .str("OMP_NUM_THREADS", bk::env("OMP_NUM_THREADS"))
        .close();
    char line[1024];
    std::snprintf(line, sizeof line, "BLAS: dgemm_ from %s | openblas threads %d | config %s | core %s | OPENBLAS_NUM_THREADS=%s OMP_NUM_THREADS=%s",
                  lib.c_str(), g ? g() : -1, c ? c() : "-", cn ? cn() : "-",
                  bk::env("OPENBLAS_NUM_THREADS"), bk::env("OMP_NUM_THREADS"));
    return line;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: bench_umf <prefix> [reps=3] [refacs=reps]\n");
        return 2;
    }
    const std::string pre = argv[1];
    const int reps = argc > 2 ? std::max(1, std::atoi(argv[2])) : 3;
    const int refacs = argc > 3 ? std::max(0, std::atoi(argv[3])) : reps;

    bk::Json j;
    j.str("solver", "umfpack").str("capture", pre).integer("reps", reps).integer("refacs", refacs);
    bk::Capture c;
    const std::string err = bk::loadCapture(pre, c);
    if (!err.empty())
    {
        std::fprintf(stderr, "bench_umf: %s\n", err.c_str());
        j.str("error", err);
        bk::emit(j);
        return 2;
    }
    const int n = c.n;
    j.integer("n", n).integer("nnz", c.Ap[n]).boolean("csc_sorted", c.sorted).boolean("has_solver_x", c.hasXs);

    char ver[64];
    std::snprintf(ver, sizeof ver, "%d.%d.%d", UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION, UMFPACK_SUBSUB_VERSION);
    typedef void (*vfn)(int*);   // UMFPACK >= 6.2: void umfpack_version(int version[3])
    vfn uv = (vfn) dlsym(RTLD_DEFAULT, "umfpack_version");
    int rv[3] = { -1, -1, -1 };
    if (uv) uv(rv);
    char rver[64];
    std::snprintf(rver, sizeof rver, "%d.%d.%d", rv[0], rv[1], rv[2]);
    j.str("umfpack_header", ver).str("umfpack_runtime", uv ? rver : "(umfpack_version absent)");
    const std::string blasLine = blasReport(j);
    std::printf("%s n %d nnz %d sorted %d | UMFPACK header %s runtime %s\n%s\n", pre.c_str(), n, c.Ap[n],
                int(c.sorted), ver, uv ? rver : "?", blasLine.c_str());

    double control[UMFPACK_CONTROL], info[UMFPACK_INFO];
    umfpack_di_defaults(control);
    control[UMFPACK_PRL] = 0;
    if (c.hasCtl)
    {
        int mism = 0;
        const size_t m = std::min(c.ctl.size(), size_t(UMFPACK_CONTROL));
        for (size_t k = 0; k < m; ++k)
        {
            const bool same = (c.ctl[k] == control[k]) || (std::isnan(c.ctl[k]) && std::isnan(control[k]));
            if (!same) { ++mism; std::printf("CONTROL MISMATCH [%zu] capture %.17g here %.17g\n", k, c.ctl[k], control[k]); }
        }
        if (c.ctl.size() != size_t(UMFPACK_CONTROL)) ++mism;
        j.integer("control_mismatches_vs_capture", mism);
    }
    else j.null("control_mismatches_vs_capture");
    for (int k = 0; k < UMFPACK_INFO; ++k) info[k] = -1;

    std::vector<double> tSym, tNum, tSol, tRef, tRefSol;
    double lnz = NAN, unz = NAN, rcond = NAN, flops = NAN, peak = NAN, ordUsed = NAN, stratUsed = NAN, irTaken = NAN;
    int worst = 0;
    auto note = [&](int st) { if (st != UMFPACK_OK && worst == 0) worst = st; };
    bk::Accuracy acc{NAN, NAN, NAN}, accRef{NAN, NAN, NAN};
    std::vector<double> x(size_t(n), 0.0);
    void* keep = nullptr;
    for (int r = 0; r < reps; ++r)
    {
        void* sym = nullptr;
        void* num = nullptr;
        double a = bk::now();
        int st = umfpack_di_symbolic(n, n, c.Ap.data(), c.Ai.data(), c.Ax.data(), &sym, control, info);
        tSym.push_back(bk::now() - a);
        note(st);
        if (st != UMFPACK_OK && st != UMFPACK_WARNING_singular_matrix) break;
        ordUsed = info[UMFPACK_ORDERING_USED];
        stratUsed = info[UMFPACK_STRATEGY_USED];
        a = bk::now();
        st = umfpack_di_numeric(c.Ap.data(), c.Ai.data(), c.Ax.data(), sym, &num, control, info);
        tNum.push_back(bk::now() - a);
        note(st);
        lnz = info[UMFPACK_LNZ];
        unz = info[UMFPACK_UNZ];
        rcond = info[UMFPACK_RCOND];
        flops = info[UMFPACK_FLOPS];
        peak = info[UMFPACK_PEAK_MEMORY] * info[UMFPACK_SIZE_OF_UNIT];
        if (num)
        {
            x.assign(size_t(n), 0.0);
            a = bk::now();
            st = umfpack_di_solve(UMFPACK_A, c.Ap.data(), c.Ai.data(), c.Ax.data(), x.data(), c.b.data(), num, control, info);
            tSol.push_back(bk::now() - a);
            note(st);
            irTaken = info[UMFPACK_IR_TAKEN];
            acc = bk::accuracy(c, x);
            umfpack_di_free_numeric(&num);
        }
        if (r == 0) keep = sym; else umfpack_di_free_symbolic(&sym);
    }
    for (int r = 0; r < refacs && keep; ++r)
    {
        void* num = nullptr;
        double a = bk::now();
        int st = umfpack_di_numeric(c.Ap.data(), c.Ai.data(), c.Ax.data(), keep, &num, control, info);
        tRef.push_back(bk::now() - a);
        note(st);
        if (!num) break;
        x.assign(size_t(n), 0.0);
        a = bk::now();
        st = umfpack_di_solve(UMFPACK_A, c.Ap.data(), c.Ai.data(), c.Ax.data(), x.data(), c.b.data(), num, control, info);
        tRefSol.push_back(bk::now() - a);
        note(st);
        accRef = bk::accuracy(c, x);
        umfpack_di_free_numeric(&num);
    }
    if (keep) umfpack_di_free_symbolic(&keep);

    const bk::Stats sS = bk::stats(tSym), sN = bk::stats(tNum), sV = bk::stats(tSol), sR = bk::stats(tRef), sRV = bk::stats(tRefSol);
    std::printf("RESULT umf %s n %d | sym med %.3f [%.3f,%.3f] | num med %.3f [%.3f,%.3f] | solve med %.3f [%.3f,%.3f] | "
                "refac med %.3f [%.3f,%.3f] refac-solve med %.3f | LU nnz %.4g (L %.4g U %.4g) | rcond %.3g | flops %.4g | "
                "umfPeak %.3f GB | ordering %g strategy %g IR %g | relres2 %.3e relresInf %.3e xdiff %.3e | refac relres2 %.3e xdiff %.3e | status %d | maxrss %.3f GB\n",
                pre.c_str(), n, sS.med, sS.min, sS.max, sN.med, sN.min, sN.max, sV.med, sV.min, sV.max,
                sR.med, sR.min, sR.max, sRV.med, lnz + unz, lnz, unz, rcond, flops, peak / 1e9, ordUsed, stratUsed, irTaken,
                acc.relres2, acc.relresInf, acc.xdiff, accRef.relres2, accRef.xdiff, worst, bk::maxrssBytes() / 1e9);
    j.stats("analysis_s", sS).stats("factor_s", sN).stats("solve_s", sV).stats("refactor_s", sR).stats("refactor_solve_s", sRV)
        .num("lu_nnz", lnz + unz).num("lnz", lnz).num("unz", unz).num("rcond", rcond).num("flops", flops)
        .num("solver_peak_bytes", peak).num("ordering_used", ordUsed).num("strategy_used", stratUsed).num("ir_taken", irTaken)
        .num("relres2", acc.relres2).num("relresInf", acc.relresInf).num("xdiff", acc.xdiff)
        .num("refactor_relres2", accRef.relres2).num("refactor_relresInf", accRef.relresInf).num("refactor_xdiff", accRef.xdiff)
        .integer("status", worst).num("maxrss_bytes", bk::maxrssBytes());
    bk::emit(j);
    return worst < 0 ? 1 : 0;
}
