// bench_cudss.cpp -- NVIDIA cuDSS (GPU direct sparse solver) on an umfdump capture.
// Host API only: compiled with g++ (no nvcc) against the pinned nvidia-cudss-cu12 0.8.0.10
// header (cudss.h, CUDSS_VERSION 800) and the CUDA runtime headers.
//
//   bench_cudss <prefix> [reps=3] [refacs=reps] [key=value ...]
//     reorder=<cudssReorderingAlg_t int>  ir=<IR steps>  pivot_type=<int>  pivot_eps=<double>
//     matching=<cudssMatchingAlg_t int>   host_nthreads=<int>  mtlib=<path to libcudss_mtlayer_*.so>
//
// The matrix is the CSR of A (transposed once from the capture's CSC, outside any timing),
// CUDSS_MTYPE_GENERAL, full view, zero-based, int32 offsets/indices, float64 values.
// Phase 1 (reps): a new cudssData per rep -- ANALYSIS, FACTORIZATION, SOLVE.
// Phase 2 (refacs): on the last rep's data -- upload the values again (timed separately,
//   the per-step H2D cost an integration would pay), REFACTORIZATION, SOLVE.
// Every phase is timed on the host around cudssExecute + cudaStreamSynchronize.
// Accuracy is measured on the host, from x copied back, exactly as for UMFPACK/PARDISO.
#include "bench_common.h"
#include <cuda_runtime_api.h>
#include <library_types.h>
#include <cudss.h>

static bk::Json* gJ = nullptr;

[[noreturn]] static void fail(const std::string& what)
{
    std::fprintf(stderr, "bench_cudss: %s\n", what.c_str());
    if (gJ) { gJ->str("error", what); bk::emit(*gJ); }
    std::exit(3);
}

#define CUDA_CK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) fail(std::string(#call) + ": " + cudaGetErrorString(e_)); } while (0)
#define DSS_CK(call) do { cudssStatus_t s_ = (call); if (s_ != CUDSS_STATUS_SUCCESS) fail(std::string(#call) + ": cudssStatus " + std::to_string(int(s_))); } while (0)

// cudssDataGet for an integer output, tolerant of a 4- or 8-byte parameter.
static long long dataInt(cudssHandle_t h, cudssData_t d, cudssDataParam_t p, size_t first, size_t* written)
{
    unsigned char buf[16];
    for (size_t sz : {first, first == 4 ? size_t(8) : size_t(4)})
    {
        std::memset(buf, 0, sizeof buf);
        size_t w = 0;
        if (cudssDataGet(h, d, p, buf, sz, &w) == CUDSS_STATUS_SUCCESS)
        {
            if (written) *written = w;
            if (w == 4) { int32_t v; std::memcpy(&v, buf, 4); return v; }
            int64_t v; std::memcpy(&v, buf, 8); return (long long) v;
        }
    }
    if (written) *written = 0;
    return -1;
}

int main(int argc, char** argv)
{
    if (argc < 2)
    {
        std::fprintf(stderr, "usage: bench_cudss <prefix> [reps=3] [refacs=reps] [reorder=N ir=N pivot_type=N pivot_eps=V matching=N host_nthreads=N mtlib=PATH]\n");
        return 2;
    }
    const std::string pre = argv[1];
    int reps = 3, refacs = -1, pos = 0;
    std::vector<std::pair<std::string, std::string>> kv;
    for (int a = 2; a < argc; ++a)
    {
        const std::string t = argv[a];
        const size_t eq = t.find('=');
        if (eq != std::string::npos) kv.push_back({t.substr(0, eq), t.substr(eq + 1)});
        else if (pos == 0) { reps = std::max(1, std::atoi(t.c_str())); ++pos; }
        else if (pos == 1) { refacs = std::max(0, std::atoi(t.c_str())); ++pos; }
    }
    if (refacs < 0) refacs = reps;

    bk::Json j;
    gJ = &j;
    j.str("solver", "cudss").str("capture", pre).integer("reps", reps).integer("refacs", refacs)
        .integer("cudss_header_version", CUDSS_VERSION);
    bk::Capture c;
    const std::string err = bk::loadCapture(pre, c);
    if (!err.empty()) fail(err);
    const int n = c.n;
    const int64_t nnz = c.Ap[n];
    j.integer("n", n).integer("nnz", nnz).boolean("csc_sorted", c.sorted);

    int rtv = 0, drv = 0, lmaj = -1, lmin = -1, lpat = -1;
    cudaRuntimeGetVersion(&rtv);
    cudaDriverGetVersion(&drv);
    cudssGetProperty(MAJOR_VERSION, &lmaj);
    cudssGetProperty(MINOR_VERSION, &lmin);
    cudssGetProperty(PATCH_LEVEL, &lpat);
    double t0 = bk::now();
    CUDA_CK(cudaSetDevice(0));
    CUDA_CK(cudaFree(nullptr));  // forces context creation, timed apart from the solver
    const double tCtx = bk::now() - t0;
    cudaDeviceProp prop;
    CUDA_CK(cudaGetDeviceProperties(&prop, 0));
    char lver[64];
    std::snprintf(lver, sizeof lver, "%d.%d.%d", lmaj, lmin, lpat);
    j.str("cudss_library_version", lver).integer("cuda_runtime", rtv).integer("cuda_driver", drv)
        .str("gpu", prop.name).str("sm", std::to_string(prop.major) + "." + std::to_string(prop.minor))
        .num("gpu_mem_bytes", double(prop.totalGlobalMem)).num("context_s", tCtx);
    std::printf("%s n %d nnz %lld | GPU %s sm %d.%d %.1f GB | CUDA runtime %d driver %d | cuDSS lib %s header %d\n",
                pre.c_str(), n, (long long) nnz, prop.name, prop.major, prop.minor, prop.totalGlobalMem / 1e9, rtv, drv,
                lver, CUDSS_VERSION);

    cudaStream_t stream;
    CUDA_CK(cudaStreamCreate(&stream));
    cudssHandle_t h;
    t0 = bk::now();
    DSS_CK(cudssCreate(&h));
    j.num("cudss_create_s", bk::now() - t0);
    DSS_CK(cudssSetStream(h, stream));
    cudssConfig_t cfg;
    DSS_CK(cudssConfigCreate(&cfg));
    j.open("config");
    for (auto& p : kv)
    {
        const std::string& k = p.first;
        const std::string& v = p.second;
        j.str(k.c_str(), v);
        if (k == "mtlib") { DSS_CK(cudssSetThreadingLayer(h, v.c_str())); continue; }
        if (k == "pivot_eps") { double d = std::atof(v.c_str()); DSS_CK(cudssConfigSet(cfg, CUDSS_CONFIG_PIVOT_EPSILON, &d, sizeof d)); continue; }
        int iv = std::atoi(v.c_str());
        if (k == "reorder") { cudssReorderingAlg_t e = cudssReorderingAlg_t(iv); DSS_CK(cudssConfigSet(cfg, CUDSS_CONFIG_REORDERING_ALG, &e, sizeof e)); }
        else if (k == "ir") DSS_CK(cudssConfigSet(cfg, CUDSS_CONFIG_IR_N_STEPS, &iv, sizeof iv));
        else if (k == "pivot_type") { cudssPivotType_t e = cudssPivotType_t(iv); DSS_CK(cudssConfigSet(cfg, CUDSS_CONFIG_PIVOT_TYPE, &e, sizeof e)); }
        else if (k == "matching") { cudssMatchingAlg_t e = cudssMatchingAlg_t(iv); DSS_CK(cudssConfigSet(cfg, CUDSS_CONFIG_MATCHING_ALG, &e, sizeof e)); }
        else if (k == "host_nthreads") DSS_CK(cudssConfigSet(cfg, CUDSS_CONFIG_HOST_NTHREADS, &iv, sizeof iv));
        else fail("unknown option " + k);
    }
    j.close();

    int32_t *dRp = nullptr, *dCj = nullptr;
    double *dV = nullptr, *dB = nullptr, *dX = nullptr;
    CUDA_CK(cudaMalloc((void**) &dRp, sizeof(int32_t) * (size_t(n) + 1)));
    CUDA_CK(cudaMalloc((void**) &dCj, sizeof(int32_t) * size_t(nnz)));
    CUDA_CK(cudaMalloc((void**) &dV, sizeof(double) * size_t(nnz)));
    CUDA_CK(cudaMalloc((void**) &dB, sizeof(double) * size_t(n)));
    CUDA_CK(cudaMalloc((void**) &dX, sizeof(double) * size_t(n)));
    CUDA_CK(cudaMemcpy(dRp, c.Rp.data(), sizeof(int32_t) * (size_t(n) + 1), cudaMemcpyHostToDevice));
    CUDA_CK(cudaMemcpy(dCj, c.Rj.data(), sizeof(int32_t) * size_t(nnz), cudaMemcpyHostToDevice));
    CUDA_CK(cudaMemcpy(dV, c.Rx.data(), sizeof(double) * size_t(nnz), cudaMemcpyHostToDevice));
    CUDA_CK(cudaMemcpy(dB, c.b.data(), sizeof(double) * size_t(n), cudaMemcpyHostToDevice));
    CUDA_CK(cudaMemset(dX, 0, sizeof(double) * size_t(n)));

    cudssMatrix_t A, B, X;
    DSS_CK(cudssMatrixCreateCsr(&A, n, n, nnz, dRp, nullptr, dCj, dV, CUDSS_R_32I, CUDSS_R_32I, CUDSS_R_64F,
                                CUDSS_MTYPE_GENERAL, CUDSS_MVIEW_FULL, CUDSS_BASE_ZERO));
    DSS_CK(cudssMatrixCreateDn(&B, n, 1, n, dB, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));
    DSS_CK(cudssMatrixCreateDn(&X, n, 1, n, dX, CUDSS_R_64F, CUDSS_LAYOUT_COL_MAJOR));

    size_t free0 = 0, total = 0, freeNow = 0;
    CUDA_CK(cudaMemGetInfo(&free0, &total));
    double devUsedMax = 0;
    std::vector<double> x(size_t(n), 0.0);
    std::vector<double> tA, tF, tS, tD2H, tH2D, tR, tRS;
    long long luNnz = -1, npiv = -1, npivRef = -1, info = -1, infoRef = -1, irSteps = -1;
    size_t wLu = 0, wPiv = 0, wInfo = 0, wIr = 0;
    bk::Accuracy acc{NAN, NAN, NAN}, accRef{NAN, NAN, NAN};
    cudssData_t keep = nullptr;
    auto timed = [&](int phase, cudssData_t d, std::vector<double>& into, const char* name)
    {
        CUDA_CK(cudaStreamSynchronize(stream));
        const double a = bk::now();
        const cudssStatus_t s = cudssExecute(h, phase, cfg, d, A, X, B);
        const cudaError_t e = cudaStreamSynchronize(stream);
        into.push_back(bk::now() - a);
        if (s != CUDSS_STATUS_SUCCESS) fail(std::string("cudssExecute ") + name + ": cudssStatus " + std::to_string(int(s)));
        if (e != cudaSuccess) fail(std::string("sync after ") + name + ": " + cudaGetErrorString(e));
    };
    auto fetchX = [&](bool time)
    {
        const double a = bk::now();
        CUDA_CK(cudaMemcpy(x.data(), dX, sizeof(double) * size_t(n), cudaMemcpyDeviceToHost));
        if (time) tD2H.push_back(bk::now() - a);
    };
    for (int r = 0; r < reps; ++r)
    {
        cudssData_t d;
        DSS_CK(cudssDataCreate(h, &d));
        timed(CUDSS_PHASE_ANALYSIS, d, tA, "ANALYSIS");
        timed(CUDSS_PHASE_FACTORIZATION, d, tF, "FACTORIZATION");
        CUDA_CK(cudaMemGetInfo(&freeNow, &total));
        devUsedMax = std::max(devUsedMax, double(free0) - double(freeNow));
        luNnz = dataInt(h, d, CUDSS_DATA_LU_NNZ, 8, &wLu);
        npiv = dataInt(h, d, CUDSS_DATA_NPIVOTS, 4, &wPiv);
        info = dataInt(h, d, CUDSS_DATA_INFO, 4, &wInfo);
        CUDA_CK(cudaMemset(dX, 0, sizeof(double) * size_t(n)));
        timed(CUDSS_PHASE_SOLVE, d, tS, "SOLVE");
        irSteps = dataInt(h, d, CUDSS_DATA_IR_N_STEPS, 4, &wIr);
        fetchX(true);
        acc = bk::accuracy(c, x);
        if (r + 1 < reps) DSS_CK(cudssDataDestroy(h, d)); else keep = d;
    }
    for (int r = 0; r < refacs && keep; ++r)
    {
        const double a = bk::now();
        CUDA_CK(cudaMemcpyAsync(dV, c.Rx.data(), sizeof(double) * size_t(nnz), cudaMemcpyHostToDevice, stream));
        CUDA_CK(cudaStreamSynchronize(stream));
        tH2D.push_back(bk::now() - a);
        timed(CUDSS_PHASE_REFACTORIZATION, keep, tR, "REFACTORIZATION");
        npivRef = dataInt(h, keep, CUDSS_DATA_NPIVOTS, 4, nullptr);
        infoRef = dataInt(h, keep, CUDSS_DATA_INFO, 4, nullptr);
        CUDA_CK(cudaMemset(dX, 0, sizeof(double) * size_t(n)));
        timed(CUDSS_PHASE_SOLVE, keep, tRS, "SOLVE(refactor)");
        fetchX(false);
        accRef = bk::accuracy(c, x);
    }
    if (keep) cudssDataDestroy(h, keep);
    cudssMatrixDestroy(A);
    cudssMatrixDestroy(B);
    cudssMatrixDestroy(X);
    cudssConfigDestroy(cfg);
    cudssDestroy(h);
    cudaFree(dRp); cudaFree(dCj); cudaFree(dV); cudaFree(dB); cudaFree(dX);
    cudaStreamDestroy(stream);

    const bk::Stats sA = bk::stats(tA), sF = bk::stats(tF), sS = bk::stats(tS), sR = bk::stats(tR), sRS = bk::stats(tRS),
                    sH = bk::stats(tH2D), sD = bk::stats(tD2H);
    std::printf("RESULT cudss %s n %d | analysis med %.3f [%.3f,%.3f] | factor med %.3f [%.3f,%.3f] | solve med %.3f | "
                "refactor med %.3f [%.3f,%.3f] refac-solve med %.3f | H2D values med %.4f D2H x med %.4f | LU nnz %lld | "
                "npivots %lld (refac %lld) info %lld (refac %lld) IR steps %lld | device used %.3f GB | "
                "relres2 %.3e relresInf %.3e xdiff %.3e | refac relres2 %.3e xdiff %.3e\n",
                pre.c_str(), n, sA.med, sA.min, sA.max, sF.med, sF.min, sF.max, sS.med, sR.med, sR.min, sR.max, sRS.med,
                sH.med, sD.med, luNnz, npiv, npivRef, info, infoRef, irSteps, devUsedMax / 1e9,
                acc.relres2, acc.relresInf, acc.xdiff, accRef.relres2, accRef.xdiff);
    j.stats("analysis_s", sA).stats("factor_s", sF).stats("solve_s", sS).stats("refactor_s", sR).stats("refactor_solve_s", sRS)
        .stats("h2d_values_s", sH).stats("d2h_x_s", sD)
        .num("lu_nnz", double(luNnz)).integer("npivots", npiv).integer("npivots_refactor", npivRef)
        .integer("info", info).integer("info_refactor", infoRef).integer("ir_steps", irSteps)
        .integer("bytes_written_lu_nnz", (long long) wLu).integer("bytes_written_npivots", (long long) wPiv)
        .integer("bytes_written_info", (long long) wInfo).integer("bytes_written_ir_steps", (long long) wIr)
        .num("device_used_bytes", devUsedMax).num("solver_peak_bytes", devUsedMax)
        .num("relres2", acc.relres2).num("relresInf", acc.relresInf).num("xdiff", acc.xdiff)
        .num("refactor_relres2", accRef.relres2).num("refactor_relresInf", accRef.relresInf).num("refactor_xdiff", accRef.xdiff)
        .integer("status", info > 0 ? info : 0).num("maxrss_bytes", bk::maxrssBytes());
    bk::emit(j);
    return 0;
}
