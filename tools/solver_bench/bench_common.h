// bench_common.h -- shared by bench_umf, bench_pardiso, bench_cudss and test_common.
// Loads an umfdump capture (<prefix>.Ap/.Ai/.Ax = CSC of A, .b, and the solver's .x),
// transposes it to CSR once (outside any timing), and measures accuracy the same way
// for every solver:
//   relres2   = ||A x - b||_2   / ||b||_2
//   relresInf = ||A x - b||_inf / ||b||_inf
//   xdiff     = max|x - x_solver| / max|x_solver|
// The residual is a CSR row gather in double, the same arithmetic as the probe
// numerics-analyst-02-lubench.cpp used for its relres (so the numbers compare).
#pragma once
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <sys/resource.h>
#include <sys/stat.h>

namespace bk
{

inline bool exists(const std::string& f)
{
    struct stat st;
    return ::stat(f.c_str(), &st) == 0;
}

template<class T>
inline bool readBin(const std::string& f, std::vector<T>& v, std::string& err)
{
    std::ifstream s(f, std::ios::binary | std::ios::ate);
    if (!s) { err = "missing " + f; return false; }
    const std::streamsize bytes = s.tellg();
    if (bytes < 0 || bytes % std::streamsize(sizeof(T)) != 0)
    {
        err = f + ": size " + std::to_string((long long) bytes) + " is not a multiple of " + std::to_string(sizeof(T));
        return false;
    }
    s.seekg(0);
    v.resize(size_t(bytes) / sizeof(T));
    if (bytes > 0 && !s.read(reinterpret_cast<char*>(v.data()), bytes)) { err = "short read " + f; return false; }
    return true;
}

// Checks a square CSC (or CSR: the test is symmetric in the roles) for the invariants
// every solver here assumes. Returns "" when valid.
inline std::string validateCompressed(int n, const std::vector<int32_t>& p, const std::vector<int32_t>& i, size_t nx)
{
    if (n <= 0) return "n <= 0";
    if (p.size() != size_t(n) + 1) return "pointer array length " + std::to_string(p.size()) + " != n+1";
    if (p[0] != 0) return "p[0] != 0";
    for (int c = 0; c < n; ++c) if (p[c+1] < p[c]) return "pointers decrease at " + std::to_string(c);
    if (size_t(p[n]) != i.size()) return "p[n] " + std::to_string(p[n]) + " != index count " + std::to_string(i.size());
    if (nx != i.size()) return "value count " + std::to_string(nx) + " != index count " + std::to_string(i.size());
    for (size_t k = 0; k < i.size(); ++k) if (i[k] < 0 || i[k] >= n) return "index out of range at " + std::to_string(k);
    return "";
}

inline bool sortedNoDup(int n, const std::vector<int32_t>& p, const std::vector<int32_t>& i)
{
    for (int c = 0; c < n; ++c) for (int k = p[c] + 1; k < p[c+1]; ++k) if (i[k-1] >= i[k]) return false;
    return true;
}

// Counting transpose: the CSC of A (column pointers Ap, row indices Ai) becomes the CSR
// of A (row pointers Rp, column indices Rj). Columns are visited in increasing order, so
// the column indices of every row come out sorted; duplicates, if any, are kept.
inline void cscToCsr(int n, const int32_t* Ap, const int32_t* Ai, const double* Ax,
                     std::vector<int32_t>& Rp, std::vector<int32_t>& Rj, std::vector<double>& Rx)
{
    const int64_t nnz = Ap[n];
    Rp.assign(size_t(n) + 1, 0);
    Rj.assign(size_t(nnz), 0);
    Rx.assign(size_t(nnz), 0.0);
    for (int64_t k = 0; k < nnz; ++k) ++Rp[size_t(Ai[k]) + 1];
    for (int r = 0; r < n; ++r) Rp[r+1] += Rp[r];
    std::vector<int32_t> next(Rp.begin(), Rp.end() - 1);
    for (int c = 0; c < n; ++c)
        for (int32_t k = Ap[c]; k < Ap[c+1]; ++k)
        {
            const int32_t dst = next[size_t(Ai[k])]++;
            Rj[size_t(dst)] = c;
            Rx[size_t(dst)] = Ax[k];
        }
}

struct Residual { double relres2, relresInf; };

// r = A x - b by CSR row gather (the reference used for every solver).
inline Residual residualCsr(int n, const int32_t* Rp, const int32_t* Rj, const double* Rx,
                            const double* x, const double* b)
{
    double r2 = 0, b2 = 0, rinf = 0, binf = 0;
    for (int k = 0; k < n; ++k)
    {
        double r = -b[k];
        for (int32_t j = Rp[k]; j < Rp[k+1]; ++j) r += Rx[j] * x[Rj[j]];
        r2 += r * r;
        b2 += b[k] * b[k];
        rinf = std::max(rinf, std::fabs(r));
        binf = std::max(binf, std::fabs(b[k]));
        if (std::isnan(r)) rinf = NAN;
    }
    return { std::sqrt(r2 / b2), rinf / binf };
}

// r = A x - b by CSC column scatter: an independent path, used only to cross-check.
inline Residual residualCsc(int n, const int32_t* Ap, const int32_t* Ai, const double* Ax,
                            const double* x, const double* b)
{
    std::vector<double> r(static_cast<size_t>(n));
    for (int k = 0; k < n; ++k) r[size_t(k)] = -b[k];
    for (int c = 0; c < n; ++c) for (int32_t j = Ap[c]; j < Ap[c+1]; ++j) r[size_t(Ai[j])] += Ax[j] * x[c];
    double r2 = 0, b2 = 0, rinf = 0, binf = 0;
    for (int k = 0; k < n; ++k)
    {
        r2 += r[size_t(k)] * r[size_t(k)];
        b2 += b[k] * b[k];
        rinf = std::max(rinf, std::fabs(r[size_t(k)]));
        binf = std::max(binf, std::fabs(b[k]));
        if (std::isnan(r[size_t(k)])) rinf = NAN;
    }
    return { std::sqrt(r2 / b2), rinf / binf };
}

// max|x - xs| / max|xs|; NaN anywhere in x gives NaN.
inline double xdiff(int n, const double* x, const double* xs)
{
    double d = 0, m = 0;
    for (int k = 0; k < n; ++k)
    {
        const double e = std::fabs(x[k] - xs[k]);
        if (std::isnan(e)) return NAN;
        d = std::max(d, e);
        m = std::max(m, std::fabs(xs[k]));
    }
    return m > 0 ? d / m : (d > 0 ? INFINITY : 0.0);
}

struct Capture
{
    std::string prefix;
    int n = 0;
    std::vector<int32_t> Ap, Ai, Rp, Rj;
    std::vector<double> Ax, Rx, b, xs, ctl;
    bool hasXs = false, hasCtl = false, sorted = false;
};

// Loads <prefix>.Ap/.Ai/.Ax/.b (required) and .x/.ctl (optional), validates, and builds
// the CSR. Returns "" on success.
inline std::string loadCapture(const std::string& prefix, Capture& c)
{
    std::string err;
    c.prefix = prefix;
    if (!readBin(prefix + ".Ap", c.Ap, err) || !readBin(prefix + ".Ai", c.Ai, err) ||
        !readBin(prefix + ".Ax", c.Ax, err) || !readBin(prefix + ".b", c.b, err)) return err;
    if (c.Ap.size() < 2) return "Ap too short";
    c.n = int(c.Ap.size()) - 1;
    err = validateCompressed(c.n, c.Ap, c.Ai, c.Ax.size());
    if (!err.empty()) return "CSC: " + err;
    if (c.b.size() != size_t(c.n)) return "b length " + std::to_string(c.b.size()) + " != n";
    if (exists(prefix + ".x"))
    {
        if (!readBin(prefix + ".x", c.xs, err)) return err;
        if (c.xs.size() != size_t(c.n)) return "x length != n";
        c.hasXs = true;
    }
    if (exists(prefix + ".ctl") && readBin(prefix + ".ctl", c.ctl, err)) c.hasCtl = true;
    c.sorted = sortedNoDup(c.n, c.Ap, c.Ai);
    cscToCsr(c.n, c.Ap.data(), c.Ai.data(), c.Ax.data(), c.Rp, c.Rj, c.Rx);
    return "";
}

struct Accuracy { double relres2, relresInf, xdiff; };

inline Accuracy accuracy(const Capture& c, const std::vector<double>& x)
{
    const Residual r = residualCsr(c.n, c.Rp.data(), c.Rj.data(), c.Rx.data(), x.data(), c.b.data());
    return { r.relres2, r.relresInf, c.hasXs ? xdiff(c.n, x.data(), c.xs.data()) : NAN };
}

inline double now()
{
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct Stats { double med = NAN, min = NAN, max = NAN; int n = 0; };

inline Stats stats(std::vector<double> v)
{
    Stats s;
    s.n = int(v.size());
    if (v.empty()) return s;
    std::sort(v.begin(), v.end());
    const size_t m = v.size();
    s.med = (m % 2) ? v[m/2] : 0.5 * (v[m/2 - 1] + v[m/2]);
    s.min = v.front();
    s.max = v.back();
    return s;
}

inline double maxrssBytes()
{
    rusage r;
    getrusage(RUSAGE_SELF, &r);
#ifdef __APPLE__
    return double(r.ru_maxrss);
#else
    return double(r.ru_maxrss) * 1024.0;
#endif
}

inline const char* env(const char* k) { const char* v = std::getenv(k); return v ? v : ""; }

// Minimal JSON object builder (one line, non-finite numbers as null).
class Json
{
    std::string s_;
    std::vector<bool> first_;
    void sep() { if (!first_.back()) s_ += ','; first_.back() = false; }
    void key(const char* k) { sep(); s_ += '"'; s_ += k; s_ += "\":"; }
    void numRaw(double v)
    {
        if (!std::isfinite(v)) { s_ += "null"; return; }
        char b[40];
        std::snprintf(b, sizeof b, "%.10g", v);
        s_ += b;
    }
    void strRaw(const std::string& v)
    {
        s_ += '"';
        for (unsigned char ch : v)
        {
            if (ch == '"' || ch == '\\') { s_ += '\\'; s_ += char(ch); }
            else if (ch < 0x20) { char b[8]; std::snprintf(b, sizeof b, "\\u%04x", ch); s_ += b; }
            else s_ += char(ch);
        }
        s_ += '"';
    }
public:
    Json() : s_("{"), first_{true} {}
    Json& num(const char* k, double v) { key(k); numRaw(v); return *this; }
    Json& integer(const char* k, long long v) { key(k); s_ += std::to_string(v); return *this; }
    Json& str(const char* k, const std::string& v) { key(k); strRaw(v); return *this; }
    Json& boolean(const char* k, bool v) { key(k); s_ += v ? "true" : "false"; return *this; }
    Json& null(const char* k) { key(k); s_ += "null"; return *this; }
    Json& stats(const char* k, const Stats& st)
    {
        key(k);
        s_ += "{\"med\":"; numRaw(st.med); s_ += ",\"min\":"; numRaw(st.min);
        s_ += ",\"max\":"; numRaw(st.max); s_ += ",\"n\":" + std::to_string(st.n) + "}";
        return *this;
    }
    Json& arr(const char* k, const std::vector<double>& v)
    {
        key(k);
        s_ += '[';
        for (size_t i = 0; i < v.size(); ++i) { if (i) s_ += ','; numRaw(v[i]); }
        s_ += ']';
        return *this;
    }
    Json& open(const char* k) { key(k); s_ += '{'; first_.push_back(true); return *this; }
    Json& close() { s_ += '}'; first_.pop_back(); return *this; }
    std::string text() const { return s_ + "}"; }
};

// Every benchmark ends with exactly one line "RESULT_JSON {...}" (run_bench.py parses it).
inline void emit(const Json& j)
{
    std::printf("RESULT_JSON %s\n", j.text().c_str());
    std::fflush(stdout);
}

} // namespace bk
