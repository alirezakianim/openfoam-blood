/*
 * umfdump.c -- LD_PRELOAD shim that captures what UMFPACK is given and what it
 * returns, inside an unmodified, production-identical solver run.
 *
 * It intercepts umfpack_di_symbolic, umfpack_di_numeric and umfpack_di_solve
 * and forwards every call, with the caller's own arguments, through
 * dlsym(RTLD_NEXT). It never alters an argument, a return value or errno, and
 * it writes nothing to stdout/stderr unless UMFDUMP_VERBOSE=1.
 *
 * Environment (read once, at the first intercepted call -- never at load):
 *   UMFDUMP_DIR      absolute directory for the capture (created if missing).
 *                    Use a FRESH directory per run: the first process to create
 *                    UMFDUMP_DIR/owner is the only one that captures; any other
 *                    process (a second rank, a rerun into the same dir) is inert.
 *   UMFDUMP_CALLS    1-based indices of umfpack_di_numeric calls to capture,
 *                    comma-separated ("1,2", "5,400"). Tokens <= 0 or that do
 *                    not parse are ignored; an empty list makes the shim inert.
 *   UMFDUMP_VERBOSE  1 = one stderr line per capture event (default silent).
 *
 * Output, for each listed numeric call k (n = n_col of the Symbolic object the
 * call was handed, from a Symbolic -> n map filled by the symbolic wrapper):
 *   call<k>.Ap  int32[n+1]   } the CSC arrays exactly as passed to numeric,
 *   call<k>.Ai  int32[nnz]   } written BEFORE the call is forwarded, so a
 *   call<k>.Ax  float64[nnz] } factorisation that crashes still leaves them
 *   call<k>.ctl float64[UMFPACK_CONTROL] (only when Control != NULL)
 *   call<k>.b   float64[n]   B of the first umfpack_di_solve with that Numeric
 *   call<k>.x   float64[n]   X after that solve returns (the solver's answer)
 *   call<k>.json n, nnz, numeric wall time, Info/Control extracts and raw arrays,
 *               the Symbolic it used, the solve's status/IR/wall time, state.
 *   DONE        written once every listed call is complete (all files above).
 *   FAILED      appended to when a listed call cannot be completed (reason).
 * Every file is written to a dot-temp name and renamed, so a file that exists
 * under its final name is complete.
 *
 * Build (Linux):  gcc -O2 -shared -fPIC -fvisibility=hidden \
 *                     -I/usr/include/suitesparse umfdump.c -o libumfdump.so -ldl -lpthread
 * Build (macOS):  clang -O2 -dynamiclib -fvisibility=hidden -I<ss>/include/suitesparse \
 *                     umfdump.c -o libumfdump.dylib
 * Use:  LD_PRELOAD=/abs/libumfdump.so UMFDUMP_DIR=/abs/dump UMFDUMP_CALLS=1,400 <solver> ...
 *       (macOS local test: DYLD_INSERT_LIBRARIES=... into a host linked -Wl,-flat_namespace;
 *        dyld4 ignores DYLD_FORCE_FLAT_NAMESPACE, see README "Local tests")
 */
#ifndef _GNU_SOURCE
#define _GNU_SOURCE 1
#endif
#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <time.h>
#include <unistd.h>

#include <umfpack.h>

#ifdef __APPLE__
#include <mach-o/dyld.h>
#endif

#ifndef O_CLOEXEC
#define O_CLOEXEC 0
#endif

#define UD_EXPORT __attribute__((visibility("default")))
#define UD_MAX_LISTED 32
#define UD_MAX_SYMBOLIC 256
#define UD_PATH 4096
#define UD_JSON 32768

typedef int (*ud_symbolic_fn)(int32_t, int32_t, const int32_t *, const int32_t *,
                              const double *, void **, const double *, double *);
typedef int (*ud_numeric_fn)(const int32_t *, const int32_t *, const double *, void *,
                             void **, const double *, double *);
typedef int (*ud_solve_fn)(int, const int32_t *, const int32_t *, const double *, double *,
                           const double *, void *, const double *, double *);

static ud_symbolic_fn ud_real_symbolic;
static ud_numeric_fn ud_real_numeric;
static ud_solve_fn ud_real_solve;
static pthread_once_t ud_once = PTHREAD_ONCE_INIT;

#ifdef __APPLE__
/* Local-test path only. dyld4 (macOS 12+) resolves RTLD_NEXT from an inserted image
 * to nothing, so scan the loaded images for the first definition that is not this
 * shim's own. On Linux the standard dlsym(RTLD_NEXT) lookup above is the one used. */
static void *ud_next_apple(const char *name)
{
    Dl_info self;
    if (!dladdr((void *) &ud_next_apple, &self)) return NULL;
    const uint32_t count = _dyld_image_count();
    for (uint32_t i = 0; i < count; ++i)
    {
        const char *img = _dyld_get_image_name(i);
        if (!img) continue;
        void *h = dlopen(img, RTLD_LAZY | RTLD_NOLOAD);
        if (!h) continue;
        void *f = dlsym(h, name);
        dlclose(h);
        Dl_info di;
        if (f && dladdr(f, &di) && di.dli_fbase != self.dli_fbase) return f;
    }
    return NULL;
}
#endif

/* Linux fallback, used only when dlsym(RTLD_NEXT) finds nothing. RTLD_NEXT
 * searches the global scope, which holds UMFPACK when the solver plugin was
 * dlopen()ed RTLD_GLOBAL (OpenFOAM's dlOpen: RTLD_LAZY|RTLD_GLOBAL). A plugin
 * opened RTLD_LOCAL keeps its libumfpack out of that scope, so look the symbol
 * up in the already-loaded library itself; RTLD_NOLOAD loads nothing new.
 * Compiled on every platform so the local build checks it; called on Linux. */
__attribute__((unused)) static void *ud_next_soname(const char *name)
{
    static const char *const sonames[] = {"libumfpack.so.6", "libumfpack.so.5", "libumfpack.so", NULL};
    Dl_info self;
    if (!dladdr((void *) &ud_next_soname, &self)) return NULL;
    for (int i = 0; sonames[i] != NULL; ++i)
    {
        void *h = dlopen(sonames[i], RTLD_LAZY | RTLD_NOLOAD);
        if (!h) continue;
        void *f = dlsym(h, name);
        dlclose(h);
        Dl_info di;
        if (f && dladdr(f, &di) && di.dli_fbase != self.dli_fbase) return f;
    }
    return NULL;
}

static void *ud_next(const char *name)
{
    void *f = dlsym(RTLD_NEXT, name);
#ifdef __APPLE__
    if (!f) { (void) dlerror(); f = ud_next_apple(name); (void) dlerror(); }
#else
    if (!f) { (void) dlerror(); f = ud_next_soname(name); (void) dlerror(); }
#endif
    return f;
}

static void ud_resolve_all(void)
{
    ud_real_symbolic = (ud_symbolic_fn) ud_next("umfpack_di_symbolic");
    ud_real_numeric = (ud_numeric_fn) ud_next("umfpack_di_numeric");
    ud_real_solve = (ud_solve_fn) ud_next("umfpack_di_solve");
}

/* Called only from inside an intercepted UMFPACK call, so the next definition
 * exists in any process that could have made the call. If it does not, there
 * is nothing to forward to: fail loudly rather than invent a status. */
static void *ud_need(void *fn, const char *name)
{
    if (!fn)
    {
        fprintf(stderr, "umfdump: dlsym(RTLD_NEXT, \"%s\") found no definition; aborting\n", name);
        abort();
    }
    return fn;
}

/* ---- state (all guarded by ud_mu) ---------------------------------------- */

enum { UD_UNINIT = 0, UD_ACTIVE = 1, UD_FINISHED = 2, UD_INERT = -1 };
enum { REC_WAITING = 0, REC_AWAIT_SOLVE = 1, REC_IN_SOLVE = 2, REC_COMPLETE = 3, REC_FAILED = 4 };

typedef struct
{
    void *ptr;
    int32_t n_row, n_col;
    long call;
    int status;
    double wall;
    double ord_used, strat_used;
} ud_sym;

typedef struct
{
    long call;
    int state;
    void *numeric;
    int sym_known;
    ud_sym sym;
    int32_t n;
    int64_t nnz;
    const void *Ap, *Ai, *Ax;
    int num_status;
    double num_wall;
    int have_info, have_ctl;
    double info[UMFPACK_INFO], ctl[UMFPACK_CONTROL];
    long numeric_calls_at, symbolic_calls_at, solve_calls_at;
    int solved;
    int sys, sol_status;
    double sol_wall;
    int have_sinfo;
    double sinfo[UMFPACK_INFO];
    int same_arrays;
    long solve_call;
    char err[256];
} ud_rec;

static pthread_mutex_t ud_mu = PTHREAD_MUTEX_INITIALIZER;
static int ud_state = UD_UNINIT;
static int ud_verbose = 0;
static char ud_dir[UD_PATH];
static ud_rec ud_recs[UD_MAX_LISTED];
static int ud_nrec = 0;
static ud_sym ud_syms[UD_MAX_SYMBOLIC];
static int ud_sym_next = 0;
static long ud_numeric_calls = 0, ud_symbolic_calls = 0, ud_solve_calls = 0;
static char ud_json[UD_JSON];

static double ud_now(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double) ts.tv_sec + 1e-9 * (double) ts.tv_nsec;
}

static void ud_log(const char *fmt, ...)
{
    if (!ud_verbose) return;
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "umfdump[%ld]: ", (long) getpid());
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
}

static int ud_path(char *out, const char *name)
{
    int k = snprintf(out, UD_PATH, "%s/%s", ud_dir, name);
    return (k > 0 && k < UD_PATH) ? 0 : -1;
}

/* Write `bytes` from `data` to UMFDUMP_DIR/name via a dot-temp file + rename. */
static int ud_write_file(const char *name, const void *data, size_t bytes)
{
    char path[UD_PATH], tmpname[600], tmp[UD_PATH];
    if (snprintf(tmpname, sizeof tmpname, ".%s.tmp", name) >= (int) sizeof tmpname) return -1;
    if (ud_path(path, name) || ud_path(tmp, tmpname)) return -1;
    int fd = open(tmp, O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644);
    if (fd < 0) return -1;
    const char *p = (const char *) data;
    size_t left = bytes;
    while (left > 0)
    {
        size_t chunk = left > ((size_t) 1 << 30) ? ((size_t) 1 << 30) : left;
        ssize_t w = write(fd, p, chunk);
        if (w < 0)
        {
            if (errno == EINTR) continue;
            close(fd);
            unlink(tmp);
            return -1;
        }
        p += w;
        left -= (size_t) w;
    }
    if (close(fd) != 0) { unlink(tmp); return -1; }
    if (rename(tmp, path) != 0) { unlink(tmp); return -1; }
    return 0;
}

static void ud_append_failed(const ud_rec *r)
{
    char path[UD_PATH], line[400];
    if (ud_path(path, "FAILED")) return;
    int fd = open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (fd < 0) return;
    int k = snprintf(line, sizeof line, "call%ld %s\n", r->call, r->err);
    if (k > 0) { ssize_t w = write(fd, line, (size_t) (k < (int) sizeof line ? k : (int) sizeof line - 1)); (void) w; }
    close(fd);
}

static void ud_fail(ud_rec *r, const char *why)
{
    r->state = REC_FAILED;
    snprintf(r->err, sizeof r->err, "%s", why);
    ud_append_failed(r);
    ud_log("call %ld FAILED: %s", r->call, why);
}

/* ---- JSON ------------------------------------------------------------------ */

static size_t ud_jlen;
static int ud_joverflow;

static void ud_jp(const char *fmt, ...)
{
    if (ud_joverflow) return;
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(ud_json + ud_jlen, UD_JSON - ud_jlen, fmt, ap);
    va_end(ap);
    if (k < 0 || (size_t) k >= UD_JSON - ud_jlen) { ud_joverflow = 1; return; }
    ud_jlen += (size_t) k;
}

static void ud_jnum(double v)
{
    if (isfinite(v)) ud_jp("%.17g", v);
    else ud_jp("null");
}

static void ud_jkv(const char *k, double v, int comma)
{
    ud_jp("\"%s\":", k);
    ud_jnum(v);
    if (comma) ud_jp(",");
}

static void ud_jarr(const char *k, const double *v, int m)
{
    ud_jp("\"%s\":[", k);
    for (int i = 0; i < m; ++i) { if (i) ud_jp(","); ud_jnum(v[i]); }
    ud_jp("]");
}

static const char *ud_state_name(int s)
{
    switch (s)
    {
        case REC_WAITING: return "waiting";
        case REC_AWAIT_SOLVE: return "awaiting_solve";
        case REC_IN_SOLVE: return "in_solve";
        case REC_COMPLETE: return "complete";
        default: return "failed";
    }
}

static void ud_write_json(ud_rec *r)
{
    char name[64];
    ud_jlen = 0;
    ud_joverflow = 0;
    ud_jp("{\"call\":%ld,\"pid\":%ld,\"umfpack_header\":\"%d.%d.%d\",\"state\":\"%s\",",
          r->call, (long) getpid(), UMFPACK_MAIN_VERSION, UMFPACK_SUB_VERSION,
          UMFPACK_SUBSUB_VERSION, ud_state_name(r->state));
    if (r->state == REC_FAILED) ud_jp("\"error\":\"%s\",", r->err); else ud_jp("\"error\":null,");
    ud_jp("\"n\":%ld,\"nnz\":%lld,", (long) r->n, (long long) r->nnz);
    ud_jp("\"counters_at_numeric\":{\"numeric\":%ld,\"symbolic\":%ld,\"solve\":%ld},",
          r->numeric_calls_at, r->symbolic_calls_at, r->solve_calls_at);
    ud_jp("\"numeric\":{\"status\":%d,\"wall_s\":", r->num_status);
    ud_jnum(r->num_wall);
    ud_jp(",\"numeric_null\":%s},", r->numeric ? "false" : "true");
    if (r->sym_known)
    {
        ud_jp("\"symbolic\":{\"known\":true,\"call\":%ld,\"status\":%d,\"n_row\":%ld,\"n_col\":%ld,",
              r->sym.call, r->sym.status, (long) r->sym.n_row, (long) r->sym.n_col);
        ud_jkv("wall_s", r->sym.wall, 1);
        ud_jkv("ORDERING_USED", r->sym.ord_used, 1);
        ud_jkv("STRATEGY_USED", r->sym.strat_used, 0);
        ud_jp("},");
    }
    else ud_jp("\"symbolic\":{\"known\":false},");
    if (r->have_info)
    {
        const double *I = r->info;
        ud_jp("\"info\":{");
        ud_jkv("STATUS", I[UMFPACK_STATUS], 1);
        ud_jkv("LNZ", I[UMFPACK_LNZ], 1);
        ud_jkv("UNZ", I[UMFPACK_UNZ], 1);
        ud_jkv("LU_NNZ", I[UMFPACK_LNZ] + I[UMFPACK_UNZ], 1);
        ud_jkv("RCOND", I[UMFPACK_RCOND], 1);
        ud_jkv("FLOPS", I[UMFPACK_FLOPS], 1);
        ud_jkv("PEAK_MEMORY_UNITS", I[UMFPACK_PEAK_MEMORY], 1);
        ud_jkv("SIZE_OF_UNIT", I[UMFPACK_SIZE_OF_UNIT], 1);
        ud_jkv("PEAK_MEMORY_BYTES", I[UMFPACK_PEAK_MEMORY] * I[UMFPACK_SIZE_OF_UNIT], 1);
        ud_jkv("NUMERIC_SIZE_UNITS", I[UMFPACK_NUMERIC_SIZE], 1);
        ud_jkv("NUMERIC_WALLTIME", I[UMFPACK_NUMERIC_WALLTIME], 1);
        ud_jkv("ORDERING_USED", I[UMFPACK_ORDERING_USED], 1);
        ud_jkv("STRATEGY_USED", I[UMFPACK_STRATEGY_USED], 0);
        ud_jp("},");
    }
    else ud_jp("\"info\":null,");
    if (r->have_ctl)
    {
        const double *C = r->ctl;
        ud_jp("\"control\":{");
        ud_jkv("PRL", C[UMFPACK_PRL], 1);
        ud_jkv("STRATEGY", C[UMFPACK_STRATEGY], 1);
        ud_jkv("ORDERING", C[UMFPACK_ORDERING], 1);
        ud_jkv("SCALE", C[UMFPACK_SCALE], 1);
        ud_jkv("IRSTEP", C[UMFPACK_IRSTEP], 0);
        ud_jp("},");
    }
    else ud_jp("\"control\":null,");
    if (r->solved)
    {
        ud_jp("\"solve\":{\"call\":%ld,\"sys\":%d,\"status\":%d,\"arrays_same_as_numeric\":%s,",
              r->solve_call, r->sys, r->sol_status, r->same_arrays ? "true" : "false");
        ud_jkv("wall_s", r->sol_wall, 1);
        if (r->have_sinfo)
        {
            ud_jkv("IR_TAKEN", r->sinfo[UMFPACK_IR_TAKEN], 1);
            ud_jkv("IR_ATTEMPTED", r->sinfo[UMFPACK_IR_ATTEMPTED], 1);
            ud_jkv("SOLVE_FLOPS", r->sinfo[UMFPACK_SOLVE_FLOPS], 0);
        }
        else ud_jp("\"info\":null");
        ud_jp("},");
    }
    else ud_jp("\"solve\":null,");
    if (r->have_info) { ud_jarr("info_raw", r->info, UMFPACK_INFO); ud_jp(","); }
    if (r->have_ctl) { ud_jarr("control_raw", r->ctl, UMFPACK_CONTROL); ud_jp(","); }
    if (r->have_sinfo) { ud_jarr("solve_info_raw", r->sinfo, UMFPACK_INFO); ud_jp(","); }
    ud_jp("\"end\":true}\n");
    if (ud_joverflow) return;
    snprintf(name, sizeof name, "call%ld.json", r->call);
    if (ud_write_file(name, ud_json, ud_jlen) != 0 && r->state != REC_FAILED)
    {
        ud_fail(r, "write failed: json");
    }
}

/* ---- bookkeeping ----------------------------------------------------------- */

static void ud_init_locked(void)
{
    if (ud_state != UD_UNINIT) return;
    ud_state = UD_INERT;
    const char *v = getenv("UMFDUMP_VERBOSE");
    ud_verbose = (v && v[0] == '1');
    const char *dir = getenv("UMFDUMP_DIR");
    const char *calls = getenv("UMFDUMP_CALLS");
    if (!dir || !dir[0] || !calls || !calls[0]) return;
    const char *s = calls;
    while (*s && ud_nrec < UD_MAX_LISTED)
    {
        char *end = NULL;
        errno = 0;
        long k = strtol(s, &end, 10);
        if (end == s) { ++s; continue; }
        s = end;
        if (errno != 0 || k <= 0) continue;
        int dup = 0;
        for (int i = 0; i < ud_nrec; ++i) if (ud_recs[i].call == k) dup = 1;
        if (dup) continue;
        memset(&ud_recs[ud_nrec], 0, sizeof(ud_rec));
        ud_recs[ud_nrec].call = k;
        ud_recs[ud_nrec].state = REC_WAITING;
        ud_recs[ud_nrec].n = -1;
        ud_recs[ud_nrec].num_wall = NAN;
        ud_recs[ud_nrec].sol_wall = NAN;
        ++ud_nrec;
    }
    if (ud_nrec == 0) return;
    if (snprintf(ud_dir, sizeof ud_dir, "%s", dir) >= (int) sizeof ud_dir) return;
    mkdir(ud_dir, 0755);
    char path[UD_PATH];
    if (ud_path(path, "owner")) return;
    int fd = open(path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644);
    if (fd < 0)
    {
        ud_log("cannot create %s (exists or unwritable): inert", path);
        return;
    }
    char line[256];
    int k = snprintf(line, sizeof line, "pid %ld\ncalls %s\n", (long) getpid(), calls);
    if (k > 0) { ssize_t w = write(fd, line, (size_t) (k < (int) sizeof line ? k : (int) sizeof line - 1)); (void) w; }
    close(fd);
    ud_state = UD_ACTIVE;
    ud_log("active: dir %s, %d listed numeric call(s)", ud_dir, ud_nrec);
}

static ud_rec *ud_listed(long call)
{
    for (int i = 0; i < ud_nrec; ++i) if (ud_recs[i].call == call) return &ud_recs[i];
    return NULL;
}

static void ud_check_finished(void)
{
    int complete = 1, resolved = 1;
    for (int i = 0; i < ud_nrec; ++i)
    {
        if (ud_recs[i].state != REC_COMPLETE) complete = 0;
        if (ud_recs[i].state != REC_COMPLETE && ud_recs[i].state != REC_FAILED) resolved = 0;
    }
    if (complete)
    {
        char txt[64 * UD_MAX_LISTED + 64];
        size_t len = 0;
        len += (size_t) snprintf(txt + len, sizeof txt - len, "complete:");
        for (int i = 0; i < ud_nrec && len < sizeof txt; ++i)
            len += (size_t) snprintf(txt + len, sizeof txt - len, " %ld", ud_recs[i].call);
        if (len < sizeof txt) len += (size_t) snprintf(txt + len, sizeof txt - len, "\n");
        if (len > sizeof txt) len = sizeof txt;
        if (ud_write_file("DONE", txt, len) == 0) ud_log("DONE written");
    }
    if (resolved) ud_state = UD_FINISHED;
}

/* ---- the three intercepted entry points ------------------------------------ */

UD_EXPORT int umfpack_di_symbolic(int32_t n_row, int32_t n_col, const int32_t Ap[],
                                  const int32_t Ai[], const double Ax[], void **Symbolic,
                                  const double Control[UMFPACK_CONTROL], double Info[UMFPACK_INFO])
{
    const int saved_in = errno;
    pthread_once(&ud_once, ud_resolve_all);
    ud_symbolic_fn real = (ud_symbolic_fn) ud_need((void *) ud_real_symbolic, "umfpack_di_symbolic");
    errno = saved_in;
    const double t0 = ud_now();
    const int status = real(n_row, n_col, Ap, Ai, Ax, Symbolic, Control, Info);
    const double t1 = ud_now();
    const int saved = errno;
    pthread_mutex_lock(&ud_mu);
    ud_init_locked();
    const long call = ++ud_symbolic_calls;
    if (ud_state == UD_ACTIVE && Symbolic != NULL && *Symbolic != NULL)
    {
        int slot = -1;
        for (int i = 0; i < UD_MAX_SYMBOLIC; ++i) if (ud_syms[i].ptr == *Symbolic) { slot = i; break; }
        if (slot < 0) { slot = ud_sym_next; ud_sym_next = (ud_sym_next + 1) % UD_MAX_SYMBOLIC; }
        ud_sym *e = &ud_syms[slot];
        e->ptr = *Symbolic;
        e->n_row = n_row;
        e->n_col = n_col;
        e->call = call;
        e->status = status;
        e->wall = t1 - t0;
        e->ord_used = Info ? Info[UMFPACK_ORDERING_USED] : NAN;
        e->strat_used = Info ? Info[UMFPACK_STRATEGY_USED] : NAN;
    }
    pthread_mutex_unlock(&ud_mu);
    errno = saved;
    return status;
}

UD_EXPORT int umfpack_di_numeric(const int32_t Ap[], const int32_t Ai[], const double Ax[],
                                 void *Symbolic, void **Numeric,
                                 const double Control[UMFPACK_CONTROL], double Info[UMFPACK_INFO])
{
    const int saved_in = errno;
    pthread_once(&ud_once, ud_resolve_all);
    ud_numeric_fn real = (ud_numeric_fn) ud_need((void *) ud_real_numeric, "umfpack_di_numeric");
    pthread_mutex_lock(&ud_mu);
    ud_init_locked();
    const long call = ++ud_numeric_calls;
    ud_rec *r = (ud_state == UD_ACTIVE) ? ud_listed(call) : NULL;
    if (r && r->state == REC_WAITING)
    {
        r->numeric_calls_at = ud_numeric_calls;
        r->symbolic_calls_at = ud_symbolic_calls;
        r->solve_calls_at = ud_solve_calls;
        r->Ap = Ap; r->Ai = Ai; r->Ax = Ax;
        for (int i = 0; i < UD_MAX_SYMBOLIC; ++i)
            if (Symbolic != NULL && ud_syms[i].ptr == Symbolic) { r->sym = ud_syms[i]; r->sym_known = 1; break; }
        if (Control) { memcpy(r->ctl, Control, sizeof r->ctl); r->have_ctl = 1; }
        if (!r->sym_known) ud_fail(r, "Symbolic object not seen by the shim: n unknown");
        else if (!Ap || !Ai || !Ax) ud_fail(r, "NULL Ap/Ai/Ax");
        else if (r->sym.n_col <= 0 || Ap[0] != 0 || Ap[r->sym.n_col] < 0) ud_fail(r, "invalid column pointers");
        else
        {
            char name[64];
            r->n = r->sym.n_col;
            r->nnz = Ap[r->n];
            int bad = 0;
            snprintf(name, sizeof name, "call%ld.Ap", call);
            bad |= ud_write_file(name, Ap, sizeof(int32_t) * ((size_t) r->n + 1));
            snprintf(name, sizeof name, "call%ld.Ai", call);
            if (!bad) bad |= ud_write_file(name, Ai, sizeof(int32_t) * (size_t) r->nnz);
            snprintf(name, sizeof name, "call%ld.Ax", call);
            if (!bad) bad |= ud_write_file(name, Ax, sizeof(double) * (size_t) r->nnz);
            if (!bad && r->have_ctl)
            {
                snprintf(name, sizeof name, "call%ld.ctl", call);
                bad |= ud_write_file(name, Control, sizeof(double) * UMFPACK_CONTROL);
            }
            if (bad) ud_fail(r, "write failed: Ap/Ai/Ax/ctl");
            else ud_log("call %ld: wrote n %ld nnz %lld", call, (long) r->n, (long long) r->nnz);
        }
    }
    else r = NULL;
    pthread_mutex_unlock(&ud_mu);

    errno = saved_in;
    const double t0 = ud_now();
    const int status = real(Ap, Ai, Ax, Symbolic, Numeric, Control, Info);
    const double t1 = ud_now();
    const int saved_out = errno;

    pthread_mutex_lock(&ud_mu);
    if (ud_state == UD_ACTIVE || r)
    {
        void *num = (Numeric != NULL) ? *Numeric : NULL;
        /* A new object at an address still awaiting its first solve means the
         * old one was freed without a solve: that capture can never complete. */
        if (num)
            for (int i = 0; i < ud_nrec; ++i)
                if (&ud_recs[i] != r && ud_recs[i].state == REC_AWAIT_SOLVE && ud_recs[i].numeric == num)
                {
                    ud_recs[i].numeric = NULL;
                    ud_fail(&ud_recs[i], "Numeric object freed before any solve");
                    ud_write_json(&ud_recs[i]);
                }
        if (r)
        {
            r->num_status = status;
            r->num_wall = t1 - t0;
            if (Info) { memcpy(r->info, Info, sizeof r->info); r->have_info = 1; }
            if (r->state != REC_FAILED)
            {
                if (status < 0 || num == NULL) ud_fail(r, "numeric returned an error (no Numeric object)");
                else { r->numeric = num; r->state = REC_AWAIT_SOLVE; }
            }
            ud_write_json(r);
        }
        ud_check_finished();
    }
    pthread_mutex_unlock(&ud_mu);
    errno = saved_out;
    return status;
}

UD_EXPORT int umfpack_di_solve(int sys, const int32_t Ap[], const int32_t Ai[], const double Ax[],
                               double X[], const double B[], void *Numeric,
                               const double Control[UMFPACK_CONTROL], double Info[UMFPACK_INFO])
{
    const int saved_in = errno;
    pthread_once(&ud_once, ud_resolve_all);
    ud_solve_fn real = (ud_solve_fn) ud_need((void *) ud_real_solve, "umfpack_di_solve");
    pthread_mutex_lock(&ud_mu);
    ud_init_locked();
    const long scall = ++ud_solve_calls;
    ud_rec *r = NULL;
    if (ud_state == UD_ACTIVE && Numeric != NULL)
        for (int i = 0; i < ud_nrec; ++i)
            if (ud_recs[i].state == REC_AWAIT_SOLVE && ud_recs[i].numeric == Numeric) { r = &ud_recs[i]; break; }
    if (r)
    {
        char name[64];
        r->state = REC_IN_SOLVE;
        r->solve_call = scall;
        r->sys = sys;
        r->same_arrays = (Ap == r->Ap && Ai == r->Ai && Ax == r->Ax);
        snprintf(name, sizeof name, "call%ld.b", r->call);
        if (!B || !X) ud_fail(r, "NULL B or X in solve");
        else if (ud_write_file(name, B, sizeof(double) * (size_t) r->n) != 0) ud_fail(r, "write failed: b");
    }
    pthread_mutex_unlock(&ud_mu);

    errno = saved_in;
    const double t0 = ud_now();
    const int status = real(sys, Ap, Ai, Ax, X, B, Numeric, Control, Info);
    const double t1 = ud_now();
    const int saved_out = errno;

    if (r)
    {
        pthread_mutex_lock(&ud_mu);
        r->solved = 1;
        r->sol_status = status;
        r->sol_wall = t1 - t0;
        if (Info) { memcpy(r->sinfo, Info, sizeof r->sinfo); r->have_sinfo = 1; }
        if (r->state == REC_IN_SOLVE)
        {
            char name[64];
            snprintf(name, sizeof name, "call%ld.x", r->call);
            if (ud_write_file(name, X, sizeof(double) * (size_t) r->n) != 0) ud_fail(r, "write failed: x");
            else r->state = REC_COMPLETE;
        }
        ud_write_json(r);
        ud_log("call %ld: solve status %d, state %s", r->call, status, ud_state_name(r->state));
        ud_check_finished();
        pthread_mutex_unlock(&ud_mu);
    }
    errno = saved_out;
    return status;
}
