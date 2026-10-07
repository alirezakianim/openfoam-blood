#!/bin/bash
# plugin_selftest.sh -- Linux only. Proves the shim captures UMFPACK calls made from a
# PLUGIN, loaded the way OpenFOAM loads a solver module: dlopen(RTLD_LAZY|RTLD_GLOBAL)
# (src/OSspecific/POSIX/POSIX.cpp, Foam::dlOpen). run_bench.py --shim-only tests a host
# that links libumfpack directly, which is a different lookup path. The RTLD_LOCAL case
# exercises the shim's soname fallback and is reported, not required.
#   bash plugin_selftest.sh <abs path to libumfdump.so> <work dir>
# Ends "PLUGIN_SELFTEST PASS" and exits 0 only when the RTLD_GLOBAL capture is complete
# and byte-exact.
set -u
SHIM="$1"; W="$2"
rm -rf "$W"; mkdir -p "$W"
cat > "$W/plugin.c" <<'EOF'
#include <umfpack.h>
#include <math.h>
/* A = [[4,1,0],[1,3,1],[0,1,2]] (CSC), b = A [1,2,3] = [6,10,8] */
int plugin_run(double *x)
{
    static const int Ap[] = {0, 2, 5, 7};
    static const int Ai[] = {0, 1, 0, 1, 2, 1, 2};
    static const double Ax[] = {4, 1, 1, 3, 1, 1, 2};
    static const double b[] = {6, 10, 8};
    double ctl[UMFPACK_CONTROL], info[UMFPACK_INFO];
    void *sym = 0, *num = 0;
    umfpack_di_defaults(ctl);
    if (umfpack_di_symbolic(3, 3, Ap, Ai, Ax, &sym, ctl, info) != UMFPACK_OK) return 1;
    if (umfpack_di_numeric(Ap, Ai, Ax, sym, &num, ctl, info) != UMFPACK_OK) return 2;
    if (umfpack_di_solve(UMFPACK_A, Ap, Ai, Ax, x, b, num, ctl, info) != UMFPACK_OK) return 3;
    umfpack_di_free_numeric(&num);
    umfpack_di_free_symbolic(&sym);
    for (int i = 0; i < 3; ++i) if (fabs(x[i] - (i + 1)) > 1e-12) return 4;
    return 0;
}
EOF
cat > "$W/host.c" <<'EOF'
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
int main(int argc, char **argv)
{
    const int local = argc > 2 && strcmp(argv[2], "local") == 0;
    void *h = dlopen(argv[1], RTLD_LAZY | (local ? RTLD_LOCAL : RTLD_GLOBAL));
    if (!h) { fprintf(stderr, "dlopen: %s\n", dlerror()); return 10; }
    int (*run)(double *) = (int (*)(double *)) dlsym(h, "plugin_run");
    if (!run) { fprintf(stderr, "dlsym: %s\n", dlerror()); return 11; }
    double x[3] = {0, 0, 0};
    const int rc = run(x);
    printf("PLUGIN %s rc %d x %.17g %.17g %.17g\n", local ? "RTLD_LOCAL" : "RTLD_GLOBAL", rc, x[0], x[1], x[2]);
    return rc;
}
EOF
fail=0
gcc -O2 -shared -fPIC -I/usr/include/suitesparse "$W/plugin.c" -o "$W/libplugin.so" -lumfpack -lm || { echo "FAIL build plugin"; exit 1; }
gcc -O2 "$W/host.c" -o "$W/host" -ldl || { echo "FAIL build host"; exit 1; }
"$W/host" "$W/libplugin.so" global > "$W/plain.out" 2>&1; rc=$?
cat "$W/plain.out"
[ "$rc" -eq 0 ] && echo "PASS plain plugin (no shim) solves A x = b" || { echo "FAIL plain plugin rc $rc"; fail=1; }

check_capture() {   # $1 = capture dir, $2 = label
    python3 - "$1" "$2" <<'PY'
import os, struct, sys
d, label = sys.argv[1], sys.argv[2]
want = {"Ap": struct.pack("<4i", 0, 2, 5, 7), "Ai": struct.pack("<7i", 0, 1, 0, 1, 2, 1, 2),
        "Ax": struct.pack("<7d", 4, 1, 1, 3, 1, 1, 2), "b": struct.pack("<3d", 6, 10, 8)}
ok = os.path.isfile(os.path.join(d, "DONE"))
print(f"  {label}: DONE {'present' if ok else 'MISSING'}")
for k, v in want.items():
    p = os.path.join(d, f"call1.{k}")
    same = os.path.isfile(p) and open(p, "rb").read() == v
    print(f"  {label}: call1.{k} {'byte-exact' if same else 'WRONG or MISSING'}")
    ok &= same
p = os.path.join(d, "call1.x")
x = struct.unpack("<3d", open(p, "rb").read()) if os.path.isfile(p) and os.path.getsize(p) == 24 else None
good = x is not None and all(abs(x[i] - (i + 1)) <= 1e-12 for i in range(3))
print(f"  {label}: call1.x {x} {'ok' if good else 'WRONG or MISSING'}")
sys.exit(0 if ok and good else 1)
PY
}

env LD_PRELOAD="$SHIM" UMFDUMP_DIR="$W/cap_global" UMFDUMP_CALLS=1 "$W/host" "$W/libplugin.so" global > "$W/global.out" 2>&1; rc=$?
cat "$W/global.out"
if [ "$rc" -eq 0 ] && cmp -s <(sed 's/RTLD_GLOBAL//' "$W/plain.out") <(sed 's/RTLD_GLOBAL//' "$W/global.out") && check_capture "$W/cap_global" RTLD_GLOBAL; then
    echo "PASS RTLD_GLOBAL plugin under the shim: same answer, capture complete and byte-exact"
else
    echo "FAIL RTLD_GLOBAL plugin under the shim (rc $rc)"; fail=1
fi

env LD_PRELOAD="$SHIM" UMFDUMP_DIR="$W/cap_local" UMFDUMP_CALLS=1 "$W/host" "$W/libplugin.so" local > "$W/local.out" 2>&1; rc=$?
cat "$W/local.out"
if [ "$rc" -eq 0 ] && check_capture "$W/cap_local" RTLD_LOCAL; then
    echo "NOTE RTLD_LOCAL plugin under the shim: capture complete (soname fallback works)"
else
    echo "NOTE RTLD_LOCAL plugin under the shim did not capture (rc $rc); OpenFOAM uses RTLD_GLOBAL, so this is informative only"
fi

if [ "$fail" -eq 0 ]; then echo "PLUGIN_SELFTEST PASS"; exit 0; fi
echo "PLUGIN_SELFTEST FAIL"; exit 1
