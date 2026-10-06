"""numerics-analyst 03: is the free clot's edge broadening numerical (upwind) diffusion, stretch, or dilution?
Log fact: PHASE_CLOT_CONTENT_FCT lambdaFluxMean=0 on every step (thetaH == 0 -> its limiter is 0 on
every face -> min() over components admits no antidiffusion), so the solver transports theta by
implicit upwind alone.
(A) per-frame history: perimeter-mean q 0.25-0.75 width (coarea: 0.5*A_band/int_band|grad q|),
    interior-only version (cells > 3 um from both walls), max theta, total content, band-mean upwind
    diffusivity D_up = 0.5(|Vs_x|dx nx^2 + |Vs_y|dy ny^2), band-mean normal strain s_n = n.gradVs.n,
    and the ODE prediction d(w^2)/dt = 2 s_n w^2 + 3.64 D (with and without D).
(B) replay windows: theta(t0) carried by Vs (linear-interp faces, time-interp between 0.2 ms frames)
    with (U) implicit upwind = the solver as logged, (F) upwind + Zalesak FCT on theta alone (the fix),
    compared with the returned frame at t1.
PREDICTION (written before running): U reproduces the returned width to <~0.3 um; F's width is
smaller by ~sqrt(3.64*D_up*dt_window) in quadrature, i.e. a few um by 25 ms.
usage: python na03-width.py <case> <tmax_ms> [replay]"""
import re, sys, numpy as np
C = sys.argv[1]; TMAX = float(sys.argv[2]); REPLAY = len(sys.argv) > 3
def rd(path, vec=False, n=None):
    s = open(path).read()
    m = re.search(r"internalField\s+nonuniform\s+List<(scalar|vector)>\s*(\d+)\s*\(", s)
    if not m:
        u = re.search(r"internalField\s+uniform\s+(\S+);", s)
        if u: return np.full(n, float(u.group(1)))
        u = re.search(r"internalField\s+uniform\s+\((\S+)\s+(\S+)\s+(\S+)\);", s)
        return np.tile([float(u.group(i)) for i in (1, 2, 3)], (n, 1))
    body = s[m.end(): s.find("\n)\n", m.end())]
    if not vec: return np.array(body.split()[:int(m.group(2))], float)
    return np.array(re.findall(r"\(\s*(\S+)\s+(\S+)\s+(\S+)\s*\)", body), float)
def foamlist(path):
    s = open(path).read(); m = re.search(r"\n(\d+)\s*\n\(", s); return s[m.end():], int(m.group(1))
pm = f"{C}/constant/polyMesh"
b, n = foamlist(f"{pm}/points"); P = np.array(re.findall(r"\(([^()]+)\)", b)[:n], float).reshape(n, 3) if False else \
    np.array([l.split() for l in re.findall(r"\(([^()]+)\)", b)[:n]], float)
b, n = foamlist(f"{pm}/faces"); Fc = np.array([l.split() for l in re.findall(r"4\(([^()]+)\)", b)[:n]], int)
b, n = foamlist(f"{pm}/owner"); own = np.array(b.split(")")[0].split(), int)[:n]
b, nn = foamlist(f"{pm}/neighbour"); nei = np.array(b.split(")")[0].split(), int)[:nn]
NI = len(nei); NC = own.max() + 1
p0, p1, p2, p3 = (P[Fc[:, k]] for k in range(4))
Sf = 0.5*np.cross(p2 - p0, p3 - p1); Cf = 0.25*(p0 + p1 + p2 + p3)
V = np.zeros(NC); np.add.at(V, own, (Sf*Cf).sum(1)/3); np.add.at(V, nei, -(Sf[:NI]*Cf[:NI]).sum(1)/3)
Cc = np.c_[rd(f"{C}/0/Ccx"), rd(f"{C}/0/Ccy"), np.zeros(NC)]
o, ne = own[:NI], nei
SfI = Sf[:NI]; wL = (SfI*(Cc[ne] - Cf[:NI])).sum(1)/(SfI*(Cc[ne] - Cc[o])).sum(1)
dpos = Cc[ne] - Cc[o]
ymin, ymax = np.interp(Cc[:, 0], *[0, 0]) if False else (None, None)
def grad(f):  # Gauss linear, boundary faces take the cell value (zeroGradient)
    ff = wL*f[o] + (1 - wL)*f[ne]
    g = np.zeros((NC, 3)); np.add.at(g, o, SfI*ff[:, None]); np.add.at(g, ne, -SfI*ff[:, None])
    np.add.at(g, own[NI:], Sf[NI:]*f[own[NI:]][:, None]); return g/V[:, None]
# local spacing: cell extent in x and y
hx = np.zeros(NC); hy = np.zeros(NC)
ax = np.abs(dpos[:, 0]) > np.abs(dpos[:, 1])
np.maximum.at(hx, o[ax], np.abs(dpos[ax, 0])); np.maximum.at(hy, o[~ax], np.abs(dpos[~ax, 1]))
hx[hx == 0] = np.median(hx); hy[hy == 0] = np.median(hy[hy > 0])
# distance to walls: per column, from the extreme cell centres of that column
wallf = np.r_[NI:len(own)]; wallf = wallf[np.abs(Sf[wallf, 1]) > 1e-3*np.abs(Sf[wallf]).max()]
xs = Cf[wallf, 0]; ys = Cf[wallf, 1]
bot = ys < np.median(ys); xb, yb = xs[bot], ys[bot]; xt, yt = xs[~bot], ys[~bot]
yB = np.interp(Cc[:, 0], xb[np.argsort(xb)], yb[np.argsort(xb)]); yT = np.interp(Cc[:, 0], xt[np.argsort(xt)], yt[np.argsort(xt)])
interior = np.minimum(Cc[:, 1] - yB, yT - Cc[:, 1]) > 3e-6
def metrics(q, Vs):
    th = 0.45*(1 - q); g = grad(q); gm = np.linalg.norm(g, axis=1)
    out = {}
    for tag, msk in (("all", np.ones(NC, bool)), ("int", interior)):
        bd = (q > 0.25) & (q < 0.75) & msk
        den = (gm[bd]*V[bd]).sum(); out[tag] = 0.5*V[bd].sum()/den*1e6 if den > 0 else np.nan
    bd = (q > 0.25) & (q < 0.75) & interior
    nrm = g[bd]/np.maximum(gm[bd], 1e-30)[:, None]; wgt = gm[bd]*V[bd]
    D = 0.5*(np.abs(Vs[bd, 0])*hx[bd]*nrm[:, 0]**2 + np.abs(Vs[bd, 1])*hy[bd]*nrm[:, 1]**2)
    gx, gy = grad(Vs[:, 0]), grad(Vs[:, 1])
    sn = nrm[:, 0]**2*gx[bd, 0] + nrm[:, 0]*nrm[:, 1]*(gx[bd, 1] + gy[bd, 0]) + nrm[:, 1]**2*gy[bd, 1]
    divV = gx[:, 0] + gy[:, 1]; core = q < 0.25
    out.update(D=(D*wgt).sum()/wgt.sum(), sn=(sn*wgt).sum()/wgt.sum(), thmax=th.max(), M=(th*V).sum(),
               Acore=V[core].sum(), divcore=(divV[core]*V[core]).sum()/max(V[core].sum(), 1e-300),
               Vbody=(Vs[q < 0.5, 0]*V[q < 0.5]).sum()/V[q < 0.5].sum())
    return out
times = [0.0002] + [round(0.001*k, 4) for k in range(1, int(TMAX) + 1)]
def tdir(t):
    for s in (f"{t:g}", f"{t}"):
        try: open(f"{C}/{s}/q"); return s
        except OSError: pass
print(f"mesh: {NC} cells, dx median {np.median(hx)*1e6:.3f} um, dy median {np.median(hy)*1e6:.3f} um, min dy {hy.min()*1e6:.3f}")
print(" t_ms  w_all  w_int  w_pred(s+D) w_pred(s)  D_up um2/s  s_n 1/s  thmax  minq  M/M0-1  divcore 1/s  Vbody mm/s")
w2a = w2b = None; tprev = None; M0 = None
for t in times:
    d = tdir(t)
    if d is None: continue
    q = rd(f"{C}/{d}/q", n=NC); Vs = rd(f"{C}/{d}/Vs", True, NC)
    m = metrics(q, Vs)
    if w2a is None: w2a = w2b = m["int"]**2; M0 = m["M"]
    else:
        dt = t - tprev
        w2a += dt*(2*0.5*(m["sn"] + sp)*w2a + 3.64*0.5*(m["D"] + Dp)*1e12)
        w2b += dt*(2*0.5*(m["sn"] + sp)*w2b)
    sp, Dp, tprev = m["sn"], m["D"], t
    print(f"{t*1e3:5.1f} {m['all']:6.2f} {m['int']:6.2f} {np.sqrt(w2a):9.2f} {np.sqrt(w2b):9.2f} {m['D']*1e12:10.0f} {m['sn']:9.1f}"
          f" {m['thmax']:.4f} {1-m['thmax']/0.45:.3f} {m['M']/M0-1:+.2e} {m['divcore']:9.2f} {m['Vbody']*1e3:6.2f}")
if not REPLAY: sys.exit()
dT = 4.7619047619e-06; frames = sorted({round(0.0002*k, 4) for k in range(1, 126)})
def Vs_at(t, cache={}):
    k = min(int(t/0.0002), 124); ta, tb = 0.0002*k, 0.0002*(k + 1)
    for tt in (ta, tb):
        if tt not in cache: cache[tt] = rd(f"{C}/{tdir(round(tt, 4)) if tt > 0 else '0.0002'}/Vs", True, NC)
    a = (t - ta)/(tb - ta); return (1 - a)*cache[ta] + a*cache[tb]
def step(th, F, fct):
    up = F >= 0; don = np.where(up, o, ne); rec = np.where(up, ne, o); aF = np.abs(F)
    diag = V/dT + np.bincount(don, aF, NC); x = th.copy()
    for _ in range(30):
        xn = (V/dT*th + np.bincount(rec, aF*x[don], NC))/diag
        if np.abs(xn - x).max() < 1e-15: x = xn; break
        x = xn
    if not fct: return x
    A = F*(wL - up)*(x[o] - x[ne])  # as fctAntidiffusiveFlux (low-order field) -- see note
    hi = np.maximum(x, th); lo = np.minimum(x, th); H, L = hi.copy(), lo.copy()
    np.maximum.at(H, o, hi[ne]); np.maximum.at(H, ne, hi[o]); np.minimum.at(L, o, lo[ne]); np.minimum.at(L, ne, lo[o])
    Pp = np.zeros(NC); Pm = np.zeros(NC)
    np.add.at(Pm, o, np.maximum(A, 0)); np.add.at(Pp, ne, np.maximum(A, 0)); np.add.at(Pp, o, np.maximum(-A, 0)); np.add.at(Pm, ne, np.maximum(-A, 0))
    tiny = np.finfo(float).tiny; Qp = (H - x)*V/dT*(1 - 1e-12); Qm = (x - L)*V/dT*(1 - 1e-12)
    Rp = np.where(Qp < tiny, 0, np.where(Pp > 0, np.minimum(1, Qp/np.maximum(Pp, tiny)), 1))
    Rm = np.where(Qm < tiny, 0, np.where(Pm > 0, np.minimum(1, Qm/np.maximum(Pm, tiny)), 1))
    lam = np.where(np.abs(A) < tiny, 0, np.where(A > 0, np.minimum(Rm[o], Rp[ne]), np.minimum(Rp[o], Rm[ne])))
    dv = np.zeros(NC); np.add.at(dv, o, lam*A); np.add.at(dv, ne, -lam*A)
    step.lam.append((lam*np.abs(A)).sum()/max(np.abs(A).sum(), tiny))
    return x - dT/V*dv
step.lam = []
print("replay  window_ms  scheme  w_int  w_returned  thmax  thmax_ret  L1/M  lambdaFluxMean  M/M0-1")
WIN = [(0.0002, 0.005), (0.005, 0.01), (0.01, 0.015), (0.015, 0.02), (0.02, 0.025)]
if sys.argv[3] == "chain": WIN = [(0.0002, 0.025)]  # one 25-ms run from the first frame
for t0, t1 in WIN:
    if t1*1e3 > TMAX + 1e-9: break
    q0 = rd(f"{C}/{tdir(t0)}/q", n=NC); q1 = rd(f"{C}/{tdir(t1)}/q", n=NC); th1 = 0.45*(1 - q1)
    for fct in (False, True):
        th = 0.45*(1 - q0); M0 = (th*V).sum(); t = t0; step.lam = []
        while t < t1 - 0.5*dT:
            Vf = Vs_at(t + dT); F = ((wL[:, None]*Vf[o] + (1 - wL[:, None])*Vf[ne])*SfI).sum(1)
            th = step(th, F, fct); t += dT
            if sys.argv[3] == "chain" and abs(t*1e3/5 - round(t*1e3/5)) < 0.5*dT*1e3/5 and round(t*1e3) % 5 == 0 and t > 1e-3:
                qq = rd(f"{C}/{tdir(round(t*1e3)/1e3)}/q", n=NC); mr_ = metrics(1 - th/0.45, Vf); mq = metrics(qq, Vf)
                print(f"chain {t*1e3:5.1f} {'FCT' if fct else 'UPW'} w_int {mr_['int']:6.2f} returned {mq['int']:6.2f} minq {1-th.max()/0.45:.3f} returned {qq.min():.3f}; n(theta>0.45) {(th>0.45).sum()} vs {(qq<0).sum()}; core(q<0.25) mean theta {th[th>0.3375].dot(V[th>0.3375])/V[th>0.3375].sum():.4f} area {V[th>0.3375].sum()/V.mean():.0f} cells vs {(0.45*(1-qq))[qq<0.25].dot(V[qq<0.25])/V[qq<0.25].sum():.4f} area {V[qq<0.25].sum()/V.mean():.0f}", flush=True)
        mr = metrics(1 - th/0.45, Vf); m1 = metrics(q1, Vf)
        print(f"replay {t0*1e3:5.1f}-{t1*1e3:4.1f}  {'FCT' if fct else 'UPW'}  {mr['int']:6.2f} {m1['int']:6.2f}  {th.max():.4f} {th1.max():.4f}"
              f"  {np.abs(th - th1).dot(V)/M0:.3f}  {np.mean(step.lam) if fct else 0:.3f}  {(th*V).sum()/M0-1:+.1e}")
