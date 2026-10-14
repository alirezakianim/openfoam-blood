#!/usr/bin/env python3
"""na09_roundtrip: rebuild a mesh from a blockMeshDict TEXT FILE ONLY (no optimiser state) and report
OF-style counts, checkMesh metrics and (optionally) the point-by-point distance to a reference point set.

Parser: vertices, hex blocks (simpleGrading 1 only), arc / polyLine edges, boundary faces.
Edge points: lineDivide lambda = i/n (lineDivide.C:89), arcEdge uniform angle (arcEdge.C position),
polyLine uniform in cumulative length (polyLine.C calcParam/position). Block interior: blockCreate.C
replica (na06_ymesh.block_points). Points are merged with a tolerance of 1e-3 x the shortest edge,
so a crack wider than that shows up as extra points and extra boundary faces.
Usage: python3 na09_roundtrip.py <dict> [--plane z|x] [--ref points.npy] [--real <polyMesh/points>]
Run with /usr/local/bin/python3 (scipy)."""
import re, sys, math, numpy as np
from scipy.spatial import cKDTree
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from na06_ymesh import block_points, metrics

def strip(t):
    t = re.sub(r"/\*.*?\*/", " ", t, flags=re.S); return re.sub(r"//[^\n]*", " ", t)

def section(t, name):
    m = re.search(r"\b" + name + r"\s*\(", t); i = m.end(); depth = 1; j = i
    while depth:
        depth += {"(": 1, ")": -1}.get(t[j], 0); j += 1
    return t[i:j - 1]

FLOAT = r"[-+0-9.eE]+"
def parse(path):
    t = strip(open(path).read())
    ctm = float(re.search(r"convertToMeters\s+(" + FLOAT + ")", t).group(1)) if "convertToMeters" in t else 1.0
    V = np.array([[float(a) for a in m] for m in re.findall(r"\(\s*(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*\)", section(t, "vertices"))]) * ctm
    B = []
    for m in re.finditer(r"hex\s*\(([\d\s]+)\)\s*\(\s*(\d+)\s+(\d+)\s+(\d+)\s*\)\s*simpleGrading\s*\(\s*(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*\)", section(t, "blocks")):
        assert all(float(m.group(k)) == 1.0 for k in (5, 6, 7)), "grading != 1 not replicated"
        B.append((list(map(int, m.group(1).split())), int(m.group(2)), int(m.group(3)), int(m.group(4))))
    E = {}
    es = section(t, "edges")
    for m in re.finditer(r"(arc|polyLine)\s+(\d+)\s+(\d+)\s*\(", es):
        i = m.end(); depth = 1; j = i
        while depth:
            depth += {"(": 1, ")": -1}.get(es[j], 0); j += 1
        body = es[m.end() - 1:j]
        pts = np.array([[float(a) for a in q] for q in re.findall(r"\(\s*(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*\)", body)]) * ctm
        if m.group(1) == "arc" and len(pts) == 0:     # arc a b (x y z): no inner parentheses
            pts = np.array([[float(a) for a in body.strip("() ").split()]]) * ctm
        E[(int(m.group(2)), int(m.group(3)))] = (m.group(1), pts)
    bs = section(t, "boundary"); P = {}
    for m in re.finditer(r"(\w+)\s*\{[^}]*?type\s+(\w+)\s*;[^}]*?faces\s*\((.*?)\)\s*;\s*\}", bs, flags=re.S):
        P[m.group(1)] = (m.group(2), [list(map(int, f.split())) for f in re.findall(r"\(([\d\s]+)\)", m.group(3))])
    return V, B, E, P

def arc_pos(a, m, b, lam):
    a, m, b = map(np.asarray, (a, m, b))
    r1, r2 = m - a, b - a; ax = np.cross(r1, r2); n = ax / np.linalg.norm(ax)
    # circumcentre in 3-D
    c = a + (np.dot(r1, r1) * np.cross(r2, ax) + np.dot(r2, r2) * np.cross(ax, r1)) / (2 * np.dot(ax, ax))
    u0 = a - c; R = np.linalg.norm(u0); v0 = np.cross(n, u0)
    ang = lambda p: math.atan2(np.dot(p - c, v0), np.dot(p - c, u0))
    th = ang(b) % (2 * math.pi)                       # a -> m -> b, counter-clockwise about n by construction
    return c + np.outer(np.cos(th * lam), u0) + np.outer(np.sin(th * lam), v0)

def edge_points(V, E, a, b, n):
    lam = np.arange(n + 1) / n
    for (u, v, rev) in ((a, b, False), (b, a, True)):
        if (u, v) in E:
            kd, d = E[(u, v)]; L = lam[::-1] if rev else lam
            if kd == "arc":
                return arc_pos(V[u], d[0], V[v], L)
            q = np.concatenate([[V[u]], d, [V[v]]]); s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(q, axis=0), axis=1))]) / \
                np.sum(np.linalg.norm(np.diff(q, axis=0), axis=1))
            return np.stack([np.interp(L, s, q[:, k]) for k in range(3)], 1)
    return V[a] + lam[:, None] * (V[b] - V[a])

def build2d(V, B, E, drop):
    keep = [k for k in range(3) if k != drop]
    pts, cells, base, tag = [], [], 0, []
    for bi, (h, ni, nj, nk) in enumerate(B):
        v0, v1, v2, v3 = h[:4]
        e = lambda a, b, n: edge_points(V, E, a, b, n)[:, keep]
        X = block_points(V[v0][keep], V[v1][keep], V[v2][keep], V[v3][keep], e(v0, v1, ni), e(v3, v2, ni), e(v0, v3, nj), e(v1, v2, nj))
        pts.append(X.reshape(-1, 2))
        J, I = np.meshgrid(np.arange(nj), np.arange(ni), indexing="ij"); p = (base + J * (ni + 1) + I).ravel()
        cells.append(np.stack([p, p + 1, p + ni + 2, p + ni + 1], 1)); base += (ni + 1) * (nj + 1); tag += [bi] * (ni * nj)
    Pall = np.concatenate(pts); C = np.concatenate(cells)
    emin = min(np.linalg.norm(np.diff(X.reshape(-1, 2), axis=0), axis=1)[np.linalg.norm(np.diff(X.reshape(-1, 2), axis=0), axis=1) > 0].min() for X in [Pall])
    tol = 1e-3 * emin
    tree = cKDTree(Pall); pairs = tree.query_pairs(tol, output_type="ndarray")
    parent = np.arange(len(Pall))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for i, j in pairs:
        ri, rj = find(i), find(j)
        if ri != rj: parent[max(ri, rj)] = min(ri, rj)
    root = np.array([find(i) for i in range(len(Pall))])
    uq, inv = np.unique(root, return_inverse=True)
    Pu = Pall[uq]; return Pu, inv[C], np.array(tag), tol

def of_counts(V, B, P, Pu, Cu, m, drop):
    nc = len(Cu); n_int = len(m["no"]); nb2d = len(m["skb"])
    cnt = {}; inward = []
    for nm, (ty, faces) in P.items():
        tot = 0
        for f in faces:
            blk = [b for b in B if set(f) <= set(b[0])]
            assert len(blk) >= 1, f"patch face {f} not on a block"
            h, ni, nj, nk = blk[0]; pos = sorted(h.index(v) for v in f)
            if pos in ([0, 1, 2, 3], [4, 5, 6, 7]): tot += ni * nj
            elif pos in ([0, 1, 4, 5], [2, 3, 6, 7]): tot += ni * nk
            else: tot += nj * nk
            # outward check (blockMeshCheck.C:177-209): face normal (right-hand) vs block centre
            q = V[f]; nrm = np.cross(q[1] - q[0], q[2] - q[0]) + np.cross(q[2] - q[0], q[3] - q[0])
            if np.dot(nrm, q.mean(0) - V[h].mean(0)) <= 0: inward.append((nm, f))
        cnt[nm] = (ty, tot)
    side = sum(v[1] for k, v in cnt.items() if v[0] != "empty")
    assert side == nb2d, f"2-D boundary edges {nb2d} != non-empty patch faces {side}: crack or uncovered face"
    return dict(inward=inward, nPoints=2 * len(Pu), nCells=nc, nInternalFaces=n_int, nFaces=n_int + sum(v[1] for v in cnt.values()), patches=cnt)

def main():
    a = sys.argv[1:]; path = a[0]
    drop = 2 if "--plane" not in a else {"z": 2, "x": 0}[a[a.index("--plane") + 1]]
    V, B, E, P = parse(path)
    # 2-D: the first four hex vertices must share the dropped coordinate
    z0 = V[B[0][0][0]][drop]
    Pu, Cu, tag, tol = build2d(V, B, E, drop)
    A = 0.5 * np.sum(Pu[Cu][:, :, 0] * np.roll(Pu[Cu][:, :, 1], -1, 1) - np.roll(Pu[Cu][:, :, 0], -1, 1) * Pu[Cu][:, :, 1], 1)
    sgn = 1.0 if (A > 0).all() else (-1.0 if (A < 0).all() else 0.0)
    assert sgn != 0, "mixed cell orientation"
    if sgn < 0: Cu = Cu[:, ::-1]
    m = metrics(Pu, Cu)
    if "--pointsonly" in a:
        c = dict(nPoints=len(Pu), nCells=len(Cu), nFaces=-1, nInternalFaces=len(m["no"]), patches={}, inward=[])
    else:
        c = of_counts(V, B, P, Pu, Cu, m, drop)
    print(f"[roundtrip {path.rsplit('/',1)[-1]}] blocks {len(B)} vertices {len(V)} curved edges {len(E)} (arc {sum(1 for v in E.values() if v[0]=='arc')}, polyLine {sum(1 for v in E.values() if v[0]=='polyLine')}); merge tol {tol:.3e} m")
    print(f"  OF counts: nPoints {c['nPoints']} nCells {c['nCells']} nFaces {c['nFaces']} nInternalFaces {c['nInternalFaces']}  patches " +
          ", ".join(f"{k} ({v[0]}) {v[1]}" for k, v in c["patches"].items()))
    print(f"  patch faces pointing toward their block centre (geometric inward): {len(c['inward'])} {c['inward'][:3]}")
    print(f"  metrics: nonOrth max {m['nomax']:.4f} avg {m['noavg']:.4f} (>25: {m['n25']})  skew max {m['skmax']:.4f}  cell area {abs(A).min():.6e} .. {abs(A).max():.6e} m2  max edge {m['elen_max']:.6e} m")
    np.save(path + ".roundtrip_points.npy", Pu)
    for flag in ("--ref", "--real"):
        if flag in a:
            f = a[a.index(flag) + 1]
            if f.endswith(".npy"): R = np.load(f)
            else:   # OpenFOAM ascii points file, keep the plane of the first block's k=0 face
                txt = open(f).read(); body = txt[txt.index("(", txt.index("\n", txt.index("}")) ) :]
                R3 = np.array(re.findall(r"\(\s*(" + FLOAT + r")\s+(" + FLOAT + r")\s+(" + FLOAT + r")\s*\)", body), float)
                R = R3[np.abs(R3[:, drop] - z0) < 1e-12][:, [k for k in range(3) if k != drop]]
            d1, _ = cKDTree(R).query(Pu); d2, _ = cKDTree(Pu).query(R)
            print(f"  vs {flag} {f.rsplit('/',1)[-1]}: n {len(R)} vs {len(Pu)}; max |dx| replica->ref {d1.max():.3e} m, ref->replica {d2.max():.3e} m")

if __name__ == "__main__":
    main()
