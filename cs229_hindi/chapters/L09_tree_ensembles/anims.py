"""Lecture 9: Decision Trees and Ensemble Methods: 3D animations."""
import numpy as np

from lib.anim import C, CLASS_COLORS, animation, ease, grid, lerp, plane, surface

_LIM = (-1.0, 1.0)


def _stems(ax, x1, x2, z0, z1, **kw):
    """Vertical line segments from (x1, x2, z0) to (x1, x2, z1) as one artist."""
    nan = np.full(len(x1), np.nan)
    return ax.plot(np.c_[x1, x1, nan].ravel(), np.c_[x2, x2, nan].ravel(),
                   np.c_[z0, z1, nan].ravel(), **kw)


def _best_reg_split(X, y, idx, min_leaf=2):
    """Find best (feat, thr, gain) reducing SSE on subset idx."""
    if len(idx) < 2 * min_leaf:
        return None
    Xi, yi = X[idx], y[idx]
    base_sse = np.sum((yi - yi.mean()) ** 2)
    best = None
    for j in range(X.shape[1]):
        order = np.argsort(Xi[:, j])
        xj, yj = Xi[order, j], yi[order]
        csum = np.cumsum(yj)
        csum2 = np.cumsum(yj ** 2)
        total, total2, m = csum[-1], csum2[-1], len(yj)
        for k in range(min_leaf - 1, m - min_leaf):
            if xj[k] == xj[k + 1]:
                continue
            nl, nr = k + 1, m - (k + 1)
            sl, sr = csum[k], total - csum[k]
            sse = (csum2[k] - sl * sl / nl) + ((total2 - csum2[k]) - sr * sr / nr)
            gain = base_sse - sse
            if best is None or gain > best[0]:
                best = (gain, j, 0.5 * (xj[k] + xj[k + 1]))
    return best


def _fit_reg_tree_depth(X, y, max_depth=2, min_leaf=3):
    """Standard recursive CART regression tree up to max_depth."""
    def build(idx, depth):
        val = float(y[idx].mean())
        if depth >= max_depth:
            return ("leaf", val)
        sp = _best_reg_split(X, y, idx, min_leaf=min_leaf)
        if sp is None or sp[0] <= 1e-12:
            return ("leaf", val)
        _, j, thr = sp
        left = idx[X[idx, j] <= thr]
        right = idx[X[idx, j] > thr]
        return ("node", j, thr, build(left, depth + 1), build(right, depth + 1))

    return build(np.arange(len(y)), 0)


def _pred_tree(tree, X):
    out = np.empty(len(X), dtype=float)

    def walk(node, mask):
        if not np.any(mask):
            return
        if node[0] == "leaf":
            out[mask] = node[1]
            return
        _, j, thr, left, right = node
        go_left = mask & (X[:, j] <= thr)
        walk(left, go_left)
        walk(right, mask & ~go_left)

    walk(tree, np.ones(len(X), dtype=bool))
    return out


# =====================================================================
# Animation 1: Regression tree piecewise surface refining toward target
# =====================================================================
@animation("Regression tree refining surface",
           caption=r"""स्लेटी जाली असली smooth function $h^\star(x_1, x_2)$ है और काले बिंदु noisy training data हैं।
           शुरू में सिर्फ़ $1$ leaf है, इसलिए prediction पूरे क्षेत्र पर एक सपाट औसत सतह है। हर कदम पर CART सबसे ज़्यादा
           squared error घटाने वाला axis-aligned split चुनता है और एक सीढ़ी को दो नई ऊँचाइयों में तोड़ देता है। जैसे-जैसे
           leaves बढ़ते हैं, यह सीढ़ीदार (piecewise-constant) सतह असली मुड़ी हुई सतह का आकार पकड़ती जाती है।""",
           frames=42, keys=(0.0, 0.45, 1.0))
def reg_tree_refine(s, n):
    rng = np.random.default_rng(4)
    f_true = lambda a, b: 1.05 * np.sin(2.1 * a) + 0.85 * np.cos(2.2 * b)
    c = (np.arange(9) + 0.5) / 9 * 2 - 1
    A, B = np.meshgrid(c, c)
    x1 = np.clip(A.ravel() + rng.uniform(-0.09, 0.09, 81), -0.98, 0.98)
    x2 = np.clip(B.ravel() + rng.uniform(-0.09, 0.09, 81), -0.98, 0.98)
    y = f_true(x1, x2) + rng.normal(0, 0.16, len(x1))
    X = np.c_[x1, x2]

    G1, G2 = grid(_LIM, _LIM, 36)
    GX = np.c_[G1.ravel(), G2.ravel()]

    # Best-first leaf splitting so each stage adds concrete visible steps
    leaves = [{"idx": np.arange(len(y)), "gmask": np.ones(len(GX), dtype=bool)}]
    stages_keep = [1, 2, 3, 4, 6, 9, 13, 18, 25]
    surfs, preds, counts = [], [], []

    for step in range(1, max(stages_keep) + 1):
        if step in stages_keep:
            zg = np.empty(len(GX))
            zp = np.empty(len(y))
            for lf in leaves:
                val = y[lf["idx"]].mean()
                zg[lf["gmask"]] = val
                zp[lf["idx"]] = val
            surfs.append(zg.reshape(G1.shape))
            preds.append(zp)
            counts.append(len(leaves))
        if step == max(stages_keep):
            break
        best_leaf, best_sp = -1, None
        for li, lf in enumerate(leaves):
            sp = _best_reg_split(X, y, lf["idx"], min_leaf=2)
            if sp is not None and (best_sp is None or sp[0] > best_sp[0]):
                best_leaf, best_sp = li, sp
        if best_sp is None:
            break
        lf = leaves.pop(best_leaf)
        _, j, thr = best_sp
        idx = lf["idx"]
        gm = lf["gmask"]
        leaves.append({"idx": idx[X[idx, j] <= thr], "gmask": gm & (GX[:, j] <= thr)})
        leaves.append({"idx": idx[X[idx, j] > thr], "gmask": gm & (GX[:, j] > thr)})

    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$y$"), elev=27, azim=-58,
                xlim=_LIM, ylim=_LIM, zlim=(-2.3, 2.3), ticks=False)
    surface(ax, f_true, _LIM, _LIM, n=28, wire=True, alpha=0.45, color=C["dark"])
    ax.scatter(x1, x2, y, color=C["dark"], s=11, depthshade=False)

    def update(i):
        s.clear()
        t = i / (n - 1) * (len(surfs) - 1)
        k = min(int(t), len(surfs) - 2)
        u = ease((t - k) / 0.78)
        Z = lerp(surfs[k], surfs[k + 1], u)
        P = lerp(preds[k], preds[k + 1], u)
        s.add(ax.plot_surface(G1, G2, Z, cmap="viridis", vmin=-1.9, vmax=1.9,
                              alpha=0.48, linewidth=0, rstride=1, cstride=1))
        s.add(_stems(ax, x1, x2, y, P, color=C["red"], lw=0.75, alpha=0.7))
        j = k + (u > 0.5)
        mse = float(np.mean((preds[j] - y) ** 2))
        s.status(f"leaves = {counts[j]}    train MSE = {mse:.3f}")
        s.spin(i, n, sweep=42)

    return update


# =====================================================================
# Animation 2: Axis-aligned split planes carving 3D space into boxes
# =====================================================================
def _gini(labels, n_classes=3):
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels, minlength=n_classes) / len(labels)
    return float(1.0 - np.sum(counts ** 2))


def _box_edges(ax, box, color=C["gray"], lw=0.75, alpha=0.65):
    (x0, x1), (y0, y1), (z0, z1) = box
    segs = [
        ([x0, x1], [y0, y0], [z0, z0]), ([x0, x1], [y1, y1], [z0, z0]),
        ([x0, x1], [y0, y0], [z1, z1]), ([x0, x1], [y1, y1], [z1, z1]),
        ([x0, x0], [y0, y1], [z0, z0]), ([x1, x1], [y0, y1], [z0, z0]),
        ([x0, x0], [y0, y1], [z1, z1]), ([x1, x1], [y0, y1], [z1, z1]),
        ([x0, x0], [y0, y0], [z0, z1]), ([x1, x1], [y0, y0], [z0, z1]),
        ([x0, x0], [y1, y1], [z0, z1]), ([x1, x1], [y1, y1], [z0, z1]),
    ]
    xs, ys, zs = [], [], []
    for a, b, c in segs:
        xs += [a[0], a[1], np.nan]
        ys += [b[0], b[1], np.nan]
        zs += [c[0], c[1], np.nan]
    return ax.plot(xs, ys, zs, color=color, lw=lw, alpha=alpha)


def _split_patch(ax, box, axis, thr, frac, color, alpha):
    """Draw axis-aligned split plane inside parent box, grown by frac in [0, 1]."""
    b = [list(r) for r in box]
    other = [d for d in range(3) if d != axis]
    # Grow along the first non-split coordinate for smooth carving effect
    d_grow = other[0]
    lo, hi = b[d_grow]
    b[d_grow][1] = lo + (hi - lo) * frac
    u = np.linspace(b[other[0]][0], b[other[0]][1], 4)
    v = np.linspace(b[other[1]][0], b[other[1]][1], 4)
    U, V = np.meshgrid(u, v)
    W = np.full_like(U, thr)
    coords = [None, None, None]
    coords[axis] = W
    coords[other[0]] = U
    coords[other[1]] = V
    return ax.plot_surface(coords[0], coords[1], coords[2], color=color,
                           alpha=alpha, linewidth=0.6, edgecolor=color, shade=False)


@animation("Axis-aligned splits in 3D space",
           caption=r"""यहाँ तीनों axes input features $(x_1, x_2, x_3)$ हैं और तीन रंग तीन classes दिखाते हैं।
           CART classification tree हर कदम पर सबसे ज़्यादा Gini impurity घटाने वाला feature $j$ और threshold $t$ चुनता है।
           हर नया split $x_j = t$ एक सपाट दीवार की तरह अपने parent डिब्बे को दो छोटे axis-aligned डिब्बों में काटता है।
           कुछ ही splits में पूरा 3D घन लगभग शुद्ध रंग वाले बक्सों में बँट जाता है।""",
           frames=42, keys=(0.12, 0.52, 1.0))
def splits_3d_boxes(s, n):
    rng = np.random.default_rng(12)
    centers = [(-0.52, -0.48, -0.45, 0), (0.52, -0.45, 0.48, 1),
               (-0.48, 0.52, 0.45, 2), (0.50, 0.50, -0.45, 0),
               (0.48, -0.50, -0.48, 2), (-0.48, -0.45, 0.50, 1),
               (0.48, 0.48, 0.48, 2)]
    pts, labs = [], []
    for cx, cy, cz, cl in centers:
        p = rng.normal([cx, cy, cz], 0.24, size=(18, 3))
        pts.append(np.clip(p, -0.94, 0.94))
        labs.append(np.full(18, cl))
    X = np.vstack(pts)
    y = np.concatenate(labs)

    # Run real 3D Gini CART best-first for 6 splits
    root_box = ((-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0))
    regions = [{"idx": np.arange(len(y)), "box": root_box}]
    splits = []
    gini_hist = [_gini(y)]
    box_hist = [[root_box]]

    for _ in range(6):
        best_ri, best_sp = -1, None
        for ri, reg in enumerate(regions):
            idx = reg["idx"]
            if len(idx) < 6 or _gini(y[idx]) < 1e-6:
                continue
            g0 = _gini(y[idx])
            for j in range(3):
                vals = np.sort(np.unique(X[idx, j]))
                for thr in 0.5 * (vals[:-1] + vals[1:]):
                    lmask = X[idx, j] <= thr
                    nl, nr = int(lmask.sum()), int((~lmask).sum())
                    if nl < 2 or nr < 2:
                        continue
                    gain = (len(idx) / len(y)) * (
                        g0 - (nl / len(idx)) * _gini(y[idx[lmask]])
                        - (nr / len(idx)) * _gini(y[idx[~lmask]])
                    )
                    if best_sp is None or gain > best_sp[0]:
                        best_sp = (gain, j, float(thr))
                        best_ri = ri
        if best_sp is None:
            break
        reg = regions.pop(best_ri)
        _, j, thr = best_sp
        pbox = reg["box"]
        idx = reg["idx"]
        b_left = tuple((lo, thr) if d == j else (lo, hi) for d, (lo, hi) in enumerate(pbox))
        b_right = tuple((thr, hi) if d == j else (lo, hi) for d, (lo, hi) in enumerate(pbox))
        regions.append({"idx": idx[X[idx, j] <= thr], "box": b_left})
        regions.append({"idx": idx[X[idx, j] > thr], "box": b_right})
        splits.append((pbox, j, thr))
        wg = sum(len(r["idx"]) / len(y) * _gini(y[r["idx"]]) for r in regions)
        gini_hist.append(wg)
        box_hist.append([r["box"] for r in regions])

    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$x_3$"), elev=24, azim=-54,
                xlim=_LIM, ylim=_LIM, zlim=_LIM, ticks=False)
    _box_edges(ax, root_box, color=C["dark"], lw=0.9, alpha=0.55)
    for cl, name in enumerate(("class 0", "class 1", "class 2")):
        m = y == cl
        ax.scatter(X[m, 0], X[m, 1], X[m, 2], color=CLASS_COLORS[cl], s=16,
                   depthshade=False, label=name)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    plane_cols = [C["teal"], C["purple"], C["orange"], C["blue"], C["pink"], C["green"]]

    def update(i):
        s.clear()
        total_splits = len(splits)
        pos = (i / (n - 1)) * total_splits
        done = min(int(pos), total_splits)
        frac = ease(min(1.0, (pos - done) / 0.75)) if done < total_splits else 1.0

        for k in range(done):
            pbox, j, thr = splits[k]
            s.add(_split_patch(ax, pbox, j, thr, 1.0, plane_cols[k], alpha=0.22))
        if done < total_splits and frac > 0.02:
            pbox, j, thr = splits[done]
            s.add(_split_patch(ax, pbox, j, thr, frac, plane_cols[done], alpha=0.38))

        cur_stage = done + (1 if (done < total_splits and frac > 0.65) else 0)
        for bx in box_hist[cur_stage]:
            s.add(_box_edges(ax, bx, color=C["gray"], lw=0.55, alpha=0.45))

        s.status(f"splits: {cur_stage}   boxes: {cur_stage + 1}   Gini = {gini_hist[cur_stage]:.3f}")
        s.spin(i, n, sweep=48)

    return update


# =====================================================================
# Animation 3: Bagging smoothing a jagged tree surface
# =====================================================================
@animation("Bagging smooths jagged trees",
           caption=r"""अकेला गहरा regression tree ($M = 1$) noisy बिंदुओं को रटने के चक्कर में बेहद उबड़-खाबड़ और सीढ़ीदार सतह
           बनाता है (high variance)। हर frame में हम एक नया bootstrap sample खींचकर उस पर उगाया गहरा tree औसत में जोड़ते
           हैं। जैसे-जैसे $M$ बढ़ता है, अलग-अलग पेड़ों की बेतरतीब उछालें एक-दूसरे को काट देती हैं। औसत सतह चिकनी होकर
           स्लेटी जाली ($h^\star$) के करीब आ जाती है और test MSE साफ़ गिरता है।""",
           frames=40, keys=(0.0, 0.35, 1.0))
def bagging_smooth(s, n):
    rng = np.random.default_rng(21)
    f_true = lambda a, b: np.sin(2.2 * a) + 0.85 * np.cos(2.2 * b)
    m = 70
    x1, x2 = rng.uniform(-1, 1, m), rng.uniform(-1, 1, m)
    y = f_true(x1, x2) + rng.normal(0, 0.38, m)
    X = np.c_[x1, x2]

    G1, G2 = grid(_LIM, _LIM, 36)
    GX = np.c_[G1.ravel(), G2.ravel()]
    Z_true = f_true(G1, G2)

    max_trees = 40
    tree_preds = []
    for _ in range(max_trees):
        boot = rng.integers(0, m, size=m)
        tr = _fit_reg_tree_depth(X[boot], y[boot], max_depth=6, min_leaf=2)
        tree_preds.append(_pred_tree(tr, GX).reshape(G1.shape))
    tree_preds = np.array(tree_preds)
    cum_avg = np.cumsum(tree_preds, axis=0) / np.arange(1, max_trees + 1)[:, None, None]

    # Map frames smoothly across M = 1 .. 40 (spend more frames early where change is dramatic)
    counts = np.unique(np.round(1 + (max_trees - 1) * (np.linspace(0, 1, n) ** 1.5)).astype(int))
    while len(counts) < n:
        counts = np.round(1 + (max_trees - 1) * (np.linspace(0, 1, n) ** 1.35)).astype(int)
        break

    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$\hat{h}_{\mathrm{bag}}(x)$"), elev=27, azim=-60,
                xlim=_LIM, ylim=_LIM, zlim=(-2.3, 2.3), ticks=False)
    surface(ax, f_true, _LIM, _LIM, n=28, wire=True, alpha=0.45, color=C["dark"])
    ax.scatter(x1, x2, y, color=C["dark"], s=10, depthshade=False)

    def update(i):
        s.clear()
        M = int(counts[i])
        Z = cum_avg[M - 1]
        s.add(ax.plot_surface(G1, G2, Z, cmap="viridis", vmin=-1.9, vmax=1.9,
                              alpha=0.48, linewidth=0, rstride=1, cstride=1))
        mse = float(np.mean((Z - Z_true) ** 2))
        s.status(f"bagged trees M = {M}    MSE vs h* = {mse:.3f}")
        s.spin(i, n, sweep=38)

    return update


# =====================================================================
# Animation 4: AdaBoost sample-weight stems growing on misclassified points
# =====================================================================
def _fit_stump(X, y, w):
    """Best weighted decision stump returning (feat, thr, polarity, err, pred)."""
    best = None
    for j in range(X.shape[1]):
        vals = np.sort(np.unique(X[:, j]))
        thrs = np.r_[vals[0] - 0.1, 0.5 * (vals[:-1] + vals[1:]), vals[-1] + 0.1]
        for thr in thrs:
            for pol in (1.0, -1.0):
                pred = np.where(X[:, j] <= thr, pol, -pol)
                err = float(np.sum(w[pred != y]))
                if best is None or err < best[3]:
                    best = (j, float(thr), pol, err, pred)
    return best


@animation("AdaBoost sample weights growing",
           caption=r"""फ़र्श ($z = 0$) पर दो मुड़ी हुई classes के बिंदु हैं और खड़ी डंडियों की ऊँचाई हर बिंदु का
           scaled sample weight $m \cdot w_i^{(t)}$ दिखाती है। शुरू में सब डंडियाँ बराबर ($1$) हैं। हर round में एक
           सरल decision stump (टूटी रेखा) लगता है: जिन बिंदुओं पर गलती होती है (नारंगी), उनका weight $e^{+\alpha_t}$ से
           उछलता है और आसान बिंदुओं का घटता है। फ़र्श पर काली boundary सारे stumps के $\alpha_t$-weighted वोट से मुड़कर
           पूरे डेटा को अलग कर देती है।""",
           frames=40, keys=(0.0, 0.45, 1.0))
def adaboost_weights(s, n):
    rng = np.random.default_rng(9)
    m_half = 30
    ang0 = np.linspace(0.05 * np.pi, 0.95 * np.pi, m_half)
    ang1 = np.linspace(0.05 * np.pi, 0.95 * np.pi, m_half)
    X0 = np.c_[np.cos(ang0), np.sin(ang0)] + rng.normal(0, 0.14, (m_half, 2))
    X1 = np.c_[1.0 - np.cos(ang1), 0.62 - np.sin(ang1)] + rng.normal(0, 0.13, (m_half, 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.ones(m_half), -np.ones(m_half)]
    m = len(y)

    xlim, ylim = (-1.35, 2.25), (-0.95, 1.35)
    G1, G2 = grid(xlim, ylim, 36)
    GX = np.c_[G1.ravel(), G2.ravel()]

    rounds = 26
    w = np.full(m, 1.0 / m)
    F_grid = np.zeros(len(GX))
    F_tr = np.zeros(m)
    hist = []

    for _ in range(rounds):
        j, thr, pol, err, pred = _fit_stump(X, y, w)
        err = np.clip(err, 1e-6, 1 - 1e-6)
        alpha = 0.5 * np.log((1.0 - err) / err)
        F_tr = F_tr + alpha * pred
        gpred = np.where(GX[:, j] <= thr, pol, -pol)
        F_grid = F_grid + alpha * gpred
        mis = pred != y
        hist.append({
            "w_scaled": (m * w).copy(),
            "mis": mis.copy(),
            "j": j,
            "thr": thr,
            "alpha": alpha,
            "F_grid": F_grid.reshape(G1.shape).copy(),
            "acc": float(np.mean(np.sign(F_tr) == y)),
        })
        w = w * np.exp(-alpha * y * pred)
        w /= w.sum()

    zmax = 5.2
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$m \cdot w_i$"), elev=30, azim=-56,
                xlim=xlim, ylim=ylim, zlim=(0, zmax), ticks=False)
    plane(ax, lambda a, b: 0 * a, xlim, ylim, color=C["gray"], alpha=0.12)
    pos, neg = y == 1, y == -1
    ax.scatter(X[pos, 0], X[pos, 1], np.zeros(pos.sum()), color=C["blue"], s=14,
               depthshade=False, label="class +1")
    ax.scatter(X[neg, 0], X[neg, 1], np.zeros(neg.sum()), color=C["red"], s=14,
               depthshade=False, label="class -1")
    ax.plot([], [], [], color=C["orange"], lw=2, label="misclassified by stump")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        r = min(int(round(i / (n - 1) * (rounds - 1))), rounds - 1)
        st = hist[r]
        ws = np.clip(st["w_scaled"], 0.0, zmax - 0.15)
        mis = st["mis"]
        ok = ~mis

        if np.any(ok):
            s.add(_stems(ax, X[ok, 0], X[ok, 1], np.zeros(ok.sum()), ws[ok],
                         color=C["teal"], lw=1.2, alpha=0.75))
            s.add(ax.scatter(X[ok, 0], X[ok, 1], ws[ok], color=C["teal"], s=20, depthshade=False))
        if np.any(mis):
            s.add(_stems(ax, X[mis, 0], X[mis, 1], np.zeros(mis.sum()), ws[mis],
                         color=C["orange"], lw=1.9, alpha=0.95))
            s.add(ax.scatter(X[mis, 0], X[mis, 1], ws[mis], color=C["orange"], s=34,
                             edgecolor="white", linewidth=0.5, depthshade=False))

        # Current stump line on floor z = 0
        if st["j"] == 0:
            s.add(ax.plot([st["thr"], st["thr"]], list(ylim), [0, 0],
                          color=C["purple"], lw=1.3, ls="--"))
        else:
            s.add(ax.plot(list(xlim), [st["thr"], st["thr"]], [0, 0],
                          color=C["purple"], lw=1.3, ls="--"))

        # Ensemble decision boundary F(x) = 0 on floor
        Fg = st["F_grid"]
        if Fg.min() < 0 < Fg.max():
            s.add(ax.contour(G1, G2, Fg, levels=[0.0], zdir="z", offset=0.01,
                             colors=[C["dark"]], linewidths=1.8))

        s.status(f"round {r + 1}   alpha={st['alpha']:.2f}   train acc={st['acc']:.0%}")
        s.spin(i, n, sweep=36)

    return update


# =====================================================================
# Animation 5: Gradient boosting residual surface flattening
# =====================================================================
@animation("Boosting flattens residual surface",
           caption=r"""यहाँ ऊँचाई खुद target नहीं बल्कि बची हुई गलती यानी **residual** $y - F_t(x)$ है, और बीच का स्लेटी
           समतल शून्य गलती ($z = 0$) दिखाता है। $t = 0$ पर $F_0(x) = \bar{y}$ है, इसलिए लाल पहाड़ और नीली घाटियाँ बहुत
           गहरी हैं। हर round में depth-2 का एक छोटा regression tree इन्हीं residuals पर fit होता है और $\nu h_t(x)$
           घटा देता है। देखते ही देखते पूरी residual सतह दबकर $z = 0$ के सपाट तल पर बैठ जाती है।""",
           frames=42, keys=(0.0, 0.38, 1.0))
def gbm_residuals_flatten(s, n):
    rng = np.random.default_rng(15)
    f_true = lambda a, b: 1.15 * np.sin(2.1 * a) + 0.9 * np.cos(2.3 * b)
    c = (np.arange(8) + 0.5) / 8 * 2 - 1
    A, B = np.meshgrid(c, c)
    x1 = np.clip(A.ravel() + rng.uniform(-0.08, 0.08, 64), -0.97, 0.97)
    x2 = np.clip(B.ravel() + rng.uniform(-0.08, 0.08, 64), -0.97, 0.97)
    y = f_true(x1, x2) + rng.normal(0, 0.12, 64)
    X = np.c_[x1, x2]

    G1, G2 = grid(_LIM, _LIM, 34)
    GX = np.c_[G1.ravel(), G2.ravel()]
    Z_true = f_true(G1, G2)

    lr = 0.25
    n_rounds = 26
    F_tr = np.full(len(y), y.mean())
    F_gr = np.full(len(GX), y.mean())
    res_surfs = [(Z_true - F_gr.reshape(G1.shape)).copy()]
    res_pts = [(y - F_tr).copy()]

    for _ in range(n_rounds):
        r_cur = y - F_tr
        tr = _fit_reg_tree_depth(X, r_cur, max_depth=2, min_leaf=3)
        F_tr = F_tr + lr * _pred_tree(tr, X)
        F_gr = F_gr + lr * _pred_tree(tr, GX)
        res_surfs.append((Z_true - F_gr.reshape(G1.shape)).copy())
        res_pts.append((y - F_tr).copy())

    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$y - F_t(x)$"), elev=25, azim=-60,
                xlim=_LIM, ylim=_LIM, zlim=(-2.1, 2.1), ticks=False)
    plane(ax, lambda a, b: 0 * a, _LIM, _LIM, color=C["gray"], alpha=0.18)

    def update(i):
        s.clear()
        pos = (i / (n - 1)) * n_rounds
        k = min(int(pos), n_rounds - 1)
        u = ease(pos - k)
        R_surf = lerp(res_surfs[k], res_surfs[k + 1], u)
        R_pt = lerp(res_pts[k], res_pts[k + 1], u)

        s.add(ax.plot_surface(G1, G2, R_surf, cmap="coolwarm", vmin=-1.7, vmax=1.7,
                              alpha=0.47, linewidth=0, rstride=1, cstride=1))
        s.add(_stems(ax, x1, x2, np.zeros_like(R_pt), R_pt, color=C["dark"], lw=0.8, alpha=0.7))
        s.add(ax.scatter(x1, x2, R_pt, color=C["dark"], s=11, depthshade=False))
        rd = int(round(pos))
        rms = float(np.sqrt(np.mean(res_pts[rd] ** 2)))
        s.status(f"round t = {rd}    residual RMS = {rms:.3f}")
        s.spin(i, n, sweep=40)

    return update
