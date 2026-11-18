"""Lecture 12: Practical Advice for ML Projects & Evaluation Metrics: 3D animations."""
import numpy as np

from lib.anim import C, animation, ease, grid, lerp, plane, surface


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))


def _fit_logreg(X, y, lam=0.05, steps=35):
    """Fit L2-regularized logistic regression with Newton steps from scratch."""
    Xb = np.c_[np.ones(len(X)), X]
    th = np.zeros(Xb.shape[1])
    reg = np.diag([0.0] + [lam] * X.shape[1])
    for _ in range(steps):
        p = _sigmoid(Xb @ th)
        w = np.maximum(p * (1.0 - p), 1e-6)
        grad = Xb.T @ (p - y) / len(y) + reg @ th
        H = (Xb.T * w) @ Xb / len(y) + reg
        th -= np.linalg.solve(H, grad)
    return th


def _predict_prob(th, X):
    return _sigmoid(np.c_[np.ones(len(X)), X] @ th)


def _curtain(ax, xs, y_const, zs, color, alpha=0.42):
    """Vertical translucent 3D curtain under (xs, y_const, zs)."""
    if len(xs) < 2:
        return None
    X = np.vstack([xs, xs])
    Y = np.full_like(X, y_const, dtype=float)
    Z = np.vstack([np.zeros_like(zs), zs])
    return ax.plot_surface(X, Y, Z, color=color, alpha=alpha, linewidth=0, shade=False)


# ---------------------------------------------------------------------------
# 1. Training and test error surfaces over (training size m, complexity d)
# ---------------------------------------------------------------------------
@animation("Error surfaces: size vs complexity",
           caption=r"""यहाँ ज़मीन पर training size $m$ और polynomial degree $d$ हैं, जबकि ऊँचाई MSE दिखाती है।
           नीली पारदर्शी सतह training error है और लाल सतह test error है। जैसे-जैसे स्लाइस $d=1$ (high bias) से
           $d=9$ (high variance) तक चलती है, देखिए कि कम complexity पर दोनों curves पास-पास और ऊँचे रहते हैं।
           ऊँची complexity और छोटे $m$ पर दोनों के बीच भारी generalization gap खुलता है, जो $m$ बढ़ने पर सिकुड़ता है।""",
           frames=42, keys=(0.0, 0.38, 1.0))
def error_surfaces(s, n):
    rng = np.random.default_rng(12)
    f_true = lambda x: np.sin(2.6 * x) + 0.45 * np.cos(4.8 * x)
    noise = 0.28

    x_te = np.linspace(-1.0, 1.0, 800)
    y_te = f_true(x_te) + rng.normal(0.0, noise, len(x_te))

    ms = np.linspace(14, 110, 15).astype(int)
    ds = np.arange(1, 10)
    V_te_full = np.polynomial.legendre.legvander(x_te, ds[-1])

    tr_surf = np.zeros((len(ds), len(ms)))
    te_surf = np.zeros((len(ds), len(ms)))
    trials = 40
    for j, m in enumerate(ms):
        for _ in range(trials):
            x_tr = rng.uniform(-1.0, 1.0, m)
            y_tr = f_true(x_tr) + rng.normal(0.0, noise, m)
            V_tr_full = np.polynomial.legendre.legvander(x_tr, ds[-1])
            for i_d, d in enumerate(ds):
                A = V_tr_full[:, : d + 1]
                th = np.linalg.solve(A.T @ A + 1e-4 * np.eye(d + 1), A.T @ y_tr)
                tr_surf[i_d, j] += np.mean((A @ th - y_tr) ** 2)
                te_surf[i_d, j] += np.mean((V_te_full[:, : d + 1] @ th - y_te) ** 2)
    tr_surf = np.clip(tr_surf / trials, 0.0, 1.15)
    te_surf = np.clip(te_surf / trials, 0.0, 1.15)

    M, D = np.meshgrid(ms, ds)
    ax = s.ax3d(labels=("train size m", "degree d", "MSE"), elev=25, azim=-132,
                xlim=(14, 110), ylim=(1, 9), zlim=(0, 1.15))
    ax.plot_surface(M, D, tr_surf, color=C["blue"], alpha=0.30, linewidth=0, shade=True)
    ax.plot_surface(M, D, te_surf, color=C["red"], alpha=0.32, linewidth=0, shade=True)
    ax.plot([], [], [], color=C["blue"], lw=2.2, label="train error")
    ax.plot([], [], [], color=C["red"], lw=2.2, label="test error")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    best_i, best_j = np.unravel_index(np.argmin(te_surf), te_surf.shape)
    ax.scatter([ms[best_j]], [ds[best_i]], [te_surf[best_i, best_j]],
               marker="*", s=95, color=C["green"], depthshade=False)

    def update(i):
        s.clear()
        pos = (i / (n - 1)) * (len(ds) - 1)
        k = min(int(pos), len(ds) - 2)
        u = pos - k
        d_val = lerp(ds[k], ds[k + 1], u)
        tr_c = lerp(tr_surf[k], tr_surf[k + 1], u)
        te_c = lerp(te_surf[k], te_surf[k + 1], u)
        ys = np.full_like(ms, d_val, dtype=float)

        s.add(ax.plot(ms, ys, tr_c, color=C["blue"], lw=2.6))
        s.add(ax.plot(ms, ys, te_c, color=C["red"], lw=2.6))
        nan = np.full(len(ms), np.nan)
        s.add(ax.plot(np.c_[ms, ms, nan].ravel(), np.c_[ys, ys, nan].ravel(),
                      np.c_[tr_c, te_c, nan].ravel(), color=C["purple"], lw=1.0, alpha=0.8))
        tag = "high bias" if d_val < 2.6 else ("sweet spot" if d_val < 5.3 else "high var")
        s.status(f"d={d_val:.1f} ({tag})  m=110 test={te_c[-1]:.2f}")
        s.spin(i, n, sweep=36)

    return update


# ---------------------------------------------------------------------------
# 2. Class-score distributions with threshold sweep & confusion 3D bars
# ---------------------------------------------------------------------------
@animation("Score distributions & confusion 3D",
           caption=r"""आगे की दो पट्टियाँ Logistic Regression के predicted scores $s(x)$ का बंटवारा दिखाती हैं:
           $y=0$ (आगे) और $y=1$ (बीच में)। पारदर्शी दीवार threshold $\tau$ है जो बाएँ से दाएँ खिसकती है।
           पीछे खड़े चार 3D खंभे उसी पल के **TN** (नीला), **FP** (लाल), **FN** (नारंगी), और **TP** (हरा)
           दिखाते हैं। देखिए कैसे $\tau$ बढ़ाने पर झूठे अलार्म (FP) घटते हैं पर छूटे हुए मामले (FN) बढ़ जाते हैं।""",
           frames=40, keys=(0.12, 0.5, 0.88))
def score_threshold_bars(s, n):
    rng = np.random.default_rng(21)
    m0, m1 = 130, 80
    X0 = rng.normal([-0.75, -0.45], [0.85, 0.85], size=(m0, 2))
    X1 = rng.normal([0.85, 0.65], [0.85, 0.85], size=(m1, 2))
    X = np.vstack([X0, X1])
    y = np.r_[np.zeros(m0), np.ones(m1)]
    th = _fit_logreg(X, y, lam=0.04)
    sc0, sc1 = _predict_prob(th, X0), _predict_prob(th, X1)

    sg = np.linspace(0.01, 0.99, 90)
    bw = 0.085
    d0 = np.mean(np.exp(-0.5 * ((sg[:, None] - sc0[None, :]) / bw) ** 2), axis=1)
    d1 = np.mean(np.exp(-0.5 * ((sg[:, None] - sc1[None, :]) / bw) ** 2), axis=1)
    scale = 0.88 / max(d0.max(), d1.max())
    d0, d1 = d0 * scale, d1 * scale

    ax = s.ax3d(labels=("score s(x)", "", "scaled count"), elev=26, azim=-62,
                xlim=(0, 1), ylim=(-0.2, 2.7), zlim=(0, 1.02))
    ax.set_yticks([0.0, 0.95, 2.15])
    ax.set_yticklabels(["y=0", "y=1", "counts"])
    ax.plot(sg, np.zeros_like(sg), d0, color=C["dark"], lw=1.2, alpha=0.7)
    ax.plot(sg, np.full_like(sg, 0.95), d1, color=C["dark"], lw=1.2, alpha=0.7)

    bx = [0.06, 0.30, 0.54, 0.78]
    bcols = [C["blue"], C["red"], C["orange"], C["green"]]
    bnames = ["TN", "FP", "FN", "TP"]
    thrs = np.linspace(0.12, 0.88, n)

    def update(i):
        s.clear()
        tau = thrs[i]
        m_left = sg <= tau
        m_right = sg >= tau
        s.add(_curtain(ax, sg[m_left], 0.0, d0[m_left], C["blue"], 0.44))
        s.add(_curtain(ax, sg[m_right], 0.0, d0[m_right], C["red"], 0.44))
        s.add(_curtain(ax, sg[m_left], 0.95, d1[m_left], C["orange"], 0.44))
        s.add(_curtain(ax, sg[m_right], 0.95, d1[m_right], C["green"], 0.44))

        WY, WZ = np.meshgrid([-0.12, 1.15], [0.0, 0.92])
        WX = np.full_like(WY, tau)
        s.add(ax.plot_surface(WX, WY, WZ, color=C["dark"], alpha=0.22, linewidth=0, shade=False))
        s.add(ax.plot([tau, tau], [-0.12, 1.15], [0.92, 0.92], color=C["dark"], lw=1.8))

        tn = int(np.sum(sc0 < tau))
        fp = int(np.sum(sc0 >= tau))
        fn = int(np.sum(sc1 < tau))
        tp = int(np.sum(sc1 >= tau))
        counts = [tn, fp, fn, tp]
        for x0, cnt, col, nm in zip(bx, counts, bcols, bnames):
            h = max(0.015, cnt / 135.0)
            s.add(ax.bar3d(x0, 1.95, 0.0, 0.15, 0.36, h, color=col, alpha=0.82, shade=True))
            s.add(ax.text(x0 + 0.075, 2.13, h + 0.04, f"{nm}:{cnt}",
                          ha="center", fontsize=7.8, color=col, fontweight="bold"))
        s.status(f"thr={tau:.2f}  TP={tp} FP={fp} FN={fn} TN={tn}")
        s.spin(i, n, sweep=28)

    return update


# ---------------------------------------------------------------------------
# 3. ROC curve as a 3D trajectory over (FPR, TPR, threshold)
# ---------------------------------------------------------------------------
@animation("3D ROC curve over (FPR, TPR, thr)",
           caption=r"""यहाँ ROC curve को तीन आयामों $(\mathrm{FPR}, \mathrm{TPR}, \tau)$ में दिखाया गया है। ऊपर
           $\tau \approx 1$ पर बिंदु $(0, 0, 1)$ से शुरू होकर $\tau$ घटने के साथ $(1, 1, 0)$ तक उतरता है।
           फ़र्श ($z = 0$) पर इसकी परछाईं परिचित 2D ROC curve है और हरा क्षेत्रफल अब तक जुड़ा हुआ AUC दिखाता
           है। स्लेटी विकर्ण बिना सोचे-समझे सिक्का उछालने वाले classifier की रेखा है।""",
           frames=40, keys=(0.12, 0.5, 0.95))
def roc_3d_curve(s, n):
    rng = np.random.default_rng(33)
    m0, m1 = 260, 160
    X0 = rng.normal([-0.7, -0.5], [0.95, 0.9], size=(m0, 2))
    X1 = rng.normal([0.8, 0.75], [0.9, 0.95], size=(m1, 2))
    th = _fit_logreg(np.vstack([X0, X1]), np.r_[np.zeros(m0), np.ones(m1)], lam=0.05)
    s0, s1 = _predict_prob(th, X0), _predict_prob(th, X1)

    taus = np.linspace(0.995, 0.005, 160)
    fpr = np.array([np.mean(s0 >= t) for t in taus])
    tpr = np.array([np.mean(s1 >= t) for t in taus])
    taus_full = np.r_[1.0, taus, 0.0]
    fpr_full = np.r_[0.0, fpr, 1.0]
    tpr_full = np.r_[0.0, tpr, 1.0]
    auc_total = float(np.trapezoid(tpr_full, fpr_full))

    ax = s.ax3d(labels=("FPR", "TPR", r"threshold $\tau$"), elev=25, azim=-56,
                xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    u = np.linspace(0, 1, 20)
    ax.plot(u, u, np.zeros_like(u), color=C["gray"], ls="--", lw=1.0, label="random (0.50)")
    ax.plot(u, u, 1.0 - u, color=C["gray"], ls=":", lw=0.9, alpha=0.7)
    ax.plot(fpr_full, tpr_full, taus_full, color=C["blue"], lw=1.0, alpha=0.30)
    ax.plot(fpr_full, tpr_full, np.zeros_like(fpr_full), color=C["blue"], ls="--", lw=1.1, alpha=0.45)
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    idxs = np.linspace(2, len(taus_full) - 1, n).astype(int)

    def update(i):
        s.clear()
        k = idxs[i]
        fx, ty, tz = fpr_full[: k + 1], tpr_full[: k + 1], taus_full[: k + 1]
        s.add(ax.plot(fx, ty, tz, color=C["red"], lw=2.5))
        s.add(ax.plot(fx, ty, np.zeros_like(fx), color=C["teal"], lw=2.0))

        # Floor AUC ribbon from (fx, 0, 0) to (fx, ty, 0)
        FX = np.vstack([fx, fx])
        FY = np.vstack([np.zeros_like(ty), ty])
        FZ = np.zeros_like(FX)
        s.add(ax.plot_surface(FX, FY, FZ, color=C["green"], alpha=0.22, linewidth=0, shade=False))

        s.add(ax.plot([fx[-1], fx[-1]], [ty[-1], ty[-1]], [0.0, tz[-1]],
                      color=C["dark"], ls="--", lw=1.2))
        s.add(ax.scatter([fx[-1]], [ty[-1]], [tz[-1]], color=C["red"], s=52,
                         depthshade=False, edgecolor="white"))
        s.add(ax.scatter([fx[-1]], [ty[-1]], [0.0], color=C["teal"], s=34, depthshade=False))
        s.status(f"thr={tz[-1]:.2f}  FPR={fx[-1]:.2f}  TPR={ty[-1]:.2f}  AUC={auc_total:.2f}")
        s.spin(i, n, sweep=38)

    return update


# ---------------------------------------------------------------------------
# 4. F1 surface over (Precision, Recall) with classifier operating point
# ---------------------------------------------------------------------------
@animation("F1 surface & PR operating point",
           caption=r"""रंगीन सतह harmonic mean $F_1(P, R) = \frac{2PR}{P+R}$ है और ऊपर हल्की जाली साधारण औसत
           $\frac{P+R}{2}$ है। ध्यान दें कि अगर $P$ या $R$ में से एक भी शून्य के पास जाए तो $F_1$ सतह तेज़ी से
           फ़र्श पर गिर जाती है। असंतुलित data पर $\tau$ घटाने से लाल बिंदु इस सतह पर चढ़कर हरे तारे (अधिकतम
           $F_1$) तक पहुँचता है और फिर कम Precision के कारण नीचे उतर जाता है।""",
           frames=42, keys=(0.08, 0.52, 0.95))
def f1_surface_operating_point(s, n):
    rng = np.random.default_rng(44)
    m0, m1 = 920, 80                              # 8% imbalanced positive class
    X0 = rng.normal([-0.4, -0.3], [1.0, 1.0], size=(m0, 2))
    X1 = rng.normal([1.35, 1.20], [0.9, 0.9], size=(m1, 2))
    th = _fit_logreg(np.vstack([X0, X1]), np.r_[np.zeros(m0), np.ones(m1)], lam=0.03)
    sc0, sc1 = _predict_prob(th, X0), _predict_prob(th, X1)

    thrs = np.linspace(0.86, 0.03, n)
    ps, rs, f1s = [], [], []
    for t in thrs:
        tp = float(np.sum(sc1 >= t))
        fp = float(np.sum(sc0 >= t))
        fn = float(np.sum(sc1 < t))
        p = tp / max(1.0, tp + fp)
        r = tp / max(1.0, tp + fn)
        f1 = 2.0 * p * r / max(1e-8, p + r)
        ps.append(p)
        rs.append(r)
        f1s.append(f1)
    ps, rs, f1s = np.array(ps), np.array(rs), np.array(f1s)
    best = int(np.argmax(f1s))

    ax = s.ax3d(labels=("Precision P", "Recall R", r"$F_1$"), elev=27, azim=-128,
                xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    f1_fn = lambda P, R: 2.0 * P * R / np.maximum(1e-6, P + R)
    _, GX, GY, GZ = surface(ax, f1_fn, (0.02, 1.0), (0.02, 1.0), n=32, cmap="viridis", alpha=0.44)
    surface(ax, lambda P, R: 0.5 * (P + R), (0.02, 1.0), (0.02, 1.0),
            n=16, wire=True, alpha=0.22, color=C["gray"])
    ax.contour(GX, GY, GZ, levels=np.linspace(0.15, 0.9, 6), zdir="z", offset=0.0,
               cmap="viridis", linewidths=0.7, alpha=0.65)
    ax.scatter([ps[best]], [rs[best]], [f1s[best] + 0.02], marker="*", s=110,
               color=C["green"], depthshade=False)

    def update(i):
        s.clear()
        s.add(ax.plot(ps[: i + 1], rs[: i + 1], f1s[: i + 1] + 0.012, color=C["red"], lw=2.4))
        s.add(ax.plot(ps[: i + 1], rs[: i + 1], np.zeros(i + 1), color=C["orange"], ls="--", lw=1.4))
        s.add(ax.plot([ps[i], ps[i]], [rs[i], rs[i]], [0.0, f1s[i]], color=C["dark"], ls=":", lw=1.2))
        s.add(ax.scatter([ps[i]], [rs[i]], [f1s[i] + 0.015], color=C["red"], s=56,
                         depthshade=False, edgecolor="white"))
        s.status(f"thr={thrs[i]:.2f}  P={ps[i]:.2f}  R={rs[i]:.2f}  F1={f1s[i]:.2f}")
        s.spin(i, n, sweep=34)

    return update


# ---------------------------------------------------------------------------
# 5. Pipeline accuracy & F1 bars rising as components get ground truth
# ---------------------------------------------------------------------------
@animation("Pipeline ceiling analysis in 3D",
           caption=r"""चार-चरणीय ML pipeline में एक-एक करके हर हिस्से को $100\%$ सही ground truth से बदला जा रहा
           है। आगे की कतार pipeline की कुल **Accuracy** है और पीछे की कतार **$F_1$ score** है। पहला चरण
           (`+Pre`) बदलने पर खंभा मुश्किल से हिलता है, पर दूसरा चरण (`+Det`, नारंगी) बदलते ही भारी उछाल
           आता है। Ceiling analysis से तुरंत साफ हो जाता है कि असली रुकावट कहाँ है।""",
           frames=44, keys=(0.05, 0.52, 1.0))
def pipeline_ceiling_bars(s, n):
    rng = np.random.default_rng(55)
    m = 600
    y_true = (rng.uniform(0, 1, m) < 0.38).astype(float)
    sign = np.where(y_true > 0.5, 1.0, -1.0)

    # Ground-truth latent signals for stages 1..3 & trained logistic classifier for stage 4
    z1_gt = np.c_[sign * 1.15 + rng.normal(0, 0.32, m), sign * 0.95 + rng.normal(0, 0.32, m)]
    z2_gt = z1_gt * np.array([1.1, 1.2]) + sign[:, None] * 0.35
    z3_gt = np.c_[z2_gt[:, 0] + 0.4 * z2_gt[:, 1], z2_gt[:, 1] - 0.2 * z2_gt[:, 0]]
    th_cls = _fit_logreg(z3_gt, y_true, lam=0.05)

    # Noisy learned stage errors: Stage 1 tiny noise, Stage 2 severe detector noise (bottleneck!), Stage 3 medium
    e1 = rng.normal(0, 0.18, z1_gt.shape)
    e2 = rng.normal(0, 1.85, z2_gt.shape)
    e3 = rng.normal(0, 0.72, z3_gt.shape)

    def eval_pipeline(a1, a2, a3, a4):
        s1 = z1_gt + (1.0 - a1) * e1
        s2 = (s1 * np.array([1.1, 1.2]) + sign[:, None] * 0.35 * a2
              + (1.0 - a2) * (e2 + sign[:, None] * 0.35))
        s3 = np.c_[s2[:, 0] + 0.4 * s2[:, 1], s2[:, 1] - 0.2 * s2[:, 0]] + (1.0 - a3) * e3
        prob = (1.0 - a4) * _predict_prob(th_cls, s3) + a4 * y_true
        pred = (prob >= 0.5).astype(float)
        acc = float(np.mean(pred == y_true))
        tp = float(np.sum((pred == 1) & (y_true == 1)))
        fp = float(np.sum((pred == 1) & (y_true == 0)))
        fn = float(np.sum((pred == 0) & (y_true == 1)))
        f1 = 2.0 * tp / max(1.0, 2.0 * tp + fp + fn)
        return acc, f1

    anchors = [
        eval_pipeline(0, 0, 0, 0),
        eval_pipeline(1, 0, 0, 0),
        eval_pipeline(1, 1, 0, 0),
        eval_pipeline(1, 1, 1, 0),
        eval_pipeline(1, 1, 1, 1),
    ]
    names = ["Base", "+Pre", "+Det", "+Feat", "+Cls"]

    ax = s.ax3d(labels=("pipeline stage", "", "score"), elev=24, azim=-58,
                xlim=(-0.5, 4.6), ylim=(-0.35, 1.55), zlim=(0.0, 1.08))
    ax.set_xticks(range(5))
    ax.set_xticklabels(names)
    ax.set_yticks([0.05, 0.95])
    ax.set_yticklabels(["Acc", "F1"])
    plane(ax, lambda x, y: np.ones_like(x), (-0.4, 4.5), (-0.25, 1.35), color=C["green"], alpha=0.14)

    def update(i):
        s.clear()
        prog = (i / (n - 1)) * 4.0                 # progresses across transitions 0->1, 1->2, 2->3, 3->4
        cur_acc, cur_f1 = [anchors[0][0]], [anchors[0][1]]
        for st in range(1, 5):
            u = ease(prog - (st - 1))
            prev_a, prev_f = cur_acc[-1], cur_f1[-1]
            cur_acc.append(lerp(prev_a, anchors[st][0], u))
            cur_f1.append(lerp(prev_f, anchors[st][1], u))

        for k in range(5):
            col_a = C["orange"] if k == 2 else C["blue"]
            col_f = C["red"] if k == 2 else C["teal"]
            s.add(ax.bar3d(k - 0.24, -0.15, 0.0, 0.44, 0.36, cur_acc[k],
                           color=col_a, alpha=0.84, shade=True))
            s.add(ax.bar3d(k - 0.24, 0.75, 0.0, 0.44, 0.36, cur_f1[k],
                           color=col_f, alpha=0.78, shade=True))
            if k > 0 and prog >= k - 0.35:
                gain = 100.0 * (cur_acc[k] - cur_acc[k - 1])
                s.add(ax.text(k, 0.03, cur_acc[k] + 0.03, f"+{gain:.1f}%",
                              ha="center", fontsize=7.8, fontweight="bold",
                              color=C["red"] if k == 2 else C["dark"]))
        idx = min(4, int(prog + 0.5))
        s.status(f"{names[idx]}: Acc={100*cur_acc[idx]:.1f}%  F1={100*cur_f1[idx]:.1f}%")
        s.spin(i, n, sweep=30)

    return update
