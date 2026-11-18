"""Lecture 5: Gaussian Discriminant Analysis and Naive Bayes: 3D animations."""
import numpy as np

import matplotlib.pyplot as plt

from lib.anim import C, animation, ease, grid, lerp

_XLIM, _YLIM = (-3.4, 3.4), (-3.4, 3.4)


def _gauss(X, Y, mu, S):
    """Bivariate Gaussian density N(mu, S) evaluated on a grid."""
    d = np.stack([X - mu[0], Y - mu[1]], axis=-1)
    q = np.einsum("...i,ij,...j->...", d, np.linalg.inv(S), d)
    return np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(np.linalg.det(S)))


def _patch(mu, S, rmax=2.7, nr=18, nt=44):
    """Polar grid over the rmax-sigma ellipse of N(mu, S): returns X, Y (clipped to the box) and density Z."""
    L = np.linalg.cholesky(S)
    r, th = np.meshgrid(np.linspace(0, rmax, nr), np.linspace(0, 2 * np.pi, nt))
    U = np.stack([r * np.cos(th), r * np.sin(th)], axis=-1) @ L.T
    X = np.clip(mu[0] + U[..., 0], *_XLIM)
    Y = np.clip(mu[1] + U[..., 1], *_YLIM)
    return X, Y, _gauss(X, Y, mu, S)


def _boundary(theta, theta0):
    """End points of the line theta^T x + theta0 = 0, clipped to the plotting box."""
    d = np.array([-theta[1], theta[0]]) / np.linalg.norm(theta)      # direction along the line
    p = -theta0 * theta / (theta @ theta)                             # closest point to the origin
    lo, hi = -20.0, 20.0
    for k, (a, b) in enumerate((_XLIM, _YLIM)):
        if abs(d[k]) > 1e-9:
            t1, t2 = sorted(((a - p[k]) / d[k], (b - p[k]) / d[k]))
            lo, hi = max(lo, t1), min(hi, t2)
    ts = np.linspace(lo, hi, 40)
    return p[0] + ts * d[0], p[1] + ts * d[1]


def _gda_theta(phi, mu0, mu1, S):
    """GDA posterior is sigmoid(theta^T x + theta0)."""
    P = np.linalg.inv(S)
    return P @ (mu1 - mu0), 0.5 * (mu0 @ P @ mu0 - mu1 @ P @ mu1) + np.log(phi / (1 - phi))


# ---- two fixed class-conditional Gaussians used by animations 3 and 4 ----
_MU0, _MU1 = np.array([-0.9, -0.6]), np.array([0.9, 0.6])
_SIG = np.array([[0.55, 0.15], [0.15, 0.45]])


@animation("GDA: fitting two Gaussians",
           caption=r"""फ़र्श पर दो features $x_1, x_2$ हैं और ऊँचाई density $p(x \mid y)$ है। Training examples
           एक-एक करके आते हैं (रंगीन बिंदु देखे जा चुके हैं, धूसर अभी नहीं)। हर frame में GDA वही करता है जो
           formulas कहते हैं: $\phi$ के लिए गिनती, $\mu_0$ और $\mu_1$ के लिए हर class का औसत, और साझा $\Sigma$।
           शुरू में थोड़े data पर दोनों पहाड़ियाँ गलत जगह और गलत आकार की हैं। जैसे-जैसे $m$ बढ़ता है, वे अपने
           बिंदुओं के ऊपर जम जाती हैं। ध्यान दें कि दोनों पहाड़ियों का आकार हमेशा एक जैसा है, क्योंकि $\Sigma$ एक ही है।""",
           frames=40, keys=(0.0, 0.4, 1.0))
def gda_fit(s, n):
    rng = np.random.default_rng(11)
    S_true = np.array([[0.75, 0.38], [0.38, 0.6]])
    mus = (np.array([-1.2, -0.8]), np.array([1.2, 0.9]))
    x0 = np.clip(rng.multivariate_normal(mus[0], S_true, 42), -3.0, 3.0)
    x1 = np.clip(rng.multivariate_normal(mus[1], S_true, 28), -3.0, 3.0)
    # arrival order: three of each class first (so the first estimate exists), then random
    rest = rng.permutation(np.r_[np.c_[x0[3:], np.zeros(39)], np.c_[x1[3:], np.ones(25)]])
    data = np.r_[np.c_[x0[:3], np.zeros(3)], np.c_[x1[:3], np.ones(3)], rest]
    pts, lab = data[:, :2], data[:, 2].astype(int)
    m = len(lab)
    zmax = 0.45
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$p(x \mid y)$"), elev=30, azim=-62,
                xlim=_XLIM, ylim=_YLIM, zlim=(0, zmax))
    ax.set_zticks([0, 0.1, 0.2, 0.3, 0.4])
    ax.zaxis.labelpad = 1
    cols = (C["blue"], C["red"])
    for c in (0, 1):
        ax.plot([], [], [], color=cols[c], lw=3, label=rf"$\mathcal{{N}}(\mu_{c}, \Sigma)$,  y = {c}")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        k = int(round(6 * (m / 6) ** (i / (n - 1))))            # examples seen so far
        p, l = pts[:k], lab[:k]
        mu = [p[l == c].mean(axis=0) for c in (0, 1)]
        d = p - np.array(mu)[l]
        S = d.T @ d / k                                          # shared covariance (MLE)
        s.add(ax.scatter(pts[k:, 0], pts[k:, 1], 0, color="#c4c8d0", s=7, depthshade=False))
        for c in (0, 1):
            s.add(ax.scatter(p[l == c, 0], p[l == c, 1], 0, color=cols[c], s=11, depthshade=False))
            X, Y, Z = _patch(mu[c], S)
            Z = np.minimum(Z, zmax - 0.004)
            s.add(ax.plot_surface(X, Y, Z, color=cols[c], alpha=0.42, linewidth=0, rstride=1, cstride=1))
            s.add(ax.plot([mu[c][0]] * 2, [mu[c][1]] * 2, [0, Z.max()], color=cols[c], lw=1.2, ls="--"))
        s.status(rf"examples seen m = {k}     $\phi$ = {l.mean():.2f}")
        s.spin(i, n, sweep=35)

    return update


@animation(r"How $\Sigma$ shapes a Gaussian",
           caption=r"""एक ही Gaussian $\mathcal{N}(0, \Sigma)$ है, बस $\Sigma$ बदल रहा है। ऊँचाई density $p(x)$ है और
           नीचे फ़र्श पर उसके contours हैं। क्रम यह है: $\Sigma = I$ (गोल), फिर छोटा $\Sigma$ (पतली, ऊँची पहाड़ी),
           फिर बड़ा $\Sigma$ (चौड़ी, नीची), फिर diagonal entries अलग (एक axis की ओर खिंचा ellipse), और अंत में
           off-diagonal entries positive और फिर negative (झुका हुआ ellipse)। पहाड़ी के नीचे का कुल volume हमेशा
           1 रहता है, इसलिए फैलने पर ऊँचाई घटती है।""",
           frames=41, keys=(0.2, 0.6, 1.0))
def sigma_shapes(s, n):
    stages = [("identity", [[1, 0], [0, 1]]), ("small variance", [[0.45, 0], [0, 0.45]]),
              ("large variance", [[1.5, 0], [0, 1.5]]), ("stretched along $x_1$", [[1.5, 0], [0, 0.4]]),
              ("positive correlation", [[1, 0.8], [0.8, 1]]), ("negative correlation", [[1, -0.8], [-0.8, 1]])]
    mats = [np.array(mat, float) for _, mat in stages]
    floor = -0.14
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", r"$p(x)$"), elev=28, azim=-60,
                xlim=_XLIM, ylim=_YLIM, zlim=(floor, 0.36))
    ax.set_zticks([0, 0.1, 0.2, 0.3])
    ax.zaxis.labelpad = 1
    seg = (n - 1) / (len(stages) - 1)

    def update(i):
        s.clear()
        j = min(int(i // seg), len(stages) - 2)
        t = ease((i - j * seg) / seg)
        S = lerp(mats[j], mats[j + 1], t)
        X, Y, Z = _patch((0, 0), S)
        s.add(ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.65, linewidth=0, rstride=1, cstride=1,
                              vmin=0, vmax=0.3))
        for r in (0.6, 1.2, 1.8, 2.4):                       # contours = level sets, drawn on the floor
            cx, cy, cz = _patch((0, 0), S, rmax=r, nr=2, nt=60)
            s.add(ax.plot(cx[:, 1], cy[:, 1], floor, color=plt.cm.viridis(cz[0, 1] / 0.3), lw=1.2))
        name = stages[j + 1][0] if t > 0.5 else stages[j][0]
        s.status(rf"{name}:  $\Sigma$ = [[{S[0, 0]:.2f}, {S[0, 1]:.2f}], [{S[1, 0]:.2f}, {S[1, 1]:.2f}]]")
        s.spin(i, n, sweep=30)

    return update


@animation("Bayes rule: joint to posterior",
           caption=r"""शुरू में दो पहाड़ियाँ joint densities हैं: लाल $p(x \mid y=1)\,p(y=1)$ और नीली
           $p(x \mid y=0)\,p(y=0)$। Bayes rule कहता है कि इन्हें $p(x)$ (दोनों के जोड़) से divide करो। Animation
           में यही division धीरे-धीरे होता है, और दोनों पहाड़ियाँ posterior surfaces $p(y=1 \mid x)$ और
           $p(y=0 \mid x)$ बन जाती हैं, जो sigmoid आकार की हैं। काली line वह जगह है जहाँ दोनों surfaces बराबर
           हैं। वह divide करने से हिलती नहीं (इसीलिए argmax के लिए $p(x)$ छोड़ सकते हैं) और सीधी है: यही
           linear decision boundary है, जो अंत में ऊँचाई 0.5 पर पहुँच जाती है।""",
           frames=42, keys=(0.0, 0.52, 1.0))
def joint_to_posterior(s, n):
    phi = 0.4
    sig = np.array([[0.34, 0.09], [0.09, 0.28]])
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", "value"), elev=26, azim=-110,
                xlim=_XLIM, ylim=_YLIM, zlim=(0, 1))
    X, Y = grid(_XLIM, _YLIM, 38)
    A = phi * _gauss(X, Y, _MU1, sig)                     # p(x|y=1) p(y=1)
    B = (1 - phi) * _gauss(X, Y, _MU0, sig)               # p(x|y=0) p(y=0)
    theta, theta0 = _gda_theta(phi, _MU0, _MU1, sig)
    bx, by = _boundary(theta, theta0)
    bz = phi * _gauss(bx, by, _MU1, sig)                  # joint density along the boundary (A = B there)
    ax.plot(bx, by, 0 * bx, color=C["dark"], lw=1.6)
    ax.plot([], [], [], color=C["red"], lw=3, label="class 1")
    ax.plot([], [], [], color=C["blue"], lw=3, label="class 0")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        t = ease((i / (n - 1) - 0.12) / 0.76)             # 0: joint densities, 1: divided by p(x)
        norm = (A + B) ** t
        for Z, col in ((B / norm, C["blue"]), (A / norm, C["red"])):
            s.add(ax.plot_surface(X, Y, Z, color=col, alpha=0.4, linewidth=0, rstride=1, cstride=1))
        s.add(ax.plot(bx, by, bz / (2 * bz) ** t + 0.004, color=C["dark"], lw=2.0))
        if t <= 0:
            s.status(r"joint:  $p(x \mid y)\,p(y)$")
        elif t >= 1:
            s.status(r"posterior:  $p(y \mid x)$,  boundary at 0.5")
        else:
            s.status(rf"dividing by $p(x)$ ...  {100 * t:.0f}%")
        s.spin(i, n, sweep=-40)

    return update


@animation(r"Class prior $\phi$ moves the boundary",
           caption=r"""दोनों class Gaussians ($\mu_0, \mu_1, \Sigma$) स्थिर हैं, सिर्फ़ prior $\phi = p(y=1)$ को 0.1 से
           0.9 तक बढ़ाया जा रहा है। ऊँचाई weighted densities हैं: लाल $\phi\, p(x \mid y=1)$ और नीली
           $(1-\phi)\, p(x \mid y=0)$। $\phi$ बढ़ने पर लाल पहाड़ी ऊँची और नीली नीची होती है, इसलिए दोनों के
           बराबर होने की जगह (काली line) नीली class की तरफ़ खिसकती है। फ़र्श का रंग बताता है कि कहाँ कौन
           सी class जीतती है। Boundary की दिशा नहीं बदलती, क्योंकि $\phi$ सिर्फ़ $\theta_0$ में आता है; धूसर
           dashed line $\phi = 0.5$ वाली boundary है।""",
           frames=40)
def prior_sweep(s, n):
    ax = s.ax3d(labels=(r"$x_1$", r"$x_2$", "weighted density"), elev=30, azim=-112,
                xlim=_XLIM, ylim=_YLIM, zlim=(0, 0.34))
    ax.zaxis.labelpad = 3
    X, Y = grid(_XLIM, _YLIM, 8)
    P0, P1 = _patch(_MU0, _SIG, rmax=3.0), _patch(_MU1, _SIG, rmax=3.0)
    ax.plot(*_boundary(*_gda_theta(0.5, _MU0, _MU1, _SIG)), 0, color=C["gray"], lw=1.2, ls="--")
    ax.plot([], [], [], color=C["red"], lw=3, label=r"$\phi\,p(x \mid y=1)$")
    ax.plot([], [], [], color=C["blue"], lw=3, label=r"$(1-\phi)\,p(x \mid y=0)$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        phi = lerp(0.1, 0.9, ease(i / (n - 1)))
        theta, theta0 = _gda_theta(phi, _MU0, _MU1, _SIG)
        s.add(ax.contourf(X, Y, X * theta[0] + Y * theta[1] + theta0, levels=[-1e9, 0, 1e9], zdir="z",
                          offset=0, colors=[C["blue"], C["red"]], alpha=0.16))
        for (PX, PY, PZ), wgt, col in ((P0, 1 - phi, C["blue"]), (P1, phi, C["red"])):
            s.add(ax.plot_surface(PX, PY, wgt * PZ, color=col, alpha=0.45, linewidth=0, rstride=1, cstride=1))
        bx, by = _boundary(theta, theta0)
        s.add(ax.plot(bx, by, 0 * bx, color=C["dark"], lw=1.8))
        s.add(ax.plot(bx, by, phi * _gauss(bx, by, _MU1, _SIG) + 0.002, color=C["dark"], lw=1.8))
        s.status(rf"$\phi$ = {phi:.2f}     $\theta_0$ = {theta0:+.2f}")
        s.spin(i, n, sweep=-30)

    return update


@animation("Naive Bayes: counting words",
           caption=r"""यह Naive Bayes की training है। एक axis पर vocabulary के पाँच शब्द हैं, दूसरी पर class
           (spam या not spam), और ऊँचाई estimate $\phi_{j \mid y}$ है, यानी उस class के कितने हिस्से emails में
           वह शब्द आया। Emails एक-एक करके गिने जाते हैं (ऊपर की line में अभी वाला email और उसके शब्द दिखते हैं)
           और हर bar बस एक अनुपात है: गिनती / उस class के emails। काली छोटी lines वे असली probabilities हैं
           जिनसे data बना था: शुरू में bars उछलते हैं, फिर उनके पास जम जाते हैं। ध्यान दें कि शुरू में कई bars
           ठीक zero हैं, सिर्फ़ इसलिए कि वह शब्द अभी तक उस class में दिखा नहीं।""",
           frames=41, keys=(0.1, 0.4, 1.0))
def nb_counting(s, n):
    rng = np.random.default_rng(4)
    words = ["free", "win", "cash", "team", "report"]
    true = np.array([[0.15, 0.05, 0.10, 0.60, 0.50],       # p(word | not spam)
                     [0.75, 0.60, 0.50, 0.10, 0.05]])      # p(word | spam)
    y = (rng.random(n - 1) < 0.45).astype(int)
    x = (rng.random((n - 1, 5)) < true[y]).astype(int)
    ax = s.ax3d(labels=("", "", r"$\phi_{j \mid y}$"), elev=33, azim=-86,
                xlim=(-0.6, 4.6), ylim=(-0.5, 1.5), zlim=(0, 1), zoom=0.8)
    ax.set_box_aspect((5.2, 3.4, 3.0), zoom=0.8)
    ax.zaxis.labelpad = 0
    ax.set_xticks(range(5))
    ax.set_xticklabels(words, fontsize=8)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["not spam", "spam"], fontsize=8, ha="left")
    cols = (C["blue"], C["red"])
    w = 0.5
    for c in (0, 1):
        for j in range(5):
            ax.plot([j - 0.36, j + 0.36], [c - 0.2] * 2, [true[c, j]] * 2, color=C["dark"], lw=1.8)
    jj, cc = np.meshgrid(np.arange(5), np.arange(2))

    def update(i):
        s.clear()
        seen_y, seen_x = y[:i], x[:i]
        est = np.zeros((2, 5))
        for c in (0, 1):
            if (seen_y == c).any():
                est[c] = seen_x[seen_y == c].mean(axis=0)
        s.add(ax.bar3d(jj.ravel() - w / 2, cc.ravel() - 0.18, 0, w, 0.36, est.ravel(),
                       color=[cols[c] for c in cc.ravel()], alpha=0.75, shade=True))
        if i == 0:
            s.status("no emails counted yet")
        else:
            in_mail = " ".join(wd for wd, f in zip(words, x[i - 1]) if f) or "(none of these)"
            s.status(f"email {i} ({'spam' if y[i - 1] else 'not spam'}): {in_mail}")
        s.spin(i, n, sweep=24)

    return update
