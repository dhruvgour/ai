"""Lecture 17: Value Iteration, Policy Iteration, LQR, and LQG: 3D animations."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.anim import C, animation, ease, surface

# ---- shared 4x3 gridworld from Lecture 16: state = (column, row), (1, 1) bottom-left ----
COLS, ROWS = 4, 3
WALL, GOAL, PIT, START = (2, 2), (4, 3), (4, 2), (1, 1)
STATES = [(c, r) for r in range(1, ROWS + 1) for c in range(1, COLS + 1) if (c, r) != WALL]
IDX = {st: i for i, st in enumerate(STATES)}
NS = len(STATES)
TERMINAL = [IDX[GOAL], IDX[PIT]]
LIVE = [i for i in range(NS) if i not in TERMINAL]
MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]          # actions 0..3 = N, E, S, W
ARROWS = "^>v<"
GAMMA = 0.99


def _move(st, a):
    nxt = (st[0] + MOVES[a][0], st[1] + MOVES[a][1])
    return nxt if nxt in IDX else st


P = np.zeros((4, NS, NS))
for _st in STATES:
    if _st in (GOAL, PIT):
        continue
    for _a in range(4):
        for _b, _p in ((_a, 0.8), ((_a - 1) % 4, 0.1), ((_a + 1) % 4, 0.1)):
            P[_a, IDX[_st], IDX[_move(_st, _b)]] += _p

R_GRID = np.full(NS, -0.02)
R_GRID[IDX[GOAL]], R_GRID[IDX[PIT]] = 1.0, -1.0

_X = np.array([st[0] for st in STATES], float)
_Y = np.array([st[1] for st in STATES], float)


def _policy(text):
    pi = np.zeros(NS, int)
    for k, line in enumerate(text.split()):
        for c, ch in enumerate(line, 1):
            if (c, ROWS - k) in IDX and ch in ARROWS:
                pi[IDX[(c, ROWS - k)]] = ARROWS.index(ch)
    return pi


def _evaluate(pi):
    return np.linalg.solve(np.eye(NS) - GAMMA * P[pi, np.arange(NS)], R_GRID)


def _stage_grid(s, zlabel=r"$V(s)$", zlim=(-1.15, 1.15), elev=34, azim=-64):
    ax = s.ax3d(labels=("column", "row", zlabel), elev=elev, azim=azim,
                xlim=(0.5, 4.5), ylim=(0.5, 3.5), zlim=zlim)
    ax.set_box_aspect((4, 3, 2.9), zoom=0.93)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([1, 2, 3])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.zaxis.labelpad = -1
    ax.computed_zorder = False
    z = zlim[0]
    tiles, faces = [], []
    for c in range(1, COLS + 1):
        for r in range(1, ROWS + 1):
            tiles.append([(c - .5, r - .5, z), (c + .5, r - .5, z),
                          (c + .5, r + .5, z), (c - .5, r + .5, z)])
            faces.append({WALL: (0.25, 0.27, 0.32, 0.85), GOAL: (0.09, 0.64, 0.29, 0.3),
                          PIT: (0.86, 0.15, 0.15, 0.3)}.get((c, r), (1, 1, 1, 0.0)))
    ax.add_collection3d(Poly3DCollection(tiles, facecolors=faces, edgecolors=(0.45, 0.47, 0.52, 0.9),
                                         linewidths=0.6, zorder=1))
    ax.plot([0.5, 4.5, 4.5, 0.5, 0.5], [0.5, 0.5, 3.5, 3.5, 0.5], [0] * 5,
            color=C["dark"], lw=0.9, ls=(0, (4, 3)), zorder=1.5)
    return ax


def _bars(ax, V, width=0.62, alpha=0.84, base=-1.15):
    V = np.asarray(V, float)
    colors = plt.cm.RdYlGn(0.5 + 0.5 * np.clip(V, -1, 1))
    return ax.bar3d(_X - width / 2, _Y - width / 2, base, width, width,
                    np.maximum(V - base, 1e-3), color=colors, alpha=alpha, shade=True,
                    edgecolor=(0, 0, 0, 0.3), linewidth=0.3, zorder=2)


def _outline(ax, x, y, z, width=0.62, base=-1.15):
    h = width / 2
    cx = np.array([-h, h, h, -h, -h]) + x
    cy = np.array([-h, -h, h, h, -h]) + y
    xs, ys, zs = list(cx), list(cy), [z] * 5
    for k in range(4):
        xs += [np.nan, cx[k], cx[k]]
        ys += [np.nan, cy[k], cy[k]]
        zs += [np.nan, base, z]
    return ax.plot(xs, ys, zs, color=(0.07, 0.09, 0.15, 0.65), lw=0.7, zorder=4)


def _arrows(ax, pi, z, idx=None, color=C["dark"], length=0.44, lw=1.7):
    idx = [i for i in (LIVE if idx is None else idx)]
    if not idx:
        return None
    d = np.array([MOVES[pi[i]] for i in idx], float)
    z = np.broadcast_to(np.asarray(z, float), (NS,))[idx]
    return ax.quiver(_X[idx] - 0.5 * length * d[:, 0], _Y[idx] - 0.5 * length * d[:, 1], z,
                     d[:, 0], d[:, 1], 0 * d[:, 0], length=length, color=color, linewidth=lw,
                     arrow_length_ratio=0.5, zorder=6)


# =====================================================================================
@animation("Value Iteration on 4x3 Gridworld",
           caption=r"""शुरुआत $V_0(s) = 0$ से होती है और हर sweep में Bellman optimality update
           $V_{k+1}(s) = R(s) + \gamma \max_a \sum_{s'} P_{sa}(s') V_k(s')$ चलता है। पहले कदम पर सिर्फ़
           $+1$ और $-1$ वाले terminal cells उठते-गिरते हैं, फिर $+1$ की ऊँचाई लहर की तरह दीवार के चारों ओर
           एक-एक cell आगे फैलती है। काले wire boxes असली $V^*$ हैं और ऊपर के तीर वर्तमान greedy policy दिखाते
           हैं; नारंगी तीर उस sweep में बदले हुए actions हैं। देखिए कि $V_k$ के पूरी तरह टिकने से काफ़ी पहले ही
           सारे तीर सही $\pi^*$ पर स्थिर हो जाते हैं।""",
           frames=38, keys=(0.08, 0.24, 1.0))
def value_iteration_bars(s, n):
    ax = _stage_grid(s, r"$V_k(s)$")
    V_hist = [np.zeros(NS)]
    pi_hist = [np.zeros(NS, int)]
    for _ in range(60):
        Q = P @ V_hist[-1]
        V_hist.append(R_GRID + GAMMA * Q.max(axis=0))
        pi_hist.append(Q.argmax(axis=0))
    V_star = V_hist[-1]
    for _ in range(500):
        V_star = R_GRID + GAMMA * (P @ V_star).max(axis=0)
    for j in range(NS):
        _outline(ax, _X[j], _Y[j], V_star[j])

    # Slow 1-by-1 sweeps early when the wave moves across cells, faster at the tail
    steps = [i if i <= 18 else int(round(18 + (i - 18) / (n - 1 - 18) * 32)) for i in range(n)]

    def update(i):
        s.clear()
        k = steps[i]
        V, pi = V_hist[k], pi_hist[k]
        prev = pi_hist[max(0, k - 1)]
        changed = [j for j in LIVE if k > 0 and pi[j] != prev[j]]
        s.add(_bars(ax, V))
        if k > 0:
            s.add(_arrows(ax, pi, V + 0.05, idx=[j for j in LIVE if j not in changed]))
            s.add(_arrows(ax, pi, V + 0.05, idx=changed, color=C["orange"], lw=2.8))
        err = np.abs(V - V_star).max()
        s.status(rf"sweep {k}   max $|V_k - V^*|$ = {err:.3f}")
        s.spin(i, n, sweep=28)

    return update


# =====================================================================================
@animation("Policy Iteration: Eval & Improve",
           caption=r"""Policy Iteration दो कदमों की जोड़ी दोहराता है: **Evaluate** (वर्तमान $\pi$ के लिए linear
           system हल करके पूरा $V^\pi$ निकालना) और **Improve** ($V^\pi$ पर greedy होकर तीरों को सुधारना)। शुरू में
           खराब policy से कई bars शून्य से नीचे हैं। हर Improve कदम पर जिन states को बेहतर रास्ता मिलता है उनके तीर
           नारंगी चमकते हैं, और अगले ही Evaluate कदम पर सारे bars बड़ी छलाँग लगाकर ऊपर उठ जाते हैं। मात्र कुछ ही
           चक्रों में कोई तीर नहीं बदलता और हरे तीर exact $\pi^*$ दे देते हैं।""",
           frames=40, keys=(0.10, 0.45, 1.0))
def policy_iteration_cycle(s, n):
    ax = _stage_grid(s, r"$V^\pi(s)$")
    pi = _policy(">>>. v.>. >>^^")                   # poor starting policy from L16
    rounds = []                                      # list of (pi_old, V_old, pi_new, changed)
    while True:
        V = _evaluate(pi)
        pi_new = (P @ V).argmax(axis=0)
        changed = [j for j in LIVE if pi_new[j] != pi[j]]
        rounds.append((pi.copy(), V, pi_new.copy(), changed))
        if not changed:
            break
        pi = pi_new
    V_star = rounds[-1][1]
    for j in range(NS):
        _outline(ax, _X[j], _Y[j], V_star[j])

    # Build timeline: for each non-final round, morph V -> show V -> flip arrows; final holds V*
    M = len(rounds) - 1
    per = (n - 6) // M

    def update(i):
        s.clear()
        r = min(i // per, M)
        sub = i - r * per
        pi_old, V_old, pi_new, changed = rounds[r]
        V_prev = np.zeros(NS) if r == 0 else rounds[r - 1][1]
        if r == M:
            s.add(_bars(ax, V_old))
            s.add(_arrows(ax, pi_old, V_old + 0.05, color=C["green"], lw=2.3))
            s.status(f"round {r + 1}: converged! 0 arrows flip")
        elif sub < per // 2:
            t = ease((sub + 1) / max(1, per // 2))
            V = (1 - t) * V_prev + t * V_old
            s.add(_bars(ax, V))
            s.add(_arrows(ax, pi_old, V + 0.05))
            s.status(rf"round {r + 1} eval: $V^\pi$(1,1) = {V[IDX[START]]:+.2f}")
        else:
            s.add(_bars(ax, V_old))
            s.add(_arrows(ax, pi_new, V_old + 0.05, idx=[j for j in LIVE if j not in changed]))
            s.add(_arrows(ax, pi_new, V_old + 0.05, idx=changed, color=C["orange"], lw=3.0))
            s.status(f"round {r + 1} improve: {len(changed)} arrows flip")
        s.spin(i, n, sweep=28)

    return update


# =====================================================================================
@animation("Curse of Dimensionality",
           caption=r"""Continuous state के हर अक्ष को $k = 5$ बराबर टुकड़ों में काटने पर cells की संख्या $k^d$ की
           रफ़्तार से विस्फोट करती है। पहले $d = 1$ (नीली रेखा) में सिर्फ़ $5^1 = 5$ डिब्बे हैं। दूसरा आयाम जुड़ते
           ही $d = 2$ (हरा फ़र्श) में $5^2 = 25$ डिब्बों का वर्ग बन जाता है, और तीसरे आयाम $d = 3$ में यही
           $5^3 = 125$ डिब्बों का घन बन जाता है। चौथे या बारहवें आयाम तक पहुँचते-पहुँचते डिब्बों की संख्या इतनी
           विशाल हो जाती है कि लाल बिंदु जैसा वास्तविक अनुभव लगभग सारे डिब्बों को खाली छोड़ देता है।""",
           frames=40, keys=(0.20, 0.55, 1.0))
def curse_of_dimensionality(s, n):
    k = 5
    ax = s.ax3d(labels=(r"$s_1$", r"$s_2$", r"$s_3$"), elev=25, azim=-52,
                xlim=(0, 1), ylim=(0, 1), zlim=(0, 1))
    ax.set_xticks([0, 0.5, 1])
    ax.set_yticks([0, 0.5, 1])
    ax.set_zticks([0, 0.5, 1])
    # Faint unit-cube frame
    for a in (0, 1):
        for b in (0, 1):
            ax.plot([0, 1], [a, a], [b, b], color=C["gray"], lw=0.5, alpha=0.35)
            ax.plot([a, a], [0, 1], [b, b], color=C["gray"], lw=0.5, alpha=0.35)
            ax.plot([a, a], [b, b], [0, 1], color=C["gray"], lw=0.5, alpha=0.35)

    step = 1.0 / k
    pad = 0.022
    w = step - 2 * pad
    c1 = np.arange(k) * step + pad
    # Sample trajectory dots to highlight sparsity in 3D
    ts = np.linspace(0.08, 0.92, 14)
    tx, ty, tz = ts, 0.5 + 0.35 * np.sin(4 * ts), 0.5 + 0.35 * np.cos(4 * ts)

    def update(i):
        s.clear()
        if i <= 9:                                   # Phase 1: 1D line (5 cells)
            m = min(k, int(np.ceil((i + 1) / 9 * k)))
            xs = c1[:m]
            ys = np.zeros(m) + pad
            zs = np.zeros(m) + pad
            s.add(ax.bar3d(xs, ys, zs, w, w, w, color=C["blue"], alpha=0.48,
                           edgecolor=(0, 0, 0, 0.35), linewidth=0.4))
            s.add(ax.scatter(tx, np.full_like(tx, 0.1), np.full_like(tx, 0.1),
                             color=C["red"], s=16, depthshade=False))
            s.status(f"d = 1 (line):  {k}^1 = {m} cells")
        elif i <= 23:                                # Phase 2: 2D square (25 cells)
            rows = min(k, int(np.ceil((i - 9) / 13 * k)))
            gx, gy = np.meshgrid(c1, c1[:rows])
            xs, ys = gx.ravel(), gy.ravel()
            zs = np.zeros_like(xs) + pad
            s.add(ax.bar3d(xs, ys, zs, w, w, w, color=C["teal"], alpha=0.46,
                           edgecolor=(0, 0, 0, 0.3), linewidth=0.35))
            s.add(ax.scatter(tx, ty, np.full_like(tx, 0.1),
                             color=C["red"], s=16, depthshade=False))
            s.status(f"d = 2 (square): {k}x{rows} = {len(xs)} cells")
        else:                                        # Phase 3: 3D cube (125 cells)
            layers = min(k, int(np.ceil((i - 23) / 13 * k)))
            gx, gy, gz = np.meshgrid(c1, c1, c1[:layers])
            xs, ys, zs = gx.ravel(), gy.ravel(), gz.ravel()
            s.add(ax.bar3d(xs, ys, zs, w, w, w, color=C["orange"], alpha=0.38,
                           edgecolor=(0, 0, 0, 0.22), linewidth=0.28))
            s.add(ax.scatter(tx, ty, tz, color=C["red"], s=18, depthshade=False))
            s.status(f"d = 3 (cube):  {k}x{k}x{layers} = {len(xs)} cells")
        s.spin(i, n, sweep=38)

    return update


# =====================================================================================
@animation("LQR Bowl: Backward Riccati",
           caption=r"""Double integrator के state $s = (p, v)$ के ऊपर LQR का optimal cost-to-go कटोरा
           $J_t^*(s) = s^T \Phi_t s$ दिखाया गया है। अंतिम कदम $t = T$ पर कटोरा उथला है (हल्का धूसर जाल
           $s^T U_T s$)। जैसे-जैसे Riccati recursion पीछे $t = T-1, \dots, 0$ की ओर चलती है, आगे के कदमों का
           खर्च जुड़ने से कटोरा तेज़ी से गहरा और खड़ी ढलान वाला होता जाता है। फ़र्श के समोच्च (contours) सिकुड़ते
           और तिरछे होते दिखते हैं क्योंकि एक ही दिशा के $p$ और $v$ साथ मिलकर बड़ा खर्च पैदा करते हैं।""",
           frames=38, keys=(0.0, 0.35, 1.0))
def lqr_riccati_bowl(s, n):
    dt, T = 0.25, 22
    A = np.array([[1.0, dt], [0.0, 1.0]])
    B = np.array([[0.5 * dt**2], [dt]])
    U = np.diag([0.65, 0.22])
    W = np.array([[0.35]])
    U_T = np.diag([0.30, 0.15])

    # Run Riccati backward from t = T down to t = 0
    phis = [U_T]
    for _ in range(T):
        Phi = phis[-1]
        K = np.linalg.solve(W + B.T @ Phi @ B, B.T @ Phi @ A)
        Phi_prev = U + A.T @ Phi @ A - A.T @ Phi @ B @ K
        phis.append(0.5 * (Phi_prev + Phi_prev.T))   # phis[0]=Phi_T ... phis[T]=Phi_0

    lim, zmax = (-1.4, 1.4), 10.5
    ax = s.ax3d(labels=("pos $p$", "vel $v$", r"$J_t^*(s)$"), elev=30, azim=-58,
                xlim=lim, ylim=lim, zlim=(0, zmax))

    def quad(Phi):
        return lambda X, Y: np.minimum(
            Phi[0, 0] * X**2 + 2 * Phi[0, 1] * X * Y + Phi[1, 1] * Y**2, zmax
        )

    surface(ax, quad(phis[0]), lim, lim, n=28, wire=True, alpha=0.38, color=C["gray"])
    tau = np.linspace(0, T, n)

    def update(i):
        s.clear()
        u = tau[i]
        idx = min(int(np.floor(u)), T - 1)
        frac = u - idx
        Phi = (1 - frac) * phis[idx] + frac * phis[idx + 1]
        art, X, Y, Z = surface(ax, quad(Phi), lim, lim, n=32, cmap="viridis", alpha=0.45)
        s.add(art)
        s.add(ax.contour(X, Y, Z, levels=np.linspace(0.6, 9.0, 7), zdir="z", offset=0,
                         cmap="viridis", linewidths=0.85, alpha=0.85))
        t_rem = T - int(round(u))
        s.status(rf"Riccati t = {t_rem}/{T}   tr($\Phi_t$) = {np.trace(Phi):.2f}")
        s.spin(i, n, sweep=42)

    return update


# =====================================================================================
@animation("LQR Control vs Uncontrolled Drift",
           caption=r"""यहाँ फ़र्श पर $(p, v)$ हैं और ऊर्ध्वाधर अक्ष समय $t$ है; बीच की हरी खड़ी रेखा लक्ष्य $(0, 0)$ है।
           दोनों पथ एक ही शुरुआती स्थिति और एक ही random process noise $w_t$ झेलते हैं। बिना नियंत्रण ($a_t = 0$,
           लाल पथ) के वेग और स्थिति भटककर बाहर भाग जाते हैं। वहीं LQR की linear policy $a_t = L_t s_t$ (नीला पथ)
           पहले वेग मोड़कर पथ को घुमावदार सर्पिल (spiral) में भीतर लाती है और शोर के बावजूद $(0, 0)$ के गिर्द बाँधे
           रखती है।""",
           frames=40, keys=(0.15, 0.55, 1.0))
def lqr_spiral_vs_drift(s, n):
    dt, T = 0.22, 38
    # Lightly unstable oscillatory 2D dynamics so uncontrolled drifts & LQR spirals cleanly
    A = np.array([[1.0, dt], [-0.55 * dt, 1.0 + 0.16 * dt]])
    B = np.array([[0.5 * dt**2], [dt]])
    U = np.diag([1.6, 0.08])
    W = np.array([[1.1]])
    Phi = np.diag([3.0, 0.6])
    gains = [None] * T
    for t in range(T - 1, -1, -1):
        L = -np.linalg.solve(W + B.T @ Phi @ B, B.T @ Phi @ A)
        gains[t] = L
        Phi = U + A.T @ Phi @ A + A.T @ Phi @ B @ L

    rng = np.random.default_rng(12)
    noise = rng.normal(0.0, 0.045, size=(T, 2))
    s0 = np.array([1.55, 0.85])
    ctrl, free = [s0.copy()], [s0.copy()]
    for t in range(T):
        a = gains[t] @ ctrl[-1]
        ctrl.append(A @ ctrl[-1] + B @ a + noise[t])
        free.append(np.clip(A @ free[-1] + noise[t], -2.5, 2.5))
    ctrl, free = np.array(ctrl), np.array(free)
    ts = np.arange(T + 1, dtype=float)

    ax = s.ax3d(labels=("pos $p$", "vel $v$", "time $t$"), elev=24, azim=-55,
                xlim=(-2.3, 2.3), ylim=(-2.3, 2.3), zlim=(0, T))
    ax.plot([0, 0], [0, 0], [0, T], color=C["green"], lw=1.6, ls="--")
    ax.plot([], [], [], color=C["blue"], lw=2, label="LQR ($a_t = L_t s_t$)")
    ax.plot([], [], [], color=C["red"], lw=2, label="uncontrolled ($a_t = 0$)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        k = max(1, int(round(i / (n - 1) * T)))
        c, f, tt = ctrl[: k + 1], free[: k + 1], ts[: k + 1]
        s.add(ax.plot(c[:, 0], c[:, 1], tt, color=C["blue"], lw=2.1))
        s.add(ax.plot(c[:, 0], c[:, 1], 0 * tt, color=C["blue"], lw=0.8, alpha=0.45, ls=":"))
        s.add(ax.scatter(c[-1:, 0], c[-1:, 1], tt[-1:], color=C["blue"], s=48,
                         depthshade=False, edgecolor="white"))
        s.add(ax.plot(f[:, 0], f[:, 1], tt, color=C["red"], lw=1.8))
        s.add(ax.plot(f[:, 0], f[:, 1], 0 * tt, color=C["red"], lw=0.8, alpha=0.45, ls=":"))
        s.add(ax.scatter(f[-1:, 0], f[-1:, 1], tt[-1:], color=C["red"], s=48,
                         depthshade=False, edgecolor="white"))
        n1, n2 = np.linalg.norm(c[-1]), np.linalg.norm(f[-1])
        s.status(f"t = {k}/{T}   ||s|| LQR={n1:.2f}  drift={n2:.2f}")
        s.spin(i, n, sweep=44)

    return update


# =====================================================================================
@animation("Kalman Filter & Uncertainty",
           caption=r"""असली 2D पथ (हरी रेखा) समय $t$ के साथ ऊपर उठता है, पर हमें सिर्फ़ भारी शोर वाले लाल मापन
           $y_t = C s_t + v_t$ मिलते हैं। शुरू में अनिश्चितता $\Sigma_{0|0}$ बहुत बड़ी है, इसलिए $2\sigma$ अंडाकार
           (नीला घेरा और फ़र्श पर उसकी परछाईं) काफ़ी चौड़ा है। हर कदम पर Predict से नारंगी टूटा घेरा थोड़ा फैलता है और
           Update मापन देखकर नीले घेरे को सिकोड़ देता है। कुछ ही कदमों में अंडाकार छोटा हो जाता है और नीला अनुमान
           $s_{t|t}$ असली पथ से चिपककर चलता है।""",
           frames=40, keys=(0.05, 0.35, 1.0))
def kalman_filter_track(s, n):
    T = 26
    th = 0.13
    A = 0.99 * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    C_obs = np.eye(2)
    Sw = np.diag([0.012, 0.012])
    Sv = np.diag([0.32, 0.32])

    rng = np.random.default_rng(21)
    s_true = [np.array([1.5, -0.4])]
    for _ in range(T):
        s_true.append(A @ s_true[-1] + rng.multivariate_normal(np.zeros(2), Sw))
    s_true = np.array(s_true)
    y_obs = s_true + rng.multivariate_normal(np.zeros(2), Sv, size=T + 1)

    # Kalman filter: store (pred_mean, pred_cov, post_mean, post_cov)
    mu_p = np.array([0.0, 0.0])
    Sig_p = np.diag([1.65, 1.65])
    pred_mu, pred_S, post_mu, post_S = [], [], [], []
    for t in range(T + 1):
        if t > 0:
            mu_p = A @ post_mu[-1]
            Sig_p = A @ post_S[-1] @ A.T + Sw
        pred_mu.append(mu_p.copy())
        pred_S.append(Sig_p.copy())
        K = Sig_p @ C_obs.T @ np.linalg.inv(C_obs @ Sig_p @ C_obs.T + Sv)
        post_mu.append(mu_p + K @ (y_obs[t] - C_obs @ mu_p))
        post_S.append((np.eye(2) - K @ C_obs) @ Sig_p)
    post_mu = np.array(post_mu)
    ang = np.linspace(0, 2 * np.pi, 48)
    unit = np.c_[np.cos(ang), np.sin(ang)]

    def ell_pts(mu, Sig):
        vals, vecs = np.linalg.eigh(Sig)
        L = vecs @ np.diag(2.0 * np.sqrt(np.maximum(vals, 1e-6)))
        return mu + unit @ L.T

    ax = s.ax3d(labels=(r"$s_1$", r"$s_2$", "time $t$"), elev=26, azim=-56,
                xlim=(-2.6, 2.6), ylim=(-2.6, 2.6), zlim=(0, T))
    ax.plot([], [], [], color=C["green"], lw=2, label="true $s_t$")
    ax.plot([], [], [], color=C["blue"], lw=2, label=r"estimate $s_{t|t}$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    ts = np.arange(T + 1, dtype=float)

    def update(i):
        s.clear()
        k = min(T, int(round(i / (n - 1) * T)))
        s.add(ax.plot(s_true[: k + 1, 0], s_true[: k + 1, 1], ts[: k + 1], color=C["green"], lw=1.9))
        s.add(ax.scatter(y_obs[: k + 1, 0], y_obs[: k + 1, 1], ts[: k + 1],
                         color=C["red"], s=14, alpha=0.65, depthshade=False))
        s.add(ax.plot(post_mu[: k + 1, 0], post_mu[: k + 1, 1], ts[: k + 1], color=C["blue"], lw=2.0))
        s.add(ax.scatter(post_mu[k : k + 1, 0], post_mu[k : k + 1, 1], [ts[k]],
                         color=C["blue"], s=46, depthshade=False, edgecolor="white"))

        # Predicted 2-sigma ellipse (dashed orange) vs updated 2-sigma ellipse (filled blue)
        ep = ell_pts(pred_mu[k], pred_S[k])
        eu = ell_pts(post_mu[k], post_S[k])
        s.add(ax.plot(ep[:, 0], ep[:, 1], np.full(len(ang), ts[k]),
                      color=C["orange"], lw=1.2, ls="--"))
        s.add(ax.plot(eu[:, 0], eu[:, 1], np.full(len(ang), ts[k]), color=C["blue"], lw=1.8))
        s.add(ax.plot(eu[:, 0], eu[:, 1], np.zeros(len(ang)), color=C["blue"], lw=1.1, alpha=0.55))
        verts = [list(zip(eu[:, 0], eu[:, 1], np.full(len(ang), ts[k])))]
        poly = Poly3DCollection(verts, facecolors=[(0.15, 0.39, 0.92, 0.25)], edgecolors="none")
        ax.add_collection3d(poly)
        s.add(poly)
        s.status(rf"step {k}/{T}   tr($\Sigma_{{t|t}}$) = {np.trace(post_S[k]):.3f}")
        s.spin(i, n, sweep=40)

    return update
