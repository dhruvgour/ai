"""Lecture 19: Policy Search, REINFORCE and POMDPs: 3D animations."""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.anim import C, animation, arrow, ease, lerp, surface


def _sig(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


# ---- shared 1D stabilization MDP for Animation 1 (7 states, exact J + real REINFORCE) ----
_XS = np.arange(-3, 4, dtype=float)          # states -3..+3, target x* = 1
_NS1 = len(_XS)
_START1 = 2                                  # x_0 = -1
_HORIZON1 = 6
_REW1 = np.exp(-0.5 * (_XS - 1.0) ** 2)      # +1 at target x=1, decays away


def _exact_J_grid(t1, t2):
    """Exact expected return J(theta1, theta2) via Markov transition matrices (works on grids)."""
    t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
    shape = np.broadcast(t1, t2).shape
    p_right = _sig(t1[..., None] + t2[..., None] * (_XS - 1.0))          # (..., 7)
    # Transition matrix T[..., s, s']: action +1 moves right (0.9) / stays (0.1), action -1 moves left
    T = np.zeros(shape + (_NS1, _NS1))
    for s in range(_NS1):
        sr, sl = min(_NS1 - 1, s + 1), max(0, s - 1)
        pr = p_right[..., s]
        T[..., s, sr] += pr * 0.9
        T[..., s, s] += pr * 0.1 + (1.0 - pr) * 0.1
        T[..., s, sl] += (1.0 - pr) * 0.9
    dist = np.zeros(shape + (_NS1,))
    dist[..., _START1] = 1.0
    total = np.zeros(shape)
    for _ in range(_HORIZON1):
        dist = np.einsum("...s,...st->...t", dist, T)
        total = total + np.einsum("...t,t->...", dist, _REW1)
    return total


def _run_reinforce_1d(start_th, steps, batch=6, alpha=0.22, seed=4):
    """Run real REINFORCE with reward-to-go and per-step baseline on the 1D stabilization MDP."""
    rng = np.random.default_rng(seed)
    th = np.array(start_th, float)
    path = [th.copy()]
    b_run = np.zeros(_HORIZON1)
    for k in range(steps):
        ep_feats, ep_acts, ep_Gs = [], [], []
        for _ in range(batch):
            s = _START1
            states, acts, rews = [], [], []
            for _t in range(_HORIZON1):
                feat = np.array([1.0, _XS[s] - 1.0])
                pr = float(_sig(th @ feat))
                a = 1 if rng.random() < pr else 0
                if rng.random() < 0.9:
                    s = min(_NS1 - 1, s + 1) if a == 1 else max(0, s - 1)
                states.append(feat)
                acts.append(a)
                rews.append(_REW1[s])
            ep_feats.append(states)
            ep_acts.append(acts)
            ep_Gs.append(np.cumsum(rews[::-1])[::-1])
        G_mat = np.array(ep_Gs)
        b_run = 0.75 * b_run + 0.25 * G_mat.mean(axis=0) if k > 0 else G_mat.mean(axis=0)
        grad = np.zeros(2)
        for states, acts, G in zip(ep_feats, ep_acts, G_mat):
            for t_i, (feat, a, gt) in enumerate(zip(states, acts, G)):
                grad += (gt - b_run[t_i]) * (a - _sig(th @ feat)) * feat
        th = np.clip(th + alpha * (grad / batch), [-3.1, -3.4], [3.1, 2.4])
        path.append(th.copy())
    return np.array(path)


def _exact_ascent_1d(start_th, steps, alpha=0.28):
    th = np.array(start_th, float)
    path = [th.copy()]
    eps = 1e-4
    for _ in range(steps):
        g = np.array([
            (_exact_J_grid(th[0] + eps, th[1]) - _exact_J_grid(th[0] - eps, th[1])) / (2 * eps),
            (_exact_J_grid(th[0], th[1] + eps) - _exact_J_grid(th[0], th[1] - eps)) / (2 * eps),
        ])
        th = np.clip(th + alpha * g, [-3.1, -3.4], [3.1, 2.4])
        path.append(th.copy())
    return np.array(path)


# =====================================================================================
@animation(r"REINFORCE Ascent on $J(\theta)$",
           caption=r"""यहाँ ज़मीन पर policy के दो parameters $(\theta_1, \theta_2)$ हैं और ऊँचाई सच्चा expected return
           $J(\theta) = E_{\tau \sim p_\theta}[R(\tau)]$ है। काली टूटी रेखा सच्चे gradient $\nabla_\theta J(\theta)$ का
           रास्ता है, जबकि नारंगी रेखा असली sampled trajectories से चलने वाले REINFORCE का रास्ता है। हर कदम पर सिर्फ़
           चंद rollouts से gradient का अंदाज़ा लगता है, इसलिए रास्ता थोड़ा टेढ़ा-मेढ़ा (noisy) है। फिर भी unbiased होने के
           कारण यह नीचे की घाटी से चढ़कर ऊँची चोटी तक पहुँच जाता है।""",
           frames=42, keys=(0.05, 0.48, 1.0))
def reinforce_surface(s, n):
    xlim, ylim = (-3.2, 3.2), (-3.5, 2.5)
    ax = s.ax3d(labels=(r"$\theta_1$", r"$\theta_2$", r"$J(\theta)$"), elev=31, azim=-56,
                xlim=xlim, ylim=ylim, zlim=(0.0, 5.2))
    _, X, Y, Z = surface(ax, _exact_J_grid, xlim, ylim, n=34, cmap="viridis", alpha=0.44)
    ax.contour(X, Y, Z, levels=np.linspace(0.6, 4.6, 9), zdir="z", offset=0.0, cmap="viridis",
               linewidths=0.7, alpha=0.75)

    start = (-2.2, 1.5)
    sub = 2
    rf_path = _run_reinforce_1d(start, (n - 1) * sub, batch=5, alpha=0.22, seed=7)
    rf_z = _exact_J_grid(rf_path[:, 0], rf_path[:, 1])
    ex_path = _exact_ascent_1d(start, n - 1, alpha=0.34)
    ex_z = _exact_J_grid(ex_path[:, 0], ex_path[:, 1])

    ax.plot(ex_path[:, 0], ex_path[:, 1], ex_z + 0.05, color=C["dark"], lw=1.3, ls="--",
            label=r"exact $\nabla J$")
    ax.plot([], [], [], color=C["orange"], lw=2.0, label="REINFORCE")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        k = i * sub
        p, zz = rf_path[: k + 1], rf_z[: k + 1]
        s.add(ax.plot(p[:, 0], p[:, 1], zz + 0.06, color=C["orange"], lw=1.9))
        s.add(ax.plot(p[:, 0], p[:, 1], np.zeros_like(zz), color=C["orange"], lw=0.9, ls=":"))
        s.add(ax.scatter(p[-1:, 0], p[-1:, 1], zz[-1:] + 0.07, color=C["red"], s=52,
                         depthshade=False, edgecolor="white"))
        s.status(f"step {k:2d}   theta=({p[-1, 0]:+.2f},{p[-1, 1]:+.2f})   J={zz[-1]:.2f}")
        s.spin(i, n, sweep=42)

    return update


# =====================================================================================
@animation(r"Policy $\pi_\theta$ Over 2D State",
           caption=r"""यहाँ ज़मीन पर 2D state $(s_1, s_2)$ (position और velocity) है और ऊँचाई logistic policy
           $\pi_\theta(a{=}1 \mid s) = \sigma(\theta_0 + \theta_1 s_1 + \theta_2 s_2)$ है। शुरुआत में $\theta = 0$
           है, इसलिए surface $0.5$ पर बिल्कुल सपाट चादर जैसी है और agent सिक्का उछालने जैसा व्यवहार करता है। जैसे-जैसे
           REINFORCE सीखता है, $\|\theta\|$ बढ़ता है और यह चादर मुड़कर एक पैनी S-आकार की ढलान बन जाती है। फ़र्श पर
           नीले और लाल बिंदु उस दौर में लिए गए असली actions हैं, जो साफ़ तौर पर boundary के दोनों ओर बँट जाते हैं।""",
           frames=40, keys=(0.0, 0.42, 1.0))
def policy_sharpening(s, n):
    rng = np.random.default_rng(12)
    ax = s.ax3d(labels=(r"pos $s_1$", r"vel $s_2$", r"$\pi_\theta(a{=}1|s)$"), elev=27, azim=-132,
                xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), zlim=(0.0, 1.05))

    # Train 3-parameter logistic thruster policy on 2D (pos, vel) stabilization
    th = np.zeros(3)
    total_iters = 156
    snap_idx = set(np.round(np.linspace(0, total_iters, n)).astype(int))
    snaps = []

    for it in range(total_iters + 1):
        ep_data, pts_s, pts_a = [], [], []
        for ep in range(20):
            st = rng.uniform(-1.6, 1.6, 2)
            feats, acts, rews = [], [], []
            for _t in range(12):
                f = np.array([1.0, st[0], st[1]])
                p1 = float(_sig(th @ f))
                a = 1 if rng.random() < p1 else 0
                u = 1.0 if a == 1 else -1.0
                if ep < 6 and _t < 8:
                    pts_s.append(st.copy())
                    pts_a.append(a)
                st = np.array([
                    np.clip(st[0] + 0.35 * st[1], -2.2, 2.2),
                    np.clip(st[1] + 0.45 * u + rng.normal(0.0, 0.12), -2.2, 2.2),
                ])
                r = float(np.exp(-0.6 * (st[0] ** 2 + 0.5 * st[1] ** 2)))
                feats.append(f)
                acts.append(a)
                rews.append(r)
            ep_data.append((feats, acts, np.cumsum(rews[::-1])[::-1]))
        G_mat = np.array([d[2] for d in ep_data])
        b_step = G_mat.mean(axis=0)
        grad = np.zeros(3)
        for feats, acts, G in ep_data:
            for t_i, (f, a, gt) in enumerate(zip(feats, acts, G)):
                grad += (gt - b_step[t_i]) * (a - _sig(th @ f)) * f
        mean_r = float(G_mat[:, 0].mean())
        if it in snap_idx:
            snaps.append((th.copy(), mean_r, np.array(pts_s), np.array(pts_a), it))
        th = th + 0.30 * (grad / 20.0)

    # Ensure exact frame count matches n
    while len(snaps) < n:
        snaps.append(snaps[-1])

    xs = np.linspace(-2.0, 2.0, 32)
    S1, S2 = np.meshgrid(xs, xs)

    def update(i):
        s.clear()
        w, mr, ps, pa, it = snaps[i]
        Z = _sig(w[0] + w[1] * S1 + w[2] * S2)
        s.add(ax.plot_surface(S1, S2, Z, cmap="coolwarm", vmin=0.0, vmax=1.0,
                              alpha=0.45, linewidth=0, antialiased=True))
        m1 = pa == 1
        p_clipped = np.clip(ps, -1.95, 1.95)
        if np.any(m1):
            s.add(ax.scatter(p_clipped[m1, 0], p_clipped[m1, 1], np.zeros(m1.sum()),
                             color=C["red"], s=11, alpha=0.75, depthshade=False))
        if np.any(~m1):
            s.add(ax.scatter(p_clipped[~m1, 0], p_clipped[~m1, 1], np.zeros((~m1).sum()),
                             color=C["blue"], s=11, alpha=0.75, depthshade=False))
        if abs(w[2]) > 0.15:
            bx = np.linspace(-2.0, 2.0, 40)
            by = -(w[0] + w[1] * bx) / w[2]
            ok = np.abs(by) <= 2.0
            if np.any(ok):
                s.add(ax.plot(bx[ok], by[ok], np.zeros(ok.sum()), color=C["dark"], lw=1.6, ls="--"))
        s.status(f"iter {it:3d}   ||theta||={np.linalg.norm(w):.2f}   return={mr:.2f}")
        s.spin(i, n, sweep=38)

    return update


# =====================================================================================
@animation("Trajectories Funnel to Goal",
           caption=r"""यहाँ $(x, y)$ फ़र्श पर 2D मैदान है और ऊर्ध्वाधर अक्ष समय $t = 0 \dots 16$ है। हरा स्तंभ
           लक्ष्य (goal) की जगह है। शुरू में 4-action softmax policy यादृच्छिक है, इसलिए 14 sampled trajectories
           हर दिशा में बिखर जाती हैं। REINFORCE जैसे-जैसे अच्छे trajectories के actions को बढ़ावा देता है, सारे रास्ते
           मुड़कर एक साफ़ कुप्पी (funnel) की तरह सीधे हरे लक्ष्य स्तंभ में समाने लगते हैं।""",
           frames=40, keys=(0.0, 0.16, 1.0))
def trajectory_funnel(s, n):
    rng = np.random.default_rng(21)
    T_hor = 16
    goal = np.array([1.15, 1.15])
    start_mu = np.array([-1.2, -1.1])
    moves = np.array([[0.0, 0.38], [0.38, 0.0], [0.0, -0.38], [-0.38, 0.0]])

    ax = s.ax3d(labels=("x", "y", "time t"), elev=25, azim=-58,
                xlim=(-2.0, 2.0), ylim=(-2.0, 2.0), zlim=(0, T_hor))
    # Static goal pillar through spacetime
    ax.plot([goal[0], goal[0]], [goal[1], goal[1]], [0, T_hor], color=C["green"], lw=2.6)
    ang = np.linspace(0, 2 * np.pi, 40)
    for z0 in (0.0, float(T_hor)):
        ax.plot(goal[0] + 0.25 * np.cos(ang), goal[1] + 0.25 * np.sin(ang),
                np.full_like(ang, z0), color=C["green"], lw=1.1, alpha=0.8)
    ax.scatter([goal[0]], [goal[1]], [T_hor], marker="*", s=110, color=C["green"], depthshade=False)
    ax.scatter([start_mu[0]], [start_mu[1]], [0.0], marker="o", s=45, color=C["dark"], depthshade=False)

    # Train 4-action linear softmax policy: logits(s) = W @ [x - gx, y - gy]
    W = np.zeros((4, 2))
    total_epochs = 95
    snap_epochs = set(np.round(np.linspace(0, total_epochs, n)).astype(int))
    history = []

    def softmax_probs(feat, weights):
        z = weights @ feat
        z = z - z.max()
        ez = np.exp(z)
        return ez / ez.sum()

    for ep in range(total_epochs + 1):
        batch_data, vis_trajs = [], []
        for m in range(28):
            # Diverse training starts around the arena; first 14 start from start_mu for clean funnel display
            if m < 14:
                pos = np.clip(start_mu + rng.normal(0.0, 0.13, 2), -1.85, 1.85)
            else:
                pos = rng.uniform(-1.7, 1.7, 2)
            pts = [pos.copy()]
            feats, acts, probs_l, rews = [], [], [], []
            for _t in range(T_hor):
                feat = np.array([pos[0] - goal[0], pos[1] - goal[1]])
                pr = softmax_probs(feat, W)
                a = int(rng.choice(4, p=pr))
                pos = np.clip(pos + moves[a] + rng.normal(0.0, 0.045, 2), -1.92, 1.92)
                d2 = float(np.sum((pos - goal) ** 2))
                r = -0.25 * d2 + 1.5 * float(np.exp(-2.5 * d2))
                pts.append(pos.copy())
                feats.append(feat)
                acts.append(a)
                probs_l.append(pr)
                rews.append(r)
            G = np.cumsum(rews[::-1])[::-1]
            batch_data.append((feats, acts, probs_l, G))
            if m < 14:
                vis_trajs.append(np.array(pts))
        Gs_mat = np.array([d[3] for d in batch_data])       # (28, T_hor)
        b_step = Gs_mat.mean(axis=0)                        # per-step baseline b_t
        grad = np.zeros_like(W)
        for feats, acts, probs_l, G in batch_data:
            for t_i, (feat, a, pr, gt) in enumerate(zip(feats, acts, probs_l, G)):
                one_hot = np.zeros(4)
                one_hot[a] = 1.0
                grad += (gt - b_step[t_i]) * np.outer(one_hot - pr, feat)
        mr = float(Gs_mat[:14, 0].mean())
        if ep in snap_epochs:
            end_dist = float(np.mean([np.linalg.norm(tr[-1] - goal) for tr in vis_trajs]))
            history.append((vis_trajs, mr, end_dist, ep))
        W = W + 0.22 * (grad / len(batch_data))

    while len(history) < n:
        history.append(history[-1])

    ts = np.arange(T_hor + 1, dtype=float)

    def update(i):
        s.clear()
        trajs, mr, edist, ep = history[i]
        xs, ys, zs = [], [], []
        end_xy = np.array([tr[-1] for tr in trajs])
        for tr in trajs:
            xs += list(tr[:, 0]) + [np.nan]
            ys += list(tr[:, 1]) + [np.nan]
            zs += list(ts) + [np.nan]
        col = C["blue"] if edist > 0.55 else C["teal"]
        s.add(ax.plot(xs, ys, zs, color=col, lw=1.15, alpha=0.72))
        near = np.linalg.norm(end_xy - goal, axis=1) < 0.45
        pt_cols = np.where(near, C["green"], C["orange"])
        s.add(ax.scatter(end_xy[:, 0], end_xy[:, 1], np.full(len(trajs), float(T_hor)),
                         color=pt_cols, s=26, depthshade=False, edgecolor="white"))
        s.status(f"epoch {ep:3d}   return={mr:+.1f}   goal dist={edist:.2f}")
        s.spin(i, n, sweep=44)

    return update


# =====================================================================================
@animation("Gradient Clouds: Baseline Effect",
           caption=r"""यहाँ तीनों अक्ष तीन policy parameters के gradient estimates $(\hat{g}_1, \hat{g}_2, \hat{g}_3)$ हैं।
           जब हर कदम पर धनात्मक survival reward जुड़ता है, तो बिना baseline के एक-एक trajectory के estimates (लाल बादल)
           चारों ओर बहुत दूर-दूर बिखरते हैं। Baseline घटाते ही estimates (नीला बादल) बेहद सघन हो जाते हैं। दोनों बादलों का
           औसत तीर अंततः एक ही सच्चे $\nabla_\theta J(\theta)$ (हरा तारा) पर जाता है क्योंकि $E[b\,\nabla_\theta \log \pi_\theta] = 0$,
           पर नीले बादल का variance कई गुना छोटा है।""",
           frames=40, keys=(0.1, 0.5, 1.0))
def baseline_clouds(s, n):
    rng = np.random.default_rng(33)
    th = np.array([0.15, -1.05, -0.75])

    def sample_grads(m_ep):
        """Sample single-trajectory REINFORCE gradients in R^3 without and with baseline."""
        raw_list, rtg_list, scores_step, Gs_step = [], [], [], []
        for _ in range(m_ep):
            st = rng.uniform(-1.4, 1.4, 2)
            sc_ep, rw_ep = [], []
            for _t in range(10):
                f = np.array([1.0, st[0], st[1]])
                p1 = float(_sig(th @ f))
                a = 1 if rng.random() < p1 else 0
                u = 1.0 if a == 1 else -1.0
                st = np.array([
                    np.clip(st[0] + 0.35 * st[1], -2.2, 2.2),
                    np.clip(st[1] + 0.45 * u + rng.normal(0.0, 0.15), -2.2, 2.2),
                ])
                # Positive survival offset +1.6 makes raw R(tau) ~ 20, exposing baseline role clearly
                r = 1.6 + float(np.exp(-0.6 * (st[0] ** 2 + 0.5 * st[1] ** 2)))
                sc_ep.append((a - p1) * f)
                rw_ep.append(r)
            sc_arr = np.array(sc_ep)                        # (10, 3)
            G = np.cumsum(rw_ep[::-1])[::-1]                # (10,)
            raw_list.append(G[0] * sc_arr.sum(axis=0))
            scores_step.append(sc_arr)
            Gs_step.append(G)
        Gs_mat = np.array(Gs_step)                          # (m_ep, 10)
        b_step = Gs_mat.mean(axis=0)                        # state-independent time baseline b_t
        for sc_arr, G in zip(scores_step, Gs_step):
            rtg_list.append(((G - b_step)[:, None] * sc_arr).sum(axis=0))
        return np.array(raw_list), np.array(rtg_list)

    # High-sample reference for true expected gradient, plus 130 display trajectories
    _, ref_rtg = sample_grads(5000)
    true_g = ref_rtg.mean(axis=0)
    g_raw, g_base = sample_grads(130)

    lim = 48.0
    ax = s.ax3d(labels=(r"$\hat{g}_1$", r"$\hat{g}_2$", r"$\hat{g}_3$"), elev=24, azim=-52,
                xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
    ax.scatter([0.0], [0.0], [0.0], color=C["dark"], s=25, marker="+", depthshade=False)
    ax.scatter([true_g[0]], [true_g[1]], [true_g[2]], color=C["green"], s=110, marker="*",
               depthshade=False, zorder=9)
    ax.plot([], [], [], "o", color=C["red"], ms=5, label="no baseline")
    ax.plot([], [], [], "o", color=C["blue"], ms=5, label="with baseline")
    ax.plot([], [], [], "*", color=C["green"], ms=8, label=r"true $\nabla J$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    # Unit sphere mesh (20x20 <= 40x40) for translucent covariance ellipsoids
    u = np.linspace(0, 2 * np.pi, 20)
    v = np.linspace(0, np.pi, 16)
    SPH = np.stack([np.outer(np.cos(u), np.sin(v)),
                    np.outer(np.sin(u), np.sin(v)),
                    np.outer(np.ones_like(u), np.cos(v))], axis=-1)

    counts = np.round(np.linspace(8, len(g_raw), n)).astype(int)

    def _ellipsoid(pts, color, alpha):
        cov = np.cov(pts.T) + 1e-3 * np.eye(3)
        vals, vecs = np.linalg.eigh(cov)
        radii = np.sqrt(np.maximum(vals, 1e-3))
        E = pts.mean(axis=0) + SPH @ (vecs * radii).T
        E = np.clip(E, -lim * 0.98, lim * 0.98)
        return ax.plot_surface(E[..., 0], E[..., 1], E[..., 2], color=color, alpha=alpha,
                               linewidth=0, shade=False)

    def update(i):
        s.clear()
        k = counts[i]
        r_sub = np.clip(g_raw[:k], -lim * 0.96, lim * 0.96)
        b_sub = np.clip(g_base[:k], -lim * 0.96, lim * 0.96)
        s.add(ax.scatter(r_sub[:, 0], r_sub[:, 1], r_sub[:, 2], color=C["red"], s=14,
                         alpha=0.45, depthshade=False))
        s.add(ax.scatter(b_sub[:, 0], b_sub[:, 1], b_sub[:, 2], color=C["blue"], s=18,
                         alpha=0.82, depthshade=False))
        s.add(_ellipsoid(g_raw[:k], C["red"], 0.12))
        s.add(_ellipsoid(g_base[:k], C["blue"], 0.28))
        mr = g_raw[:k].mean(axis=0)
        mb = g_base[:k].mean(axis=0)
        s.add(arrow(ax, (0, 0, 0), mr * 5.0, color=C["red"], lw=2.0))
        s.add(arrow(ax, (0, 0, 0), mb * 5.0, color=C["blue"], lw=2.2))
        s.add(arrow(ax, (0, 0, 0), true_g * 5.0, color=C["green"], lw=1.8))
        vr = float(np.trace(np.cov(g_raw[:k].T)))
        vb = float(np.trace(np.cov(g_base[:k].T)))
        s.status(f"N={k:3d}  Var(no b)={vr:.0f}  Var(base)={vb:.1f}")
        s.spin(i, n, sweep=46)

    return update


# =====================================================================================
@animation("POMDP Belief Update (Bayes Filter)",
           caption=r"""फ़र्श पर 4x3 कमरे के 11 खुले खाने हैं ((2,2) पर दीवार है) और ऊँचाई belief state $b_t(s)$ है। शुरू में
           robot को अपनी जगह बिल्कुल पता नहीं है, इसलिए सारे bars बराबर ($1/11$) हैं। कदम 1 पर "उत्तर और दक्षिण में
           दीवार" दिखते ही दो एक जैसे गलियारों $(2,1)$ और $(2,3)$ पर दो जुड़वाँ चोटियाँ उभरती हैं। जैसे ही robot पूर्व और
           उत्तर की ओर चलता है और नए observations आते हैं, Bayes filter गलत चोटी को खत्म कर देता है और असली
           जगह (हरा तारा) पर एक ऊँचा, पैना belief bar खड़ा हो जाता है।""",
           frames=44, keys=(0.0, 0.12, 1.0))
def pomdp_belief(s, n):
    cols, rows = 4, 3
    wall = (2, 2)
    states = [(c, r) for r in range(1, rows + 1) for c in range(1, cols + 1) if (c, r) != wall]
    idx = {st: i for i, st in enumerate(states)}
    ns = len(states)
    moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]      # N, E, S, W
    anames = ["N", "E", "S", "W"]

    def step_cell(st, a):
        nxt = (st[0] + moves[a][0], st[1] + moves[a][1])
        return nxt if nxt in idx else st

    # Transition P[a, s, s']: 0.85 intended, 0.075 each perpendicular
    P = np.zeros((4, ns, ns))
    for st in states:
        for a in range(4):
            for b_act, pr in ((a, 0.85), ((a - 1) % 4, 0.075), ((a + 1) % 4, 0.075)):
                P[a, idx[st], idx[step_cell(st, b_act)]] += pr

    # 4-bit wall signature (N, E, S, W blocked?) with sensor accuracy 0.90 per bit
    sigs = np.zeros((ns, 4), int)
    for st in states:
        for d, mv in enumerate(moves):
            sigs[idx[st], d] = 0 if (st[0] + mv[0], st[1] + mv[1]) in idx else 1

    def obs_likelihood(obs_bits):
        match = (sigs == obs_bits[None, :])
        probs = np.where(match, 0.90, 0.10)
        return np.prod(probs, axis=1)

    # True robot path starting at (2, 1):
    # t=0: uniform -> obs at (2,1) gives twin peaks at (2,1) & (2,3)!
    # t=1: move E to (3,1) + obs -> disambiguates (3,1)!
    # t=2: move N to (3,2) + obs -> tracks (3,2)
    # t=3: move N to (3,3) + obs -> tracks (3,3)
    # t=4: move E to (4,3) + obs -> locks onto (4,3)
    true_cells = [(2, 1), (3, 1), (3, 2), (3, 3), (4, 3)]
    taken_acts = [1, 0, 0, 1]                       # E, N, N, E

    # Build sequence of belief snapshots: (belief_vec, true_cell, label_str)
    b = np.full(ns, 1.0 / ns)
    stages = [(b.copy(), true_cells[0], "t=0 prior (uniform)")]
    # First observation at t=1 before moving
    L0 = obs_likelihood(sigs[idx[true_cells[0]]])
    b = (b * L0) / np.sum(b * L0)
    stages.append((b.copy(), true_cells[0], "t=1 obs (N,S walls): twin peaks"))
    for step_k, a in enumerate(taken_acts, start=2):
        b_pred = b @ P[a]
        true_st = true_cells[step_k - 1]
        stages.append((b_pred.copy(), true_st, f"t={step_k} predict after move {anames[a]}"))
        L = obs_likelihood(sigs[idx[true_st]])
        b = (b_pred * L) / np.sum(b_pred * L)
        stages.append((b.copy(), true_st, f"t={step_k} Bayes update"))

    ax = s.ax3d(labels=("column", "row", r"belief $b_t(s)$"), elev=33, azim=-60,
                xlim=(0.5, 4.5), ylim=(0.5, 3.5), zlim=(0.0, 1.0))
    ax.set_box_aspect((4, 3, 2.7), zoom=0.92)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([1, 2, 3])
    ax.computed_zorder = False

    tiles, faces = [], []
    for c in range(1, cols + 1):
        for r in range(1, rows + 1):
            tiles.append([(c - 0.5, r - 0.5, 0.0), (c + 0.5, r - 0.5, 0.0),
                          (c + 0.5, r + 0.5, 0.0), (c - 0.5, r + 0.5, 0.0)])
            faces.append((0.25, 0.27, 0.32, 0.85) if (c, r) == wall else (0.96, 0.97, 0.99, 0.5))
    ax.add_collection3d(Poly3DCollection(tiles, facecolors=faces, edgecolors=(0.45, 0.47, 0.52, 0.9),
                                         linewidths=0.6, zorder=1))

    xs = np.array([st[0] for st in states], float)
    ys = np.array([st[1] for st in states], float)
    w_bar = 0.58
    m_st = len(stages) - 1

    def update(i):
        s.clear()
        pos_f = (i / max(1, n - 1)) * m_st
        k = min(m_st - 1, int(np.floor(pos_f)))
        t_loc = ease(pos_f - k)
        b_cur = lerp(stages[k][0], stages[k + 1][0], t_loc)
        true_c, label = stages[k + 1 if t_loc > 0.5 else k][1], stages[k + 1 if t_loc > 0.5 else k][2]

        cols_bar = plt.cm.YlGnBu(0.22 + 0.78 * np.clip(b_cur / 0.85, 0.0, 1.0))
        s.add(ax.bar3d(xs - w_bar / 2, ys - w_bar / 2, np.zeros(ns), w_bar, w_bar,
                       np.maximum(b_cur, 1e-3), color=cols_bar, alpha=0.88, shade=True,
                       edgecolor=(0, 0, 0, 0.3), linewidth=0.3, zorder=3))
        s.add(ax.scatter([true_c[0]], [true_c[1]], [b_cur[idx[true_c]] + 0.06],
                         marker="*", s=115, color=C["green"], depthshade=False, zorder=8))
        ent = -float(np.sum(b_cur * np.log(np.maximum(b_cur, 1e-12))))
        s.status(f"{label}   H={ent:.2f}")
        s.spin(i, n, sweep=28)

    return update
