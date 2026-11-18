"""Lecture 18: Q-Learning and Value Function Approximation: 3D animations."""
import numpy as np
from matplotlib import colormaps

from lib.anim import C, animation

# ---- shared 4x3 gridworld from Lecture 16 (wall, +1 goal, -1 pit, noisy moves) ----
_W, _H, _GAMMA = 4, 3, 0.99
_WALL, _GOAL, _PIT, _START = (1, 1), (3, 2), (3, 1), (0, 0)
_MOVES = [(0, 1), (1, 0), (0, -1), (-1, 0)]            # actions: up, right, down, left
_CELLS = [(x, y) for y in range(_H) for x in range(_W)]
_NS = len(_CELLS)
_TERMINAL = np.array([c in (_GOAL, _PIT) for c in _CELLS])
_FREE = np.array([c != _WALL for c in _CELLS])
_R = np.full(_NS, -0.02)
_R[_CELLS.index(_GOAL)], _R[_CELLS.index(_PIT)] = 1.0, -1.0


def _move(c, a):
    nxt = (c[0] + _MOVES[a][0], c[1] + _MOVES[a][1])
    inside = 0 <= nxt[0] < _W and 0 <= nxt[1] < _H and nxt != _WALL
    return nxt if inside else c


_NEXT = np.array([[_CELLS.index(_move(c, a)) for a in range(4)] for c in _CELLS])


def _grid_step(rng, s, a):
    """Intended move with probability 0.8, slip to either side with 0.1 each."""
    u = rng.random()
    a = a if u < 0.8 else ((a + 1) % 4 if u < 0.9 else (a - 1) % 4)
    return _NEXT[s, a]


def _grid_vstar():
    P = np.zeros((_NS, 4, _NS))
    for s in range(_NS):
        for a in range(4):
            for a2, p in ((a, 0.8), ((a + 1) % 4, 0.1), ((a - 1) % 4, 0.1)):
                P[s, a, _NEXT[s, a2]] += p
    V = np.zeros(_NS)
    for _ in range(2000):
        Q = _R[:, None] + _GAMMA * P @ V
        Q[_TERMINAL] = _R[_TERMINAL, None]
        V = Q.max(axis=1)
    return V


def _grid_q_learning(episodes, seed=0):
    """Tabular Q-learning; returns Q after every episode (index 0 = before learning)."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((_NS, 4))
    Q[_TERMINAL] = _R[_TERMINAL, None]
    N = np.zeros((_NS, 4))
    snaps = [Q.copy()]
    for ep in range(episodes):
        eps = max(0.1, 1.0 - ep / (0.5 * episodes))
        s = _CELLS.index(_START)
        while not _TERMINAL[s]:
            a = int(rng.integers(4)) if rng.random() < eps else int(Q[s].argmax())
            s2 = _grid_step(rng, s, a)
            N[s, a] += 1
            Q[s, a] += N[s, a] ** -0.7 * (_R[s] + _GAMMA * Q[s2].max() - Q[s, a])
            s = s2
        snaps.append(Q.copy())
    return np.array(snaps)


def _schedule(n, total):
    """n strictly increasing integer checkpoints from 0 to total, dense at the start."""
    raw = (total + 1.0) ** (np.arange(n) / (n - 1)) - 1.0
    out = [0]
    for r in raw[1:]:
        out.append(max(out[-1] + 1, int(round(r))))
    return np.minimum(out, total)


@animation("Q-learning fills in the Q-table",
           caption=r"""फ़र्श पर Lecture 16 वाला 4x3 gridworld है और हर state पर bar की ऊँचाई $\max_a Q(s,a)$ है।
           काले चौकोर frames value iteration से निकला सही जवाब $V^*(s)$ दिखाते हैं, bars के ऊपर के तीर
           greedy action $\arg\max_a Q(s,a)$ हैं, और स्लेटी पट्टी दीवार है। शुरू में सिर्फ़ terminal states (हरा $+1$, लाल $-1$) की
           value पता है। जैसे-जैसे episodes चलते हैं, value goal से **पीछे की ओर** फैलती है: पहले goal के
           पड़ोसी भरते हैं, फिर उनके पड़ोसी। अंत में हर bar अपने frame को छू लेता है, यानी बिना $P_{sa}$ जाने
           Q-learning वहीं पहुँच गया जहाँ value iteration पहुँचता है।""",
           frames=44, keys=(0.0, 0.3, 1.0))
def q_table_fills(s, n):
    episodes = 3000
    snaps = _grid_q_learning(episodes)
    vstar = _grid_vstar()
    eps_at = _schedule(n, episodes)
    ax = s.ax3d(labels=("column", "row", r"$\max_a Q(s,a)$"), elev=30, azim=-62,
                xlim=(0, 4), ylim=(0, 3), zlim=(-1.05, 1.05))
    ax.computed_zorder = False
    ax.set_xticks([0.5, 1.5, 2.5, 3.5], ["1", "2", "3", "4"])
    ax.set_yticks([0.5, 1.5, 2.5], ["1", "2", "3"])
    ax.set_zticks([-1, -0.5, 0, 0.5, 1])
    ax.set_zlabel(r"$\max_a Q(s,a)$", labelpad=1)
    ax.set_box_aspect((4, 3, 2.6), zoom=0.88)
    w = 0.7
    free = np.flatnonzero(_FREE)
    cx = np.array([_CELLS[i][0] + 0.5 for i in free])
    cy = np.array([_CELLS[i][1] + 0.5 for i in free])
    # wall slab and the exact V* frames (static)
    ax.bar3d(_WALL[0] + 0.1, _WALL[1] + 0.1, 0, 0.8, 0.8, 0.04, color=C["gray"], zorder=1)
    sq = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [-1, -1]]) * w / 2
    for x, y, v in zip(cx, cy, vstar[free]):
        ax.plot(x + sq[:, 0], y + sq[:, 1], np.full(5, v), color=C["dark"], lw=0.9, zorder=5)
    ax.plot([], [], [], color=C["dark"], lw=0.9, label=r"exact $V^*$ (value iteration)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    cmap = colormaps["RdYlGn"]

    def update(i):
        s.clear()
        Q = snaps[eps_at[i]]
        v = Q.max(axis=1)[free]
        s.add(ax.bar3d(cx - w / 2, cy - w / 2, np.zeros_like(v), w, w, v,
                       color=cmap(0.5 + 0.5 * v), shade=True, zorder=2))
        for k, idx in enumerate(free):
            if _TERMINAL[idx] or not np.any(Q[idx]):
                continue
            dx, dy = _MOVES[int(Q[idx].argmax())]
            top = max(v[k], 0.0) + 0.04
            s.add(ax.quiver(cx[k] - 0.2 * dx, cy[k] - 0.2 * dy, top, 0.4 * dx, 0.4 * dy, 0,
                            color=C["blue"], linewidth=1.6, arrow_length_ratio=0.45, zorder=7))
        err = np.abs(v - vstar[free]).max()
        s.status(rf"episode {eps_at[i]}    max $|\max_a Q - V^*|$ = {err:.2f}")
        s.spin(i, n, sweep=30)

    return update


# ---- exploration: 5x5 grid with a small reward close by and a big one further away ----
_EX_START, _EX_SMALL, _EX_BIG = (0, 0), (2, 0), (2, 3)


def _explore_run(eps_of, episodes, seed, size=5, r_small=0.3, gamma=0.9, alpha=0.5, max_steps=40):
    """Q-learning on the deterministic 5x5 grid. Returns visit counts and returns per episode."""
    rng = np.random.default_rng(seed)
    Q = np.zeros((size, size, 4))
    visits = np.zeros((size, size))
    counts, returns = [visits.copy()], []
    for ep in range(episodes):
        eps = eps_of(ep)
        x, y = _EX_START
        visits[x, y] += 1
        ret = 0.0
        for t in range(max_steps):
            q = Q[x, y]
            if rng.random() < eps:
                a = int(rng.integers(4))
            else:
                a = int(rng.choice(np.flatnonzero(q == q.max())))       # greedy, ties at random
            nx = min(max(x + _MOVES[a][0], 0), size - 1)
            ny = min(max(y + _MOVES[a][1], 0), size - 1)
            r = r_small if (nx, ny) == _EX_SMALL else (1.0 if (nx, ny) == _EX_BIG else 0.0)
            target = r if r > 0 else gamma * Q[nx, ny].max()
            Q[x, y, a] += alpha * (target - Q[x, y, a])
            x, y = nx, ny
            visits[x, y] += 1
            if r > 0:
                ret = r * gamma ** t
                break
        returns.append(ret)
        counts.append(visits.copy())
    return np.array(counts), np.array(returns)


@animation(r"Exploring: greedy vs $\epsilon$-greedy",
           caption=r"""दो एक जैसे 5x5 grids हैं: बायाँ (लाल) हमेशा greedy action लेता है, दायाँ (नीला) $\epsilon$-greedy
           है जिसमें $\epsilon$ धीरे-धीरे $1$ से $0.1$ तक घटता है। दोनों कोने S से चलते हैं। पास में छोटा reward
           $+0.3$ है और थोड़ा दूर बड़ा reward $+1$। Bar की ऊँचाई बताती है कि agent उस cell में कितनी बार गया
           (log scale)। Greedy agent को जैसे ही छोटा reward मिलता है, वह उसी पतले रास्ते पर अटक जाता है और
           बाकी grid कभी नहीं देखता। $\epsilon$-greedy agent के visits पूरे grid पर फैलते हैं, उसे $+1$ मिल
           जाता है, और उसका average return greedy से काफ़ी ऊपर निकल जाता है। यही exploration की कीमत और फ़ायदा है।""",
           frames=40, keys=(0.08, 0.4, 1.0))
def exploration(s, n):
    episodes = 300
    runs = [_explore_run(lambda ep: 0.0, episodes, seed=4),
            _explore_run(lambda ep: max(0.1, 1.0 - ep / 150), episodes, seed=4)]
    gap = 6.5
    ax = s.ax3d(labels=("", "", "visits"), elev=34, azim=-76,
                xlim=(0, gap + 5), ylim=(0, 5), zlim=(0, 3.2), ticks=False)
    ax.set_box_aspect((gap + 5, 5, 4.6), zoom=0.98)
    ax.set_zticks([0, 1, 2, 3], ["0", "9", "99", "999"])
    ax.set_zlabel("visits", labelpad=0)
    ax.computed_zorder = False
    tile = np.array([[0.06, 0.06], [0.94, 0.06], [0.94, 0.94], [0.06, 0.94], [0.06, 0.06]])
    for off in (0.0, gap):
        for a in range(5):
            for b in range(5):
                ax.plot(off + a + tile[:, 0], b + tile[:, 1], 0, color=C["gray"], lw=0.5, alpha=0.6, zorder=1)
    ax.plot([], [], [], color=C["red"], lw=5, label=r"greedy ($\epsilon$ = 0)")
    ax.plot([], [], [], color=C["blue"], lw=5, label=r"$\epsilon$-greedy ($\epsilon$: 1 to 0.1)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    gx, gy = np.meshgrid(np.arange(5), np.arange(5), indexing="ij")
    gx, gy = gx.ravel(), gy.ravel()
    # draw far rows first so that near bars paint over them
    order = np.argsort(-gy, kind="stable")
    gx, gy = gx[order], gy[order]
    w = 0.74
    tags = {_EX_START: "S", _EX_SMALL: "+0.3", _EX_BIG: "+1"}
    ep_at = np.round(episodes * (np.arange(n) / (n - 1)) ** 1.6).astype(int)

    def update(i):
        s.clear()
        ep = ep_at[i]
        for (counts, _), off, col in zip(runs, (0.0, gap), (C["red"], C["blue"])):
            h = np.log10(1.0 + counts[ep][gx, gy])
            cols = np.array([C["green"] if (a, b) == _EX_BIG else C["yellow"] if (a, b) == _EX_SMALL
                             else col for a, b in zip(gx, gy)])
            seen = (h > 0) | (cols != col)               # goal tiles are always drawn
            hh = np.maximum(h[seen], 0.03)
            s.add(ax.bar3d(off + gx[seen] + (1 - w) / 2, gy[seen] + (1 - w) / 2, 0 * hh, w, w, hh,
                           color=list(cols[seen]), shade=True, zorder=2))
            for cell, tag in tags.items():
                top = np.log10(1.0 + counts[ep][cell]) + 0.12
                s.add(ax.text(off + cell[0] + 0.5, cell[1] + 0.5, top, tag, ha="center", fontsize=8,
                              fontweight="bold", color=C["dark"], zorder=8))
        lo = max(0, ep - 30)
        avg = [r[1][lo:ep].mean() if ep > 0 else 0.0 for r in runs]
        s.status(f"ep {ep}   return: greedy {avg[0]:.2f}, eps {avg[1]:.2f}")
        s.spin(i, n, sweep=14)

    return update


# ---- fitted value iteration on a point mass: state (position, velocity), 3 actions ----
_PM_DT, _PM_GAMMA = 0.2, 0.9
_PM_ACTIONS = (-1.0, 0.0, 1.0)
_PM_CENTERS = np.array([(a, b) for a in np.linspace(-1, 1, 6) for b in np.linspace(-1, 1, 6)])


def _pm_sim(S, a, rng=None, noise=0.0):
    """Simulator: accelerate by a, move, add noise, stay inside the box [-1, 1]^2."""
    v = S[..., 1] + _PM_DT * a
    x = S[..., 0] + _PM_DT * v
    out = np.stack([x, v], axis=-1)
    if noise:
        out = out + rng.normal(0, noise, out.shape)
    return np.clip(out, -1, 1)


def _pm_phi(S):
    """Features: a constant plus a 6x6 grid of Gaussian bumps."""
    d2 = ((S[..., None, :] - _PM_CENTERS) ** 2).sum(axis=-1)
    return np.concatenate([np.ones(S.shape[:-1] + (1,)), np.exp(-d2 / (2 * 0.4 ** 2))], axis=-1)


@animation("Fitted Value Iteration",
           caption=r"""फ़र्श पर एक continuous state space है: एक गाड़ी की position $x$ और velocity $v$, और reward
           $R(s) = -x^2$ है। काले बिंदु sampled states $s^{(i)}$ पर निकाले गए targets
           $y^{(i)} = \max_a \left[R(s^{(i)}) + \gamma E[V(s')]\right]$ हैं, और रंगीन surface fitted value function
           $V(s) = \theta^T\phi(s)$ है। हर iteration में surface को regression से targets पर fit किया जाता है,
           फिर नई surface से नए targets बनते हैं जो थोड़ा और नीचे होते हैं (लाल रेखाएँ यही फ़ासला हैं)। देखिए
           कि surface सिर्फ़ $x$ पर निर्भर नाली से शुरू होकर तिरछी हो जाती है: तेज़ी से दूर भागती state की value
           कम है। अंत में targets surface पर ही बैठ जाते हैं, यानी $V$ अपने ही Bellman backup का fit बन गया।""",
           frames=40, keys=(0.0, 0.2, 1.0))
def fitted_vi(s, n):
    rng = np.random.default_rng(2)
    m, k = 110, 6
    S = rng.uniform(-1, 1, (m, 2))
    Phi = _pm_phi(S)
    theta = np.zeros(Phi.shape[1])
    thetas, targets = [], []
    for _ in range(n):
        q = [-(S[:, 0] ** 2) + _PM_GAMMA * np.mean(
            [_pm_phi(_pm_sim(S, a, rng, 0.05)) @ theta for _ in range(k)], axis=0) for a in _PM_ACTIONS]
        y = np.max(q, axis=0)
        thetas.append(theta)
        targets.append(y)
        theta = np.linalg.lstsq(Phi, y, rcond=None)[0]
    ax = s.ax3d(labels=("position $x$", "velocity $v$", r"$V(s)$"), elev=24, azim=-58,
                xlim=(-1, 1), ylim=(-1, 1), zlim=(-6.6, 0.6))
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    g = np.linspace(-1, 1, 30)
    X, Y = np.meshgrid(g, g)
    PhiG = _pm_phi(np.stack([X, Y], axis=-1))
    ax.scatter([], [], [], color=C["dark"], s=10, label=r"targets $y^{(i)}$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)
    nan = np.full(m, np.nan)

    def update(i):
        s.clear()
        th, y = thetas[i], targets[i]
        fit = Phi @ th
        Z = np.clip(PhiG @ th, -6.6, 0.6)
        s.add(ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.5, linewidth=0, vmin=-6.0, vmax=0.0))
        s.add(ax.plot(np.c_[S[:, 0], S[:, 0], nan].ravel(), np.c_[S[:, 1], S[:, 1], nan].ravel(),
                      np.c_[y, fit, nan].ravel(), color=C["red"], lw=0.8, alpha=0.8))
        s.add(ax.scatter(S[:, 0], S[:, 1], y, color=C["dark"], s=7, depthshade=False))
        s.status(rf"iter {i}   mean $|y - \theta^T\phi|$ = {np.abs(y - fit).mean():.3f}")
        s.spin(i, n, sweep=28)

    return update


@animation("TD error creeps back from the goal",
           caption=r"""यह एक सीधा corridor है: 6 states और दाएँ सिरे पर goal (हरा, value $1$)। Axes हैं: state,
           episode का नंबर, और ऊँचाई $Q(s, \text{right})$। लाल गेंद agent है जो हर episode में बाएँ से दाएँ
           चलता है, और हर कदम पर एक Q-learning update होता है (तस्वीर साफ़ रखने के लिए $\alpha = 1$,
           $\gamma = 0.9$)। पहले episode में TD error $\delta$ सिर्फ़ goal के ठीक पहले वाली state पर nonzero है
           (नारंगी तीर), बाकी जगह $0$। दूसरे episode में उससे एक state पीछे, और इसी तरह। Value की जानकारी हर
           episode में सिर्फ़ **एक कदम** पीछे खिसकती है, इसलिए bars की सीढ़ी तिरछी बनती है। सातवें episode में
           हर जगह $\delta = 0$ है: सीखना पूरा हो गया।""",
           frames=42, keys=(0.12, 0.5, 1.0))
def td_corridor(s, n):
    L, alpha, gamma = 6, 1.0, 0.9
    episodes = n // L
    Q = np.zeros(L + 1)
    Q[L] = 1.0                                     # goal: Q(goal, .) = R(goal) = 1
    steps, rows = [], []                           # one entry per update: (episode, state, delta, row)
    for ep in range(episodes):
        row = np.full(L, np.nan)                   # bars of this episode appear as the agent passes
        for st in range(L):
            delta = 0.0 + gamma * Q[st + 1] - Q[st]          # R(s) = 0 inside the corridor
            Q[st] += alpha * delta
            row = row.copy()
            row[st] = Q[st]
            steps.append((ep, st, delta, row, Q[st + 1]))
        rows.append(row)
    ax = s.ax3d(labels=("state", "episode", r"$Q(s,\mathrm{right})$"), elev=30, azim=-62,
                xlim=(0, L + 1), ylim=(0.5, episodes + 0.5), zlim=(0, 1.1))
    ax.computed_zorder = False
    ax.set_xticks(np.arange(L + 1) + 0.5, [str(k + 1) for k in range(L)] + ["goal"])
    ax.set_yticks(np.arange(1, episodes + 1))
    ax.set_zlabel(r"$Q(s,\mathrm{right})$", labelpad=0)
    cmap = colormaps["plasma"]
    wx, wy = 0.7, 0.5

    def draw_row(ep, vals):
        z = 2 + 2 * (episodes - ep)
        xs = np.flatnonzero(~np.isnan(vals) & (np.nan_to_num(vals) > 0))
        s.add(ax.bar3d(L + 0.15, ep + 1 - wy / 2, 0, wx, wy, 1.0, color=C["green"], shade=True, zorder=z))
        if len(xs):
            s.add(ax.bar3d(xs + 0.15, ep + 1 - wy / 2, 0 * xs, wx, wy, vals[xs],
                           color=cmap(0.1 + 0.75 * vals[xs]), shade=True, zorder=z + 1))

    def update(i):
        s.clear()
        ep, st, delta, row, q_next = steps[min(i, len(steps) - 1)]
        for e in range(ep + 1):
            draw_row(e, rows[e] if e < ep else row)
        if delta > 1e-9:                                   # the backup: value flows from s' to s
            s.add(ax.quiver(st + 1.5, ep + 1, q_next + 0.06, -1.0, 0, row[st] - q_next,
                            color=C["orange"], linewidth=2.2, arrow_length_ratio=0.3, zorder=60))
        s.add(ax.scatter([st + 0.5], [ep + 1], [row[st] + 0.06], color=C["red"], s=55,
                         depthshade=False, edgecolor="white", zorder=61))
        s.status(rf"episode {ep + 1}, state {st + 1}:   TD error $\delta$ = {delta:.2f}")
        s.spin(i, n, sweep=20)

    return update


@animation("Q(s,a) surface and its greedy ridge",
           caption=r"""यहाँ state $s$ एक line पर position है, action $a$ यह है कि अगले कदम में कितना खिसकना है
           (ज़्यादा से ज़्यादा $0.4$), और reward $R(s) = -s^2$ है। ऊँचाई tabular Q-learning से सीखा $Q(s,a)$ है।
           Agent पूरी तरह **random** actions लेकर घूमता है, फिर भी update में $\max_{a'}$ होने से वह greedy
           policy की value सीखता है: यही off-policy होना है। लाल रेखा हर $s$ के लिए सबसे ऊँचा बिंदु
           $\arg\max_a Q(s,a)$ है और फ़र्श पर उसकी परछाईं policy $\pi(s)$ है। शुरू में ridge बेमतलब है। अंत में
           वह हरी रेखा (सही policy: origin की ओर जितना हो सके उतना चलो) पर बैठ जाती है।""",
           frames=40, keys=(0.1, 0.45, 1.0))
def q_surface_ridge(s, n):
    ns, na, gamma, alpha = 31, 13, 0.8, 0.5
    states = np.linspace(-1, 1, ns)
    shifts = np.arange(na) - na // 2               # action = shift by this many grid cells
    actions = shifts * (states[1] - states[0])
    rng = np.random.default_rng(5)
    total = 60000
    marks = _schedule(n, total)
    mark_set = set(marks.tolist())
    Q = np.zeros((ns, na))
    snaps = {0: Q.copy()}
    si = int(rng.integers(ns))
    for t in range(1, total + 1):
        if t % 25 == 0:
            si = int(rng.integers(ns))             # restart somewhere new now and then
        ai = int(rng.integers(na))                 # behaviour policy: uniformly random
        s2 = int(np.clip(si + shifts[ai], 0, ns - 1))
        Q[si, ai] += alpha * (-states[si] ** 2 + gamma * Q[s2].max() - Q[si, ai])
        si = s2
        if t in mark_set:
            snaps[t] = Q.copy()
    zlo = -1.5
    ax = s.ax3d(labels=("state $s$", "action $a$", r"$Q(s,a)$"), elev=30, azim=-128,
                xlim=(-1, 1), ylim=(actions[0], actions[-1]), zlim=(zlo, 0.05))
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-0.4, -0.2, 0, 0.2, 0.4])
    ax.set_zticks([-1.5, -1, -0.5, 0])
    ax.set_zlabel(r"$Q(s,a)$", labelpad=0)
    A, Sg = np.meshgrid(actions, states)
    best = np.clip(-states, actions[0], actions[-1])
    ax.plot(states, best, zlo, color=C["green"], lw=2.2, label=r"optimal $\pi^*(s)$")
    ax.plot([], [], [], color=C["red"], lw=2, label=r"greedy ridge $\arg\max_a Q$")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.98), frameon=False)

    def update(i):
        s.clear()
        Qi = snaps[marks[i]]
        s.add(ax.plot_surface(Sg, A, Qi, cmap="viridis", alpha=0.5, linewidth=0, vmin=zlo, vmax=0.0))
        g = Qi.argmax(axis=1)
        s.add(ax.plot(states, actions[g], Qi.max(axis=1) + 0.02, color=C["red"], lw=2.0))
        s.add(ax.plot(states, actions[g], zlo + 0.01, color=C["red"], lw=1.2, ls="--"))
        agree = np.mean(np.abs(actions[g] - best) < 1e-9)
        s.status(f"step {marks[i]}   greedy = optimal: {100 * agree:.0f}% states")
        s.spin(i, n, sweep=28)

    return update
