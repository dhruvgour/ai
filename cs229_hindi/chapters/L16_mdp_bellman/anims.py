"""Lecture 16: Markov Decision Processes and Bellman Equations: 3D animations.

The 4x3 gridworld of the CS229 notes lies on the floor (column, row); heights are values or payoffs.
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from lib.anim import C, animation, ease

# ---- the 4x3 gridworld: states are (column, row), (1, 1) is the bottom-left corner ----
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
    return nxt if nxt in IDX else st                 # wall or edge: stay where you are


# P[a, s, s']: 0.8 intended direction, 0.1 each perpendicular one. Terminal rows stay zero.
P = np.zeros((4, NS, NS))
for _st in STATES:
    if _st in (GOAL, PIT):
        continue
    for _a in range(4):
        for _b, _p in ((_a, 0.8), ((_a - 1) % 4, 0.1), ((_a + 1) % 4, 0.1)):
            P[_a, IDX[_st], IDX[_move(_st, _b)]] += _p


def _rewards(step=-0.02):
    R = np.full(NS, float(step))
    R[IDX[GOAL]], R[IDX[PIT]] = 1.0, -1.0
    return R


def _policy(text):
    """Rows from top to bottom, one character per cell: ^ > v < (anything else is ignored)."""
    pi = np.zeros(NS, int)
    for k, line in enumerate(text.split()):
        for c, ch in enumerate(line, 1):
            if (c, ROWS - k) in IDX and ch in ARROWS:
                pi[IDX[(c, ROWS - k)]] = ARROWS.index(ch)
    return pi


def _evaluate(pi, gamma=GAMMA, R=None):
    """Exact V^pi from the Bellman linear system (I - gamma P^pi) V = R."""
    R = _rewards() if R is None else R
    return np.linalg.solve(np.eye(NS) - gamma * P[pi, np.arange(NS)], R)


def _optimal(R, gamma=GAMMA):
    """V* and the greedy policy (value iteration, explained in Lecture 17)."""
    V = np.zeros(NS)
    for _ in range(2000):
        V = R + gamma * (P @ V).max(axis=0)
    return V, (P @ V).argmax(axis=0)


PI_GOOD = _policy(">>>. ^.^. ^<<<")      # the optimal policy of the notes
PI_BAD = _policy(">>>. v.>. >>^^")       # the poor policy the notes use to illustrate V^pi
PI_RISKY = _policy(">>>. ^.^. >>^<")     # short cut along the bottom, right past the -1 state

_X = np.array([st[0] for st in STATES], float)
_Y = np.array([st[1] for st in STATES], float)


# ---- drawing helpers ----
def _stage(s, zlabel, zlim=(-1.15, 1.15), elev=34, azim=-64, floor=None, zticks=(-1, -0.5, 0, 0.5, 1)):
    """Axes with the gridworld drawn as tiles at height ``floor`` (default: the bottom of the box)."""
    ax = s.ax3d(labels=("column", "row", zlabel), elev=elev, azim=azim,
                xlim=(0.5, 4.5), ylim=(0.5, 3.5), zlim=zlim)
    ax.set_box_aspect((4, 3, 2.9), zoom=0.93)
    ax.set_xticks([1, 2, 3, 4])
    ax.set_yticks([1, 2, 3])
    ax.set_zticks(list(zticks))
    ax.zaxis.labelpad = -1
    ax.computed_zorder = False                       # we order artists by hand (floor < bars < arrows)
    z = zlim[0] if floor is None else floor
    tiles, faces = [], []
    for c in range(1, COLS + 1):
        for r in range(1, ROWS + 1):
            tiles.append([(c - .5, r - .5, z), (c + .5, r - .5, z), (c + .5, r + .5, z), (c - .5, r + .5, z)])
            faces.append({WALL: (0.25, 0.27, 0.32, 0.85), GOAL: (0.09, 0.64, 0.29, 0.3),
                          PIT: (0.86, 0.15, 0.15, 0.3)}.get((c, r), (1, 1, 1, 0.0)))
    ax.add_collection3d(Poly3DCollection(tiles, facecolors=faces, edgecolors=(0.45, 0.47, 0.52, 0.9),
                                         linewidths=0.6, zorder=1))
    if zlim[0] < 0 and floor is None:                # mark the level V = 0 around the grid
        ax.plot([0.5, 4.5, 4.5, 0.5, 0.5], [0.5, 0.5, 3.5, 3.5, 0.5], [0] * 5, color=C["dark"], lw=0.9,
                ls=(0, (4, 3)), zorder=1.5)
        ax.text(0.5, 3.5, 0.0, "V = 0 ", fontsize=7.5, ha="right", va="center", color=C["dark"], zorder=9)
    return ax


def _colors(V):
    return plt.cm.RdYlGn(0.5 + 0.5 * np.clip(np.asarray(V, float), -1, 1))


def _bars(ax, V, idx=None, width=0.62, alpha=0.93, colors=None, zorder=2, base=-1.15):
    """Value bars: every bar rises from the floor (z = base) up to the height V(s)."""
    idx = np.arange(NS) if idx is None else np.asarray(idx)
    V = np.asarray(V, float)[idx]
    colors = _colors(V) if colors is None else colors
    return ax.bar3d(_X[idx] - width / 2, _Y[idx] - width / 2, base, width, width,
                    np.maximum(V - base, 1e-3), color=colors, alpha=alpha, shade=True,
                    edgecolor=(0, 0, 0, 0.3), linewidth=0.3, zorder=zorder)


def _block(ax, x, y, z0, z1, color, width=0.62, alpha=0.95, zorder=3):
    lo, hi = min(z0, z1), max(z0, z1)
    return ax.bar3d(x - width / 2, y - width / 2, lo, width, width, max(hi - lo, 1e-3), color=color,
                    alpha=alpha, shade=True, edgecolor=(0, 0, 0, 0.4), linewidth=0.3, zorder=zorder)


def _outline(ax, x, y, z, width=0.62, color=C["dark"], lw=1.0, zorder=4, base=-1.15):
    """Wire box from the floor (z = base) to height z: the target value of a bar."""
    h = width / 2
    cx = np.array([-h, h, h, -h, -h]) + x
    cy = np.array([-h, -h, h, h, -h]) + y
    xs, ys, zs = list(cx), list(cy), [z] * 5
    for k in range(4):
        xs += [np.nan, cx[k], cx[k]]
        ys += [np.nan, cy[k], cy[k]]
        zs += [np.nan, base, z]
    return ax.plot(xs, ys, zs, color=color, lw=lw, zorder=zorder)


def _arrows(ax, pi, z, idx=None, color=C["dark"], length=0.46, lw=1.8):
    idx = [i for i in (LIVE if idx is None else idx)]
    if not idx:
        return None
    d = np.array([MOVES[pi[i]] for i in idx], float)
    z = np.broadcast_to(np.asarray(z, float), (NS,))[idx] if np.ndim(z) else np.full(len(idx), z)
    return ax.quiver(_X[idx] - 0.5 * length * d[:, 0], _Y[idx] - 0.5 * length * d[:, 1], z,
                     d[:, 0], d[:, 1], 0 * d[:, 0], length=length, color=color, linewidth=lw,
                     arrow_length_ratio=0.5, zorder=6)


# =====================================================================================
@animation("Episodes under a fixed policy",
           caption=r"""फ़र्श पर 4x3 gridworld है (हरा cell $+1$, लाल cell $-1$, गहरा cell दीवार) और ऊँचाई अब तक का
           discounted payoff $R(s_0) + \gamma R(s_1) + \dots$ है। Policy $\pi$ (फ़र्श के तीर) fixed है, पर चाल noisy है:
           0.8 probability से agent तीर की दिशा में जाता है, 0.1-0.1 से बगल में फिसल जाता है। पहला episode धीरे-धीरे
           चलता है: हर step पर $-0.02$ जुड़ता है और अंत में terminal reward की छलाँग लगती है। फिर सैकड़ों episodes
           जुड़ते हैं: कुछ $+1$ पर खत्म होते हैं, कुछ $-1$ पर। इन सबके payoff का औसत (नीला निशान) धीरे-धीरे
           $V^\pi(s_0)$ (काली रेखा) पर टिक जाता है। यही value function की परिभाषा है: payoff का expectation।""",
           frames=44, keys=(0.16, 0.45, 1.0))
def episodes(s, n):
    ax = _stage(s, "payoff so far", zlim=(-1.4, 1.2), elev=28, azim=-66, floor=0.0)
    _outline(ax, 4, 3, 1.2, width=0.9, color=C["green"], lw=0.7, base=0.0)      # payoffs of +1 endings
    _outline(ax, 4, 2, -1.4, width=0.9, color=C["red"], lw=0.7, base=0.0)       # payoffs of -1 endings
    ax.text(4.75, 3, 1.0, "+1", color=C["green"], fontsize=10, ha="center", fontweight="bold", zorder=9)
    ax.text(4.75, 2, -1.3, "-1", color=C["red"], fontsize=10, ha="center", fontweight="bold", zorder=9)
    _arrows(ax, PI_RISKY, 0.0, color=(0.42, 0.45, 0.5, 0.75), lw=1.3, length=0.4)
    R = _rewards()
    v_start = _evaluate(PI_RISKY)[IDX[START]]
    rng = np.random.default_rng(7)

    def rollout():
        i, t, total = IDX[START], 0, 0.0
        cells, heights = [], []
        while True:
            total += GAMMA**t * R[i]
            cells.append(STATES[i])
            heights.append(total)
            if i in TERMINAL:
                return np.array(cells, float), np.array(heights)
            i, t = rng.choice(NS, p=P[PI_RISKY[i], i]), t + 1

    first = rollout()
    while not (9 <= len(first[1]) <= 11 and first[1][-1] > 0):     # a typical successful episode
        first = rollout()
    rest = [rollout() for _ in range(320)]
    jitter = rng.uniform(-0.24, 0.24, (len(rest), 2))
    slow = len(first[1]) + 2                                      # frames spent on the first episode
    counts = np.round(np.linspace(0, 1, n - slow) ** 1.6 * len(rest)).astype(int)
    ends = np.array([h[-1] for _, h in rest])
    ax.plot([0.6, 0.6], [1, 1], [0, 1.0], color=C["dark"], lw=0.9, zorder=7)     # gauge at the start cell
    ax.plot([0.45, 0.9], [1, 1], [v_start, v_start], color=C["dark"], lw=2.4, zorder=8)
    ax.text(0.5, 1, 1.06, rf"$V^\pi(s_0)$", fontsize=9, ha="center", zorder=9)

    def update(i):
        s.clear()
        k = min(i, len(first[1]) - 1)
        cells, h = first[0][: k + 1], first[1][: k + 1]
        s.add(ax.plot(cells[:, 0], cells[:, 1], h, color=C["blue"], lw=2.0, zorder=7))
        s.add(ax.plot([cells[-1, 0]] * 2, [cells[-1, 1]] * 2, [0, h[-1]], color=C["blue"], lw=0.8, ls=":",
                      zorder=7))
        s.add(ax.scatter(cells[-1:, 0], cells[-1:, 1], h[-1:], color=C["blue"], s=60, depthshade=False,
                         edgecolor="white", zorder=8))
        if i < slow:
            s.status(f"episode 1   t = {k}   payoff so far = {h[-1]:+.2f}")
        else:
            m = counts[i - slow]
            xs, ys, zs = [], [], []
            for (cl, hh), (jx, jy) in zip(rest[:m], jitter[:m]):
                xs += list(cl[:, 0] + jx) + [np.nan]
                ys += list(cl[:, 1] + jy) + [np.nan]
                zs += list(hh) + [np.nan]
            s.add(ax.plot(xs, ys, zs, color=C["gray"], lw=0.5, alpha=0.35, zorder=5))
            if m:
                ex = np.array([cl[-1, 0] for cl, _ in rest[:m]]) + jitter[:m, 0]
                ey = np.array([cl[-1, 1] for cl, _ in rest[:m]]) + jitter[:m, 1]
                col = np.where(ends[:m] > 0, C["green"], C["red"])
                s.add(ax.scatter(ex, ey, ends[:m], color=col, s=9, depthshade=False, zorder=6))
            mean = (first[1][-1] + ends[:m].sum()) / (m + 1)
            s.add(ax.scatter([0.6], [1], [mean], color=C["blue"], s=42, marker="D", depthshade=False,
                             edgecolor="white", zorder=9))
            s.status(rf"episodes: {m + 1}   mean payoff = {mean:.2f}   $V^\pi(s_0)$ = {v_start:.2f}")
        s.spin(i, n, sweep=24)

    return update


# =====================================================================================
@animation("Iterative policy evaluation",
           caption=r"""फ़र्श पर gridworld और तीरों में एक fixed (और काफ़ी खराब) policy $\pi$ है। हर bar की ऊँचाई उस
           state का value estimate है; शून्य से नीचे लटके bars negative values हैं। शुरुआत $V(s) = 0$ से होती है
           और हर sweep में सारे states पर Bellman equation का दायाँ हिस्सा लगाया जाता है:
           $V(s) := R(s) + \gamma \sum_{s'} P_{s\pi(s)}(s') V(s')$। देखिए कि जानकारी terminal states से बाहर की ओर
           फैलती है: $+1$ के पास वाले bars ऊपर उठते हैं, $-1$ की ओर ले जाने वाले states नीचे धँसते हैं। काले
           wire boxes linear system से निकला सही $V^\pi$ हैं; कुछ ही sweeps में bars उनमें ठीक बैठ जाते हैं।""",
           frames=34, keys=(0.1, 0.27, 1.0))
def policy_evaluation(s, n):
    ax = _stage(s, r"$V(s)$")
    R = _rewards()
    target = _evaluate(PI_BAD)
    Ppi = P[PI_BAD, np.arange(NS)]
    sweeps = [np.zeros(NS)]
    for _ in range(n):
        sweeps.append(R + GAMMA * Ppi @ sweeps[-1])
    for i in range(NS):
        _outline(ax, _X[i], _Y[i], target[i], lw=0.7, color=(0.07, 0.09, 0.15, 0.75))

    def update(i):
        s.clear()
        k = min(i, 28)
        V = sweeps[k]
        s.add(_bars(ax, V, alpha=0.8))
        s.add(_arrows(ax, PI_BAD, V + 0.04, lw=1.5, length=0.4))
        s.status(rf"sweep {k}    max $|V - V^\pi|$ = {np.abs(V - target).max():.3f}")
        s.spin(i, n, sweep=30)

    return update


# =====================================================================================
@animation(r"Discount $\gamma$: near-sighted to far",
           caption=r"""Policy वही रहती है (notes वाली अच्छी policy, फ़र्श के तीर), सिर्फ़ discount factor $\gamma$ को 0.05
           से 0.99 तक बढ़ाया जा रहा है, और हर $\gamma$ के लिए सही $V^\pi$ दिखाया गया है। $\gamma$ छोटा हो तो agent
           near-sighted है: सिर्फ़ $+1$ से सटे state को कुछ value मिलती है, बाकी सब लगभग शून्य हैं क्योंकि
           दूर का reward $\gamma^t$ से दबकर गायब हो जाता है। जैसे-जैसे $\gamma$ 1 के पास जाता है, $+1$ की रोशनी
           पूरे grid में फैल जाती है और start state (1,1) भी ऊँचा हो जाता है। Status में $1/(1-\gamma)$ मोटे तौर पर
           बताता है कि agent कितने steps आगे तक देख रहा है।""",
           frames=40, keys=(0.0, 0.55, 1.0))
def discount_sweep(s, n):
    ax = _stage(s, r"$V^\pi(s)$")
    gammas = 0.05 + (0.99 - 0.05) * (1 - (1 - np.linspace(0, 1, n)) ** 1.7)
    values = [_evaluate(PI_GOOD, gamma=g) for g in gammas]

    def update(i):
        s.clear()
        V = values[i]
        s.add(_bars(ax, V, alpha=0.85))
        s.add(_arrows(ax, PI_GOOD, V + 0.04, lw=1.5, length=0.4))
        s.add(ax.text(0.45, 1, V[IDX[START]], f"{V[IDX[START]]:.2f} ", fontsize=9, ha="right", va="center",
                      zorder=9))
        s.status(rf"$\gamma$ = {gammas[i]:.2f}    horizon $1/(1-\gamma)$ = {1 / (1 - gammas[i]):.1f} steps")
        s.spin(i, n, sweep=30)

    return update


# =====================================================================================
@animation("One Bellman backup, close-up",
           caption=r"""यहाँ सिर्फ़ एक state $s = (3,2)$ की Bellman equation को टुकड़ों में जोड़ा जा रहा है। Policy का action
           "ऊपर" है: 0.8 probability से agent $(3,3)$ में पहुँचता है, 0.1 से दीवार से टकराकर वहीं $(3,2)$ में रहता है,
           और 0.1 से $-1$ वाले $(4,2)$ में गिरता है। हर अगले state के value bar का उतना हिस्सा कटकर $s$ पर आता है
           जितनी उसकी probability है: $0.8 \times 0.93$, फिर $0.1 \times 0.69$, फिर $0.1 \times (-1)$, जो ढेर को
           घटा देता है। अंत में ढेर को $\gamma$ से गुणा करके $R(s) = -0.02$ जोड़ते हैं, और वह काले wire box
           $V^\pi(s) = 0.69$ में ठीक बैठ जाता है। ध्यान दें कि $V^\pi(s)$ खुद अपनी equation के दाईं ओर भी आया है,
           इसीलिए सारे states को साथ में हल करना पड़ता है।""",
           frames=46, keys=(0.3, 0.72, 1.0))
def bellman_backup(s, n):
    ax = _stage(s, r"$V^\pi$", zlim=(0, 1.15), elev=30, azim=-40, zticks=(0, 0.25, 0.5, 0.75, 1))
    V = _evaluate(PI_GOOD)
    R = _rewards()
    me, up, pit = IDX[(3, 2)], IDX[(3, 3)], IDX[PIT]
    others = [i for i in range(NS) if i not in (me, up, pit)]
    _bars(ax, V, idx=others, alpha=0.1, colors=[(0.5, 0.52, 0.56)] * len(others), base=0.0)
    _bars(ax, V, idx=[up], alpha=0.3, base=0.0)
    _outline(ax, 3, 2, V[me], lw=1.4, base=0.0, zorder=7)
    ax.text(4, 2, 0.03, "V = -1", fontsize=9, ha="center", color=C["red"], fontweight="bold", zorder=9)
    ax.text(3, 3, V[up] + 0.06, f"{V[up]:.2f}", fontsize=9, ha="center", color=C["green"], zorder=9)
    # the three terms of sum_{s'} P(s') V(s'): (source cell, probability, value, colour)
    terms = [((3, 3), 0.8, V[up], C["green"]), ((3, 2), 0.1, V[me], C["teal"]), ((4, 2), 0.1, V[pit], C["red"])]
    partial = np.cumsum([p * v for _, p, v, _ in terms])
    final = R[me] + GAMMA * partial[-1]
    seg = [(3, 13), (15, 24), (26, 36)]              # frames in which each block travels
    last = seg[-1][1]

    def update(i):
        s.clear()
        stack, msg = 0.0, "state s = (3,2), action: up"
        for k, ((cx, cy), p, v, col) in enumerate(terms):
            a, b = seg[k]
            if i < a:
                break
            t = ease((i - a) / (b - a))
            base = partial[k - 1] if k else 0.0
            h = abs(p * v)
            x, y = (1 - t) * cx + t * 3, (1 - t) * cy + t * 2
            # blocks 0 and 2 start on their own tile; the "stay" block (s' = s) drops in from above.
            # The red block is negative: it lands inside the top of the stack and removes that much.
            z_from = 1.15 - h if k == 1 else 0.0
            z_to = base - h if v < 0 else base
            z = (1 - t) * z_from + t * z_to
            if t < 1:
                stack = base
                s.add(_block(ax, x, y, z, z + h, col, width=0.5, zorder=5))
            else:
                stack = partial[k]
            msg = rf"{'+ ' if k else ''}{p} $\times$ ({v:.2f}) = {p * v:+.2f}    sum = {partial[k]:.2f}"
        if i > last:
            t = ease((i - last) / 6)
            stack = (1 - t) * partial[-1] + t * final
            msg = rf"$R + \gamma \times$ {partial[-1]:.2f} = {final:.2f} = $V^\pi(s)$"
        if stack > 0:
            s.add(_block(ax, 3, 2, 0, stack, C["green"] if i >= last + 6 else C["blue"], width=0.5,
                         alpha=0.9, zorder=3))
        s.status(msg)
        s.spin(i, n, sweep=-14)

    return update


# =====================================================================================
@animation("Changing the policy changes V",
           caption=r"""Value function policy पर निर्भर है: हर policy का अपना landscape $V^\pi$ है। शुरुआत notes वाली खराब
           policy से होती है, जिसमें नीचे के सारे states $-1$ की ओर बहते हैं और उनके bars negative हैं। फिर एक-एक
           state का तीर बदला जाता है (नारंगी तीर), और हर बार पूरा $V^\pi$ दोबारा निकाला जाता है। देखिए कि एक state का
           action बदलने से सिर्फ़ उसी का bar नहीं, उस तक पहुँचने वाले सारे states के bars उठ जाते हैं। अंत की policy
           हर state में पहली से बेहतर है: $V^{\pi_2}(s) \ge V^{\pi_1}(s)$। जो policy हर state में सबसे ऊँची हो, वही
           $\pi^*$ है।""",
           frames=44, keys=(0.0, 0.42, 1.0))
def policy_change(s, n):
    ax = _stage(s, r"$V^\pi(s)$")
    order = [(3, 2), (4, 1), (1, 2), (1, 1), (2, 1), (3, 1)]
    policies = [PI_BAD.copy()]
    for st in order:
        nxt = policies[-1].copy()
        nxt[IDX[st]] = PI_GOOD[IDX[st]]
        policies.append(nxt)
    values = [_evaluate(p) for p in policies]
    per = 6                                          # frames per change (the first 3 frames show the start)

    def update(i):
        s.clear()
        k = int(np.clip((i - 3) // per + 1, 0, len(order))) if i >= 3 else 0
        t = ease(((i - 3) % per + 1) / (per - 1)) if 0 < k and i < 3 + per * len(order) else 1.0
        V = values[k] if k == 0 else (1 - t) * values[k - 1] + t * values[k]
        pi = policies[k]
        top = V + 0.05
        s.add(_bars(ax, V, alpha=0.85))
        changed = [IDX[order[k - 1]]] if k and i < 3 + per * len(order) else []
        s.add(_arrows(ax, pi, top, idx=[j for j in LIVE if j not in changed]))
        s.add(_arrows(ax, pi, top, idx=changed, color=C["orange"], lw=3.0))
        s.status(rf"actions changed: {k}/6    $V^\pi$(start) = {V[IDX[START]]:+.2f}")
        s.spin(i, n, sweep=30)

    return update


# =====================================================================================
@animation("Optimal policy vs step reward",
           caption=r"""अब bars optimal value function $V^*$ हैं और तीर optimal policy $\pi^*$, जो हर state में उस action को
           चुनती है जिससे अगले state का expected $V^*$ सबसे बड़ा हो: तीर landscape की चढ़ाई की ओर हैं। यहाँ हर
           साधारण state का reward $R(s)$ को 0 से $-0.5$ तक घटाया जा रहा है। $R(s) = 0$ पर agent को कोई जल्दी नहीं,
           वह $-1$ से पूरी तरह बचता है (दीवार की ओर मुँह कर लेता है)। $-0.02$ पर notes वाली policy मिलती है। Penalty
           बढ़ने पर एक-एक करके तीर पलटते हैं (नारंगी) और agent $-1$ के पास से छोटा रास्ता लेने लगता है; बहुत बड़ी
           penalty पर $(4,1)$ से वह सीधे $-1$ में कूद जाता है। यानी $\pi^*$ reward function की बनावट से तय होती है।""",
           frames=46, keys=(0.2, 0.55, 1.0))
def step_reward(s, n):
    ax = _stage(s, r"$V^*(s)$", zlim=(-2.4, 1.15), zticks=(-2, -1, 0, 1))
    step = -0.5 * np.linspace(0, 1, n) ** 2
    step[int(round(0.2 * (n - 1)))] = -0.02          # land exactly on the value used in the notes
    sols = [_optimal(_rewards(r)) for r in step]

    def update(i):
        s.clear()
        V, pi = sols[i]
        prev = sols[max(i - 2, 0)][1]
        changed = [j for j in LIVE if pi[j] != prev[j]]
        top = V + 0.07
        s.add(_bars(ax, V, colors=plt.cm.RdYlGn(0.5 + 0.5 * np.clip(V / 1.6, -1, 1)), alpha=0.85, base=-2.4))
        s.add(_arrows(ax, pi, top, idx=[j for j in LIVE if j not in changed]))
        s.add(_arrows(ax, pi, top, idx=changed, color=C["orange"], lw=3.0))
        s.status(rf"step reward $R(s)$ = {step[i]:.3f}    $V^*$(start) = {V[IDX[START]]:+.2f}")
        s.spin(i, n, sweep=30)

    return update
