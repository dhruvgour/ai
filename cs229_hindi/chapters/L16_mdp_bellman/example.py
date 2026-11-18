"""Lecture 16 example: 4x3 gridworld पर policy evaluation शुरू से (सिर्फ़ NumPy).

CS229 notes वाला gridworld बनाते हैं (noisy चाल: 0.8 सीधा, 0.1-0.1 बगल में)। फिर एक fixed policy का
V^pi तीन तरीकों से निकालते हैं: (1) Bellman linear system हल करके, (2) Bellman equation को बार-बार
लगाकर, (3) Monte Carlo rollouts से। अंत में दो policies की तुलना करते हैं।
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
GAMMA = 0.99

# ---------- 1. Gridworld: state = (column, row), (1, 1) नीचे-बाएँ कोने में ----------
COLS, ROWS = 4, 3
WALL, GOAL, PIT, START = (2, 2), (4, 3), (4, 2), (1, 1)
states = [(c, r) for r in range(1, ROWS + 1) for c in range(1, COLS + 1) if (c, r) != WALL]
index = {s: i for i, s in enumerate(states)}
n_states = len(states)                                   # 11 states
terminal = [index[GOAL], index[PIT]]
moves = {"^": (0, 1), ">": (1, 0), "v": (0, -1), "<": (-1, 0)}
actions = list(moves)                                    # क्रम घड़ी की दिशा में: ^ > v <

R = np.full(n_states, -0.02)                             # हर साधारण state का reward
R[index[GOAL]], R[index[PIT]] = 1.0, -1.0


def move(s, a):
    """Action a की दिशा में एक कदम; दीवार या किनारे से टकराएँ तो वहीं रहें।"""
    nxt = (s[0] + moves[a][0], s[1] + moves[a][1])
    return nxt if nxt in index else s


# ---------- 2. Transition probabilities P[a, s, s'] ----------
# Terminal states की rows zero रहती हैं: वहाँ reward मिलकर episode खत्म हो जाता है।
P = np.zeros((4, n_states, n_states))
for s in states:
    if s in (GOAL, PIT):
        continue
    for k, a in enumerate(actions):
        left, right = actions[(k - 1) % 4], actions[(k + 1) % 4]     # a के दोनों perpendicular
        for direction, prob in ((a, 0.8), (left, 0.1), (right, 0.1)):
            P[k, index[s], index[move(s, direction)]] += prob


def make_policy(rows):
    """ऊपर की row से नीचे की ओर लिखे तीरों को action indices के array में बदलें।"""
    pi = np.zeros(n_states, dtype=int)
    for k, line in enumerate(rows):
        for c, ch in enumerate(line, start=1):
            if (c, ROWS - k) in index and ch in moves:
                pi[index[(c, ROWS - k)]] = actions.index(ch)
    return pi


policies = {"1 (good)": make_policy([">>>.", "^.^.", "^<<<"]),     # notes की optimal policy
            "2 (bad)": make_policy([">>>.", "v.>.", ">>^^"])}      # notes की खराब policy


# ---------- 3. Policy evaluation, तरीका 1: linear system (I - gamma P_pi) V = R ----------
def evaluate_exact(pi):
    P_pi = P[pi, np.arange(n_states)]                    # row s = P_{s, pi(s)}
    return np.linalg.solve(np.eye(n_states) - GAMMA * P_pi, R)


# ---------- 4. तरीका 2: Bellman equation को बार-बार लगाना ----------
def evaluate_iterative(pi, sweeps):
    P_pi = P[pi, np.arange(n_states)]
    V, history = np.zeros(n_states), []
    for _ in range(sweeps):
        V = R + GAMMA * P_pi @ V                         # सारे states एक साथ update (synchronous)
        history.append(V.copy())
    return V, history


# ---------- 5. तरीका 3: Monte Carlo, यानी सचमुच episodes चलाकर payoff का औसत ----------
def rollout(pi, start=START):
    i, discount, payoff = index[start], 1.0, 0.0
    while True:
        payoff += discount * R[i]                        # gamma^t * R(s_t)
        if i in terminal:
            return payoff, i
        i = rng.choice(n_states, p=P[pi[i], i])          # noisy चाल
        discount *= GAMMA


def show(V, pi):
    for r in range(ROWS, 0, -1):
        cells = []
        for c in range(1, COLS + 1):
            if (c, r) == WALL:
                cells.append("   ####")
            else:
                i = index[(c, r)]
                cells.append(f"{V[i]:+.2f} {' ' if i in terminal else actions[pi[i]]}")
        print("   " + "   ".join(cells))


exact, histories = {}, {}
for name, pi in policies.items():
    exact[name] = evaluate_exact(pi)
    histories[name] = evaluate_iterative(pi, 40)[1]
    print(f"Policy {name}: linear system से V^pi (तीर = policy का action)")
    show(exact[name], pi)

print("\nIterative evaluation: max |V_k - V_exact|")
print("  sweeps k    policy 1     policy 2")
for k in (1, 5, 10, 20, 40):
    gaps = [np.abs(histories[name][k - 1] - exact[name]).max() for name in policies]
    print(f"  {k:8d}    {gaps[0]:.2e}     {gaps[1]:.2e}")

print("\nMonte Carlo check: start (1,1) से 4000 episodes")
print("  policy      exact V(start)    MC mean +- std.err    ends at +1")
for name, pi in policies.items():
    runs = [rollout(pi) for _ in range(4000)]
    payoffs = np.array([p for p, _ in runs])
    at_goal = np.mean([end == index[GOAL] for _, end in runs])
    print(f"  {name:<10}  {exact[name][index[START]]:+.4f}           {payoffs.mean():+.4f} +- "
          f"{payoffs.std() / np.sqrt(len(payoffs)):.4f}     {100 * at_goal:5.1f}%")

# ---------- 6. Bellman equation को हाथ से जाँचें: state (3,1), policy 1 का V ----------
V = exact["1 (good)"]
s = index[(3, 1)]
q = {a: P[k, s] @ V for k, a in enumerate(actions)}      # sum_s' P_sa(s') V(s')
print("\nState (3,1) पर sum_s' P_sa(s') V(s'):  " + "   ".join(f"{a} {q[a]:.3f}" for a in actions))
print(f"सबसे अच्छा action '<' है, और R + gamma * {q['<']:.3f} = {R[s] + GAMMA * q['<']:.3f} = V(3,1)")
better = np.all(exact["1 (good)"] >= exact["2 (bad)"])
print(f"क्या policy 1 हर state में policy 2 से अच्छी या बराबर है? {'हाँ' if better else 'नहीं'}")

# ---------- 7. Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, (name, pi) in zip(axes, policies.items()):
    grid = np.full((ROWS, COLS), np.nan)
    for (c, r), i in index.items():
        grid[r - 1, c - 1] = exact[name][i]
    ax.imshow(grid, origin="lower", cmap="RdYlGn", vmin=-1, vmax=1, extent=(0.5, 4.5, 0.5, 3.5))
    ax.add_patch(plt.Rectangle((1.5, 1.5), 1, 1, color="0.3"))
    for (c, r), i in index.items():
        ax.text(c, r - 0.22, f"{exact[name][i]:.2f}", ha="center", va="center", fontsize=10)
        if i not in terminal:
            dx, dy = moves[actions[pi[i]]]
            ax.arrow(c - 0.17 * dx, r + 0.12 - 0.17 * dy, 0.34 * dx, 0.34 * dy, head_width=0.11, color="k")
    ax.set(title=f"V^pi of policy {name}", xticks=[1, 2, 3, 4], yticks=[1, 2, 3], xlabel="column", ylabel="row")
for name in policies:
    gap = [np.abs(h - exact[name]).max() for h in histories[name]]
    axes[2].semilogy(range(1, len(gap) + 1), gap, "o-", ms=3, label=f"policy {name}")
axes[2].set(xlabel="sweep k", ylabel="max |V_k - V_exact|", title="Iterative evaluation converges")
axes[2].legend()
fig.tight_layout()
fig.savefig("result.png", dpi=130)
