"""Lecture 18 example: 4x3 gridworld पर Tabular Q-Learning शुरू से (सिर्फ़ NumPy).

Lecture 16 वाला noisy 4x3 gridworld environment बनाते हैं (0.8 सीधा, 0.1-0.1 बगल में फिसलन)।
बिना P_sa जाने epsilon-greedy tabular Q-learning चलाते हैं, प्रति episode औसत return का बढ़ना
देखते हैं, और सीखे हुए max_a Q तथा greedy policy की तुलना exact Value Iteration से करते हैं।
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)
GAMMA = 0.99

# ---------- 1. 4x3 Gridworld environment: state = (column, row) ----------
COLS, ROWS = 4, 3
WALL, GOAL, PIT, START = (2, 2), (4, 3), (4, 2), (1, 1)
states = [(c, r) for r in range(1, ROWS + 1) for c in range(1, COLS + 1) if (c, r) != WALL]
index = {s: i for i, s in enumerate(states)}
n_states = len(states)                                    # 11 states (9 non-terminal + 2 terminal)
terminal = {index[GOAL], index[PIT]}
moves = {"^": (0, 1), ">": (1, 0), "v": (0, -1), "<": (-1, 0)}
actions = list(moves)                                     # घड़ी की दिशा में: ^ > v <

R = np.full(n_states, -0.02)
R[index[GOAL]], R[index[PIT]] = 1.0, -1.0


def move(s, a_idx):
    dx, dy = moves[actions[a_idx]]
    nxt = (s[0] + dx, s[1] + dy)
    return index[nxt] if nxt in index else index[s]


NEXT = np.array([[move(s, a) for a in range(4)] for s in states])


def env_step(s, a):
    """Model-free simulator कदम: 0.8 चुनी दिशा, 0.1 बाएँ, 0.1 दाएँ।"""
    u = rng.random()
    actual = a if u < 0.8 else ((a - 1) % 4 if u < 0.9 else (a + 1) % 4)
    return R[s], int(NEXT[s, actual])


# ---------- 2. Exact जवाब: Value Iteration (सिर्फ़ तुलना के लिए P_sa बनाकर) ----------
P = np.zeros((n_states, 4, n_states))
for s in range(n_states):
    if s not in terminal:
        for a in range(4):
            for a2, prob in ((a, 0.8), ((a - 1) % 4, 0.1), ((a + 1) % 4, 0.1)):
                P[s, a, NEXT[s, a2]] += prob

Q_star = np.zeros((n_states, 4))
Q_star[list(terminal)] = R[list(terminal), None]
for _ in range(1500):
    V_prev = Q_star.max(axis=1)
    for s in range(n_states):
        if s not in terminal:
            Q_star[s] = R[s] + GAMMA * (P[s] @ V_prev)
V_star = Q_star.max(axis=1)
pi_star = Q_star.argmax(axis=1)

# ---------- 3. Model-free Tabular Q-Learning (epsilon-greedy के साथ) ----------
EPISODES = 25000
Q = np.zeros((n_states, 4))
Q[list(terminal)] = R[list(terminal), None]              # terminal states का अंतिम मान
visits = np.zeros((n_states, 4))
non_term = np.array([s for s in range(n_states) if s not in terminal])

ep_returns, val_errors, pol_matches = np.zeros(EPISODES), np.zeros(EPISODES), np.zeros(EPISODES, int)

for ep in range(EPISODES):
    eps = max(0.03, 1.0 - (ep / 21000.0) ** 1.8)          # पर्याप्त explore, अंत में exploit
    s = int(rng.choice(non_term))                         # हर non-terminal state से शुरुआत
    discount, total_ret = 1.0, 0.0
    while s not in terminal:
        if rng.random() < eps:
            a = int(rng.integers(4))
        else:
            best = np.flatnonzero(Q[s] == Q[s].max())
            a = int(rng.choice(best))                     # बराबर मान हों तो random tie-break
        r, s_next = env_step(s, a)
        total_ret += discount * r
        discount *= GAMMA
        visits[s, a] += 1
        alpha = visits[s, a] ** -0.65                     # Robbins-Monro step size
        td_error = r + GAMMA * Q[s_next].max() - Q[s, a]
        Q[s, a] += alpha * td_error
        s = s_next
    total_ret += discount * R[s]                          # terminal state (+1 या -1) का इनाम
    ep_returns[ep] = total_ret
    val_errors[ep] = np.abs(Q.max(axis=1) - V_star).max()
    pol_matches[ep] = np.sum(Q[non_term].argmax(axis=1) == pi_star[non_term])

# ---------- 4. परिणाम छापें ----------
print("Episode प्रगति (random non-terminal start, noisy चालें):")
print("  episodes        eps_end   avg return   max |max_a Q - V*|   policy match")
checkpoints = [(0, 1000), (1000, 6000), (6000, 13000), (13000, 20000), (20000, 25000)]
for lo, hi in checkpoints:
    eps_end = max(0.03, 1.0 - ((hi - 1) / 21000.0) ** 1.8)
    print(f"  {lo + 1:5d}-{hi:<5d}      {eps_end:5.2f}      {ep_returns[lo:hi].mean():+7.3f}"
          f"             {val_errors[hi - 1]:6.3f}             {pol_matches[hi - 1]}/9")


def show_grid(V, pi, title):
    print(f"\n{title}")
    for r in range(ROWS, 0, -1):
        row = []
        for c in range(1, COLS + 1):
            if (c, r) == WALL:
                row.append("   ####")
            else:
                i = index[(c, r)]
                arrow = " " if i in terminal else actions[pi[i]]
                row.append(f"{V[i]:+.2f} {arrow}")
        print("  " + "   ".join(row))


V_q, pi_q = Q.max(axis=1), Q.argmax(axis=1)
show_grid(V_star, pi_star, "Exact Value Iteration: V*(s) और optimal policy pi*(s)")
show_grid(V_q, pi_q, "Tabular Q-Learning: max_a Q(s,a) और सीखी हुई greedy policy")

matches, opt_mean, s32 = int(np.sum(pi_q[non_term] == pi_star[non_term])), V_star[non_term].mean(), index[(3, 2)]
print(f"\nNon-terminal states पर greedy policy match: {matches}/{len(non_term)} "
      f"(state (3,2) पर exact Q*: ^ {Q_star[s32, 0]:+.4f} बनाम < {Q_star[s32, 3]:+.4f})")
print(f"अंतिम max |max_a Q(s,a) - V*(s)| = {np.abs(V_q - V_star).max():.4f},  max |Q - Q*| = {np.abs(Q - Q_star).max():.4f}")
print(f"Optimal mean V* = {opt_mean:+.4f},  आख़िरी 2500 episodes का औसत return = {ep_returns[-2500:].mean():+.4f}")

# ---------- 5. Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
win = 200
axes[0].plot(np.arange(win, EPISODES + 1), np.convolve(ep_returns, np.ones(win) / win, mode="valid"),
             color="#2563eb", label=f"{win}-ep moving avg")
axes[0].axhline(opt_mean, color="#16a34a", ls="--", label="exact mean V*")
axes[0].set(xlabel="episode", ylabel="discounted return", title="Average return per episode improves")
axes[0].legend()
axes[1].semilogy(np.arange(1, EPISODES + 1), val_errors, color="#dc2626", lw=1.2)
axes[1].set(xlabel="episode", ylabel="max |max_a Q(s,a) - V*(s)|", title="Convergence to Value Iteration")
grid = np.full((ROWS, COLS), np.nan)
for (c, r), i in index.items():
    grid[r - 1, c - 1] = V_q[i]
axes[2].imshow(grid, origin="lower", cmap="RdYlGn", vmin=-1, vmax=1, extent=(0.5, 4.5, 0.5, 3.5))
axes[2].add_patch(plt.Rectangle((1.5, 1.5), 1, 1, color="0.3"))
for (c, r), i in index.items():
    axes[2].text(c, r - 0.22, f"{V_q[i]:.2f}", ha="center", va="center", fontsize=9.5)
    if i not in terminal:
        dx, dy = moves[actions[pi_q[i]]]
        axes[2].arrow(c - 0.17 * dx, r + 0.12 - 0.17 * dy, 0.34 * dx, 0.34 * dy, head_width=0.11, color="k")
axes[2].set(title="Learned max_a Q & greedy policy", xticks=[1, 2, 3, 4], yticks=[1, 2, 3], xlabel="column", ylabel="row")
fig.tight_layout()
fig.savefig("result.png", dpi=130)
