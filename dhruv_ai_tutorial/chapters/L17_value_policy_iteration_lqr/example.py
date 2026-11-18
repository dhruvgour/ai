"""Lecture 17 example: Gridworld पर Value/Policy Iteration और Double Integrator पर LQR (NumPy).

भाग 1: Lecture 16 वाले 4x3 gridworld पर Value Iteration और Policy Iteration की तुलना।
भाग 2: Double integrator पर finite-horizon LQR (backward Riccati recursion) और शोर के साथ
नियंत्रित बनाम अनियंत्रित simulation।
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)
GAMMA = 0.99

# ---------- 1. Lecture 16 वाला 4x3 Gridworld मॉडल ----------
COLS, ROWS = 4, 3
WALL, GOAL, PIT, START = (2, 2), (4, 3), (4, 2), (1, 1)
states = [(c, r) for r in range(1, ROWS + 1) for c in range(1, COLS + 1) if (c, r) != WALL]
index = {s: i for i, s in enumerate(states)}
n_states = len(states)
terminal = [index[GOAL], index[PIT]]
live = [i for i in range(n_states) if i not in terminal]
moves = {"^": (0, 1), ">": (1, 0), "v": (0, -1), "<": (-1, 0)}
actions = list(moves)

R = np.full(n_states, -0.02)
R[index[GOAL]], R[index[PIT]] = 1.0, -1.0

P = np.zeros((4, n_states, n_states))
for s in states:
    if s in (GOAL, PIT):
        continue
    for k, a in enumerate(actions):
        for d, prob in ((a, 0.8), (actions[(k - 1) % 4], 0.1), (actions[(k + 1) % 4], 0.1)):
            nxt = (s[0] + moves[d][0], s[1] + moves[d][1])
            P[k, index[s], index[nxt if nxt in index else s]] += prob


# ---------- 2. Value Iteration और Policy Iteration ----------
def value_iteration(tol=1e-10):
    V, hist, pi_hist = np.zeros(n_states), [], []
    while True:
        Q = P @ V                                        # shape: (4, n_states)
        V_new = R + GAMMA * Q.max(axis=0)
        hist.append(V_new.copy())
        pi_hist.append(Q.argmax(axis=0))
        if np.max(np.abs(V_new - V)) < tol:
            return V_new, Q.argmax(axis=0), hist, pi_hist
        V = V_new


def policy_iteration():
    pi = np.full(n_states, actions.index("v"))           # शुरुआती policy: हर जगह नीचे ('v')
    hist = []
    while True:
        P_pi = P[pi, np.arange(n_states)]
        V = np.linalg.solve(np.eye(n_states) - GAMMA * P_pi, R)   # Policy Evaluation
        hist.append(V.copy())
        pi_new = (P @ V).argmax(axis=0)                           # Policy Improvement
        if np.array_equal(pi_new[live], pi[live]):
            return V, pi, hist
        pi = pi_new


V_vi, pi_vi, vi_hist, vi_pi_hist = value_iteration()
V_pi, pi_pi, pi_hist = policy_iteration()
pi_lock = next(k + 1 for k, p in enumerate(vi_pi_hist) if np.array_equal(p[live], pi_vi[live]))

print("=== 1. 4x3 Gridworld: Value Iteration बनाम Policy Iteration ===")
print(f"Value Iteration sweeps (tol=1e-10): {len(vi_hist)}  (सही pi* मिल गई sweep {pi_lock} पर)")
print(f"Policy Iteration outer rounds     : {len(pi_hist)}")
print(f"अधिकतम अंतर max |V_VI - V_PI|     : {np.max(np.abs(V_vi - V_pi)):.2e}")
print(f"क्या सभी live states पर pi समान है? : {'हाँ' if np.array_equal(pi_vi[live], pi_pi[live]) else 'नहीं'}")
for r in range(ROWS, 0, -1):
    row_str = [
        "   ####" if (c, r) == WALL else
        f"{V_pi[index[(c, r)]]:+.2f} {' ' if index[(c, r)] in terminal else actions[pi_pi[index[(c, r)]]]}"
        for c in range(1, COLS + 1)
    ]
    print("   " + "   ".join(row_str))

# ---------- 3. Double Integrator पर Finite-Horizon LQR (Riccati) ----------
dt, T = 0.2, 40
A = np.array([[1.0, dt], [0.0, 1.0]])                    # s = [position, velocity]
B = np.array([[0.5 * dt**2], [dt]])                      # a = acceleration
U, W, U_T = np.diag([1.0, 0.2]), np.array([[0.15]]), np.diag([6.0, 1.5])
Sigma_w = np.diag([0.004, 0.008])

Phi = [None] * (T + 1)
L = [None] * T
Phi[T], psi = U_T, 0.0
for t in range(T - 1, -1, -1):                           # पीछे की ओर Riccati recursion
    L[t] = -np.linalg.solve(W + B.T @ Phi[t + 1] @ B, B.T @ Phi[t + 1] @ A)
    Phi[t] = U + A.T @ Phi[t + 1] @ A + A.T @ Phi[t + 1] @ B @ L[t]
    psi += np.trace(Phi[t + 1] @ Sigma_w)


def simulate_lqr(s0, controlled=True, n_runs=2000):
    costs, paths = np.zeros(n_runs), np.zeros((n_runs, T + 1, 2))
    paths[:, 0] = s0
    noise = rng.multivariate_normal(np.zeros(2), Sigma_w, size=(n_runs, T))
    for t in range(T):
        s = paths[:, t]
        a = (s @ L[t].T) if controlled else np.zeros((n_runs, 1))
        costs += np.einsum("bi,ij,bj->b", s, U, s) + np.einsum("bi,ij,bj->b", a, W, a)
        paths[:, t + 1] = s @ A.T + a @ B.T + noise[:, t]
    sT = paths[:, T]
    costs += np.einsum("bi,ij,bj->b", sT, U_T, sT)
    return costs, paths


s0 = np.array([3.0, -1.0])
exact_J0 = float(s0 @ Phi[0] @ s0 + psi)
c_lqr, p_lqr = simulate_lqr(s0, controlled=True)
c_free, p_free = simulate_lqr(s0, controlled=False)

print("\n=== 2. Finite-Horizon LQR: Double Integrator (T = 40, s0 = [3.0, -1.0]) ===")
print(f"शुरुआती feedback gain L_0         : [{L[0][0, 0]:.3f}, {L[0][0, 1]:.3f}]")
print(f"सैद्धांतिक optimal cost J_0*(s0)   : {exact_J0:.2f}  (noise हिस्सा psi_0 = {psi:.2f})")
print(f"LQR नियंत्रित औसत cost (2000 runs): {c_lqr.mean():.2f} +- {c_lqr.std() / np.sqrt(len(c_lqr)):.2f}")
print(f"बिना नियंत्रण (a_t = 0) औसत cost  : {c_free.mean():.2f}")
print(f"अंतिम ||s_T|| का औसत (LQR बनाम 0): {np.linalg.norm(p_lqr[:, -1], axis=1).mean():.3f}"
      f"  बनाम  {np.linalg.norm(p_free[:, -1], axis=1).mean():.3f}")

# ---------- 4. Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
axes[0].semilogy(range(1, len(vi_hist) + 1), [np.max(np.abs(v - V_pi)) + 1e-16 for v in vi_hist],
                 label="Value Iteration")
axes[0].semilogy(range(1, len(pi_hist) + 1), [np.max(np.abs(v - V_pi)) + 1e-16 for v in pi_hist],
                 "o-", label="Policy Iteration")
axes[0].set(xlabel="iteration", ylabel=r"$\max_s |V_k(s) - V^*(s)|$", title="Gridworld convergence")
axes[0].legend()

ts = np.arange(T + 1) * dt
for k in range(12):
    axes[1].plot(ts, p_lqr[k, :, 0], color="#2563eb", alpha=0.35, lw=1)
    axes[1].plot(ts, p_free[k, :, 0], color="#dc2626", alpha=0.35, lw=1)
axes[1].plot([], [], color="#2563eb", lw=2, label=r"LQR position $p_t$")
axes[1].plot([], [], color="#dc2626", lw=2, label=r"No control ($a_t = 0$)")
axes[1].axhline(0, color="gray", ls="--", lw=0.8)
axes[1].set(xlabel="time (s)", ylabel=r"position $p_t$", title="Noisy rollouts: LQR vs free drift")
axes[1].legend()

gains = np.array([g[0] for g in L])
axes[2].plot(np.arange(T), gains[:, 0], label=r"$L_{t,\mathrm{pos}}$")
axes[2].plot(np.arange(T), gains[:, 1], label=r"$L_{t,\mathrm{vel}}$")
axes[2].set(xlabel="step $t$", ylabel="gain value", title=r"Riccati feedback gains $L_t$")
axes[2].legend()
fig.tight_layout()
fig.savefig("result.png", dpi=130)
