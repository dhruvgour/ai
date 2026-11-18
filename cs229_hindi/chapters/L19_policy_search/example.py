"""Lecture 19 example: Softmax REINFORCE (baseline के साथ और बिना) तथा POMDP Bayes filter.

सिर्फ़ NumPy से: (1) 3x3 noisy gridworld में softmax policy पर REINFORCE चलाकर कई random seeds पर
average return और gradient variance की तुलना, और (2) 5-खानों के गलियारे (hallway POMDP) में Bayes filter।
"""
import numpy as np
import matplotlib.pyplot as plt

# ---------- 1. Self-contained 3x3 Noisy Gridworld (धनात्मक survival + goal rewards) ----------
N_STATES, N_ACTIONS, HORIZON, GAMMA = 9, 4, 14, 0.98
START, TRAP, GOAL = 0, 4, 8
MOVES = [(-1, 0), (0, 1), (1, 0), (0, -1)]   # Down, Right, Up, Left


def step_env(s, a, rng):
    """0.85 probability से चुना हुआ action, 0.15 से random फिसलन।"""
    if rng.random() < 0.15:
        a = int(rng.integers(N_ACTIONS))
    r, c = divmod(s, 3)
    nr, nc = int(np.clip(r + MOVES[a][0], 0, 2)), int(np.clip(c + MOVES[a][1], 0, 2))
    ns = 3 * nr + nc
    if ns == GOAL:
        return ns, 16.0, True
    if ns == TRAP:
        return ns, 4.0, True
    return ns, 1.2, False


def softmax_policy(logits):
    z = logits - logits.max(axis=-1, keepdims=True)
    ez = np.exp(z)
    return ez / ez.sum(axis=-1, keepdims=True)


def sample_episode(theta, rng):
    probs = softmax_policy(theta)
    s, states, acts, rews = START, [], [], []
    for _ in range(HORIZON):
        a = int(rng.choice(N_ACTIONS, p=probs[s]))
        ns, rew, done = step_env(s, a, rng)
        states.append(s); acts.append(a); rews.append(rew)
        s = ns
        if done:
            break
    G, running = np.zeros(len(rews)), 0.0
    for t in range(len(rews) - 1, -1, -1):
        running = rews[t] + GAMMA * running
        G[t] = running
    return probs, states, acts, G


def grad_from_ep(probs, states, acts, G, b_vec=None, use_baseline=False):
    grad = np.zeros((N_STATES, N_ACTIONS))
    for t, (st, ac) in enumerate(zip(states, acts)):
        w = (G[t] - b_vec[st]) if use_baseline else G[0]
        score = -probs[st].copy()
        score[ac] += 1.0
        grad[st] += w * score
    return grad


# ---------- 2. एक fixed policy पर Unbiasedness और Variance की सीधी जाँच ----------
rng0 = np.random.default_rng(0)
theta_probe = rng0.normal(0.0, 0.35, (N_STATES, N_ACTIONS))
# अलग pilot rollouts से state baseline b(s) का अंदाज़ा लगाएँ ताकि estimator पूरी तरह unbiased रहे
b_sum, b_cnt = np.zeros(N_STATES), np.full(N_STATES, 1e-8)
for _ in range(600):
    _, sts, _, Gs = sample_episode(theta_probe, rng0)
    for st, gt in zip(sts, Gs):
        b_sum[st] += gt; b_cnt[st] += 1.0
b_probe = b_sum / b_cnt

eps_eval = [sample_episode(theta_probe, rng0) for _ in range(2500)]
g_raw = np.array([grad_from_ep(*ep, use_baseline=False) for ep in eps_eval])
g_bas = np.array([grad_from_ep(*ep, b_probe, use_baseline=True) for ep in eps_eval])
var_raw, var_bas = float(g_raw.var(axis=0).sum()), float(g_bas.var(axis=0).sum())

print("=== 1. Fixed Softmax Policy पर Gradient Check (2500 rollouts) ===")
print(f"  No-baseline total grad variance   : {var_raw:8.2f}")
print(f"  With-baseline total grad variance : {var_bas:8.2f}  ({var_raw / var_bas:.1f}x कम!)")
print(f"  Max |E[g_no_base] - E[g_base]|    : {np.max(np.abs(g_raw.mean(0) - g_bas.mean(0))):8.3f}  (दोनों unbiased हैं)\n")


# ---------- 3. कई Seeds पर REINFORCE Training: Baseline बनाम बिना Baseline ----------
def train_reinforce(use_baseline, seeds=8, updates=90, batch=4, alpha=0.045):
    ret_curves, var_curves = np.zeros((seeds, updates)), np.zeros((seeds, updates))
    for sd in range(seeds):
        rng = np.random.default_rng(100 + sd)
        theta, b_state = np.zeros((N_STATES, N_ACTIONS)), np.full(N_STATES, 11.0)
        for u in range(updates):
            eps = [sample_episode(theta, rng) for _ in range(batch)]
            grads = np.array([grad_from_ep(*ep, b_state, use_baseline=use_baseline) for ep in eps])
            for _, sts, _, Gs in eps:
                for st, gt in zip(sts, Gs):
                    b_state[st] = 0.90 * b_state[st] + 0.10 * gt
            ret_curves[sd, u] = np.mean([ep[3][0] for ep in eps])
            var_curves[sd, u] = grads.var(axis=0).sum()
            theta += alpha * grads.mean(axis=0)
    return ret_curves, var_curves


ret_no, var_no = train_reinforce(use_baseline=False)
ret_bs, var_bs = train_reinforce(use_baseline=True)
print("=== 2. REINFORCE Training (8 seeds, हर update में 4 episodes, कुल 360 episodes) ===")
print("  Update   No-Baseline Return   With-Baseline Return   GradVar(No)   GradVar(Base)")
for u in (0, 20, 50, 89):
    r0, r1 = ret_no[:, max(0, u - 4):u + 1].mean(1), ret_bs[:, max(0, u - 4):u + 1].mean(1)
    print(f"  {u + 1:6d}    {r0.mean():6.2f} +- {r0.std():4.2f}       {r1.mean():6.2f} +- {r1.std():4.2f}"
          f"        {var_no[:, u].mean():8.1f}       {var_bs[:, u].mean():8.2f}")

# ---------- 4. छोटा POMDP Bayes-Filter Demo: 5-खानों का गलियारा ----------
DOOR_MAP = np.array([0, 1, 0, 1, 0])          # खाने 1 और 3 पर door (1), बाकी पर wall (0)
P_RIGHT = np.zeros((5, 5))
for s in range(5):
    P_RIGHT[s, min(4, s + 1)] += 0.85; P_RIGHT[s, s] += 0.15

belief = np.full(5, 0.2)
belief_hist = [("t=0 uniform prior", belief.copy())]
steps_pomdp = [(None, 1, "t=1 obs=door"), ("right", 0, "t=2 move+obs=wall"), ("right", 1, "t=3 move+obs=door")]
print("\n=== 3. POMDP Hallway में Bayes Filter अपडेट (असली रास्ता: 1 -> 2 -> 3) ===")
print(f"  {'t=0 prior':<20}: " + "  ".join(f"s{i}:{p:.2f}" for i, p in enumerate(belief)))
for act, obs, label in steps_pomdp:
    if act == "right":
        belief = belief @ P_RIGHT
    like = np.where(DOOR_MAP == obs, 0.88, 0.12)
    belief = (belief * like) / np.sum(belief * like)
    belief_hist.append((label, belief.copy()))
    print(f"  {label:<20}: " + "  ".join(f"s{i}:{p:.2f}" for i, p in enumerate(belief)))

# ---------- 5. Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(12.6, 3.7))
xs = np.arange(1, ret_no.shape[1] + 1)
for curves, col, lbl in [(ret_no, "#dc2626", "No baseline"), (ret_bs, "#2563eb", "With baseline")]:
    sm = np.array([np.convolve(c, np.ones(5) / 5, mode="valid") for c in curves])
    axes[0].plot(xs[4:], sm.mean(0), color=col, lw=2, label=lbl)
    axes[0].fill_between(xs[4:], sm.mean(0) - sm.std(0), sm.mean(0) + sm.std(0), color=col, alpha=0.18)
axes[0].set(xlabel="update step", ylabel="average return", title="REINFORCE Return (8 seeds)")
axes[0].legend()
axes[1].semilogy(xs, var_no.mean(0), color="#dc2626", lw=1.8, label="No baseline")
axes[1].semilogy(xs, var_bs.mean(0), color="#2563eb", lw=1.8, label="With baseline")
axes[1].set(xlabel="update step", ylabel="gradient variance (log)", title="Baseline Cuts Grad Variance")
axes[1].legend()
for k, (lbl, b_vec) in enumerate(belief_hist):
    axes[2].bar(np.arange(5) + (k - 1.5) * 0.2, b_vec, width=0.2, label=lbl)
axes[2].set(xticks=range(5), xticklabels=[f"s{i}\n({'D' if DOOR_MAP[i] else 'W'})" for i in range(5)],
            ylabel="belief b_t(s)", title="POMDP Bayes Filter")
axes[2].legend(fontsize=7.8)
fig.tight_layout()
fig.savefig("result.png", dpi=130)
