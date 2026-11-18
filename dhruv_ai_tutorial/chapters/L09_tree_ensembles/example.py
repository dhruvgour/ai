"""Lecture 9 example: CART Tree (Gini), Bagging और AdaBoost शुरू से (सिर्फ़ NumPy).

Two-moons dataset पर पहले Gini index से पूरा CART classification tree बनाते हैं और
depth बढ़ने पर overfitting देखते हैं। फिर गहरे trees की Bagging (OOB accuracy सहित)
और decision stumps पर AdaBoost चलाकर train और test accuracy की तुलना करते हैं।
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# ---------- 1. Data: noisy two moons (250 train, 750 test) ----------
X_all, y_all = make_moons(1000, noise=0.30, random_state=8)
X_tr, y_tr = X_all[:250], y_all[:250]                  # labels {0, 1}
X_te, y_te = X_all[250:], y_all[250:]


# ---------- 2. CART Classification Tree (Gini impurity, सिर्फ़ NumPy) ----------
def gini_of(y):
    p = np.mean(y)
    return 2.0 * p * (1.0 - p)                         # K=2 के लिए 1 - p^2 - (1-p)^2 = 2p(1-p)


def best_gini_split(X, y):
    """सभी features और midpoints पर सबसे बड़े Gini gain वाला split ढूँढें।"""
    m = len(y)
    if m < 2 or gini_of(y) == 0.0:
        return None
    parent_g, best = gini_of(y), None
    n_l = np.arange(1, m)
    n_r = m - n_l
    for j in range(X.shape[1]):
        order = np.argsort(X[:, j])
        xj, yj = X[order, j], y[order]
        mask = xj[:-1] < xj[1:]
        if not np.any(mask):
            continue
        l_pos = np.cumsum(yj)[:-1]
        p_l, p_r = l_pos / n_l, (yj.sum() - l_pos) / n_r
        w_gini = np.where(mask, (n_l * 2 * p_l * (1 - p_l) + n_r * 2 * p_r * (1 - p_r)) / m, np.inf)
        k = int(np.argmin(w_gini))
        gain = parent_g - w_gini[k]
        if gain > 1e-12 and (best is None or gain > best[0]):
            best = (gain, j, 0.5 * (xj[k] + xj[k + 1]))
    return best


def fit_cart(X, y, max_depth=10):
    prob = float(np.mean(y))
    sp = None if max_depth == 0 else best_gini_split(X, y)
    if sp is None:
        return ("leaf", prob)
    _, j, thr = sp
    left = X[:, j] <= thr
    return ("node", j, thr, fit_cart(X[left], y[left], max_depth - 1),
            fit_cart(X[~left], y[~left], max_depth - 1))


def predict_prob_cart(node, X):
    if node[0] == "leaf":
        return np.full(len(X), node[1])
    _, j, thr, l_child, r_child = node
    return np.where(X[:, j] <= thr, predict_prob_cart(l_child, X), predict_prob_cart(r_child, X))


def acc(y_hat, y_true):
    return float(np.mean(y_hat == y_true))


# ---------- 3. Bagging + Out-of-Bag (OOB) estimate ----------
def fit_bagging(X, y, n_trees=50, max_depth=15, seed=0):
    rng = np.random.default_rng(seed)
    m = len(y)
    trees, oob_sum, oob_cnt = [], np.zeros(m), np.zeros(m)
    for _ in range(n_trees):
        boot = rng.integers(0, m, size=m)              # with replacement चुनाव
        tr = fit_cart(X[boot], y[boot], max_depth=max_depth)
        trees.append(tr)
        oob = np.ones(m, dtype=bool)
        oob[boot] = False                              # लगभग 36.8% छूटे हुए examples
        oob_sum[oob] += predict_prob_cart(tr, X[oob])
        oob_cnt[oob] += 1
    v = oob_cnt > 0
    return trees, acc((oob_sum[v] / oob_cnt[v]) >= 0.5, y[v])


def predict_bagging_prob(trees, X):
    return np.mean([predict_prob_cart(t, X) for t in trees], axis=0)


# ---------- 4. AdaBoost with Decision Stumps (labels {-1, +1}) ----------
def fit_adaboost(X, y01, n_rounds=80):
    y = np.where(y01 == 1, 1.0, -1.0)
    w = np.full(len(y), 1.0 / len(y))
    stumps = []
    for _ in range(n_rounds):
        best = None
        for j in range(X.shape[1]):
            vals = np.unique(X[:, j])
            thrs = np.r_[vals[0] - 0.1, 0.5 * (vals[:-1] + vals[1:]), vals[-1] + 0.1]
            preds = np.where(X[:, j][:, None] <= thrs[None, :], 1.0, -1.0)
            err_pos = (w[:, None] * (preds != y[:, None])).sum(axis=0)
            for pol, errs in ((1.0, err_pos), (-1.0, 1.0 - err_pos)):
                k = int(np.argmin(errs))
                if best is None or errs[k] < best[0]:
                    best = (float(errs[k]), j, float(thrs[k]), pol)
        err, j, thr, pol = best
        err = np.clip(err, 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1.0 - err) / err)        # weak learner का वोट weight
        w *= np.exp(-alpha * y * np.where(X[:, j] <= thr, pol, -pol))
        w /= w.sum()
        stumps.append((alpha, j, thr, pol))
    return stumps


def adaboost_margin(stumps, X):
    return sum(a * np.where(X[:, j] <= thr, pol, -pol) for a, j, thr, pol in stumps)


# ---------- 5. प्रयोग और तुलना ----------
print("1) अकेला CART Tree (Gini): depth का असर")
print("max_depth   train acc   test acc")
cart_models = {d: fit_cart(X_tr, y_tr, max_depth=d) for d in (1, 2, 3, 5, 10, 15)}
for d, tr in cart_models.items():
    print(f"{d:9d} {acc(predict_prob_cart(tr, X_tr) >= 0.5, y_tr):11.3f} "
          f"{acc(predict_prob_cart(tr, X_te) >= 0.5, y_te):10.3f}")

bag_trees, _ = fit_bagging(X_tr, y_tr, n_trees=50, max_depth=15, seed=7)
print("\n2) Bagging (गहरे depth=15 trees): M बढ़ाने पर variance घटना")
print("trees M     train acc   OOB acc    test acc")
for M in (1, 5, 15, 50):
    sub, oob_m = fit_bagging(X_tr, y_tr, n_trees=M, max_depth=15, seed=7)
    print(f"{M:7d} {acc(predict_bagging_prob(sub, X_tr) >= 0.5, y_tr):13.3f} "
          f"{oob_m:9.3f} {acc(predict_bagging_prob(sub, X_te) >= 0.5, y_te):11.3f}")

stumps = fit_adaboost(X_tr, y_tr, n_rounds=80)
print("\n3) AdaBoost (depth=1 decision stumps): M बढ़ाने पर bias घटना")
print("rounds M    train acc   test acc")
for M in (1, 5, 15, 40, 80):
    print(f"{M:8d} {acc(adaboost_margin(stumps[:M], X_tr) >= 0, y_tr):12.3f} "
          f"{acc(adaboost_margin(stumps[:M], X_te) >= 0, y_te):10.3f}")

# ---------- 6. Decision boundaries का चित्र ----------
g1, g2 = np.meshgrid(np.linspace(-1.7, 2.7, 200), np.linspace(-1.2, 1.7, 200))
G = np.c_[g1.ravel(), g2.ravel()]
panels = [("Deep CART (depth=15)", predict_prob_cart(cart_models[15], G),
           acc(predict_prob_cart(cart_models[15], X_te) >= 0.5, y_te)),
          ("Bagging (50 deep trees)", predict_bagging_prob(bag_trees, G),
           acc(predict_bagging_prob(bag_trees, X_te) >= 0.5, y_te)),
          ("AdaBoost (80 stumps)", 1 / (1 + np.exp(-2 * adaboost_margin(stumps, G))),
           acc(adaboost_margin(stumps, X_te) >= 0, y_te))]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
for ax, (name, prob, te_a) in zip(axes, panels):
    Z = prob.reshape(g1.shape)
    ax.contourf(g1, g2, Z, levels=20, cmap="coolwarm", alpha=0.58)
    ax.contour(g1, g2, Z, levels=[0.5], colors="k", linewidths=1.4)
    ax.scatter(X_te[:, 0], X_te[:, 1], c=y_te, cmap="coolwarm", s=13, edgecolor="k", linewidth=0.25)
    ax.set(title=f"{name}\ntest acc = {te_a:.3f}", xlabel="x1", ylabel="x2")
fig.tight_layout()
fig.savefig("result.png", dpi=130)
