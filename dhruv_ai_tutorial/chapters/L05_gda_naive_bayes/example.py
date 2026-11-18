"""Lecture 5 example: GDA और Naive Bayes शुरू से (सिर्फ़ NumPy).

भाग A: 2 features वाले data पर GDA fit करते हैं और उसकी तुलना Logistic Regression से करते हैं,
       पहले Gaussian data पर (GDA की assumption सही), फिर skewed log-normal data पर (assumption गलत)।
भाग B: हाथ से लिखे छोटे corpus पर Bernoulli Naive Bayes spam classifier।
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
PHI = 0.4                                            # असली class prior p(y=1)
MU0, MU1 = np.array([0.0, 0.0]), np.array([2.0, 1.5])
SIGMA = np.array([[1.0, 0.6], [0.6, 1.2]])


# ---------- 1. Data: Gaussian (साझा Sigma) या log-normal (skewed, Gaussian नहीं) ----------
def sample(m, gaussian=True):
    while True:
        y = (rng.random(m) < PHI).astype(int)
        if 2 <= y.sum() <= m - 2:                    # हर class के कम से कम 2 examples चाहिए
            break
    if gaussian:
        x = np.where(y[:, None] == 1, MU1, MU0) + rng.multivariate_normal([0, 0], SIGMA, m)
    else:
        x = np.exp(0.8 * rng.normal(size=(m, 2)) + np.where(y[:, None] == 1, [1.0, 0.0], [0.0, 1.0]))
    return x, y


# ---------- 2. GDA: maximum likelihood estimates (सिर्फ़ गिनती और औसत) ----------
def fit_gda(x, y):
    phi = y.mean()
    mu0, mu1 = x[y == 0].mean(axis=0), x[y == 1].mean(axis=0)
    centred = x - np.where(y[:, None] == 1, mu1, mu0)    # हर example में से उसकी class का mean घटाएँ
    Sigma = centred.T @ centred / len(y)
    return phi, mu0, mu1, Sigma


def gda_theta(phi, mu0, mu1, Sigma):
    """GDA का posterior sigmoid(theta0 + theta^T x) है: यहाँ वही theta निकलता है।"""
    S_inv = np.linalg.inv(Sigma)
    theta = S_inv @ (mu1 - mu0)
    theta0 = 0.5 * (mu0 @ S_inv @ mu0 - mu1 @ S_inv @ mu1) + np.log(phi / (1 - phi))
    return np.r_[theta0, theta]


# ---------- 3. Logistic Regression (Newton's method), तुलना के लिए ----------
def fit_logistic(x, y, iters=15, ridge=1e-3):
    X = np.c_[np.ones(len(x)), x]
    theta = np.zeros(3)
    for _ in range(iters):
        h = 1 / (1 + np.exp(-np.clip(X @ theta, -30, 30)))
        gradient = X.T @ (h - y) + ridge * theta         # छोटा ridge: separable data पर theta फटे नहीं
        hessian = (X * (h * (1 - h))[:, None]).T @ X + ridge * np.eye(3)
        theta -= np.linalg.solve(hessian, gradient)
    return theta


def accuracy(theta, x, y):
    return np.mean((theta[0] + x @ theta[1:] > 0) == y)


# ---------- 4. एक बार fit करके देखें (m = 400, Gaussian data) ----------
x_train, y_train = sample(400)
x_test, y_test = sample(5000)
phi, mu0, mu1, Sigma = fit_gda(x_train, y_train)
theta_gda, theta_lr = gda_theta(phi, mu0, mu1, Sigma), fit_logistic(x_train, y_train)
print("भाग A: GDA के सीखे parameters (m = 400)")
print(f"  phi = {phi:.3f} (सच {PHI})   mu0 = {mu0.round(2)} (सच {MU0})   mu1 = {mu1.round(2)} (सच {MU1})")
print(f"  Sigma = {Sigma.round(2).tolist()} (सच {SIGMA.tolist()})")
print("\nModel                  theta0   theta1   theta2   test accuracy")
for name, th in [("GDA (implied theta)", theta_gda), ("Logistic Regression", theta_lr)]:
    print(f"{name:<20} {th[0]:8.2f} {th[1]:8.2f} {th[2]:8.2f} {accuracy(th, x_test, y_test):12.4f}")

# ---------- 5. Training set का size बदलें: कौन कब बेहतर है? ----------
sizes, trials = [8, 16, 32, 100, 1000], 600
results = {}
print(f"\nTest accuracy ({trials} trials का औसत)")
print("            Gaussian data         log-normal data")
print("     m      GDA   Logistic         GDA   Logistic")
for gaussian in (True, False):
    x_te, y_te = sample(4000, gaussian)
    for m in sizes:
        acc = np.zeros((trials, 2))
        for t in range(trials):
            x_tr, y_tr = sample(m, gaussian)
            acc[t] = (accuracy(gda_theta(*fit_gda(x_tr, y_tr)), x_te, y_te),
                      accuracy(fit_logistic(x_tr, y_tr), x_te, y_te))
        results[gaussian, m] = acc.mean(axis=0)
for m in sizes:
    (g1, l1), (g2, l2) = results[True, m], results[False, m]
    print(f"{m:6d}   {g1:6.3f}   {l1:6.3f}      {g2:6.3f}   {l2:6.3f}")
print("Gaussian data: कम m पर GDA आगे (assumption सही)। Log-normal data: Logistic आगे (ज़्यादा robust)।")

# ---------- 6. भाग B: Bernoulli Naive Bayes spam filter ----------
spam = ["win money now", "free money offer", "win free prize now", "free offer click now",
        "cheap money win prize"]
ham = ["meeting at noon", "project report due", "lunch at noon today", "see report before meeting",
       "free for lunch today"]
vocab = sorted({w for doc in spam + ham for w in doc.split()})


def to_vector(doc):
    """x_j = 1 अगर vocabulary का j-वाँ शब्द email में है (vocabulary से बाहर के शब्द छूट जाते हैं)।"""
    return np.array([w in doc.split() for w in vocab], dtype=float)


X_nb = np.array([to_vector(d) for d in spam + ham])
y_nb = np.r_[np.ones(len(spam)), np.zeros(len(ham))]
phi_y = y_nb.mean()
phi_spam = X_nb[y_nb == 1].mean(axis=0)              # phi_{j|y=1}: कितने हिस्से spam में शब्द j है
phi_ham = X_nb[y_nb == 0].mean(axis=0)               # phi_{j|y=0}


def p_spam(doc):
    """Bayes rule: p(y=1|x). (असली vocabulary पर products की जगह logs जोड़ें।)"""
    x = to_vector(doc)
    joint1 = np.prod(np.where(x == 1, phi_spam, 1 - phi_spam)) * phi_y
    joint0 = np.prod(np.where(x == 1, phi_ham, 1 - phi_ham)) * (1 - phi_y)
    with np.errstate(invalid="ignore"):
        return joint1 / (joint1 + joint0), joint1, joint0


print(f"\nभाग B: Naive Bayes ({len(vocab)} शब्दों की vocabulary, phi_y = {phi_y})")
print("word      phi_j|spam   phi_j|ham")
for w in ("free", "win", "money", "lunch", "meeting"):
    j = vocab.index(w)
    print(f"{w:<9} {phi_spam[j]:8.1f} {phi_ham[j]:11.1f}")
print("\nmessage                        p(x,spam)   p(x,ham)   p(spam|x)")
for doc in ["win free money", "report before lunch", "free lunch today", "win a prize at the meeting"]:
    p, j1, j0 = p_spam(doc)
    print(f"{doc:<28} {j1:11.2e} {j0:10.2e} {p:10.3f}")
print("आख़िरी message: 'win' कभी ham में नहीं दिखा और 'meeting' कभी spam में नहीं,")
print("इसलिए दोनों products 0 हैं और posterior 0/0 = nan है। हल: Laplace smoothing (Lecture 6)।")

# ---------- 7. Plots ----------
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
ax = axes[0]
ax.scatter(*x_train[y_train == 0].T, s=9, color="tab:blue", label="y = 0")
ax.scatter(*x_train[y_train == 1].T, s=9, color="tab:red", label="y = 1")
xs = np.array([x_train[:, 0].min(), x_train[:, 0].max()])
for th, style, name in [(theta_gda, "k-", "GDA boundary"), (theta_lr, "g--", "Logistic boundary")]:
    ax.plot(xs, -(th[0] + th[1] * xs) / th[2], style, lw=2, label=name)
gx, gy = np.meshgrid(np.linspace(-3.5, 5.5, 80), np.linspace(-3.5, 5, 80))
S_inv = np.linalg.inv(Sigma)
for mu, col in [(mu0, "tab:blue"), (mu1, "tab:red")]:
    d = np.stack([gx - mu[0], gy - mu[1]], axis=-1)
    ax.contour(gx, gy, np.einsum("...i,ij,...j->...", d, S_inv, d), levels=[1, 4], colors=col, linewidths=1)
ax.set(xlim=(-3.5, 5.5), ylim=(-3.5, 5), xlabel="x1", ylabel="x2", title="GDA fit (m = 400)")
ax.legend(fontsize=8, loc="lower right")
ax = axes[1]
for gaussian, col, kind in [(True, "tab:purple", "Gaussian data"), (False, "tab:orange", "log-normal data")]:
    ax.plot(sizes, [results[gaussian, m][0] for m in sizes], "o-", color=col, label=f"GDA, {kind}")
    ax.plot(sizes, [results[gaussian, m][1] for m in sizes], "s--", color=col, label=f"Logistic, {kind}")
ax.set(xscale="log", xlabel="training set size m", ylabel="test accuracy", title="GDA vs Logistic Regression")
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig("result.png", dpi=130)
