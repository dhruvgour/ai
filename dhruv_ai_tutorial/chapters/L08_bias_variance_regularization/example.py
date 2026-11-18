"""Lecture 8 example: Ridge Regression, k-fold Cross Validation और Bias-Variance (सिर्फ़ NumPy).

1D data पर polynomial regression करते हैं। पहले k-fold cross validation से degree और lambda
चुनते हैं। फिर सैकड़ों training sets बनाकर bias^2 और variance को सीधे नापते हैं और देखते हैं कि
sigma^2 + bias^2 + variance सचमुच test error के बराबर आता है।
"""
import numpy as np
import matplotlib.pyplot as plt

SIGMA = 0.3                                       # noise का standard deviation


def h_star(x):
    """असली function, जो model को पता नहीं है।"""
    return np.sin(2.5 * x) + 0.7 * np.cos(3.5 * x)


def make_data(m, rng):
    x = rng.uniform(-1, 1, m)
    return x, h_star(x) + rng.normal(0, SIGMA, m)


# ---------- 1. Polynomial features और ridge का closed form ----------
def poly_features(x, degree):
    """Columns: 1, x, x^2, ..., x^degree"""
    return np.vander(x, degree + 1, increasing=True)


def ridge_fit(x, y, degree, lam):
    """theta = (X^T X + lam*I)^-1 X^T y, पर intercept theta_0 पर penalty नहीं।"""
    X = poly_features(x, degree)
    penalty = lam * np.eye(degree + 1)
    penalty[0, 0] = 0.0
    return np.linalg.solve(X.T @ X + penalty, X.T @ y)


def predict(theta, x):
    return poly_features(x, len(theta) - 1) @ theta


def mse(theta, x, y):
    return np.mean((predict(theta, x) - y) ** 2)


# ---------- 2. k-fold cross validation ----------
def kfold_error(x, y, degree, lam, k=5):
    """हर fold को एक बार बाहर रखें, बाकी पर train करें, बाहर वाले पर error नापें।"""
    folds = np.array_split(np.arange(len(x)), k)  # data पहले से random क्रम में है
    errors = []
    for held_out in folds:
        train = np.setdiff1d(np.arange(len(x)), held_out)
        theta = ridge_fit(x[train], y[train], degree, lam)
        errors.append(mse(theta, x[held_out], y[held_out]))
    return np.mean(errors)


rng = np.random.default_rng(8)
x_train, y_train = make_data(40, rng)
x_test, y_test = make_data(2000, rng)             # सिर्फ़ आख़िरी जाँच के लिए

degrees = [1, 2, 3, 4, 5, 7, 9, 12]
lambdas = [0.0, 1e-4, 1e-2, 1.0]
cv = np.array([[kfold_error(x_train, y_train, d, lam) for lam in lambdas] for d in degrees])

print("1) 5-fold CV error (m = 40). Rows: degree, columns: lambda")
print("degree " + "".join(f"{f'lam={lam:g}':>12}" for lam in lambdas) + "   train MSE (lam=0)")
for d, row in zip(degrees, cv):
    train_err = mse(ridge_fit(x_train, y_train, d, 0.0), x_train, y_train)
    print(f"{d:>6} " + "".join(f"{e:12.4f}" for e in row) + f"   {train_err:10.4f}")

best_d, best_l = np.unravel_index(np.argmin(cv), cv.shape)
d_cv, lam_cv = degrees[best_d], lambdas[best_l]
theta_cv = ridge_fit(x_train, y_train, d_cv, lam_cv)          # चुने हुए model को पूरे data पर train करें
print(f"\nCV ने चुना: degree = {d_cv}, lambda = {lam_cv:g}  (CV error {cv.min():.4f})")
print("Test MSE (2000 नए examples पर):")
for label, d, lam in [("CV का चुना हुआ model", d_cv, lam_cv), ("degree 1  (underfit)", 1, 0.0),
                      ("degree 12, lam=0 (overfit)", 12, 0.0), ("degree 12, lam=0.01", 12, 1e-2)]:
    print(f"  {mse(ridge_fit(x_train, y_train, d, lam), x_test, y_test):.4f}  {label}")
print(f"(noise का sigma^2 = {SIGMA**2:.4f}: इससे नीचे कोई model नहीं जा सकता)")


# ---------- 3. Bias-variance simulation: बहुत सारे training sets, हर एक पर नया model ----------
# हर training set में x की वही m जगहें हैं (बराबर दूरी पर) और y का noise नया है, जैसे वही
# experiment बार-बार दोहराया गया हो। इससे सिर्फ़ noise से पैदा हुआ variance साफ़ दिखता है।
x_grid = np.linspace(-1, 1, 200)                  # test points


def bias_variance(degree, lam, m=15, trials=1000, seed=0):
    sim = np.random.default_rng(seed)
    x = np.linspace(-1, 1, m)
    preds = np.empty((trials, len(x_grid)))
    test_err = 0.0
    for t in range(trials):
        y = h_star(x) + sim.normal(0, SIGMA, m)                        # नया training set
        preds[t] = predict(ridge_fit(x, y, degree, lam), x_grid)
        y_new = h_star(x_grid) + sim.normal(0, SIGMA, len(x_grid))     # नए noisy test targets
        test_err += np.mean((y_new - preds[t]) ** 2) / trials
    h_avg = preds.mean(axis=0)                                # औसत model
    variance = np.mean(preds.var(axis=0))
    # trials सीमित हैं, इसलिए h_avg में थोड़ा noise बचता है (लगभग variance / trials): उसे घटा दें
    bias2 = max(0.0, np.mean((h_avg - h_star(x_grid)) ** 2) - variance / trials)
    return bias2, variance, test_err


print("\n2) Bias-variance tradeoff (m = 15, 1000 training sets, lambda = 0)")
print("degree    bias^2   variance   sigma^2+bias^2+var   measured test MSE")
sim_degrees = [1, 2, 3, 4, 6, 8, 10]
table = []
for d in sim_degrees:
    b2, var, err = bias_variance(d, 0.0)
    table.append((b2, var, err))
    print(f"{d:>6} {b2:9.4f} {var:10.4f} {SIGMA**2 + b2 + var:20.4f} {err:19.4f}")

print("\n3) Degree 10 पर lambda और m का असर")
print("setting                 bias^2   variance")
for label, lam, m in [("lam=0,    m=15", 0.0, 15), ("lam=0.01, m=15", 1e-2, 15),
                      ("lam=1,    m=15", 1.0, 15), ("lam=0,    m=150", 0.0, 150)]:
    b2, var, _ = bias_variance(10, lam, m=m)
    print(f"{label:<20} {b2:9.4f} {var:10.4f}")

# ---------- 4. Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 3.9))
axes[0].scatter(x_train, y_train, s=14, color="black", label="training data")
axes[0].plot(x_grid, h_star(x_grid), "k--", lw=1, label="true function")
for d, lam, col in [(1, 0.0, "tab:blue"), (d_cv, lam_cv, "tab:green"), (12, 0.0, "tab:red")]:
    axes[0].plot(x_grid, predict(ridge_fit(x_train, y_train, d, lam), x_grid), color=col,
                 label=f"degree {d}, lambda={lam:g}")
axes[0].set(ylim=(-2.2, 2.2), xlabel="x", ylabel="y", title="Underfit, chosen by CV, overfit")
axes[0].legend(fontsize=7)
for j, lam in enumerate(lambdas):
    axes[1].plot(degrees, cv[:, j], "o-", ms=3, label=f"lambda={lam:g}")
axes[1].set(yscale="log", xlabel="degree", ylabel="5-fold CV error", title="Model selection")
axes[1].legend(fontsize=7)
b2s, variances, errs = np.array(table).T
axes[2].plot(sim_degrees, b2s, "o-", label="bias^2")
axes[2].plot(sim_degrees, variances, "o-", label="variance")
axes[2].plot(sim_degrees, errs, "k^-", label="test MSE")
axes[2].axhline(SIGMA**2, color="gray", ls=":", label="sigma^2 (noise)")
axes[2].set(xlabel="degree", title="Bias-variance tradeoff")
axes[2].legend(fontsize=7)
fig.tight_layout()
fig.savefig("result.png", dpi=130)
