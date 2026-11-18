"""Lecture 12 example: Imbalanced Evaluation Metrics और Learning Curves (सिर्फ़ NumPy).

दुर्लभ fraud detection (~5% positive data) पर accuracy का धोखा दिखाते हैं।
शुरू से Confusion Matrix, Precision, Recall, F1, ROC, PR, AUC और दो models के
learning curves निकालते हैं, तथा dev set से सही threshold चुनते हैं।
"""
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(42)

# ---------- 1. Imbalanced nonlinear binary dataset (~5% positive) ----------
m = 4500
X_raw = rng.normal(0.0, 1.15, size=(m, 2))
# असली fraud एक मुड़ी हुई सीमा के भीतर है (ताकि linear model को high bias रहे)
latent = -4.35 + 1.55 * (X_raw[:, 1] - 1.05 * X_raw[:, 0] ** 2 + 0.70)
y_all = (rng.uniform(0.0, 1.0, m) < 1.0 / (1.0 + np.exp(-latent))).astype(float)
# Stratified Train (50%) / Dev (25%) / Test (25%) split
pos_i, neg_i = rng.permutation(np.where(y_all == 1)[0]), rng.permutation(np.where(y_all == 0)[0])
tr_i = rng.permutation(np.r_[pos_i[: len(pos_i) // 2], neg_i[: len(neg_i) // 2]])
dv_i = rng.permutation(np.r_[pos_i[len(pos_i) // 2 : 3 * len(pos_i) // 4],
                             neg_i[len(neg_i) // 2 : 3 * len(neg_i) // 4]])
te_i = rng.permutation(np.r_[pos_i[3 * len(pos_i) // 4 :], neg_i[3 * len(neg_i) // 4 :]])
X_tr, y_tr, X_dv, y_dv, X_te, y_te = X_raw[tr_i], y_all[tr_i], X_raw[dv_i], y_all[dv_i], X_raw[te_i], y_all[te_i]
mu, sig = X_tr.mean(axis=0), X_tr.std(axis=0)


def poly_features(X, degree=1):
    """Training statistics से scale करके degree d तक के polynomial features बनाएँ।"""
    Z = (X - mu) / sig
    cols = [np.ones(len(Z))]
    for d in range(1, degree + 1):
        cols.extend((Z[:, 0] ** (d - a)) * (Z[:, 1] ** a) for a in range(d + 1))
    return np.column_stack(cols)


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -35.0, 35.0)))

def log_loss(theta, Phi, y):
    p = np.clip(sigmoid(Phi @ theta), 1e-12, 1.0 - 1e-12)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

def fit_logreg(Phi, y, lam=1e-3, steps=25):
    """Newton-Raphson से L2-regularized Logistic Regression fit करें।"""
    th = np.zeros(Phi.shape[1])
    reg = np.diag([0.0] + [lam] * (Phi.shape[1] - 1))
    for _ in range(steps):
        p = sigmoid(Phi @ th)
        H = (Phi.T * np.maximum(p * (1.0 - p), 1e-6)) @ Phi / len(y) + reg
        th -= np.linalg.solve(H, Phi.T @ (p - y) / len(y) + reg @ th)
    return th


# ---------- 2. Confusion Matrix, Precision, Recall, F1, ROC और AUC शुरू से ----------
def confusion_and_metrics(y_true, y_pred):
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    return tp, fp, fn, tn, (tp + tn) / len(y_true), prec, rec, f1


def roc_pr_auc(y_true, scores):
    taus = np.r_[1.01, np.sort(np.unique(scores))[::-1], -0.01]
    fpr, tpr, prec, rec = [], [], [], []
    for t in taus:
        tp, fp, _, tn, _, p, r, _ = confusion_and_metrics(y_true, (scores >= t).astype(float))
        fpr.append(fp / max(1, fp + tn))
        tpr.append(r)
        prec.append(p if (tp + fp) > 0 else 1.0)
        rec.append(r)
    fpr, tpr = np.array(fpr), np.array(tpr)
    diff = scores[y_true == 1][:, None] - scores[y_true == 0][None, :]
    auc_prob = float(np.mean((diff > 0) + 0.5 * (diff == 0)))
    return fpr, tpr, np.array(prec), np.array(rec), float(np.trapezoid(tpr, fpr)), auc_prob


# ---------- 3. Models train करें और Dev set से सबसे अच्छा threshold चुनें ----------
Phi1_tr, Phi1_dv, Phi1_te = (poly_features(X, 1) for X in (X_tr, X_dv, X_te))
Phi3_tr, Phi3_dv, Phi3_te = (poly_features(X, 3) for X in (X_tr, X_dv, X_te))
th_lin, th_poly = fit_logreg(Phi1_tr, y_tr, lam=1e-3), fit_logreg(Phi3_tr, y_tr, lam=3e-4)

dev_scores = sigmoid(Phi3_dv @ th_poly)
grid_tau = np.linspace(0.03, 0.85, 165)
best_tau = float(grid_tau[int(np.argmax([confusion_and_metrics(y_dv, (dev_scores >= t).astype(float))[7] for t in grid_tau]))])

test_scores, lin_scores = sigmoid(Phi3_te @ th_poly), sigmoid(Phi1_te @ th_lin)
print(f"Dataset positive अनुपात: train={y_tr.mean()*100:.1f}%, dev={y_dv.mean()*100:.1f}%, test={y_te.mean()*100:.1f}%")
print(f"Dev set पर अधिकतम F1 से चुना गया threshold: tau* = {best_tau:.2f}\n")
print("Classifier (Test set)          TP   FP   FN   TN     Acc    Prec  Recall      F1")
for name, pred in [
    ("Always-0 dummy baseline", np.zeros_like(y_te)),
    ("Linear LogReg (tau = 0.50)", (lin_scores >= 0.50).astype(float)),
    ("Poly-3 LogReg (tau = 0.50)", (test_scores >= 0.50).astype(float)),
    (f"Poly-3 LogReg (tau* = {best_tau:.2f})", (test_scores >= best_tau).astype(float)),
]:
    tp, fp, fn, tn, acc, p, r, f1 = confusion_and_metrics(y_te, pred)
    print(f"{name:<28} {tp:4d} {fp:4d} {fn:4d} {tn:4d}  {acc:6.4f}  {p:6.3f}  {r:6.3f}  {f1:6.3f}")

fpr_l, tpr_l, pr_l, rc_l, auc_l, _ = roc_pr_auc(y_te, lin_scores)
fpr_p, tpr_p, pr_p, rc_p, auc_p, auc_prob = roc_pr_auc(y_te, test_scores)
print(f"\nTest ROC-AUC -> Linear: {auc_l:.4f} | Poly-3 (trapezoid): {auc_p:.4f} | P(s+ > s-): {auc_prob:.4f}")

# ---------- 4. दो models के Learning Curves (Bias बनाम Variance, 8 trials का औसत) ----------
sizes = [120, 240, 480, 960, 1600, len(y_tr)]
lc_lin, lc_poly = [], []
for sz in sizes:
    runs1, runs3 = [], []
    for _ in range(8):
        sub = rng.choice(len(y_tr), sz, replace=False)
        t1, t3 = fit_logreg(Phi1_tr[sub], y_tr[sub], lam=2e-4), fit_logreg(Phi3_tr[sub], y_tr[sub], lam=2e-4)
        runs1.append((log_loss(t1, Phi1_tr[sub], y_tr[sub]), log_loss(t1, Phi1_dv, y_dv)))
        runs3.append((log_loss(t3, Phi3_tr[sub], y_tr[sub]), log_loss(t3, Phi3_dv, y_dv)))
    lc_lin.append(np.mean(runs1, axis=0))
    lc_poly.append(np.mean(runs3, axis=0))
lc_lin, lc_poly = np.array(lc_lin), np.array(lc_poly)

print("\nLearning curves (log-loss):   m     Lin(train/dev)      Poly-3(train/dev)")
for sz, (lt, ld), (pt, pd) in zip(sizes, lc_lin, lc_poly):
    print(f"                           {sz:4d}    {lt:.4f} / {ld:.4f}      {pt:.4f} / {pd:.4f}")

# ---------- 5. Plots ----------
fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
axes[0].plot(fpr_p, tpr_p, lw=2, label=f"Poly-3 (AUC={auc_p:.3f})")
axes[0].plot(fpr_l, tpr_l, lw=1.8, ls="--", label=f"Linear (AUC={auc_l:.3f})")
axes[0].plot([0, 1], [0, 1], "k:", lw=1, label="Random (0.50)")
axes[0].set(xlabel="FPR", ylabel="TPR (Recall)", title="ROC Curve (Test)")
axes[0].legend(fontsize=8.5)
axes[1].plot(rc_p, pr_p, lw=2, color="#dc2626", label="Poly-3 PR curve")
p50, r50 = confusion_and_metrics(y_te, (test_scores >= 0.50).astype(float))[5:7]
p_opt, r_opt = confusion_and_metrics(y_te, (test_scores >= best_tau).astype(float))[5:7]
axes[1].scatter([r50, r_opt], [p50, p_opt], c=["#111827", "#16a34a"], s=55, zorder=5)
axes[1].annotate("tau=0.50", (r50, p50), xytext=(6, -10), textcoords="offset points", fontsize=8)
axes[1].annotate(f"tau*={best_tau:.2f}", (r_opt, p_opt), xytext=(6, 4), textcoords="offset points", fontsize=8)
axes[1].axhline(y_te.mean(), color="gray", ls=":", label=f"Baseline ({y_te.mean():.2f})")
axes[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1.03), title="PR Curve & Thresholds")
axes[1].legend(fontsize=8.5)
axes[2].plot(sizes, lc_lin[:, 0], "o--", color="#2563eb", label="Linear train")
axes[2].plot(sizes, lc_lin[:, 1], "o-", color="#2563eb", label="Linear dev (high bias)")
axes[2].plot(sizes, lc_poly[:, 0], "s--", color="#ea580c", label="Poly-3 train")
axes[2].plot(sizes, lc_poly[:, 1], "s-", color="#ea580c", label="Poly-3 dev")
axes[2].set(xlabel="Training size m", ylabel="Log-loss", title="Learning Curves")
axes[2].legend(fontsize=8)
fig.tight_layout()
fig.savefig("result.png", dpi=130)
