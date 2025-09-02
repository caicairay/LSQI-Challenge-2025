import matplotlib.pyplot as plt
import numpy as np

from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def plot_pred_vs_true(y_true, y_pred_classical, y_pred_quantum, param_name="EC50"):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred_classical, alpha=0.6, label="Classical RBF", c="blue")
    plt.scatter(y_true, y_pred_quantum, alpha=0.6, label="Quantum Kernel", c="red")
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "k--", lw=2)
    plt.xlabel(f"True {param_name}")
    plt.ylabel(f"Predicted {param_name}")
    plt.legend()
    plt.title(f"Prediction vs True ({param_name})")
    plt.show()


def plot_residuals(y_true, y_pred_classical, y_pred_quantum, covariates, cov_name="BW"):
    res_classical = y_pred_classical - y_true
    res_quantum = y_pred_quantum - y_true

    plt.figure(figsize=(8, 5))
    plt.scatter(covariates, res_classical, alpha=0.6, label="Classical RBF", c="blue")
    plt.scatter(covariates, res_quantum, alpha=0.6, label="Quantum Kernel", c="red")
    plt.axhline(0, color="black", linestyle="--")
    plt.xlabel(cov_name)
    plt.ylabel("Residuals (pred - true)")
    plt.legend()
    plt.title(f"Residuals vs {cov_name}")
    plt.show()


def plot_decision_boundary(model_classical, model_quantum, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
    X_grid = np.c_[xx.ravel(), yy.ravel()]

    Zc = model_classical.predict(X_grid).reshape(xx.shape)
    Zq = model_quantum.predict(X_grid).reshape(xx.shape)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Z, title in zip(axes, [Zc, Zq], ["Classical RBF", "Quantum Kernel"]):
        cs = ax.contourf(xx, yy, Z, alpha=0.5, cmap="coolwarm")
        ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm")
        ax.set_title(title)
        plt.colorbar(cs, ax=ax)
    plt.show()


def crossval_scores(model_classical, model_quantum, X, y, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    scores_classical, scores_quantum = [], []

    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model_classical.fit(X_train, y_train)
        model_quantum.fit(X_train, y_train)

        scores_classical.append(
            mean_squared_error(y_test, model_classical.predict(X_test))
        )
        scores_quantum.append(mean_squared_error(y_test, model_quantum.predict(X_test)))

    plt.figure(figsize=(6, 4))
    plt.plot(scores_classical, "o-", label="Classical RBF")
    plt.plot(scores_quantum, "s-", label="Quantum Kernel")
    plt.ylabel("MSE")
    plt.xlabel("Fold")
    plt.legend()
    plt.title("Cross-validation MSE")
    plt.show()
