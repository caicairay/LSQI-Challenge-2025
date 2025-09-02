import os
import time
import pandas as pd
import numpy as np
import pennylane as qml
from sklearn.model_selection import train_test_split
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import mean_squared_error

from analytic_pkpd import conc_analytic
from utils import fit_pk, fit_pd
from quantum_kernel import QuantumKernel
from plotter import (
    plot_pred_vs_true,
    plot_residuals,
    plot_decision_boundary,
    crossval_scores,
)

if __name__ == "__main__":
    """
    Main workflow:
    1. Fit PK and PD per subject.
    2. Collect parameter tables.
    3. Train QSVM.
    4. Run population simulation.
    """

    # ---------------------------
    # 0) Load datasets
    # ---------------------------
    DATA_PATH = r"./EstData.csv"
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file {DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    print(f"Loaded {DATA_PATH} with {len(df)} rows.")

    subjects_ids = df["ID"].unique()
    pk_fits, pd_fits = {}, {}

    # ---------------------------
    # 1) Per-subject PK/PD target extraction
    # ---------------------------
    for sub_id in subjects_ids:
        sub = df[df["ID"] == sub_id]
        dose_rows = sub[(sub["EVID"] == 1) & (sub["AMT"] > 0)]
        dosing_times = dose_rows["TIME"].values.tolist()
        dosing_amounts = dose_rows["AMT"].values.tolist()

        # Fit PK: Compound concentration (mg/L) over time (h)
        pk_obs = sub[sub["DVID"] == 1].dropna(subset=["DV"])

        if len(pk_obs) >= 4:
            pk_fits[sub_id] = fit_pk(
                pk_obs["TIME"].values,
                pk_obs["DV"].values,
                dosing_times,
                dosing_amounts,
                BW=sub["BW"].iloc[0],  # body weight covariate
            )

        # Fit PD: Biomarker level (ng/mL) over time (h)
        pd_obs = sub[sub["DVID"] == 2].dropna(subset=["DV"])
        if len(pd_obs) >= 4 and sub_id in pk_fits:
            Cpred = conc_analytic(
                pd_obs["TIME"].values, dosing_times, dosing_amounts, **pk_fits[sub_id]
            )
            pd_fits[sub_id] = fit_pd(pd_obs["TIME"].values, pd_obs["DV"].values, Cpred)
            # prepare covariates and target row
            BW = float(sub["BW"].iloc[0]) if "BW" in sub.columns else 70.0
            COMED = int(sub["COMED"].iloc[0]) if "COMED" in sub.columns else 0
            pd_fits[sub_id].update(BW=BW, COMED=COMED)

    # ---------------------------
    # 2) Gathering fit parameters
    # ---------------------------
    pk_df = pd.DataFrame.from_dict(pk_fits, orient="index")
    pd_df = pd.DataFrame.from_dict(pd_fits, orient="index")
    print("PK summary:\n", pk_df.describe().T)
    print("PD summary:\n", pd_df.describe().T)

    pk_df.to_csv(r"./output/pk_parameters.csv")
    pd_df.to_csv(r"./output/pd_parameters.csv")

    # ---------------------------
    # 3) Train QSVM
    # ---------------------------
    # pick small number; will embed combined covariates via simple linear projection

    q_kernel = QuantumKernel(n_qubits=4, qpu="lightning.qubit")

    # Prepare covariates matrix X and target y (EC50)
    X = pd_df[["BW", "COMED"]].values
    y = pd_df["EC50"].values

    # train/test split for evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples.")

    # a) Classical RBF kernel ridge (scikit-learn)
    rbf_krr = KernelRidge(alpha=1e-2, kernel="rbf", gamma=0.1)
    rbf_krr.fit(X_train, y_train)
    y_pred_rbf = rbf_krr.predict(X_test)
    mse_rbf = mean_squared_error(y_test, y_pred_rbf)
    print(f"RBF KernelRidge test MSE: {mse_rbf:.4f}")

    # b) Quantum kernel ridge (precomputed kernel)
    print("Computing quantum kernel matrix (this runs statevector sims)...")
    t0 = time.time()
    K_train = q_kernel.quantum_kernel_matrix(X_train)
    # fit kernel ridge with precomputed kernel
    qkrr = KernelRidge(alpha=1e-2, kernel="precomputed")
    qkrr.fit(K_train, y_train)

    m_test = X_test.shape[0]
    n_train = X_train.shape[0]

    states_test = [q_kernel.compute_state_vector(x) for x in X_test]
    states_train = [q_kernel.compute_state_vector(x) for x in X_train]

    # compute kernel matrix between X_train and X_test
    K_test_train = np.zeros((m_test, n_train))
    for i in range(m_test):
        for j in range(n_train):
            K_test_train[i, j] = np.abs(np.vdot(states_test[i], states_train[j])) ** 2

    y_pred_q = qkrr.predict(K_test_train)
    mse_q = mean_squared_error(y_test, y_pred_q)
    print(f"Quantum KRR test MSE: {mse_q:.4f} (computed in {time.time()-t0:.1f}s)")

    # simple compare
    print(f"MSEs -> classical RBF: {mse_rbf:.4f}, quantum KRR: {mse_q:.4f}")

    plot_pred_vs_true(
        y_true=y_test,
        y_pred_classical=rbf_krr.predict(X_test),
        y_pred_quantum=y_pred_q,
        param_name="EC50",
    )
