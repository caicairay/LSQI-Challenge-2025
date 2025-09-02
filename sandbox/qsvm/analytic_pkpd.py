import numpy as np


def conc_analytic(t, dosing_times, dosing_amounts, ka, CL, V):
    """
    One-compartment model with first-order absorption (oral dosing).
    Uses analytic solution with superposition of doses.
    """
    t = np.array(t, dtype=float)
    k = CL / V
    conc = np.zeros_like(t, dtype=float)
    for td, dose in zip(dosing_times, dosing_amounts):
        dt = t - td
        mask = dt >= 0
        if not np.any(mask):
            continue
        dtm = dt[mask]
        if abs(ka - k) < 1e-8:
            contrib = dose * ka / V * dtm * np.exp(-k * dtm)
        else:
            contrib = (
                dose * ka / (V * (ka - k)) * (np.exp(-k * dtm) - np.exp(-ka * dtm))
            )
        conc[mask] += contrib
    return conc


def pd_emax(C, R0, Emax, EC50):
    """Simple inhibitory Emax model for PD response."""
    return R0 * (1.0 - (Emax * C) / (EC50 + C + 1e-12))
