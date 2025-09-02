import numpy as np
from scipy.optimize import least_squares
from analytic_pkpd import conc_analytic, pd_emax


def fit_pk(times, conc_obs, dosing_times, dosing_amounts, BW=70.0):
    """Fit PK parameters (ka, CL, V) for one subject."""
    p0 = [1.0, 6.0 * (BW / 70.0) ** 0.75, 40.0]
    bounds = ([0.01, 0.1, 5.0], [10.0, 200.0, 500.0])

    def resid(p):
        ka, CL, V = p
        pred = conc_analytic(times, dosing_times, dosing_amounts, ka, CL, V)
        return np.log1p(pred) - np.log1p(conc_obs)

    res = least_squares(resid, p0, bounds=bounds, method="trf", max_nfev=1000)
    return dict(ka=res.x[0], CL=res.x[1], V=res.x[2])


def fit_pd(times, resp_obs, Cpred):
    """Fit PD parameters (R0, Emax, EC50) for one subject given PK predictions."""
    p0 = [np.median(resp_obs), 0.6, 1.0]
    bounds = ([0.1, 0.01, 0.01], [20.0, 1.0, 50.0])

    def resid(p):
        R0, Emax, EC50 = p
        pred = pd_emax(Cpred, R0, Emax, EC50)
        return pred - resp_obs

    res = least_squares(resid, p0, bounds=bounds, method="trf", max_nfev=1000)
    return dict(R0=res.x[0], Emax=res.x[1], EC50=res.x[2])


def sample_population(pk_df, pd_df, n=1000, rng=None):
    """
    Monte Carlo sampling of population PK/PD parameters using log-normal (for PK, EC50) and normal (for R0, Emax).
    """
    rng = rng or np.random.default_rng()
    params = {}
    for col in ["ka", "CL", "V"]:
        mean, std = np.log(pk_df[col]).mean(), np.log(pk_df[col]).std()
        params[col] = np.exp(rng.normal(mean, std, size=n))
        params["EC50"] = np.exp(
            rng.normal(
                np.log(pd_df["EC50"]).mean(), np.log(pd_df["EC50"]).std(), size=n
            )
        )
        params["R0"] = rng.normal(pd_df["R0"].mean(), pd_df["R0"].std(), size=n)
        params["Emax"] = rng.normal(pd_df["Emax"].mean(), pd_df["Emax"].std(), size=n)
    return params


def simulate_population(
    params, dose, interval_hours=24, duration_days=28, dt=2.0, thr=3.3
):
    """
    Simulate PK/PD for a population over given dosing regimen.
    Returns fraction of patients below PD threshold at steady state.
    """
    n = len(params["ka"])
    t_grid = np.arange(0, duration_days * 24 + dt, dt)
    dosing_times = np.arange(0, duration_days * 24 + 1, interval_hours)
    dosing_amounts = np.full_like(dosing_times, dose, dtype=float)

    fractions = np.zeros(n)
    for i in range(n):
        C = conc_analytic(
            t_grid,
            dosing_times,
            dosing_amounts,
            params["ka"][i],
            params["CL"][i],
            params["V"][i],
        )
        R = pd_emax(C, params["R0"][i], params["Emax"][i], params["EC50"][i])
        last_mask = t_grid >= (duration_days * 24 - interval_hours)
        fractions[i] = np.mean(R[last_mask] < thr)
    return fractions
