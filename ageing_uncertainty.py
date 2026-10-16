"""
ageing_uncertainty.py  –  Outer-loop ageing uncertainty propagation.

Wraps run_bat_mod with a two-level Monte Carlo scheme:

  Outer loop : K samples of (capacity, resistance) drawn independently from
               the ageing GP posteriors  N(μ_C, σ_C²)  and  N(μ_R, σ_R²).
  Inner loop : M battery GP trajectories per ageing sample (the existing
               run_bat_mod Monte Carlo rollout).
  Total      : K × M trajectories pooled to produce the same output dict
               as run_bat_mod.

Note on independence
--------------------
The capacity and resistance ageing models are two separate GPs, so their
posteriors are sampled independently.  If a joint multi-output ageing model
is used in the future, replace the two rng.normal calls with a joint draw
from the 2-D posterior.

Public API
----------
run_bat_mod_with_ageing_uncertainty(
    txt_path, cumulative_charge,
    *, n_ageing_samples, n_mc_inner, **kwargs
) -> dict
    Drop-in replacement for run_bat_mod when ageing uncertainty should be
    propagated.  Returns the same dict format.
"""

import numpy as np

from ageing_model import predict_ageing
from BatMod import run_bat_mod


def run_bat_mod_with_ageing_uncertainty(
    txt_path,
    cumulative_charge,
    *,
    n_ageing_samples=40,
    n_mc_inner=50,
    battery_model_path=None,
    initial_soc=1.0,
    initial_voltage=8.4,
    initial_temperature=23.0,
    dt_s=0.000275 * 3600,
):
    """
    Two-level Monte Carlo rollout that propagates ageing parameter uncertainty
    into the battery GP output.

    Parameters
    ----------
    txt_path            : str or Path
        Discharge current profile (.txt or .csv).
    cumulative_charge   : float
        Total charge drawn from the battery [Ah], passed to the ageing GP
        to obtain the posterior mean and standard deviation for capacity
        and resistance.
    n_ageing_samples    : int
        K – number of (capacity, resistance) samples drawn from the ageing
        GP posterior.  Default 40.
    n_mc_inner          : int
        M – battery GP trajectories per ageing sample.  Default 50.
        Total trajectories = K × M  (default 40 × 50 = 2000).
    battery_model_path  : str or Path or None
        Path to the battery GP checkpoint.  None uses the default path
        inside BatMod.py.
    initial_soc         : float   Initial SOC [0–1].
    initial_voltage     : float   Initial terminal voltage [V].
    initial_temperature : float   Initial battery temperature [°C].
    dt_s                : float   Time step [s] for the time axis.

    Returns
    -------
    dict  –  same format as run_bat_mod:
        'mc_trajectories'  np.ndarray [K*M, timesteps, 3]  SOC, V [V], T [°C]
        'median'           np.ndarray [timesteps, 3]
        'lower'            np.ndarray [timesteps, 3]   10th percentile
        'upper'            np.ndarray [timesteps, 3]   90th percentile
        'time_s'           np.ndarray [timesteps]
        'current'          np.ndarray [timesteps]
        'csv_path'         Path  (from the last inner run_bat_mod call)
    """
    # ── Query the ageing GP ──────────────────────────────────────────────────
    ageing  = predict_ageing(cumulative_charge)
    mu_C    = ageing["capacity"]
    sigma_C = ageing["capacity_std"]
    mu_R    = ageing["resistance"]
    sigma_R = ageing["resistance_std"]

    # ── Draw outer samples ───────────────────────────────────────────────────
    rng       = np.random.default_rng()
    C_samples = rng.normal(mu_C, sigma_C, size=n_ageing_samples).clip(min=0.1)
    R_samples = rng.normal(mu_R, sigma_R, size=n_ageing_samples).clip(min=1e-4)

    total = n_ageing_samples * n_mc_inner
    print(f"\nAgeing uncertainty propagation:")
    print(f"  Ageing samples : {n_ageing_samples}  ×  inner MC {n_mc_inner}  =  {total} total trajectories")
    print(f"  Capacity       : {mu_C:.4f} ± {sigma_C:.4f} Ah  (1σ)")
    print(f"  Resistance     : {mu_R:.6f} ± {sigma_R:.6f} Ω   (1σ)")

    # ── Outer loop ───────────────────────────────────────────────────────────
    all_trajectories = []
    ref_time         = None
    ref_current      = None
    ref_len          = None
    last_result      = None

    for k, (C_k, R_k) in enumerate(zip(C_samples, R_samples)):
        print(f"  [Sample {k + 1:>3}/{n_ageing_samples}]  C={C_k:.4f} Ah  R={R_k:.6f} Ω")

        result = run_bat_mod(
            txt_path=txt_path,
            capacity=float(C_k),
            resistance=float(R_k),
            battery_model_path=battery_model_path,
            initial_soc=initial_soc,
            initial_voltage=initial_voltage,
            initial_temperature=initial_temperature,
            n_mc=n_mc_inner,
            dt_s=dt_s,
        )
        last_result = result
        traj = result["mc_trajectories"]  # [n_mc_inner, timesteps_k, 3]

        if ref_time is None:
            ref_time    = result["time_s"]
            ref_current = result["current"]
            ref_len     = traj.shape[1]

        # Align trajectory lengths across outer samples.
        # Different ageing parameters can cause the voltage cutoff to be hit at
        # different timesteps, producing trajectories of different length.
        # Shorter runs are NaN-padded; longer runs are trimmed to ref_len.
        t = traj.shape[1]
        if t < ref_len:
            pad  = np.full((traj.shape[0], ref_len - t, 3), np.nan)
            traj = np.concatenate([traj, pad], axis=1)
        elif t > ref_len:
            traj = traj[:, :ref_len, :]

        all_trajectories.append(traj)

    # ── Pool and compute statistics ──────────────────────────────────────────
    all_traj = np.concatenate(all_trajectories, axis=0)  # [K*M, timesteps, 3]

    # nanmedian / nanpercentile handle any NaN padding from the alignment step
    return {
        "mc_trajectories": all_traj,
        "median":          np.nanmedian(all_traj, axis=0),
        "lower":           np.nanpercentile(all_traj, 10, axis=0),
        "upper":           np.nanpercentile(all_traj, 90, axis=0),
        "time_s":          ref_time,
        "current":         ref_current,
        "csv_path":        last_result.get("csv_path") if last_result else None,
    }
