"""
battery_selector.py  -  Mission-based battery selection from a simulated pool.

Public API
----------
select_battery(pool_results, pool_metadata, cfg) -> dict or None
    Rank batteries by mission completion probability, apply hard constraints,
    print a ranked table, plot the results, and return the recommended battery.

How P(complete) is computed
---------------------------
run_bat_mod() returns mc_trajectories of shape [n_mc, timesteps, 3].
Each of the n_mc trajectories is one plausible physical discharge path.
P(complete) = fraction of trajectories whose FINAL SOC >= SOC_RESERVE_MIN.
This is a genuine stochastic metric: the GP ensemble has spread in final SOC,
so a battery whose median looks fine can still have low P(complete) if many
trajectories end close to the reserve threshold.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ── Internal helpers ──────────────────────────────────────────────────────────

def _mission_completion_prob(mc_trajectories, soc_reserve):
    """
    Fraction of MC trajectories where the final SOC >= soc_reserve.

    Using final SOC rather than a voltage cutoff is the correct stochastic
    metric here: the GP ensemble has genuine spread in final SOC, so a
    battery whose median looks adequate can still carry meaningful risk if
    many trajectories end near or below the reserve threshold.

    Parameters
    ----------
    mc_trajectories : np.ndarray [n_mc, timesteps, 3]
    soc_reserve     : float  Minimum acceptable final SOC [0-1].

    Returns
    -------
    float  in [0, 1]
    """
    final_soc = mc_trajectories[:, -1, 0]   # [n_mc] final SOC values
    return float((final_soc >= soc_reserve).mean())


def _compute_metrics(entry, soc_reserve):
    """Compute scalar end-of-mission metrics for one battery entry."""
    r = entry["results"]
    return {
        "battery_id":  entry["battery_id"],
        "p_complete":  _mission_completion_prob(r["mc_trajectories"], soc_reserve),
        "final_soc":   float(np.median(r["mc_trajectories"][:, -1, 0])),
        "min_voltage": float(r["median"][:, 1].min()),
        "max_temp":    float(r["median"][:, 2].max()),
        "duration_s":  float(r["time_s"][-1]),
    }


# ── Public function ───────────────────────────────────────────────────────────

def select_battery(pool_results, pool_metadata, cfg):
    """
    Rank batteries by mission completion probability and apply hard constraints.

    Parameters
    ----------
    pool_results   : list of dicts  {battery_id, soc, results}
    pool_metadata  : list of dicts  rows from battery_pool.csv (same order)
    cfg            : dict  USER INPUTS from main.py

    Constraint keys read from cfg (with defaults)
    ---------------------------------------------
        SOC_RESERVE_MIN : float  SOC threshold that defines mission completion.
                                 P(complete) = P(final SOC >= this).  (default 0.05)
        P_COMPLETE_MIN  : float  Minimum required P(complete) to be eligible.  (default 0.0)
    """
    soc_reserve = cfg.get("SOC_RESERVE_MIN", 0.05)
    p_min       = cfg.get("P_COMPLETE_MIN",  0.0)

    # Compute metrics for every battery
    metrics = [_compute_metrics(e, soc_reserve) for e in pool_results]

    # Apply hard constraint: P(complete) alone gates eligibility
    for m in metrics:
        m["passed"] = m["p_complete"] >= p_min

    passed = [m for m in metrics if     m["passed"]]
    failed = [m for m in metrics if not m["passed"]]

    # Rank survivors: primary sort = P(complete), tiebreak = final SOC
    ranked = sorted(passed, key=lambda m: (m["p_complete"], m["final_soc"]), reverse=True)

    # Print ranked table
    print("\n" + "=" * 72)
    print("  BATTERY SELECTION RESULTS")
    print(f"  P(complete) = P(final SOC >= {soc_reserve:.2f})  |  required P(complete) >= {p_min:.2f}")
    print("=" * 72)

    if ranked:
        print(f"\n  {'Rank':>4}  {'Bat':>4}  {'P(complete)':>12}  {'Final SOC':>10}"
              f"  {'Min V [V]':>10}  {'Max T [C]':>10}  {'Dur [s]':>9}")
        print("  " + "-" * 68)
        for rank, m in enumerate(ranked, 1):
            tag = "  <-- RECOMMENDED" if rank == 1 else ""
            print(f"  {rank:>4}  {m['battery_id']:>4}  {m['p_complete']:>12.3f}"
                  f"  {m['final_soc']:>10.3f}  {m['min_voltage']:>10.3f}"
                  f"  {m['max_temp']:>10.2f}  {m['duration_s']:>9.1f}{tag}")
    else:
        print("\n  No battery passed all constraints.")

    if failed:
        print(f"\n  Failed constraints ({len(failed)} batteries):")
        for m in failed:
            print(f"    Bat {m['battery_id']:>3}  --  P(complete)={m['p_complete']:.3f} < {p_min:.2f}")

    print("=" * 72)

    _plot_selection(metrics, ranked)

    if not ranked:
        return None

    best_id = ranked[0]["battery_id"]
    return next(e for e in pool_results if e["battery_id"] == best_id)


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_selection(metrics, ranked):
    """
    Two-panel bar chart: P(complete) and final SOC for every battery.
    Green = passed constraints, red = failed, gold outline = recommended.
    """
    x      = np.arange(len(metrics))
    ids    = [f"Bat {m['battery_id']}" for m in metrics]
    p_vals = [m["p_complete"] for m in metrics]
    s_vals = [m["final_soc"]  for m in metrics]
    colors = ["tab:green" if m["passed"] else "tab:red" for m in metrics]
    best_id = ranked[0]["battery_id"] if ranked else None

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, vals, title, ylabel in [
        (axes[0], p_vals, "Mission Completion Probability", "P(complete)"),
        (axes[1], s_vals, "Final SOC (median)",             "SOC"),
    ]:
        bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.6)
        if best_id is not None:
            best_idx = next(i for i, m in enumerate(metrics) if m["battery_id"] == best_id)
            bars[best_idx].set_edgecolor("gold")
            bars[best_idx].set_linewidth(3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(ids, rotation=40, ha="right", fontsize=8)
        ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color="0.8")
        ax.set_axisbelow(True)

    legend_items = [
        Patch(color="tab:green", label="Passed constraints"),
        Patch(color="tab:red",   label="Failed constraints"),
        Patch(facecolor="tab:green", edgecolor="gold", linewidth=3, label="Recommended"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle("Battery Selection -- Mission Suitability", fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()
