"""
fleet_scheduler.py  -  Fleet degradation scheduling.

Goal: choose which battery to use for the next mission so that the remaining
lives of all batteries stay as evenly spread as possible.  Batteries then
"die" one at a time instead of all at once, spreading capital replacement cost.

Constraints before a battery is even considered:
  - Median final SOC >= SOC_RESERVE_MIN
  - Median peak temperature < TEMP_LIMIT

Objective (minimise):
  Variance of consecutive gaps between sorted remaining lives after the choice.
  Low variance = even spacing = one replacement at a time.

Public API
----------
schedule_fleet(pool_results, pool_metadata, cfg) -> dict or None
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# ── Cost function ──────────────────────────────────────────────────────────────

def _ideal_spacing_cost(remaining_lives, eol_ah):
    """
    Sum of squared deviations from perfectly even spacing between 0 and EOL_AH.

    Ideal fleet: batteries evenly distributed so they die one at a time:
        r*_k = k * eol_ah / N   for k = 1 .. N  (sorted ascending)

    Cost = sum_k ( r_(k) - r*_k )^2

    Lower cost = closer to ideal linear spread = better for the fleet.
    """
    n      = len(remaining_lives)
    sorted_r = np.sort(remaining_lives)
    ideal  = np.linspace(eol_ah / n, eol_ah, n)   # evenly spaced targets
    return float(np.sum((sorted_r - ideal) ** 2))


# ── Mission feasibility ───────────────────────────────────────────────────────

def _mission_feasible(entry, volt_margin, temp_limit):
    """
    Hard constraints checked against the median trajectory.

      - Median minimum voltage >= 5.0 + volt_margin  (safety margin above cutoff)
      - Median peak temperature < temp_limit

    Returns
    -------
    (feasible: bool, reason: str)
    """
    VOLT_CUTOFF = 5.0
    med = entry["results"]["median"]
    min_volt  = med[:, 1].min()
    peak_temp = med[:, 2].max()
    threshold = VOLT_CUTOFF + volt_margin

    if min_volt < threshold:
        return False, f"min volt={min_volt:.3f} V < {threshold:.2f} V"
    if peak_temp >= temp_limit:
        return False, f"peak temp={peak_temp:.1f} >= {temp_limit:.1f} C"
    return True, "OK"


# ── Main scheduler ────────────────────────────────────────────────────────────

def schedule_fleet(pool_results, pool_metadata, cfg):
    """
    Recommend which battery to use for the next mission.

    Parameters
    ----------
    pool_results   : list of dicts  {battery_id, soc, results}
    pool_metadata  : list of dicts  rows from battery_pool.csv
    cfg            : dict  USER INPUTS from main.py

        Keys used
        ---------
        PROFILE_TARGET_AH  : float  Charge consumed per mission [Ah]
        EOL_AH             : float  Cumulative charge at which a battery is retired [Ah]
        VOLT_MARGIN        : float  Min voltage must be >= 5.0 + this value [V]  (default 0.5)
        TEMP_LIMIT         : float  Maximum median peak temp [C]                 (default 65.0)

    Returns
    -------
    dict  pool_results entry for the recommended battery, or None.
    """
    volt_margin = cfg.get("VOLT_MARGIN", 0.5)
    temp_limit  = cfg.get("TEMP_LIMIT",  65.0)
    mission_ah  = cfg["PROFILE_TARGET_AH"]
    eol_ah      = cfg["EOL_AH"]

    # Current cumulative charges from pool metadata
    curr_charges = {
        int(m["battery_id"]): float(m["cumulative_charge_ah"])
        for m in pool_metadata
    }

    # Remaining life = eol_ah - current_charge
    remaining = {
        bid: max(0.0, eol_ah - q)
        for bid, q in curr_charges.items()
    }

    # ── Print header ──
    print("\n" + "=" * 72)
    print("  FLEET SCHEDULER")
    print(f"  Feasibility:  min volt >= {5.0 + volt_margin:.1f} V (cutoff 5.0 + margin {volt_margin:.1f} V)  |  peak temp < {temp_limit:.1f} C")
    print(f"  Ageing:       mission = {mission_ah:.2f} Ah  |  EOL at {eol_ah:.0f} Ah")
    print("=" * 72)

    baseline_score = _ideal_spacing_cost(list(remaining.values()), eol_ah)

    print(f"\n  Current remaining lives [Ah]  (EOL at {eol_ah:.0f} Ah):\n  ideal spacing = {eol_ah/len(remaining):.1f} Ah apart")
    for bid in sorted(remaining):
        bar = "#" * int(remaining[bid] / 10)
        print(f"    Bat {bid:>3}: {remaining[bid]:>7.1f} Ah  {bar}")
    print(f"\n  Baseline cost (ideal-spacing deviation): {baseline_score:.2f}")

    # ── Evaluate each candidate ──
    print(f"\n  {'Bat':>4}  {'Feasible':>9}  {'Cost':>10}  {'Delta':>9}  Reason")
    print("  " + "-" * 58)

    candidates = []
    infeasible = []

    for entry in pool_results:
        bid = entry["battery_id"]
        feasible, reason = _mission_feasible(entry, volt_margin, temp_limit)

        if not feasible:
            print(f"  {bid:>4}  {'No':>9}  {'--':>10}  {'--':>9}  {reason}")
            infeasible.append(bid)
            continue

        # Simulate remaining lives after using this battery for one mission
        sim_remaining = dict(remaining)
        sim_remaining[bid] = max(0.0, remaining[bid] - mission_ah)
        score = _ideal_spacing_cost(list(sim_remaining.values()), eol_ah)
        delta = score - baseline_score

        print(f"  {bid:>4}  {'Yes':>9}  {score:>10.2f}  {delta:>+9.2f}")
        candidates.append((entry, score))

    print("=" * 72)

    if not candidates:
        print("\n  No battery passed all constraints.")
        _plot_fleet(remaining, pool_results, None, mission_ah, temp_limit, volt_margin, eol_ah)
        return None

    # Pick the candidate that minimises the post-mission ideal-spacing cost
    best_entry, best_score = min(candidates, key=lambda c: c[1])
    best_id = best_entry["battery_id"]
    best_delta = best_score - baseline_score

    print(f"\n  Recommended: Bat {best_id}  "
          f"(cost {best_score:.2f}, delta {best_delta:+.2f})")

    _plot_fleet(remaining, pool_results, best_id, mission_ah, temp_limit, volt_margin, eol_ah)
    return best_entry


# ── Plot ──────────────────────────────────────────────────────────────────────

def _plot_fleet(remaining, pool_results, recommended_id, mission_ah, temp_limit, volt_margin, eol_ah):
    """
    Three-panel figure:
      Left   - Remaining life bar chart with mission usage overlay
      Middle - Median minimum voltage per battery with margin line
      Right  - Median peak temperature per battery with limit line
    """
    bids   = sorted(remaining.keys())
    labels = [f"Bat {b}" for b in bids]
    x      = np.arange(len(bids))

    # Gather simulation metrics
    min_volts  = {}
    peak_temps = {}
    for entry in pool_results:
        bid = entry["battery_id"]
        med = entry["results"]["median"]
        min_volts[bid]  = float(med[:, 1].min())
        peak_temps[bid] = float(med[:, 2].max())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # ── Panel 1: Remaining life ──
    ax = axes[0]
    colors = []
    for b in bids:
        if b == recommended_id:
            colors.append("tab:green")
        elif remaining[b] == 0.0:
            colors.append("tab:red")
        else:
            colors.append("tab:blue")

    ideal_vals = np.linspace(eol_ah / len(bids), eol_ah, len(bids))
    ax.bar(x, [remaining[b] for b in bids], color=colors,
           edgecolor="white", linewidth=0.6)
    ax.plot(x, ideal_vals, color="gold", linewidth=1.5,
            linestyle="--", label="Ideal spacing")

    if recommended_id is not None:
        idx  = bids.index(recommended_id)
        base = max(0.0, remaining[recommended_id] - mission_ah)
        ax.bar(idx, mission_ah, bottom=base,
               color="darkorange", alpha=0.8, hatch="//", edgecolor="white",
               label=f"Mission usage ({mission_ah:.1f} Ah)")
    ax.legend(frameon=False, fontsize=8)

    ax.set_title("Remaining Life [Ah]")
    ax.set_ylabel("Remaining Ah")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color="0.8")
    ax.set_axisbelow(True)

    # ── Panel 2: Minimum voltage ──
    ax = axes[1]
    v_vals  = [min_volts.get(b, 0.0) for b in bids]
    colors2 = ["tab:green" if b == recommended_id else "tab:blue" for b in bids]
    ax.bar(x, v_vals, color=colors2, edgecolor="white", linewidth=0.6)
    ax.axhline(5.0 + volt_margin, color="tab:red", linestyle="--",
               linewidth=1.2, label=f"Min required ({5.0 + volt_margin:.1f} V)")
    ax.axhline(5.0, color="0.5", linestyle=":",
               linewidth=1.0, label="Cutoff (5.0 V)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Median Minimum Voltage [V]")
    ax.set_ylabel("Voltage [V]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color="0.8")
    ax.set_axisbelow(True)

    # ── Panel 3: Peak temperature ──
    ax = axes[2]
    t_vals  = [peak_temps.get(b, 0.0) for b in bids]
    colors3 = ["tab:green" if b == recommended_id else "tab:blue" for b in bids]
    ax.bar(x, t_vals, color=colors3, edgecolor="white", linewidth=0.6)
    ax.axhline(temp_limit, color="tab:red", linestyle="--",
               linewidth=1.2, label=f"Limit ({temp_limit:.0f} C)")
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Median Peak Temperature [C]")
    ax.set_ylabel("Temperature [C]")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.grid(True, axis="y", linestyle="--", linewidth=0.8, color="0.8")
    ax.set_axisbelow(True)

    legend_items = [
        Patch(color="tab:blue",  label="Available"),
        Patch(color="tab:green", label="Recommended"),
        Patch(color="tab:red",   label="End of life / infeasible"),
    ]
    fig.legend(handles=legend_items, loc="lower center", ncol=3,
               frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.04))

    title = (f"Fleet Health & Scheduling Decision"
             + (f"  --  Recommended: Bat {recommended_id}" if recommended_id else "  --  No feasible battery"))
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    plt.show()
