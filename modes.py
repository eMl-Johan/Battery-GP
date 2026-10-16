"""
modes.py  -  Implementation of the simulation modes.

Functions
---------
run_single(cfg)          - simulate one battery
run_generate_pool(cfg)   - generate and save a pool of batteries
run_simulate_pool(cfg)   - load the pool, simulate all batteries, and plot
run_schedule_fleet(cfg)  - simulate pool and recommend which battery to use
"""

import csv
from pathlib import Path

import numpy as np

from ageing_model import predict_ageing
from BatMod import run_bat_mod
from plot_results import plot_predictions, plot_pool_overlay, plot_pool_summary


def _soc_to_voltage(soc):
    """Linear approximation: 5 V at SOC=0, 8.4 V at SOC=1."""
    return 5.0 + 3.4 * float(soc)


def run_single(cfg):
    """
    Simulate a single battery discharge.

    Parameters
    ----------
    cfg : dict  –  keys from the USER INPUTS section of main.py
    """
    if cfg["CUMULATIVE_CHARGE"] == 0:
        capacity   = None
        resistance = None
    else:
        ageing     = predict_ageing(cfg["CUMULATIVE_CHARGE"])
        capacity   = ageing["capacity"]
        resistance = ageing["resistance"]
        cap_ci     = 2 * ageing["capacity_std"]
        res_ci     = 2 * ageing["resistance_std"]
        print("\nAgeing model output:")
        print(f"  Capacity   : {capacity:.4f} Ah  (+/- {cap_ci:.4f} Ah, 95% CI)")
        print(f"  Resistance : {resistance:.6f} Ohm  (+/- {res_ci:.6f} Ohm, 95% CI)")

    results = run_bat_mod(
        txt_path=cfg["DISCHARGE_PROFILE"],
        capacity=capacity,
        resistance=resistance,
        battery_model_path=cfg["BATTERY_MODEL"],
        initial_soc=cfg["INITIAL_SOC"],
        initial_voltage=cfg["INITIAL_VOLTAGE"],
        initial_temperature=cfg["INITIAL_TEMPERATURE"],
    )
    print("\nSimulation complete.")

    plot_predictions(
        results["csv_path"],
        trajectories=results["mc_trajectories"],
        show_plot=True,
    )


def run_generate_pool(cfg):
    """
    Generate a pool of N batteries with random SOC and age, run the ageing
    model for each, and save the states to a CSV file.

    Parameters
    ----------
    cfg : dict  –  keys from the USER INPUTS section of main.py
    """
    rng        = np.random.default_rng(cfg["POOL_SEED"])
    soc_values  = rng.uniform(cfg["SOC_RANGE"][0],         cfg["SOC_RANGE"][1],         size=cfg["N_BATTERIES"])
    age_values  = rng.uniform(cfg["AGE_RANGE"][0],         cfg["AGE_RANGE"][1],         size=cfg["N_BATTERIES"])
    temp_values = rng.uniform(cfg["TEMPERATURE_RANGE"][0], cfg["TEMPERATURE_RANGE"][1], size=cfg["N_BATTERIES"])

    rows = []
    print(f"\nGenerating pool of {cfg['N_BATTERIES']} batteries...\n")
    print(f"{'ID':>4}  {'SOC':>6}  {'Age [Ah]':>10}  {'Temp [°C]':>10}  {'Cap [Ah]':>10}  {'R [Ohm]':>10}  {'V_init [V]':>12}")
    print("-" * 72)

    for i, (soc, age, temp) in enumerate(zip(soc_values, age_values, temp_values)):
        if age == 0:
            capacity   = None
            resistance = None
        else:
            ageing     = predict_ageing(age)
            capacity   = ageing["capacity"]
            resistance = ageing["resistance"]

        v_init = _soc_to_voltage(soc)

        rows.append({
            "battery_id":            i + 1,
            "initial_soc":           round(float(soc), 4),
            "cumulative_charge_ah":  round(float(age), 2),
            "capacity_ah":           round(float(capacity), 4) if capacity is not None else "default",
            "resistance_ohm":        round(float(resistance), 6) if resistance is not None else "default",
            "initial_voltage_v":     round(v_init, 4),
            "initial_temperature_c": round(float(temp), 2),
        })

        cap_str = f"{capacity:.4f}" if capacity is not None else "default"
        res_str = f"{resistance:.6f}" if resistance is not None else "default"
        print(f"{i+1:>4}  {soc:>6.3f}  {age:>10.2f}  {temp:>10.1f}  {cap_str:>10}  {res_str:>10}  {v_init:>12.4f}")

    pool_path = Path(cfg["POOL_FILE"])
    pool_path.parent.mkdir(exist_ok=True)
    with open(pool_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nPool saved to: {pool_path}")


def run_simulate_pool(cfg):
    """
    Load the battery pool CSV, simulate each battery against the discharge
    profile, and plot the results.

    Parameters
    ----------
    cfg : dict  –  keys from the USER INPUTS section of main.py
    """
    pool_path = Path(cfg["POOL_FILE"])
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Pool file not found: {pool_path}\nRun MODE='generate_pool' first."
        )

    with open(pool_path, newline="") as f:
        pool = list(csv.DictReader(f))

    print(f"\nLoaded pool of {len(pool)} batteries from {pool_path}")
    print("Running simulations...\n")

    pool_results = []
    for bat in pool:
        bid  = int(bat["battery_id"])
        soc  = float(bat["initial_soc"])
        v    = float(bat["initial_voltage_v"])
        temp = float(bat["initial_temperature_c"])
        cap  = None if bat["capacity_ah"] == "default" else float(bat["capacity_ah"])
        res  = None if bat["resistance_ohm"] == "default" else float(bat["resistance_ohm"])

        print(f"  Battery {bid}: SOC={soc:.3f}, Age={bat['cumulative_charge_ah']} Ah")
        res_bat = run_bat_mod(
            txt_path=cfg["DISCHARGE_PROFILE"],
            capacity=cap,
            resistance=res,
            battery_model_path=cfg["BATTERY_MODEL"],
            initial_soc=soc,
            initial_voltage=v,
            initial_temperature=temp,
        )
        pool_results.append({"battery_id": bid, "soc": soc, "results": res_bat})

    print(f"\nAll {len(pool)} simulations complete.")

    style     = cfg["POOL_PLOT_STYLE"]
    save_plot = cfg.get("POOL_SAVE_PLOT", False)
    if style == "overlay":
        plot_pool_overlay(pool_results, pool, save_plot=save_plot)
    elif style == "summary":
        plot_pool_summary(pool_results, pool, save_plot=save_plot)
    else:
        raise ValueError(f"Unknown POOL_PLOT_STYLE: '{style}'. Choose 'overlay' or 'summary'.")


def run_schedule_fleet(cfg):
    """
    Load the battery pool CSV, simulate every battery, then run the fleet
    scheduler to recommend which battery to use for the next mission.

    The scheduler minimises the variance of consecutive gaps in remaining
    lives, subject to hard constraints on final SOC and peak temperature.

    Parameters
    ----------
    cfg : dict  -  keys from the USER INPUTS section of main.py

        Keys used (with defaults if omitted)
        -------------------------------------
        SOC_RESERVE_MIN    : float  Min median final SOC           (default 0.05)
        TEMP_LIMIT         : float  Max median peak temperature [C] (default 45.0)
        MISSION_AH         : float  Charge consumed per mission [Ah]
        CAPACITY_DEAD_FRAC : float  Dead threshold fraction         (default 0.8)
    """
    from fleet_scheduler import schedule_fleet

    pool_path = Path(cfg["POOL_FILE"])
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Pool file not found: {pool_path}\nRun MODE='generate_pool' first."
        )

    with open(pool_path, newline="") as f:
        pool = list(csv.DictReader(f))

    print(f"\nLoaded pool of {len(pool)} batteries from {pool_path}")
    print("Running simulations...\n")

    pool_results = []
    for bat in pool:
        bid  = int(bat["battery_id"])
        soc  = float(bat["initial_soc"])
        v    = float(bat["initial_voltage_v"])
        temp = float(bat["initial_temperature_c"])
        cap  = None if bat["capacity_ah"] == "default" else float(bat["capacity_ah"])
        res  = None if bat["resistance_ohm"] == "default" else float(bat["resistance_ohm"])

        print(f"  Battery {bid}: SOC={soc:.3f}, Age={bat['cumulative_charge_ah']} Ah")
        res_bat = run_bat_mod(
            txt_path=cfg["DISCHARGE_PROFILE"],
            capacity=cap,
            resistance=res,
            battery_model_path=cfg["BATTERY_MODEL"],
            initial_soc=soc,
            initial_voltage=v,
            initial_temperature=temp,
        )
        pool_results.append({"battery_id": bid, "soc": soc, "results": res_bat})

    print(f"\nAll {len(pool)} simulations complete.")

    best = schedule_fleet(pool_results, pool, cfg)
    if best is not None:
        print(f"\nUse Bat {best['battery_id']} for the next mission.")
