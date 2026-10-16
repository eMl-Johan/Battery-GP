import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def drone_discharge_profile(
    *,
    duration_s=600,
    dt_s=1.0,
    zero_prefix=9,
    takeoff_s=75,
    landing_s=110,
    wind_s=140,
    peak_takeoff_A=14.5,
    peak_landing_A=14.5,
    wind_factor=1.25,
    target_ah=2.25,
    return_info=False,
):
    n = int(np.round(duration_s / dt_s))

    n0 = int(zero_prefix)
    n_take = int(np.round(takeoff_s / dt_s))
    n_land = int(np.round(landing_s / dt_s))
    n_wind = int(np.round(wind_s / dt_s))

    n_cruise = n - (n0 + n_take + n_wind + n_land)

    if n_cruise < 0:
        raise ValueError("Segments exceed total duration.")

    t_take = n_take * dt_s
    t_land = n_land * dt_s
    t_wind = n_wind * dt_s
    t_cruise = n_cruise * dt_s

    target_as = float(target_ah) * 3600.0
    known_as = float(peak_takeoff_A) * t_take + float(peak_landing_A) * t_land

    denom = t_cruise + float(wind_factor) * t_wind

    if denom <= 0:
        raise ValueError("Cruise + wind duration must be > 0.")

    cruise_A = (target_as - known_as) / denom
    wind_A = float(wind_factor) * cruise_A

    if cruise_A < 0:
        raise ValueError(
            "Cruise current became negative. Adjust target_ah, peaks, or segment durations."
        )

    y = np.empty(n, dtype=float)

    i = 0

    y[i:i + n0] = 0.0
    i += n0

    y[i:i + n_take] = float(peak_takeoff_A)
    i += n_take

    y[i:i + n_cruise] = float(cruise_A)
    i += n_cruise

    y[i:i + n_wind] = float(wind_A)
    i += n_wind

    y[i:i + n_land] = float(peak_landing_A)
    i += n_land

    info = {
        "n": n,
        "dt_s": float(dt_s),
        "duration_s": float(n * dt_s),
        "target_ah": float(target_ah),
        "achieved_ah": float(y.sum() * dt_s / 3600.0),
        "levels_A": {
            "peak_takeoff_A": float(peak_takeoff_A),
            "peak_landing_A": float(peak_landing_A),
            "cruise_A": float(cruise_A),
            "wind_A": float(wind_A),
        },
        "segments_samples": {
            "zero": n0,
            "takeoff": n_take,
            "cruise": n_cruise,
            "wind": n_wind,
            "landing": n_land,
        },
        "segments_seconds": {
            "zero": float(n0 * dt_s),
            "takeoff": float(t_take),
            "cruise": float(t_cruise),
            "wind": float(t_wind),
            "landing": float(t_land),
        },
    }

    return (y, info) if return_info else y


def step_discharge_profile(
    *,
    n_steps=5,
    lower_A=12.0,
    upper_A=16.0,
    target_ah=None,
    duration_s=600,
    dt_s=1.0,
    rng_seed=None,
    return_info=False,
):
    """
    Generate a random step-change discharge profile.

    The total duration is divided into `n_steps` equal segments. A random
    current level is drawn uniformly from [lower_A, upper_A] for each segment
    and held constant throughout that segment.

    Parameters
    ----------
    n_steps    : int   Number of step segments (default 5)
    lower_A    : float Minimum current level [A] (default 12)
    upper_A    : float Maximum current level [A] (default 16)
    target_ah  : float or None  Desired total charge [Ah]. When provided,
                 duration_s is derived as target_ah * 3600 / mean(lower_A, upper_A).
                 Overrides duration_s when set.
    duration_s : float Total profile duration [s] (default 600, ignored if target_ah set)
    dt_s       : float Time step [s] (default 1.0)
    rng_seed   : int or None  Random seed for reproducibility
    return_info: bool  If True, return (array, info_dict)

    Returns
    -------
    y          : np.ndarray  Current profile [A], length = round(duration_s / dt_s)
    info       : dict        (only when return_info=True)
    """
    if lower_A > upper_A:
        raise ValueError("lower_A must be <= upper_A")
    if n_steps < 1:
        raise ValueError("n_steps must be >= 1")

    if target_ah is not None:
        mean_current = (lower_A + upper_A) / 2.0
        duration_s = float(target_ah) * 3600.0 / mean_current

    rng = np.random.default_rng(rng_seed)
    n = int(np.round(duration_s / dt_s))

    # Stratified sampling: divide [lower_A, upper_A] into n_steps equal bins,
    # draw one random value per bin, then shuffle the order.
    bin_edges = np.linspace(lower_A, upper_A, n_steps + 1)
    levels = rng.uniform(bin_edges[:-1], bin_edges[1:])
    rng.shuffle(levels)

    # Split n samples as evenly as possible across steps
    base = n // n_steps
    remainder = n % n_steps
    segment_lengths = [base + (1 if i < remainder else 0) for i in range(n_steps)]

    y = np.concatenate([
        np.full(seg_len, level)
        for seg_len, level in zip(segment_lengths, levels)
    ])

    info = {
        "n": n,
        "dt_s": float(dt_s),
        "duration_s": float(n * dt_s),
        "n_steps": n_steps,
        "lower_A": float(lower_A),
        "upper_A": float(upper_A),
        "target_ah": float(target_ah) if target_ah is not None else None,
        "levels_A": levels.tolist(),
        "segment_lengths_samples": segment_lengths,
        "achieved_ah": float(y.sum() * dt_s / 3600.0),
        "rng_seed": rng_seed,
    }

    return (y, info) if return_info else y


def save_and_plot_drone_profile(
    I_np,
    prof_info,
    dt_s,
    save_dir=None,
    show_plot=True,
):
    if save_dir is None:
        save_dir = Path(__file__).resolve().parent / "Data"
    else:
        save_dir = Path(save_dir)

    save_dir.mkdir(parents=True, exist_ok=True)

    time_s = np.arange(len(I_np)) * dt_s

    # Save only current values, one value per row, no header, no index
    csv_path = save_dir / "generated_profile.csv"

    np.savetxt(
        csv_path,
        I_np,
        delimiter=",",
        fmt="%.10f"
    )

    # Plot current profile
    fig, ax = plt.subplots(figsize=(8, 4))

    ax.step(time_s, I_np, where="post", linewidth=2)

    ax.set_title("Drone Discharge Current Profile")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [A]")

    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.margins(x=0)

    info_text = (
        f"Target: {prof_info['target_ah']:.3f} Ah\n"
        f"Achieved: {prof_info['achieved_ah']:.3f} Ah\n"
        f"dt: {prof_info['dt_s']:.3f} s"
    )

    ax.text(
        0.98,
        0.95,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        bbox=dict(boxstyle="round", alpha=0.15),
    )

    fig.tight_layout()

    svg_path = save_dir / "drone_discharge_profile.svg"
    png_path = save_dir / "drone_discharge_profile.png"

    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")

    print(f"Saved current-only CSV to: {csv_path}")
    print(f"Saved SVG plot to: {svg_path}")
    print(f"Saved PNG plot to: {png_path}")
    print()
    print("Profile info:")
    print(prof_info)

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

    return csv_path, svg_path, png_path


if __name__ == "__main__":
    dt_model_s = 0.000275 * 3600

    I_np, prof_info = step_discharge_profile(
        n_steps=5,
        lower_A=13.0,
        upper_A=16.0,
        duration_s=650,
        dt_s=dt_model_s,
        return_info=True,
    )

    save_dir = Path(__file__).resolve().parent / "Data"
    save_dir.mkdir(parents=True, exist_ok=True)

    time_s = np.arange(len(I_np)) * dt_model_s

    csv_path = save_dir / "generated_profile.csv"
    np.savetxt(csv_path, I_np, delimiter=",", fmt="%.10f")
    print(f"Saved profile to: {csv_path}")
    print(f"Steps: {prof_info['n_steps']}  |  Levels [A]: {[round(l, 3) for l in prof_info['levels_A']]}")
    print(f"Total charge: {prof_info['achieved_ah']:.3f} Ah  |  Duration: {prof_info['duration_s']:.1f} s")

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.step(time_s, I_np, where="post", linewidth=2)
    ax.set_title("Step Discharge Current Profile")
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Current [A]")
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.7)
    ax.margins(x=0)
    fig.tight_layout()
    png_path = save_dir / "generated_profile.png"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot to: {png_path}")
    plt.show()
