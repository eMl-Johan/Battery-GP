from pathlib import Path

import numpy as np

from ageing_model import load_ageing_models
from modes import run_single, run_generate_pool, run_simulate_pool, run_schedule_fleet





# MODE selects what this script does:
#   "single"         - run simulation for one specific battery with user-defined states and parameters 
#   "generate_pool"  - create a pool of N batteries and save their states to CSV each battery in the pool will have different levels of charge and age
#   "simulate_pool"  - load the saved pool of batteries and simulate the mission profile for all batteries in the pool
#   "schedule_fleet" - simulate pool and recommend which battery to use to keep remaining lives evenly spread
MODE = "single"

# Discharge profile generation 
# Set GENERATE_PROFILE = True to create a new profile before running simulations.
# The generated file is saved to DISCHARGE_PROFILE and used automatically.
GENERATE_PROFILE  = False
PROFILE_N_STEPS   = 6       # number of current steps
PROFILE_LOWER_A   = 13.0   # minimum current level [A], model is trained with a lower bound of 12.9 so current levels below that will give high variance due to unseen data
PROFILE_UPPER_A   = 16.0   # maximum current level [A], model is trained with a upper bound of 16.0 so current levels above that will give high variance due to unseen data
PROFILE_TARGET_AH = 2.4    # area under the curve [Ah] gives how much charge is drawn
PROFILE_SEED      = None    # random seed 

DISCHARGE_PROFILE    = "Data/X_test.csv"          # current profile (.txt or .csv)
BATTERY_MODEL        = "Trained_Models/FinalGPmodel.pth"      # battery GP
CAPACITY_MODEL       = "Trained_Models/capacity_model.pth"    # ageing GP capacity
RESISTANCE_MODEL     = "Trained_Models/Resistance_model.pth"  # ageing GP resistance

# Single-battery settings MODE = "single"
CUMULATIVE_CHARGE    = 0       # age: charge throughput [Ah];  ranging from 0 to ~600 Ah for this battery 
INITIAL_SOC          = 1.0    # initial state of charge [0–1]
INITIAL_VOLTAGE      = 8.4    # initial terminal voltage [V]
INITIAL_TEMPERATURE  = 23.0   # initial battery temperature [°C]

# Battery pool settings MODE = "generate_pool" / "simulate_pool"
N_BATTERIES          = 25      # number of batteries in the pool
SOC_RANGE            = (0.5, 1.0)   # [min, max] initial SOC sampled uniformly
AGE_RANGE            = (0, 450)     # [min, max] cumulative charge [Ah] (0 = new)
TEMPERATURE_RANGE    = (15.0, 35.0) # [min, max] initial temperature [°C] sampled uniformly
POOL_SEED            = 42           # random seed for reproducibility
POOL_FILE            = "Data/battery_pool.csv"

# POOL_PLOT_STYLE controls how simulate_pool results are displayed:
#   "overlay" - all batteries on shared axes (good for small N)
#   "summary" - bar chart of end-of-mission metrics (scales to any N)
POOL_PLOT_STYLE      = "summary"
POOL_SAVE_PLOT       = True   # save pool plot to Output

# Fleet scheduler settings (MODE = "schedule_fleet") --------------------------
EOL_AH               = 600.0  # cumulative charge at which a battery is retired [Ah]
VOLT_MARGIN          = 0.5    # battery must stay >= 5.0 + this value [V] during mission
TEMP_LIMIT           = 65.0   # maximum acceptable median peak temperature [C]


# Generate discharge profile if GENERATE_PROFILE is True
if GENERATE_PROFILE:
    from generate_profile import step_discharge_profile
    dt_model_s = 0.000275 * 3600
    I_np, prof_info = step_discharge_profile(
        n_steps=PROFILE_N_STEPS,
        lower_A=PROFILE_LOWER_A,
        upper_A=PROFILE_UPPER_A,
        target_ah=PROFILE_TARGET_AH,
        dt_s=dt_model_s,
        rng_seed=PROFILE_SEED,
        return_info=True,
    )
    profile_path = Path(DISCHARGE_PROFILE)
    profile_path.parent.mkdir(exist_ok=True)
    np.savetxt(profile_path, I_np, delimiter=",", fmt="%.10f")
    print(f"Generated profile: {PROFILE_N_STEPS} steps, {PROFILE_LOWER_A}–{PROFILE_UPPER_A} A")
    print(f"  Target: {PROFILE_TARGET_AH:.3f} Ah  |  Achieved: {prof_info['achieved_ah']:.3f} Ah")
    print(f"  Duration: {prof_info['duration_s']:.1f} s  |  Saved to: {profile_path}")

# Load ageing model checkpoints 
load_ageing_models(CAPACITY_MODEL, RESISTANCE_MODEL)


# Run the selected mode
cfg = {k: v for k, v in globals().items() if k.isupper()}

if MODE == "single":
    run_single(cfg)
elif MODE == "generate_pool":
    run_generate_pool(cfg)
elif MODE == "simulate_pool":
    run_simulate_pool(cfg)
elif MODE == "schedule_fleet":
    run_schedule_fleet(cfg)
else:
    raise ValueError(f"Unknown MODE: '{MODE}'. Choose 'single', 'generate_pool', 'simulate_pool', or 'schedule_fleet'.")


