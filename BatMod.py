"""
BatMod.py  –  Battery GP model: inference entry-point.

Public API
----------
run_bat_mod(txt_path, capacity, resistance, ...) -> dict
    Run the trained GP model on a discharge current profile.

    Inputs
    ------
    txt_path   : path to a .txt or .csv file with current values [A]
    capacity   : battery capacity [Ah]  – supplied by the ageing model
    resistance : internal resistance [Ω] – supplied by the ageing model

    Returns
    -------
    dict with:
        'mc_trajectories'  np.ndarray [n_mc, timesteps, 3]   (SOC, V [V], T [°C])
        'median'           np.ndarray [timesteps, 3]
        'lower'            np.ndarray [timesteps, 3]          10th percentile
        'upper'            np.ndarray [timesteps, 3]          90th percentile
        'time_s'           np.ndarray [timesteps]
        'current'          np.ndarray [timesteps]             input current [A]

Design note
-----------
Import this module and call run_bat_mod() – do NOT run it as a bare script

    from BatMod import run_bat_mod
    results = run_bat_mod(txt_path, capacity, resistance)

Running the file directly (python BatMod.py) is supported for quick testing.
"""

from pathlib import Path

from Physic_based_means import TempMeanHyper, AmpHourMeanHyper, VoltageMeanHyper

import numpy as np
import torch
import gpytorch
import torch.nn.functional as F



# Paths
base_dir       = Path(__file__).resolve().parent
model_dir      = base_dir / "Trained_Models"
checkpoint_path = model_dir / "FinalGPmodel.pth"



# Input loader
def load_discharge_txt(txt_path):
    """
    Load current values [A] from a plain-text discharge file.

    Supported formats
    -----------------
    - One current value per line (whitespace or comma delimited)
    - Optional non-numeric header row (skipped automatically)
    - Multi-column files: the LAST column is taken as current

    Returns
    -------
    np.ndarray of shape [N], dtype float32
    """
    txt_path = Path(txt_path)
    if not txt_path.exists():
        raise FileNotFoundError(f"Discharge profile not found: {txt_path}")

    arr = None
    for delimiter, skip in [(None, 0), (",", 0), (None, 1), (",", 1)]:
        try:
            kwargs = {"skiprows": skip}
            if delimiter is not None:
                kwargs["delimiter"] = delimiter
            arr = np.loadtxt(txt_path, **kwargs)
            break
        except ValueError:
            continue

    if arr is None:
        raise ValueError(
            f"Could not parse '{txt_path}'. "
            "Expected one current value per line, optionally with a header row."
        )

    arr = np.atleast_1d(arr)
    current_load = arr[:, -1] if arr.ndim == 2 else arr
    return current_load.astype(np.float32)


# Custom GP kernels  
class TempKernelPhys(gpytorch.kernels.Kernel):
    """
    Temperature kernel:
      - h1 = I^2
      - h2 = T_bat - T_amb

    k(x, x') = k_heat(h1, h1') + k_cool(h2, h2')
    """

    def __init__(self, T_amb=23.0):
        super().__init__()
        self.T_amb = T_amb

        self.k_heat = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )
        self.k_cool = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )

    def _features(self, x):
        # x shape: [..., 4] = [SOC, V, T_bat, I]
        I = x[..., 3:4]
        T = x[..., 2:3]

        h1 = I ** 2
        h2 = T - self.T_amb

        return h1, h2

    def forward(self, x1, x2, diag=False, **params):
        h1_1, h2_1 = self._features(x1)
        h1_2, h2_2 = self._features(x2)

        k_heat = self.k_heat(h1_1, h1_2, diag=diag, **params)
        k_cool = self.k_cool(h2_1, h2_2, diag=diag, **params)

        k = k_heat + k_cool

        if not diag and k.shape[-1] == k.shape[-2]:
            k = k.add_jitter(1e-3)

        return k


class VoltageKernelPhys(gpytorch.kernels.Kernel):
    """
    Voltage kernel:
      - h1 = I
      - h2 = (1 - SOC)^gamma
      - h3 = V_now

    gamma is forced to be > 1.
    """

    def __init__(self, gamma_init=2.0):
        super().__init__()

        self.raw_gamma = torch.nn.Parameter(torch.tensor(0.0))
        self.gamma_init = float(gamma_init)

        with torch.no_grad():
            target = max(1.0 + 1e-6, self.gamma_init)
            self.raw_gamma.copy_(
                torch.log(torch.expm1(torch.tensor(target - 1.0)))
            )

        self.k_h1 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )
        self.k_h2 = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )
        self.k_v = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=1)
        )

    def gamma(self):
        return 1.0 + F.softplus(self.raw_gamma)

    def _features(self, x):
        # x shape: [..., 4] = [SOC, V_now, T_bat, I]
        SOC = x[..., 0:1].clamp(min=1e-6, max=1.0)
        V_now = x[..., 1:2]
        I = x[..., 3:4]

        h1 = I

        dod = (1.0 - SOC).clamp(min=1e-6)
        g = self.gamma()
        h2 = dod.pow(g)

        h3 = V_now

        return h1, h2, h3

    def forward(self, x1, x2, diag=False, **params):
        h1_1, h2_1, h3_1 = self._features(x1)
        h1_2, h2_2, h3_2 = self._features(x2)

        k1 = self.k_h1(h1_1, h1_2, diag=diag, **params)
        k2 = self.k_h2(h2_1, h2_2, diag=diag, **params)
        k3 = self.k_v(h3_1, h3_2, diag=diag, **params)

        k = k1 + k2 + k3

        if not diag and k.shape[-1] == k.shape[-2]:
            k = k.add_jitter(1e-3)

        return k




# GP model wrappers for multi-output 
class SingleTaskGPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, mean_module, covar_module):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(
            inducing_points.size(0)
        )

        variational_strategy = gpytorch.variational.VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True,
        )

        super().__init__(variational_strategy)

        self.mean_module = mean_module
        self.covar_module = covar_module

    def forward(self, x):
        # x shape: [N, 4] = [SOC, V, T, I]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class MultiOutputGPWrapper(torch.nn.Module):
    def __init__(self, inducing_points_list, mean_modules, covar_modules):
        super().__init__()

        self.models = torch.nn.ModuleList(
            [
                SingleTaskGPModel(
                    inducing_points_list[i],
                    mean_modules[i],
                    covar_modules[i],
                )
                for i in range(len(mean_modules))
            ]
        )

    def forward(self, x):
        return [model(x) for model in self.models]



# Build the models
def _build_model():
    """
    Instantiate MultiOutputGPWrapper with all kernels and mean modules.
    Does NOT load trained weights.
    """
    num_inducing = 250
    input_dim    = 4

    inducing_points_list = [
        torch.zeros(num_inducing, input_dim, dtype=torch.float32)
        for _ in range(3)
    ]

    # SOC covariance
    soc_lin = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.LinearKernel(active_dims=torch.tensor([3]))
    )
    soc_rbf = gpytorch.kernels.ScaleKernel(
        gpytorch.kernels.RBFKernel(active_dims=torch.tensor([3]))
    )
    soc_lin.register_constraint("raw_outputscale",
                                gpytorch.constraints.Interval(1e-6, 5e-3))
    soc_rbf.register_constraint("raw_outputscale",
                                gpytorch.constraints.Interval(1e-6, 5e-3))
    soc_lin.outputscale = 1e-4
    soc_rbf.outputscale = 1e-4
    covar_module_0 = soc_lin + soc_rbf

    # Voltage covariance 
    covar_module_1 = VoltageKernelPhys(gamma_init=2.0)
    for sk in [covar_module_1.k_h1, covar_module_1.k_h2, covar_module_1.k_v]:
        sk.register_constraint("raw_outputscale",
                               gpytorch.constraints.Interval(1e-4, 0.5))
    covar_module_1.k_h2.base_kernel.register_constraint(
        "raw_lengthscale", gpytorch.constraints.Interval(1e-3, 0.3)
    )
    covar_module_1.k_h2.base_kernel.lengthscale = 0.05

    # Temperature covariance
    covar_module_2 = TempKernelPhys(T_amb=23.0)
    covar_module_2.k_heat.register_constraint("raw_outputscale",
                                              gpytorch.constraints.Interval(0.1, 1))
    covar_module_2.k_cool.register_constraint("raw_outputscale",
                                              gpytorch.constraints.Interval(0.1, 1))
    covar_module_2.k_heat.outputscale = 0.5
    covar_module_2.k_cool.outputscale = 0.5
    covar_module_2.k_heat.base_kernel.register_constraint(
        "raw_lengthscale", gpytorch.constraints.Interval(1e-3, 1.5)
    )
    covar_module_2.k_cool.base_kernel.register_constraint(
        "raw_lengthscale", gpytorch.constraints.Interval(1e-3, 2.0)
    )

    # Mean modules
    soc_mean     = AmpHourMeanHyper()
    voltage_mean = VoltageMeanHyper()
    temp_mean    = TempMeanHyper(voltage_mean_module=voltage_mean)

    # Assemble model and likelihood
    model = MultiOutputGPWrapper(
        inducing_points_list=inducing_points_list,
        mean_modules=torch.nn.ModuleList([soc_mean, voltage_mean, temp_mean]),
        covar_modules=torch.nn.ModuleList(
            [covar_module_0, covar_module_1, covar_module_2]
        ),
    )

    likelihood = gpytorch.likelihoods.LikelihoodList(
        *[gpytorch.likelihoods.GaussianLikelihood() for _ in range(3)]
    )
    likelihood.likelihoods[0].raw_noise_constraint = gpytorch.constraints.Interval(
        1e-6, 0.1
    )
    likelihood.likelihoods[0].noise = 0.0001

    return model, likelihood


def _set_ageing_params(model, capacity, resistance):
    """
    Override capacity and internal resistance on the mean modules with values
    from the ageing model.  Called AFTER loading the trained checkpoint.

    Parameters
    ----------
    model      : MultiOutputGPWrapper
    capacity   : float  –  battery capacity [Ah]
    resistance : float  –  internal resistance [Ω]
                 Note: VoltageMeanHyper constrains R to [0.07, 0.085];
                 values outside this range are clamped until the constraint
                 is widened to match the ageing model's range.
    """
    # SOC head: AmpHourMeanHyper  (self.C is an unconstrained nn.Parameter)
    soc_mean = model.models[0].mean_module
    with torch.no_grad():
        soc_mean.C.data.fill_(float(capacity))

    # Voltage head: VoltageMeanHyper  (R uses a constrained setter)
    voltage_mean = model.models[1].mean_module
    voltage_mean.R = float(resistance)
    # TempMeanHyper already references the same voltage_mean object, so it
    # automatically uses the updated R.



# Main public function
def _run_mc_rollout(model, likelihood, X_test, n_mc, dt_s):
    """
    Run the Monte Carlo rollout and return raw trajectory statistics.
    No plotting is performed here.
    """
    import torch

    device = next(model.parameters()).device

    if not torch.is_tensor(X_test):
        X_test = torch.tensor(X_test, dtype=torch.float32)
    X_test = X_test.to(device)

    timesteps = X_test.size(0)
    state_dim = 3  # [SOC, Voltage, Temperature]

    with torch.no_grad():
        mc_trajs = torch.zeros(n_mc, timesteps, state_dim, device=device)
        mc_trajs[:, 0] = X_test[0, :state_dim]

        for t in range(timesteps - 1):
            current_val = X_test[t + 1, 3]
            inp = torch.cat(
                [mc_trajs[:, t], current_val.expand(n_mc, 1)],
                dim=1,
            )
            outputs = model(inp)
            deltas = []
            for i in range(state_dim):
                pred_dist = likelihood.likelihoods[i](outputs[i])
                mean_i = pred_dist.mean
                var_i  = pred_dist.variance.clamp_min(1e-12)
                delta_i = mean_i + torch.sqrt(var_i) * torch.randn_like(mean_i)
                deltas.append(delta_i)
            mc_trajs[:, t + 1] = mc_trajs[:, t] + torch.stack(deltas, dim=1)

    mc_trajs_np = mc_trajs.cpu().numpy()
    time_vector  = np.arange(timesteps) * dt_s

    # Cut simulation where median voltage (index 1) first drops below 5 V
    VOLT_CUTOFF = 5.0
    median_volt = np.median(mc_trajs_np[:, :, 1], axis=0)
    below = np.where(median_volt < VOLT_CUTOFF)[0]
    if len(below) > 0:
        cutoff = below[0]
        mc_trajs_np = mc_trajs_np[:, :cutoff, :]
        time_vector  = time_vector[:cutoff]
        print(f"Simulation cut at t={time_vector[-1]:.1f} s (median voltage < {VOLT_CUTOFF} V)")

    return {
        "mc_trajectories": mc_trajs_np,
        "median":          np.median(mc_trajs_np, axis=0),
        "lower":           np.percentile(mc_trajs_np, 10, axis=0),
        "upper":           np.percentile(mc_trajs_np, 90, axis=0),
        "time_s":          time_vector,
    }


def run_bat_mod(
    txt_path,
    capacity=None,
    resistance=None,
    *,
    battery_model_path=None,
    initial_soc=1.0,
    initial_voltage=8.4,
    initial_temperature=23.0,
    n_mc=2000,
    dt_s=0.000275 * 3600,
):
    """
    Run the battery GP model on a discharge current profile.

    Parameters
    ----------
    txt_path           : str or Path
        Path to a .txt file containing current values [A], one per line.
    capacity           : float or None
        Battery capacity [Ah] – from the ageing model.
        Pass None (default) to use the value learned during training.
    resistance         : float or None
        Internal resistance [Ω] – from the ageing model.
        Pass None (default) to use the value learned during training.
    initial_soc        : float   Initial SOC  (default 1.0)
    initial_voltage    : float   Initial terminal voltage [V]  (default 8.4)
    initial_temperature: float   Initial battery temperature [°C]  (default 23.0)
    n_mc               : int     Number of Monte Carlo trajectories  (default 2000)
    dt_s               : float   Time step [s] for the time axis  (default ~0.99 s)

    Returns
    -------
    dict
        'mc_trajectories'  np.ndarray [n_mc, timesteps, 3]  – SOC, V [V], T [°C]
        'median'           np.ndarray [timesteps, 3]
        'lower'            np.ndarray [timesteps, 3]          10th percentile
        'upper'            np.ndarray [timesteps, 3]          90th percentile
        'time_s'           np.ndarray [timesteps]
        'current'          np.ndarray [timesteps]             input current [A]
        'model'            trained MultiOutputGPWrapper  (for use by the caller)
        'likelihood'       trained LikelihoodList        (for use by the caller)
        'X_test'           np.ndarray [timesteps, 4]     full GP input matrix
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load current profile ---
    current_load = load_discharge_txt(txt_path)
    n = len(current_load)
    print(f"Loaded discharge profile: {n} timesteps from '{txt_path}'")

    # Build X_test: [SOC, V, T, I] – initial state repeated, current from file
    X_test = np.column_stack([
        np.full(n, initial_soc,         dtype=np.float32),
        np.full(n, initial_voltage,     dtype=np.float32),
        np.full(n, initial_temperature, dtype=np.float32),
        current_load,
    ]).astype(np.float32)

    # --- Build and load model ---
    model, likelihood = _build_model()

    ckpt = Path(battery_model_path) if battery_model_path is not None else checkpoint_path
    if not ckpt.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt}")

    checkpoint = torch.load(ckpt, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])
    print("Model loaded from checkpoint.")

    # Override capacity and resistance with ageing-model values (if provided)
    if capacity is not None and resistance is not None:
        _set_ageing_params(model, capacity, resistance)
        print(f"Ageing params set – capacity: {capacity} Ah, resistance: {resistance} Ω")

    model.to(device)
    likelihood.to(device)
    model.eval()
    likelihood.eval()

    # --- Run MC rollout (no plotting) ---
    pred = _run_mc_rollout(model, likelihood, X_test, n_mc=n_mc, dt_s=dt_s)

    # --- Save results CSV to Output/ ---
    output_dir = base_dir / "Output"
    output_dir.mkdir(exist_ok=True)
    csv_path = output_dir / "predictions.csv"

    header = (
        "time_s,current_A,"
        "soc_median,soc_lower,soc_upper,"
        "voltage_median,voltage_lower,voltage_upper,"
        "temp_median,temp_lower,temp_upper"
    )
    csv_data = np.column_stack([
        pred["time_s"],
        current_load[:len(pred["time_s"])],
        pred["median"][:, 0], pred["lower"][:, 0], pred["upper"][:, 0],
        pred["median"][:, 1], pred["lower"][:, 1], pred["upper"][:, 1],
        pred["median"][:, 2], pred["lower"][:, 2], pred["upper"][:, 2],
    ])
    np.savetxt(csv_path, csv_data, delimiter=",", header=header, comments="")
    print(f"Results saved to: {csv_path}")

    return {
        "mc_trajectories": pred["mc_trajectories"],   # [n_mc, timesteps, 3]
        "median":          pred["median"],             # [timesteps, 3]
        "lower":           pred["lower"],              # [timesteps, 3]
        "upper":           pred["upper"],              # [timesteps, 3]
        "time_s":          pred["time_s"],             # [timesteps]
        "current":         current_load[:len(pred["time_s"])],  # [timesteps]
        "csv_path":        csv_path,                   # Path to the saved CSV
    }



# Quick test  –  python BatMod.py


if __name__ == "__main__":
    data_dir = base_dir / "Data"

    # Temporary: use the existing generated_profile.csv while a real .txt
    # discharge file is not yet available.  Replace with your .txt path.
    test_profile = data_dir / "generated_profile.csv"

    results = run_bat_mod(
        txt_path=test_profile,
        capacity=2.5,       # [Ah]  – placeholder until ageing model is ready
        resistance=0.078,   # [Ω]   – placeholder until ageing model is ready
    )

