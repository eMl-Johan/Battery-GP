"""
ageing_model.py  –  Ageing GP models for battery capacity and resistance.

Public API
----------
predict_ageing(cumulative_charge) -> dict
    Predict capacity [Ah] and internal resistance [Ω] from cumulative charge
    throughput [Ah], using two pre-trained GP models.

    Returns
    -------
    dict with:
        'capacity'          float  – predicted capacity [Ah]
        'capacity_std'      float  – 1-sigma uncertainty [Ah]
        'resistance'        float  – predicted internal resistance [Ω]
        'resistance_std'    float  – 1-sigma uncertainty [Ω]

Usage (in main.py)
------------------
    from ageing_model import predict_ageing
    ageing = predict_ageing(cumulative_charge=45.0)
    Q = ageing["capacity"]
    R = ageing["resistance"]
"""

from pathlib import Path
import torch
import gpytorch



# Paths
_base_dir  = Path(__file__).resolve().parent
_model_dir = _base_dir / "Trained_Models"

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# GP model class
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module  = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel()
        )

    def forward(self, x):
        mean_x  = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)



# Model loader
def _load_trained_gp(path: Path):
    """
    Load an ExactGP model + likelihood + training data from disk.
    Returns: model, likelihood
    """
    checkpoint = torch.load(path, map_location=_device)

    train_x = checkpoint["train_x"].to(_device)
    train_y = checkpoint["train_y"].to(_device)

    likelihood = gpytorch.likelihoods.GaussianLikelihood().to(_device)
    model      = ExactGPModel(train_x, train_y, likelihood).to(_device)

    model.load_state_dict(checkpoint["model_state_dict"])
    likelihood.load_state_dict(checkpoint["likelihood_state_dict"])

    model.eval()
    likelihood.eval()

    print(f"Loaded GP model from: {path}")
    return model, likelihood



# Load models once at import time (defaults – overridden by load_ageing_models())
_cap_model, _cap_lik = _load_trained_gp(_model_dir / "capacity_model.pth")
_R_model,   _R_lik   = _load_trained_gp(_model_dir / "Resistance_model.pth")


def load_ageing_models(capacity_model_path, resistance_model_path):
    """
    Reload the ageing GP models from custom checkpoint paths.
    Call this before predict_ageing() if you want to use non-default models.

    Parameters
    ----------
    capacity_model_path   : str or Path
    resistance_model_path : str or Path
    """
    global _cap_model, _cap_lik, _R_model, _R_lik
    _cap_model, _cap_lik = _load_trained_gp(Path(capacity_model_path))
    _R_model,   _R_lik   = _load_trained_gp(Path(resistance_model_path))



# Public function used by main.py
def predict_ageing(cumulative_charge: float) -> dict:
    """
    Predict battery capacity and internal resistance from cumulative charge
    throughput.

    Parameters
    ----------
    cumulative_charge : float
        Total charge that has passed through the battery [Ah].

    Returns
    -------
    dict
        'capacity'       float  – predicted capacity [Ah]
        'capacity_std'   float  – 1-sigma uncertainty [Ah]
        'resistance'     float  – predicted internal resistance [Ω]
        'resistance_std' float  – 1-sigma uncertainty [Ω]
    """
    x_eval = torch.tensor(
        [[float(cumulative_charge)]], dtype=torch.float32, device=_device
    )

    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        cap_pred = _cap_lik(_cap_model(x_eval))
        R_pred   = _R_lik(_R_model(x_eval))

    return {
        "capacity":       cap_pred.mean.item(),
        "capacity_std":   cap_pred.stddev.item(),
        "resistance":     R_pred.mean.item(),
        "resistance_std": R_pred.stddev.item(),
    }



# Quick test  –  python ageing_model.py
if __name__ == "__main__":
    test_ah = 45.0  # example cumulative charge throughput
    result = predict_ageing(test_ah)

    print(f"\nInput cumulative charge : {test_ah:.2f} Ah")
    print(f"Capacity   : {result['capacity']:.4f} Ah  "
          f"(± {2 * result['capacity_std']:.4f} Ah, 95% CI)")
    print(f"Resistance : {result['resistance']:.6f} Ω  "
          f"(± {2 * result['resistance_std']:.6f} Ω, 95% CI)")