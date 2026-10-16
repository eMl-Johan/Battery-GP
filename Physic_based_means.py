
import torch
import gpytorch
import torch.nn as nn
from gpytorch.constraints import Interval






""" class TempMeanHyper(gpytorch.means.Mean):
    def __init__(self, shared_R, delta_t_hours=0.000275, T_amb=23.0):
        super().__init__()
        # Shared electrical resistance (same Parameter object as in VoltageMeanHyper)
        self.R = shared_R

        # Learnable thermal capacitance C_th (J/K) – we param in log-space to keep it positive
        self.log_Cth = nn.Parameter(torch.log(torch.tensor(100.0, dtype=torch.float32)))

        # Learnable thermal conductance K (W/K). Also positive via log-param.
        self.log_K = nn.Parameter(torch.log(torch.tensor(1.0, dtype=torch.float32)))

        self.delta_t = 1.0    # delta_t in seconds to keep it physically correct
        self.T_amb = T_amb

    def forward(self, x):
        current = x[:, 3]
        T_bat  = x[:, 2]

        C_th = self.log_Cth.exp()        # > 0
        K_phys = self.log_K.exp()        # > 0

        # Compute discrete-time coefficients from continuous-time model:
        # ΔT = (Δt/C_th) * R * I^2 - (Δt/C_th) * K_phys * (T - T_amb)
        beta = (self.delta_t / C_th) * self.R
        K_eff = (self.delta_t / C_th) * K_phys

        return beta * current**2 - K_eff * (T_bat - self.T_amb) """

class AmpHourMeanHyper(gpytorch.means.Mean):
    def __init__(self, delta_t_hours=0.000275):
        super().__init__()
        self.eta     = nn.Parameter(torch.tensor(0.95, dtype=torch.float32))
        self.C       = nn.Parameter(torch.tensor(2.5,  dtype=torch.float32))  # [Ah]
        self.delta_t = float(delta_t_hours)  # hours

    def forward(self, x):
        I   = x[:, 3]   # current [A]
        dSOC = -(self.eta * self.delta_t * I) / self.C   
        return dSOC

""" class VoltageMeanHyper(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()
        #self.R = 0.08 # ohmic resistance for a battery with roughly 14 amps of
        self.R = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(0.003, dtype=torch.float32))
        self.gamma = nn.Parameter(torch.tensor(0.2, dtype=torch.float32))
        self.V_ocv = nn.Parameter(torch.tensor(8.7, dtype=torch.float32))
        self.delta_t = 0.000275  # hours

    def forward(self, x):
        current = x[:, 3]
        SOC = x[:, 0].clamp(min=1e-6)
        V_now = x[:, 1]

        V_pred = self.V_ocv - self.R * current - self.alpha * (1 - SOC**self.gamma)
        return V_pred - V_now   # ΔV """


    
""" class VoltageMeanHyper(gpytorch.means.Mean):
    def __init__(self, R_init=0.08):
        super().__init__()
        # R is a *fixed* tensor by default (no gradient)
        # You can use nn.Parameter with requires_grad=False, or just a buffer.
        self.R = nn.Parameter(
            torch.tensor(R_init, dtype=torch.float32),
            requires_grad=False
        )

        self.alpha  = nn.Parameter(torch.tensor(0.003, dtype=torch.float32))
        self.gamma  = nn.Parameter(torch.tensor(0.2,   dtype=torch.float32))
        self.V_ocv  = nn.Parameter(torch.tensor(8.7,   dtype=torch.float32))
        self.delta_t = 0.000275  # hours

    def forward(self, x):
        current = x[:, 3]
        SOC     = x[:, 0].clamp(min=1e-6)
        V_now   = x[:, 1]

        # R is just used as-is, no constraints, no transform
        V_pred = self.V_ocv - self.R * current - self.alpha * (1 - SOC**self.gamma)
        return V_pred - V_now   # ΔV

class TempMeanHyper(gpytorch.means.Mean):
    def __init__(self, shared_R, delta_t_hours=0.000275, T_amb=23.0):
        super().__init__()
        # Shared electrical resistance: SAME tensor object as in VoltageMeanHyper
        self.R = shared_R

        # Learnable thermal capacitance C_th (J/K) – param in log-space to keep it positive
        self.log_Cth = nn.Parameter(torch.log(torch.tensor(100.0, dtype=torch.float32)))

        # Learnable thermal conductance K (W/K). Also positive via log-param.
        self.log_K = nn.Parameter(torch.log(torch.tensor(1.0, dtype=torch.float32)))

        self.delta_t = 1.0    # seconds
        self.T_amb = T_amb

    def forward(self, x):
        current = x[:, 3]
        T_bat   = x[:, 2]

        C_th   = self.log_Cth.exp()   # > 0
        K_phys = self.log_K.exp()     # > 0

        # Uses the SAME R tensor as voltage mean
        beta  = (self.delta_t / C_th) * self.R
        K_eff = (self.delta_t / C_th) * K_phys

        return beta * current**2 - K_eff * (T_bat - self.T_amb) """



""" class VoltageMeanHyper(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()

        # --- raw R, unconstrained optimization variable ---
        self.register_parameter(
            name="raw_R",
            parameter=nn.Parameter(torch.zeros(1))
        )
        self.register_constraint(
            "raw_R",
            Interval(0.07, 0.085)  # physical bounds for R
        )
        # initialize R to something in the interval
        initial_R = torch.tensor(0.078)
        self.raw_R.data = self.raw_R_constraint.inverse_transform(initial_R)

        # --- raw gamma, unconstrained optimization variable ---
        self.register_parameter(
            name="raw_gamma",
            parameter=nn.Parameter(torch.zeros(1))
        )
        self.register_constraint(
            "raw_gamma",
            Interval(0.2, 0.3)  # physical bounds for gamma
        )
        # initialize gamma to something in the interval
        initial_gamma = torch.tensor(0.25)
        self.raw_gamma.data = self.raw_gamma_constraint.inverse_transform(initial_gamma)

        # other unconstrained parameters
        self.alpha = nn.Parameter(torch.tensor(0.003, dtype=torch.float32))
        self.V_ocv = nn.Parameter(torch.tensor(8.7, dtype=torch.float32))
        self.delta_t = 0.000275  # hours

    # ---- constrained properties (what the model actually uses) ----
    @property
    def R(self):
        # actual R in [0.07, 0.085]
        return self.raw_R_constraint.transform(self.raw_R)

    @property
    def gamma(self):
        # actual gamma in [0.2, 0.3]
        return self.raw_gamma_constraint.transform(self.raw_gamma)

    def forward(self, x):
        current = x[:, 3]
        SOC = x[:, 0].clamp(min=1e-6)
        V_now = x[:, 1]

        R = self.R          # constrained R
        gamma = self.gamma  # constrained gamma

        V_pred = self.V_ocv - R * current - self.alpha * (1 - SOC**gamma)
        return V_pred - V_now   # ΔV
 """

class VoltageMeanHyper(gpytorch.means.Mean):
    def __init__(self):
        super().__init__()

        # --- raw R, unconstrained optimization variable ---
        self.register_parameter(
            name="raw_R",
            parameter=nn.Parameter(torch.zeros(1))
        )
        self.register_constraint(
            "raw_R",
            Interval(0.07, 0.085)  # physical bounds for R
        )
        initial_R = torch.tensor(0.078)
        self.raw_R.data = self.raw_R_constraint.inverse_transform(initial_R)

        # --- raw gamma, unconstrained optimization variable ---
        '''self.register_parameter(
            name="raw_gamma",
            parameter=nn.Parameter(torch.zeros(1))
        )
        self.register_constraint(
            "raw_gamma",
            Interval(0.2, 0.3)  # physical bounds for gamma
        )'''
        '''initial_gamma = torch.tensor(0.25)
        self.raw_gamma.data = self.raw_gamma_constraint.inverse_transform(initial_gamma)'''

        # other unconstrained parameters
        self.gamma = nn.Parameter(torch.tensor(0.3, dtype=torch.float32))
        self.alpha = nn.Parameter(torch.tensor(0.003, dtype=torch.float32))
        self.V_ocv = nn.Parameter(torch.tensor(8.7, dtype=torch.float32))
        self.delta_t = 0.000275  # hours

    # --------- R: constrained property + setter ---------
    @property
    def R(self):
        # actual R used in the model, in [0.07, 0.085]
        return self.raw_R_constraint.transform(self.raw_R)

    @R.setter
    def R(self, value):
        # value is in *physical* units, e.g. 0.08
        v = torch.as_tensor(value, dtype=self.raw_R.dtype, device=self.raw_R.device)
        # keep it inside the allowed interval (avoid inverse_transform errors)
        v = torch.clamp(v, 0.07, 0.1) #was 0.085
        self.raw_R.data.copy_(self.raw_R_constraint.inverse_transform(v))

    """# --------- gamma: constrained property + setter ---------
    @property
    def gamma(self):
        # actual gamma used in the model, in [0.2, 0.3]
        return self.raw_gamma_constraint.transform(self.raw_gamma)

    @gamma.setter
    def gamma(self, value):
        v = torch.as_tensor(value, dtype=self.raw_gamma.dtype, device=self.raw_gamma.device)
        v = torch.clamp(v, 0.2, 0.3)
        self.raw_gamma.data.copy_(self.raw_gamma_constraint.inverse_transform(v)) """

    def forward(self, x):
        current = x[:, 3]
        SOC = x[:, 0].clamp(min=1e-6)
        V_now = x[:, 1]

        R = self.R          # constrained to physical bounds
        gamma = self.gamma  # runs unconstrained optimization

        V_pred = self.V_ocv - R * current - self.alpha * (1 - SOC**gamma)
        return V_pred - V_now







class TempMeanHyper(gpytorch.means.Mean):
    def __init__(self, voltage_mean_module: VoltageMeanHyper,
                 delta_t_seconds: float = 1.0,
                 T_amb: float = 23.0):
        """
        Args:
            voltage_mean_module: instance of VoltageMeanHyper whose R we reuse.
            delta_t_seconds: time step in seconds.
            T_amb: ambient temperature in °C.
        """
        super().__init__()

        # Store reference to the voltage mean module so we can use its R
        self.voltage_mean_module = voltage_mean_module

        # Learnable thermal capacitance C_th (J/K), positive via log-param
        self.log_Cth = nn.Parameter(torch.log(torch.tensor(100.0, dtype=torch.float32)))

        # Learnable thermal conductance K (W/K), positive via log-param
        self.log_K = nn.Parameter(torch.log(torch.tensor(1.0, dtype=torch.float32)))

        self.delta_t = delta_t_seconds
        self.T_amb = T_amb

    def forward(self, x):
        current = x[:, 3]
        T_bat  = x[:, 2] 

        # positive parameters
        C_th = self.log_Cth.exp()
        K_phys = self.log_K.exp()

        # get the SAME constrained R that the voltage model uses
        R = self.voltage_mean_module.R

        # ΔT = (Δt/C_th) * R * I^2 - (Δt/C_th) * K_phys * (T - T_amb)
        beta = (self.delta_t / C_th) * R
        K_eff = (self.delta_t / C_th) * K_phys

        return beta * current**2 - K_eff * (T_bat - self.T_amb)
    


























    

