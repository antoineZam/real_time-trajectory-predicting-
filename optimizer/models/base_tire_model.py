from abc import ABC, abstractmethod
import numpy as np


class BaseTireModel(ABC):
    """
    Abstract base class for tire force calculation models.
    """
    
    def __init__(self, tire_params: dict):
        self.params = tire_params
        super().__init__()
    
    @abstractmethod
    def calculate_forces_and_moments(self,
                                     slip_angle_rad: float,
                                     slip_ratio: float, 
                                     vertical_load: float,
                                     chamber_angle_rad: float, 
                                     current_road_friction: float = 1.0) -> dict:
        """
        Calculates the longitudinal force (Fx), lateral force (Fy),
        and aligning moment (Mz) for a single tire.
        
        Args:
            slip_angle_rad (float): Tire slip angle in radians.
            slip_ratio (float): Tire slip ratio.
            vertical_load_N (float): Vertical load on the tire in Newtons.
            camber_angle_rad (float): Camber angle in radians.
            current_road_friction (float): Effective friction coefficient of the road surface.

        Returns:
            dict: A dictionary containing {'Fx': Fx_N, 'Fy': Fy_N, 'Mz': Mz_Nm}.
                  Forces are typically in the tire's coordinate system.
        """
        pass
    
class PaceJkaTireModel(BaseTireModel):
    """
    Implements the Pacejka Magic Formula tire model.
    This version will focus on pure slip conditions.
    Coefficients are typically prefixed:
    - 'p' for general parameters (e.g., p_cx1)
    - 'b' or 'P' often for longitudinal force (Fx)
    - 'a' or 'P' often for lateral force (Fy)
    - 'c' or 'P' or 'Q' often for aligning moment (Mz)
    - 'S_v' for vertical shift, 'S_h' for horizontal shift

    We will need to parse these from the self.params dictionary.
    The structure of self.params should match a typical .tir file parameter naming.
    """
    
    def __init__(self, tire_params: dict):
        super().__init__(tire_params)
        
        # Nominal load and reference radius
        self.Fz0_nominal = self.params.get('FNOMIN', self.params.get('FZO', 4000.0)) # Nominal load (N) [cite: 1319] (FNOMIN or FZO)
        self.R0 = self.params.get('UNLOADED_RADIUS', self.params.get('RO', 0.3))    # Unloaded tire radius (m) [cite: 874, 1319]

        # Small constant for preventing division by zero
        self.epsilon = self.params.get('EPSILON', 1e-6) # For Bx, By, K_yalpha_N
        self.epsilon_V = self.params.get('EPSILON_V', 0.1) # For Vcx, Vc in denominators (gV in book, pg 185 [cite: 875])
        self.epsilon_low_speed = self.params.get('EPSILON_LOW_SPEED', 0.1) # For PVx1 term (gVx in book, pg 187 [cite: 877])

    
    def _magic_formula_std(self, X_input_shifted, B, C, D, E):
        """The standard Magic Formula: Y(x) = D * sin[C * arctan{B*x - E*(B*x - arctan(B*x))}]"""
        # x = X_input_shifted (already includes Sh)
        # This is y(x) from Eq. (4.49) [cite: 863]
        Bx_shifted = B * X_input_shifted
        arctan_Bx_shifted = np.arctan(Bx_shifted)
        return D * np.sin(C * np.arctan(Bx_shifted - E * (Bx_shifted - arctan_Bx_shifted)))

    def _magic_formula_cos(self, X_input_shifted, B, C, D, E):
        """Cosine version for trail and weighting functions: G = D * cos[C * arctan{B*x - E*(B*x - arctan(B*x))}]"""
        # This is for G(x) = D cos[C arctan(B x)] from Eq. (4.64) [cite: 869]
        # or t(alpha_t) = Dt cos[Ct arctan{Bt*alpha_t - Et*(Bt*alpha_t - arctan(Bt*alpha_t))}] from Eq. (4.E33) [cite: 878]
        Bx_shifted = B * X_input_shifted
        arctan_Bx_shifted = np.arctan(Bx_shifted)
        return D * np.cos(C * np.arctan(Bx_shifted - E * (Bx_shifted - arctan_Bx_shifted)))
    def calculate_forces_and_moments(self,
                                     slip_angle_rad: float,    # alpha
                                     slip_ratio: float,        # kappa
                                     vertical_load_N: float,   # Fz
                                     camber_angle_rad: float,  # gamma
                                     longitudinal_velocity_mps: float, # Vcx
                                     lateral_velocity_mps: float,    # Vcy
                                     current_road_friction: float = 1.0 # Overall friction multiplier
                                     ) -> dict:

        Fz_eff = max(vertical_load_N, self.epsilon * 100) # Ensure Fz is positive and non-zero

        # --- Common Basic Parameters ---
        # Eq. (4.E1) F'zo = lambda_Fz0 * Fz0 [cite: 874]
        lambda_Fzo = self.params.get('LFZO', self.params.get('LAMBDA_FZ0', 1.0)) # [cite: 876] (Name from App 3.1 is LFZO)
        F_prime_z0 = lambda_Fzo * self.Fz0_nominal

        # Eq. (4.E2) dfz = (Fz - F'z0) / F'z0 [cite: 874]
        dfz = (Fz_eff - F_prime_z0) / (F_prime_z0 + self.epsilon)


        # Eq. (4.E3) alpha_star = tan(alpha) * sgn(Vcx) [cite: 875]
        # Using alpha directly as input is common for MF if alpha is defined as effective slip angle
        # The book implies alpha here is the wheel slip angle.
        # Vcx needs to be handled for low speeds or sign changes
        Vcx_eff = longitudinal_velocity_mps
        if abs(Vcx_eff) < self.epsilon_V: # Avoid instability at zero/very low speed
            alpha_prime = slip_angle_rad # Use raw slip angle
        else:
            alpha_prime = np.tan(slip_angle_rad) * np.sign(Vcx_eff)

        # Eq. (4.E4) gamma_star = sin(gamma) [cite: 875]
        gamma_prime = np.sin(camber_angle_rad)

        # Eq. (4.E5) kappa_eff = kappa (input slip ratio) [cite: 875]
        kappa_eff = slip_ratio

        # Eq. (4.E6) cos_prime_alpha for aligning moment [cite: 875]
        Vc = np.sqrt(longitudinal_velocity_mps**2 + lateral_velocity_mps**2) # [cite: 875]
        cos_prime_N_alpha = longitudinal_velocity_mps / (Vc + self.epsilon_V)

        # Scaling factors for friction (Eq. 4.E7, 4.E8 [cite: 876])
        # Slip speed Vs (approx for pure slip, more complex for combined)
        # For friction scaling, Vs is the magnitude of slip velocity at contact.
        # Simplified Vs for scaling:
        Vs_approx_x = abs(longitudinal_velocity_mps * kappa_eff)
        Vs_approx_y = abs(longitudinal_velocity_mps * np.tan(slip_angle_rad)) # Using Vcx for Fy related slip speed
        Vs_for_mux = np.sqrt(Vs_approx_x**2 + (lateral_velocity_mps)**2) # Approximation based on components
        Vs_for_muy = np.sqrt(longitudinal_velocity_mps**2 + Vs_approx_y**2) # Approximation

        lambda_muV = self.params.get('LMUV', self.params.get('LAMBDA_MUV', 0.0)) # [cite: 876]
        V0_ref = self.params.get('LONGVL', self.params.get('VO', 16.67)) # Reference speed [cite: 874, 1319]

        # Eq (4.E7) lambda_prime_mux,y [cite: 876]
        lambda_prime_mux = self.params.get('LMUX', self.params.get('LAMBDA_MUX', 1.0)) / (1 + lambda_muV * Vs_for_mux / (V0_ref + self.epsilon))
        lambda_prime_muy = self.params.get('LMUY', self.params.get('LAMBDA_MUY', 1.0)) / (1 + lambda_muV * Vs_for_muy / (V0_ref + self.epsilon))
        
        # Apply overall road friction multiplier
        lambda_prime_mux *= current_road_friction
        lambda_prime_muy *= current_road_friction

        # Eq (4.E8) lambda_double_prime_mux,y [cite: 876] (for vertical shifts)
        A_mu = 10.0 # Suggested value in eq [cite: 876]
        lambda_double_prime_mux = A_mu * lambda_prime_mux / (1 + (A_mu - 1) * lambda_prime_mux + self.epsilon)
        lambda_double_prime_muy = A_mu * lambda_prime_muy / (1 + (A_mu - 1) * lambda_prime_muy + self.epsilon)

        # Zeta factors (turn slip influence) - for pure slip, these are 1 [cite: 875]
        zeta1 = self.params.get('FZ1', 1.0) # Name from App 3.1 not in text; default to 1 for pure slip. Usually ζ (zeta)
        zeta2 = self.params.get('FZ2', 1.0) # These are more relevant for the extended model (Section 4.3.3)
        zeta3 = self.params.get('FZ3', 1.0)
        zeta4 = self.params.get('FZ4', 1.0)
        zeta5 = self.params.get('FZ5', 1.0)
        zeta6 = self.params.get('FZ6', 1.0)
        zeta7 = self.params.get('FZ7', 1.0)
        zeta8 = self.params.get('FZ8', 1.0)
        zeta0 = self.params.get('FZ0', 1.0) # For SHy camber part

        # --- Pure Longitudinal Force Fx0 (Eq. 4.E9 - 4.E18) [cite: 877] ---
        SHx = (self.params.get('PHX1',0.0) + self.params.get('PHX2',0.0) * dfz) * self.params.get('LHX',1.0)
        kappa_x = kappa_eff + SHx
        
        Cx = self.params.get('PCX1',1.5) * self.params.get('LCX',1.0)
        
        mux = (self.params.get('PDX1',1.0) + self.params.get('PDX2',0.0) * dfz) * lambda_prime_mux
        Dx = mux * Fz_eff * zeta1
        
        Kxk = Fz_eff * (self.params.get('PKX1',20.0) + self.params.get('PKX2',0.0) * dfz) * \
              np.exp(self.params.get('PKX3',0.0) * dfz) * self.params.get('LKX',1.0)
              
        Ex_sign_term = 1.0 - self.params.get('PEX4',0.0) * np.sign(kappa_x) if 'PEX4' in self.params else 1.0
        Ex = (self.params.get('PEX1',0.0) + self.params.get('PEX2',0.0) * dfz + self.params.get('PEX3',0.0) * dfz**2) * \
             Ex_sign_term * self.params.get('LEX',1.0)
        Ex = np.clip(Ex, -float('inf'), 1.0) # Constraint Ex <= 1 [cite: 877]

        Bx = Kxk / (Cx * Dx + self.epsilon)
        
        # Using epsilon_low_speed for gVx from book for SVx denominator [cite: 877]
        SVx_Vcx_factor = abs(longitudinal_velocity_mps) / (self.epsilon_low_speed + abs(longitudinal_velocity_mps))
        SVx = Fz_eff * (self.params.get('PVX1',0.0) + self.params.get('PVX2',0.0) * dfz) * \
              SVx_Vcx_factor * self.params.get('LVX',1.0) * lambda_double_prime_mux * zeta1

        Fx0 = self._magic_formula_std(kappa_x, Bx, Cx, Dx, Ex) + SVx

        # --- Pure Lateral Force Fy0 (Eq. 4.E19 - 4.E30) ---
        Kygamma0 = Fz_eff * (self.params.get('PKY6',0.0) + self.params.get('PKY7',0.0) * dfz) * self.params.get('LKYG', self.params.get('LKYC',1.0)) # LKYG or LKYC for camber stiffness factor
        SVygamma = Fz_eff * (self.params.get('PVY3',0.0) + self.params.get('PVY4',0.0) * dfz) * \
                     gamma_prime * self.params.get('LVYG', self.params.get('LKYG',1.0)) * lambda_double_prime_muy * zeta2 # LVYG or LKYG? LKYG in book eq [cite: 878]

        Kyalpha_term_Fz = Fz_eff / ((self.params.get('PKY2',1.0) + self.params.get('PKY5',0.0) * gamma_prime**2)* F_prime_z0 + self.epsilon)
        Kyalpha = self.params.get('PKY1',-20.0) * F_prime_z0 * \
                  np.sin(self.params.get('PKY4',2.0) * np.arctan(Kyalpha_term_Fz)) * \
                  (1.0 - self.params.get('PKY3',0.0) * abs(gamma_prime)) * zeta3 * self.params.get('LKY',1.0)

        SHy_K_factor = Kygamma0 * gamma_prime - SVygamma # Numerator for the camber-induced shift term
        SHy_denom_factor = Kyalpha + self.epsilon # Denominator for camber-induced shift (K_yalphaN in book)
        SHy = (self.params.get('PHY1',0.0) + self.params.get('PHY2',0.0) * dfz) * self.params.get('LHY',1.0) + \
              SHy_K_factor * zeta0 / SHy_denom_factor + zeta4 - 1 # zeta0 for camber part of SHy, zeta4 for turnslip part [cite: 878]


        alpha_y = alpha_prime + SHy
        
        Cy = self.params.get('PCY1',1.3) * self.params.get('LCY',1.0)
        
        muy = (self.params.get('PDY1',1.0) + self.params.get('PDY2',0.0) * dfz) * \
              (1.0 - self.params.get('PDY3',0.0) * gamma_prime**2) * lambda_prime_muy
        Dy = muy * Fz_eff * zeta2
        
        Ey_sign_term = np.sign(alpha_y) if 'PEY3' in self.params or 'PEY4' in self.params else 1.0 # only if PEY3/4 are used
        Ey = (self.params.get('PEY1',-1.0) + self.params.get('PEY2',0.0) * dfz) * \
             (1.0 + self.params.get('PEY5',0.0) * gamma_prime**2 - (self.params.get('PEY3',0.0) + self.params.get('PEY4',0.0) * gamma_prime) * Ey_sign_term) * \
             self.params.get('LEY',1.0)
        Ey = np.clip(Ey, -float('inf'), 1.0) # Constraint Ey <= 1 [cite: 877]

        By = Kyalpha / (Cy * Dy + self.epsilon)
        
        SVy = Fz_eff * (self.params.get('PVY1',0.0) + self.params.get('PVY2',0.0) * dfz) * \
              self.params.get('LVY',1.0) * lambda_double_prime_muy * zeta2 + SVygamma

        Fy0_raw = self._magic_formula_std(alpha_y, By, Cy, Dy, Ey)
        Fy0 = Fy0_raw + SVy


        # --- Pure Aligning Moment Mz0 (Mz_prime in some contexts) (Eq. 4.E31 - 4.E49) ---
        # Trail t0 = t(alpha_t)
        SHt = (self.params.get('QHZ1',0.0) + self.params.get('QHZ2',0.0) * dfz) + \
              (self.params.get('QHZ3',0.0) + self.params.get('QHZ4',0.0) * dfz) * gamma_prime
        alpha_t_shifted = alpha_prime + SHt
        
        Bt = (self.params.get('QBZ1',5.0) + self.params.get('QBZ2',0.0) * dfz + self.params.get('QBZ3',0.0) * dfz**2) * \
             (1.0 + self.params.get('QBZ5',0.0) * abs(gamma_prime) + self.params.get('QBZ6',0.0) * gamma_prime**2) * \
             (self.params.get('LKY',1.0) / (lambda_prime_muy + self.epsilon)) * self.params.get('LKZ',1.0) # LKZ from App 3.1, not in Eq. 4.E40 [cite: 878]
        
        Ct = self.params.get('QCZ1',1.2) # Shape factor for trail [cite: 878]
        
        Dt0 = Fz_eff * (self.R0 / (F_prime_z0 + self.epsilon)) * \
              (self.params.get('QDZ1',0.1) + self.params.get('QDZ2',0.0) * dfz) * \
              self.params.get('LT', self.params.get('LAMBDA_T',1.0)) * np.sign(Vcx_eff) # LT instead of λt [cite: 878]
        
        Dt = Dt0 * (1.0 + self.params.get('QDZ3',0.0) * abs(gamma_prime) + self.params.get('QDZ4',0.0) * gamma_prime**2) * zeta5
        
        Et_arctan_term = np.arctan(Bt * Ct * alpha_t_shifted)
        Et = (self.params.get('QEZ1',-1.5) + self.params.get('QEZ2',0.0) * dfz + self.params.get('QEZ3',0.0) * dfz**2) * \
             (1.0 + (self.params.get('QEZ4',0.0) + self.params.get('QEZ5',0.0) * gamma_prime) * (2/np.pi) * Et_arctan_term)
        Et = np.clip(Et, -float('inf'), 1.0) # Constraint Et <= 1 [cite: 878]

        t0_trail = self._magic_formula_cos(alpha_t_shifted, Bt, Ct, Dt, Et) * cos_prime_N_alpha # Pneumatic trail t0 [cite: 878]

        # Residual Torque Mzr0 = Mzr(alpha_r)
        # SHf = SHy (alpha offset part) + SVy (Fy offset) / KyalphaN (effective cornering stiffness)
        # K_yalphaN (Kyalpha_prime in book text) = K_yalpha + epsilon_K
        SHf = SHy + SVy / (Kyalpha + self.epsilon) # Eq. 4.E38 [cite: 878]
        alpha_r_shifted = alpha_prime + SHf # Eq. 4.E37 [cite: 878]
        
        # Br for Mzr, simplified if qBz9 preferred to be 0
        if self.params.get('QBZ9',0.0) == 0: # Preferred form in book text for Br [cite: 878]
             Br = self.params.get('QBZ10',0.0) * By * Cy * zeta6
        else: # Full form Eq. 4.E45 [cite: 878]
             Br = (self.params.get('QBZ9',0.0) * (self.params.get('LKY',1.0) / (lambda_prime_muy + self.epsilon)) + \
                   self.params.get('QBZ10',0.0) * By * Cy) * zeta6


        Cr = zeta7 # Eq. 4.E46 [cite: 878]
        
        Dr_term_zeta2 = (self.params.get('QDZ6',0.0) + self.params.get('QDZ7',0.0) * dfz) * self.params.get('LMR',1.0) * zeta2
        Dr_term_zeta0_gamma = (self.params.get('QDZ8',0.0) + self.params.get('QDZ9',0.0) * dfz) * gamma_prime * \
                              self.params.get('LKZG',self.params.get('LKZC',1.0)) * zeta0 # LKZG or LKZC for Kzgamma factor
        Dr_term_zeta0_gamma_abs_gamma = (self.params.get('QDZ10',0.0) + self.params.get('QDZ11',0.0) * dfz) * \
                                       gamma_prime * abs(gamma_prime) * zeta0
        Dr_base = Fz_eff * self.R0 * (Dr_term_zeta2 + Dr_term_zeta0_gamma + Dr_term_zeta0_gamma_abs_gamma)

        Dr = Dr_base * cos_prime_N_alpha * lambda_prime_muy * np.sign(Vcx_eff) + (zeta8 -1) # (zeta8-1) is an additional turn slip term [cite: 879]
        # Note: The formula for Dr (Eq. 4.E47) in the book has lambda_muy, but residual torque is often independent of road mu.
        # Some implementations use lambda_double_prime_muy or 1.0. We follow the book here.

        Mzr0_residual = self._magic_formula_cos(alpha_r_shifted, Br, Cr, Dr, 0.0) # Et for Mzr is typically 0 or not present in this simple cos form

        # Total Mz0 (pure slip)
        # Fy0_for_Mz_calc refers to the lateral force without the total SVy vertical shift,
        # but it does include the Svy_gamma part related to camber thrust.
        # This is complex as Fyo already has SVy. The standard is -t*Fy + Mzr.
        # Fy0_raw = Fy0 - SVy (force component before vertical shift)
        Mz0 = -t0_trail * (Fy0 - SVy) + Mzr0_residual # Eq. 4.E31, 4.E32 using the part of Fy0 not from SVy total.
                                                     # The book defines Mz0_N = -t0 * Fy0, where Fy0 is the total pure lateral force.
                                                     # This interpretation: Mz0 = -t0_trail * Fy0 + Mzr0_residual is common.


        # --- Combined Slip Calculation (Eq. 4.E50 - 4.E78) ---
        # Weighting function Gxalpha for Fx
        SHxalpha = self.params.get('RHX1',0.0) # Horizontal shift for Gx_alpha [cite: 879] (rHx1)
        alpha_S_gx = alpha_prime + SHxalpha # Eq. 4.E53
        
        Bxalpha = (self.params.get('RBX1',5.0) + self.params.get('RBX3',0.0) * gamma_prime**2) * \
                  np.cos(np.arctan(self.params.get('RBX2',8.0) * kappa_eff)) * self.params.get('LXA',1.0) # LXA = lambda_xa [cite: 879]
        Cxalpha = self.params.get('RCX1',1.0) # Shape for Gx_alpha [cite: 879]
        Exalpha = (self.params.get('REX1',0.0) + self.params.get('REX2',0.0) * dfz) # Curvature for Gx_alpha [cite: 879]
        Exalpha = np.clip(Exalpha, -float('inf'), 1.0)

        Gxalpha_nom = self._magic_formula_cos(alpha_S_gx, Bxalpha, Cxalpha, 1.0, Exalpha) # D=1 for weighting
        Gxalpha_den = self._magic_formula_cos(SHxalpha, Bxalpha, Cxalpha, 1.0, Exalpha) # Eq. 4.E52
        Gxalpha = Gxalpha_nom / (Gxalpha_den + self.epsilon) # Eq. 4.E51
        Gxalpha = np.clip(Gxalpha, 0.0, float('inf')) # G must be > 0

        Fx_combined = Gxalpha * Fx0 # Eq. 4.E50

        # Weighting function Gykappa for Fy
        SHykappa = (self.params.get('RHY1',0.0) + self.params.get('RHY2',0.0) * dfz) # Horizontal shift for Gy_kappa [cite: 879] (rHy1, rHy2)
        kappa_S_gy = kappa_eff + SHykappa # Eq. 4.E61
        
        Bykappa = (self.params.get('RBY1',7.0) + self.params.get('RBY4',0.0) * gamma_prime**2) * \
                  np.cos(np.arctan(self.params.get('RBY2',5.0) * (alpha_prime - self.params.get('RBY3',0.0)))) * \
                  self.params.get('LYK',1.0) # LYK = lambda_yk [cite: 879]
        Cykappa = self.params.get('RCY1',1.0) # Shape for Gy_kappa [cite: 879]
        Eykappa = (self.params.get('REY1',0.0) + self.params.get('REY2',0.0) * dfz) # Curvature for Gy_kappa [cite: 879]
        Eykappa = np.clip(Eykappa, -float('inf'), 1.0)

        Gykappa_nom = self._magic_formula_cos(kappa_S_gy, Bykappa, Cykappa, 1.0, Eykappa)
        Gykappa_den = self._magic_formula_cos(SHykappa, Bykappa, Cykappa, 1.0, Eykappa) # Eq. 4.E60
        Gykappa = Gykappa_nom / (Gykappa_den + self.epsilon) # Eq. 4.E59
        Gykappa = np.clip(Gykappa, 0.0, float('inf'))

        # Vertical shift SVykappa (kappa-induced "ply-steer")
        DVykappa_cos_term = np.cos(np.arctan(self.params.get('RVY4',0.0) * alpha_prime)) # RVY4 not in App3.1, use rVy4 [cite: 879]
        DVykappa = muy * Fz_eff * (self.params.get('RVY1',0.0) + self.params.get('RVY2',0.0) * dfz + self.params.get('RVY3',0.0) * gamma_prime) * \
                     DVykappa_cos_term * zeta2 # Eq. 4.E67
        SVykappa = DVykappa * np.sin(self.params.get('RVY5',0.0) * np.arctan(self.params.get('RVY6',10.0) * kappa_eff)) * \
                     self.params.get('LVYK',1.0) # LVYK = lambda_Vyk [cite: 879] (Eq. 4.E66)

        Fy_combined = Gykappa * Fy0 + SVykappa # Eq. 4.E58

        # Aligning Moment Mz_combined (Eq. 4.E71 - 4.E78) [cite: 880]
        # Equivalent slip angle for trail under combined slip
        alpha_t_eq_sq_term = (Kxk / (Kyalpha + self.epsilon))**2 * kappa_eff**2
        alpha_t_eq = np.sqrt(alpha_t_shifted**2 + alpha_t_eq_sq_term) * np.sign(alpha_t_shifted) # Eq. 4.E77

        # Equivalent slip angle for residual moment under combined slip
        alpha_r_eq_sq_term = (Kxk / (Kyalpha + self.epsilon))**2 * kappa_eff**2
        alpha_r_eq = np.sqrt(alpha_r_shifted**2 + alpha_r_eq_sq_term) * np.sign(alpha_r_shifted) # Eq. 4.E78

        # Trail 't' under combined slip
        t_combined = self._magic_formula_cos(alpha_t_eq, Bt, Ct, Dt, Et) * cos_prime_N_alpha # Eq. 4.E73

        # Residual moment Mzr under combined slip
        Mzr_combined = self._magic_formula_cos(alpha_r_eq, Br, Cr, Dr, 0.0) # Eq. 4.E75 (Et for Mzr is 0)

        # Moment arm 's' for Fx contribution to Mz
        s_arm = self.R0 * (self.params.get('SSZ1',0.0) + self.params.get('SSZ2',0.0) * (Fy_combined / (F_prime_z0 +self.epsilon)) + \
                      (self.params.get('SSZ3',0.0) + self.params.get('SSZ4',0.0) * dfz) * gamma_prime) * \
                      self.params.get('LS',1.0) # LS = lambda_s [cite: 880] (Eq. 4.E76)

        Fy_prime_combined = Fy_combined - SVykappa # Force component used for trail part of Mz (Eq. 4.E74)

        Mz_combined = -t_combined * Fy_prime_combined + Mzr_combined + s_arm * Fx_combined # Eq. 4.E71


        return {
            'Fx': Fx_combined,
            'Fy': Fy_combined,
            'Mz': Mz_combined
        }