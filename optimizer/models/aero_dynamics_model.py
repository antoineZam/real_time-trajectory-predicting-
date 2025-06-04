import numpy as np


class AeroDynamicsModel:
    """
    Calculate aerodynamic forces and moments acting on a vehicle.
    """
    def __init__(self, aero_params: dict):
        """
        Initialize the AeroDynamicsModel with aerodynamic parameters.
        Args:
            aero_params (dict): Parameters like air_density (rho),
                                frontal_area (A), drag_coeff (CD),
                                lift_coeff_front (CL_front), lift_coeff_rear (CL_rear),
                                pitch_moment_coeff (Cm), etc.
                                May include sensitivities to ride height, roll, pitch.
        """
        self.rho = aero_params.get('air_density', 1.225) #kg.m^-3
        self.A = aero_params.get('frontal_area', 2.0) #m^2
        self.CD_static = aero_params.get('CD_static', 0.3)
        self.CL_static_total = aero_params.get('CL_static_total', -0.5)
        self.params = aero_params
        

    def calculate_forces_and_moments(self,
                                     body_velocity_mps: np.darray,
                                     angular_velocity_rps: np.darray,
                                     ride_height_front_m:float = 0.1,
                                     ride_height_rear_m:float = 0.1,
                                     chassis_roll_rad: float = 0.0,
                                     chassis_pitch_rad: float = 0.0
                                     ):
        """
        Calculates aerodynamic forces (Fx, Fy, Fz) and moments (Mx, My, Mz)
        acting on the vehicle body, in the vehicle body's coordinate system.

        Args:
            body_velocity_mps (np.ndarray): Vehicle's velocity vector [vx, vy, vz]
                                            in its own body coordinate system (m/s).
            angular_velocity_rps (np.ndarray): Vehicle's angular velocity vector
                                               [roll_rate, pitch_rate, yaw_rate] (rad/s).
            ride_height_front_m (float): Front ride height (m).
            ride_height_rear_m (float): Rear ride height (m).
            chassis_roll_rad (float): Chassis roll angle (rad).
            chassis_pitch_rad (float): Chassis pitch angle (rad).


        Returns:
            dict: {'Fx_aero': Fx_N, 'Fy_aero': Fy_N, 'Fz_aero': Fz_N,
                   'Mx_aero': Mx_Nm, 'My_aero': My_Nm, 'Mz_aero': Mz_Nm}
        """
        
        return {
            'Fx_aero': Fx_N, 'Fy_aero': Fy_N, 'Fz_aero': Fz_N,
                   'Mx_aero': Mx_Nm, 'My_aero': My_Nm, 'Mz_aero': Mz_Nm
        }