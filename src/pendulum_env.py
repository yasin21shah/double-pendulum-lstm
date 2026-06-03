import numpy as np
from scipy.integrate import solve_ivp

class DoublePendulum:
    
    # initialise the physical constants
    def __init__(self, L1=1.0, L2=1.0, m1=3.0, m2=1.0, g=9.81, dt=0.1):
        self.L1 = L1
        self.L2 = L2
        self.m1 = m1
        self.m2 = m2
        self.g = g
        self.dt = dt

    # calculate the time derivatives for the system
    def _derivatives(self, t, z):
        theta1, w1, theta2, w2 = z
        cos12 = np.cos(theta1 - theta2)
        sin12 = np.sin(theta1 - theta2)
        sin1 = np.sin(theta1)
        sin2 = np.sin(theta2)
        
        xi = cos12**2 * self.m2 - self.m1 - self.m2
        
        w1dot = (self.L1 * self.m2 * cos12 * sin12 * w1**2 + self.L2 * self.m2 * sin12 * w2**2 
                 - self.m2 * self.g * cos12 * sin2 + (self.m1 + self.m2) * self.g * sin1) / (self.L1 * xi)
        
        w2dot = -(self.L2 * self.m2 * cos12 * sin12 * w2**2 + self.L1 * (self.m1 + self.m2) * sin12 * w1**2 
                  + (self.m1 + self.m2) * self.g * sin1 * cos12 - (self.m1 + self.m2) * self.g * sin2) / (self.L2 * xi)
        
        return w1, w1dot, w2, w2dot

    # transform polar states to cartesian coordinates
    def _to_cartesian(self, z_states):
        theta1, w1, theta2, w2 = z_states
        x1 = self.L1 * np.sin(theta1)
        y1 = -self.L1 * np.cos(theta1)
        x2 = x1 + self.L2 * np.sin(theta2)
        y2 = y1 - self.L2 * np.cos(theta2)
        return np.column_stack((x1, y1, x2, y2))

    # generate the pendulum data over time
    def generate_data(self, z_init, t_max):
        t_eval = np.arange(0, t_max + self.dt, self.dt)
        
        # solve differential equations
        ret = solve_ivp(
            fun=self._derivatives, 
            t_span=(0, t_max), 
            y0=z_init, 
            t_eval=t_eval
        )
        
        # return the formatted coordinates
        cartesian_coords = self._to_cartesian(ret.y)
        return cartesian_coords, t_eval