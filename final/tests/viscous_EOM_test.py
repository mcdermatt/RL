import numpy as np
import cloudpickle
from scipy.integrate import odeint

rhs = cloudpickle.load(open("full_EOM_func_VISCOUS_DAMPING.txt", 'rb'))

numerical_constants = np.array([0.05,  # j0_length [m]
                             0.01,  # j0_com_length [m]
                             4.20,  # j0_mass [kg]
                             0.001,  # NOT USED j0_inertia [kg*m^2]
                             0.164,  # j1_length [m]
                             0.08,  # j1_com_length [m]
                             1.81,  # j1_mass [kg]
                             0.001,  # NOT USED j1_inertia [kg*m^2]
                             0.158,  # j2_com_length [m]
                             2.259,  # j2_mass [kg]
                             0.001,  # NOT USED j2_inertia [kg*m^2]
                             9.81, # acceleration due to gravity [m/s^2]
                             -0.15, #j0 damp
                             -0.15, #j1 damp
                             -0.15,], #j2 damp
                            ) 

numerical_specified = np.zeros(3)
args = {'constants': numerical_constants,
        'specified': numerical_specified}
	
x0 = np.zeros(6)
x0[1] = 120
t = np.linspace(0.0,1,10) #(0, end, steps per second)
y = odeint(rhs, x0, t, args=(numerical_specified, numerical_constants))

print(y)