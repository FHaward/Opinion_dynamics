import numpy as np
import matplotlib.pyplot as plt



omega_m0 = 0.1
omega_r0 = 10**-4
omega_v0 = 0.9

def omega_m(omega_m0, omega_r0, omega_v0, a):
    
    return omega_m0/(omega_m0+((1/a)*omega_r0)+(omega_v0*a**3))

def omega_r(omega_m0, omega_r0, omega_v0, a):
    
    return (omega_r0/a)/(omega_m0+((1/a)*omega_r0)+(omega_v0*a**3))

def omega_v(omega_m0, omega_r0, omega_v0, a):
    
    return (omega_v0*a**3)/(omega_m0+((1/a)*omega_r0)+(omega_v0*a**3))

# Generate values for a using a logarithmic scale
a_values = np.logspace(-5, 1, 500)  # Range from 10^-5 to 10

# Calculate omega_m for each value of a
omega_m_values = [omega_m(omega_m0, omega_r0, omega_v0, a) for a in a_values]
omega_r_values = [omega_r(omega_m0, omega_r0, omega_v0, a) for a in a_values]
omega_v_values = [omega_v(omega_m0, omega_r0, omega_v0, a) for a in a_values]

# Plot
plt.plot(a_values, omega_m_values, label=r'$\Omega_m(a)$')
plt.plot(a_values, omega_r_values, label=r'$\Omega_r(a)$')
plt.plot(a_values, omega_v_values, label=r'$\Omega_v(a)$')
plt.xscale('log')  # Set the x-axis to a logarithmic scale
plt.xlabel('Scale factor (a)')
plt.ylabel(r'$\Omega$')
plt.title(r'Plot of $\Omega$ as a function of scale factor $a$')
plt.legend()
plt.grid(True)
plt.show()


