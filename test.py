import numpy as np

def calculate_energy_change_test(J_b, h_b, J_s, spin, zealot_spin, magnetization): 

    # Calculate energy change ΔE
    social_influence = -J_b*(magnetization-spin)*spin
    internal_field = -h_b*spin
    leader_influence= -J_s*zealot_spin*spin
    
    return -2*(social_influence+internal_field+leader_influence)

def safe_exp(delta_E, temp, k_B):
    """
    Compute exp(-ΔE/kT) for large energy changes without try/except.
    Uses direct value checks to prevent overflow/underflow.
    """
    beta_delta_E = -delta_E / (k_B * temp)
    
    # For large negative values, return exp(700) as upper limit
    if beta_delta_E > 700:  # exp(700) is near the maximum float64 can handle
        return np.exp(700)
    # For large positive values, return exp(-700) as lower limit
    elif beta_delta_E < -700:
        return np.exp(-700)
    # For manageable values, compute normally
    else:
        return np.exp(beta_delta_E)
    
L=100
N=L**2
J_b = 1/(N-1)
J_s = 1.01
h_b= -1
h_s = N
k_B = 1
temp = 0.9285714   #0.84286   
m = -0.45
M = N*m
spin = -1
zealot_spin = 1

delta_E = calculate_energy_change_test(J_b, h_b, J_s, spin, zealot_spin, M)
exp = safe_exp(delta_E, temp, k_B)

print(delta_E, exp)


#magnetisation where 50% change to flip or stick from the bottom path for a spin down given that the leader is spin up 
((((np.log(2)*temp/-2)-(J_s+h_b))*9999)-1)/10001