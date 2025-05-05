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
m = -0.3493943852663016
M = N*m
spin = -1
zealot_spin = 1

delta_E = calculate_energy_change_test(J_b, h_b, J_s, spin, zealot_spin, M)
exp = safe_exp(delta_E, temp, k_B)

print(delta_E, exp)


#magnetisation where 50% change to flip or stick from the bottom path for a spin down given that the leader is spin up 
((((np.log(2)*temp/-2)-(J_s+h_b))*9999)-1)/10001


(temp*np.log(2)/-2)-(J_s+h_b)-1/N
(-2*((J_s+h_b)+m+1/N))/(k_B*np.log(2))


-2*(J_s+h_b+(m*N+1)/(N-1))/(k_B*np.log(2))


import math
import numpy as np
from scipy import stats

def binomial_pmf(k, n, p):
    """
    Calculate the binomial probability mass function for P(X = k)
    
    Parameters:
    - k: number of successes
    - n: number of trials
    - p: probability of success in a single trial
    
    Returns:
    - Probability of exactly k successes in n trials
    """
    # Calculate binomial coefficient (n choose k)
    binomial_coef = math.comb(n, k)
    
    # Calculate probability
    probability = binomial_coef * (p ** k) * ((1 - p) ** (n - k))
    
    return probability

def binomial_cdf(k, n, p):
    """
    Calculate the binomial cumulative distribution function for P(X ≤ k)
    
    Parameters:
    - k: number of successes
    - n: number of trials
    - p: probability of success in a single trial
    
    Returns:
    - Probability of k or fewer successes in n trials
    """
    cumulative_prob = 0
    for i in range(k + 1):
        cumulative_prob += binomial_pmf(i, n, p)
    
    return cumulative_prob

def binomial_cdf_using_scipy(k, n, p):
    """
    Calculate the binomial CDF using SciPy's built-in function
    """
    return stats.binom.cdf(k, n, p)

def binomial_cdf_at_least_scipy(k, n, p):
    """
    Calculate P(X ≥ k) using SciPy's survival function
    """
    return stats.binom.sf(k - 1, n, p)  # sf(k-1) = P(X > k-1) = P(X ≥ k)

# Problem: Calculate P(X = 5050) for 10000 coin tosses
n = 10000     # Number of trials
k = 4950      # Number of successes
p = 0.51       # Probability of heads
p_list = [0.51, 0.5, 0.495, 0.49, 0.485, 0.48]

# Using SciPy for validation
scipy_cumulative = binomial_cdf_using_scipy(k, n, p)
mean = n * p
std_dev = math.sqrt(n * p * (1 - p))
normal_approx = stats.norm.cdf((k - 0.5 - mean) / std_dev)

print(f"Probability of {k} or fewer spin up using SciPy: {scipy_cumulative:.10f}")
print(f"Normal approximation to P(X ≤ {k}): {normal_approx:.10f}")

p_list = [0.51, 0.5, 0.495, 0.49, 0.485, 0.48]
ratios =[]
probs =[]
for p in p_list:
    scipy_cumulative = binomial_cdf_using_scipy(k, n, p)
    probs.append(np.round(scipy_cumulative,3))
    predicted_ratio = 0.5-(0.5*scipy_cumulative)
    ratios.append(np.round(predicted_ratio,3))
    
ratios
probs

J_s = 1.01

m=-1/N-((h_b+J_s)/(J_b*N))
m


import numpy as np

(np.sqrt(2*np.pi))/(10*np.pi)
