import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import itertools
import time
import os
import json
from datetime import datetime
from Function_database import*

# Parameters
L = 100  # Size of the lattice (LxL)
N=L**2
zealot_spin = 1
k_B = 1     #.380649e-23  # Boltzmann constant
num_iterations = N*100  # Total number of iterations
J_b = 1/(N-1)  # Coupling constant
J_s = 1.01
h_b= -1
h_s = N
number_of_MC_steps = 2
seeds = np.linspace(1,1000,1000).astype(int).tolist()
temperatures = np.linspace(0.1,1.5,50).tolist()
burn_in_steps = int((num_iterations/(number_of_MC_steps*N))*0.6)
time_average_proportion = 0
initial_up_ratio = 0.5
results_path = save_results(
    simulation_type="fully_connected", 
    initial_up_ratio=initial_up_ratio,  # The varied value
    J_s=J_s,
    h_s=h_s,
    varied_param="ratio",  # Specify which parameter is being varied
    L=L,
    N=N,
    zealot_spin=zealot_spin,
    k_B=k_B,
    num_iterations=num_iterations,
    J_b=J_b,
    h_b=h_b,
    number_of_MC_steps=number_of_MC_steps,
    seeds=seeds,
    temperatures=temperatures,
    burn_in_steps=burn_in_steps
)

# Run the simulation for all temperatures in parallel
start = time.time()
simulation_results = run_parallel_simulations_fc(temperatures, seeds, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, initial_up_ratio)
end = time.time()
length = end - start
processed_results_lists = process_all_results(simulation_results, burn_in_steps, time_average_proportion)



plot_magnetization_over_time(simulation_results, results_path)
plot_average_magnetizations_vs_temperature(processed_results_lists, results_path)
plot_m_plus_minus_vs_temperature(processed_results_lists, results_path)
plot_g_plus_minus_vs_temperature(processed_results_lists, results_path)
plot_zealot_statistics_vs_temperature(processed_results_lists, results_path)



with open(os.path.join(results_path, "numerical_results.json"), "w") as f:
    # Convert numpy values to standard Python types for JSON serialization
    serializable_results = {
        key: [float(x) if isinstance(x, np.float64) else x for x in value]
        for key, value in processed_results_lists.items()
    }
    json.dump(serializable_results, f, indent=4)
