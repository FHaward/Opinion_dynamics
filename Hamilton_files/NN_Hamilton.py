import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
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
num_iterations = N*200  # Total number of iterations
J_b = 1/4  # Coupling constant
J_s = 1.01
h_b= -1
h_s = N
seeds = list(np.linspace(1,10,10, dtype = int))
number_of_MC_steps = 2
temperatures = list(np.linspace(0.1,1.5,15))
burn_in_steps = int((num_iterations/(number_of_MC_steps*N))*0.5)
time_average_proportion = 0.8
initial_up_ratio = 0.5
results_path = save_results(
    simulation_type="fully_connected", 
    initial_up_ratio=0.7,  # The varied value
    J_s=1.01,
    h_s=h_s,
    varied_param="ratio"  # Specify which parameter is being varied
)

# Run the simulation for all temperatures in parallel
start = time.time()
simulation_results = run_parallel_simulations_nn(temperatures, seeds, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, initial_up_ratio)
end = time.time()
length = end - start
print("It took", length, "seconds!")
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

