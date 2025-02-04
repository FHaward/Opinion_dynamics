import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools
import time
import os
import json
from datetime import datetime


def calculate_energy_change(lattice, L, i, j, J_b, h_b, J_s, zealot_spin, magnetization): 
    """
    Calculate the energy change that would occur if the spin at (i, j) is flipped.
    """
    # Get the spin at the selected site
    spin = lattice[i, j]
    

    # Calculate energy change ΔE
    social_influence = -J_b*(magnetization-spin)*spin
    internal_field = -h_b*spin
    leader_influence= -J_s*zealot_spin*spin
    
    return -2*(social_influence+internal_field+leader_influence)

def calculate_energy_change_zealot(zealot_spin, magnetization, L, J_s, h_s):
    """
    Calculate the energy change that would occur if the zealot spin is flipped.
    """

    leader_field = -h_s*zealot_spin
    leader_influence = -J_s*zealot_spin*magnetization 
    
    return -2*(leader_influence+leader_field)

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
   
def metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization):
    """
    Perform one Metropolis step on the lattice and update the magnetization.
    """
    i, j = np.random.randint(0, L), np.random.randint(0, L)
    delta_E = calculate_energy_change(lattice, L, i, j, J_b, h_b, J_s, zealot_spin, magnetization)
    
    if np.random.rand() < safe_exp(delta_E, temp, k_B):
        lattice[i, j] *= -1  # Flip the spin
        magnetization += 2*lattice[i, j]

    return magnetization
    
def run_individual_simulation(seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps):
    np.random.seed(seed)
    zealot_spin = np.random.choice([-1, 1])
    lattice = np.random.choice([-1, 1], size=(L, L))
    magnetization_record_interval = number_of_MC_steps * N
    magnetization = np.sum(lattice)  # Initial magnetization

    # Preallocate the array for magnetization values
    num_intervals = num_iterations // magnetization_record_interval
    total_recalculations = num_intervals + 1 + (1 if num_iterations % magnetization_record_interval > 0 else 0)
    magnetization_array = np.zeros(total_recalculations, dtype=np.float64)
    zealot_array = np.zeros(total_recalculations, dtype=np.float64)
    magnetization_array[0] = magnetization
    zealot_array[0] = zealot_spin
    recalculation_index = 1

    # Loop over each interval
    for interval in tqdm(range(num_intervals), desc=f"Temp {temp}, Seed {seed}"):
        for step in range(number_of_MC_steps):
            for _ in range(N): 
                magnetization = metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

            delta_E_zealot = calculate_energy_change_zealot(zealot_spin, magnetization, L, J_s, h_s)
            if np.random.rand() < safe_exp(delta_E_zealot, temp, k_B):
                zealot_spin *= -1  # Flip the spin

        magnetization_array[recalculation_index] = magnetization
        zealot_array[recalculation_index] = zealot_spin
        recalculation_index += 1
        
        
    # Handle any remaining steps if num_iterations is not an exact multiple of recalculation_interval
    remaining_steps = num_iterations % magnetization_record_interval
    if remaining_steps > 0:
        full_zealot_updates = remaining_steps // N
        extra_steps = remaining_steps % N

        for _ in range(full_zealot_updates):  # Full zealot updates
            for _ in range(N):
                magnetization = metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

            # Update zealot spin every L^2 steps
            delta_E_zealot = calculate_energy_change_zealot(zealot_spin, magnetization, L, J_s, h_s)
            if np.random.rand() < safe_exp(delta_E_zealot, temp, k_B):
                zealot_spin *= -1

        # Handle any remaining steps without a zealot recalculation
        for _ in range(extra_steps):
            magnetization = metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

        # Final recalculation for the last segment
        magnetization_array[recalculation_index] = magnetization
        zealot_array[recalculation_index] = zealot_spin
        
    all_magnetizations = magnetization_array/N
                   
    return temp, all_magnetizations, zealot_array

def run_parallel_simulations(temperatures, seeds, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps):
    """
    Run simulations for all temperature-seed combinations in parallel.
    """
    combinations = list(itertools.product(temperatures, seeds))
    
    # Run all combinations in parallel
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_individual_simulation)(
            seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps
        ) for temp, seed in tqdm(combinations, desc="Running simulations")
    )
    
    # Organize results by temperature
    results_dict = {}
    for temp, magnetizations, zealot_spins in results:
        if temp not in results_dict:
            results_dict[temp] = {'magnetizations': [], 'zealot_spins': []}
        results_dict[temp]['magnetizations'].append(magnetizations)
        results_dict[temp]['zealot_spins'].append(zealot_spins)
    
    return results_dict 
  
def post_process_combined_results(all_magnetizations, all_zealot_spins, burn_in_steps, time_average_proportion):

    # Convert inputs to numpy arrays if they aren't already
    all_magnetizations = np.array(all_magnetizations)
    all_zealot_spins = np.array(all_zealot_spins)
    
    # Remove burn-in steps if specified
    if burn_in_steps > 0:
        all_magnetizations = all_magnetizations[:, burn_in_steps:]
        all_zealot_spins = all_zealot_spins[:, burn_in_steps:]
    
    # Calculate last 10% index
    last_10_percent_index = int(all_magnetizations.shape[1] * time_average_proportion)
    
    # Calculate time-averaged values for the last 10%
    time_avg_magnetizations = np.mean(all_magnetizations[:, last_10_percent_index:], axis=1)
    time_avg_spins = np.mean(all_zealot_spins[:, last_10_percent_index:], axis=1)
    
    # Identify indices for positive and negative results
    mag_positive_indices = time_avg_magnetizations > 0
    mag_negative_indices = time_avg_magnetizations < 0
    spin_positive_indices = time_avg_spins > 0
    spin_negative_indices = time_avg_spins < 0
    
    # Separate magnetization data
    m_plus = all_magnetizations[mag_positive_indices]
    m_minus = all_magnetizations[mag_negative_indices]
    
    # Separate zealot spin data
    z_plus = all_zealot_spins[spin_positive_indices]
    z_minus = all_zealot_spins[spin_negative_indices]
    
    # Calculate total runs
    total_runs = all_magnetizations.shape[0]
    
    # Compile all results into a dictionary
    results = {
        # Magnetization results
        'average_magnetization': np.mean(all_magnetizations),
        'm_plus_avg': np.mean(m_plus) if m_plus.size > 0 else 0,
        'm_minus_avg': np.mean(m_minus) if m_minus.size > 0 else 0,
        'g_plus': np.count_nonzero(mag_positive_indices) / total_runs,
        'g_minus': np.count_nonzero(mag_negative_indices) / total_runs,
        
        # Zealot spin results
        'average_zealot_spin': np.mean(all_zealot_spins),
        'z_plus_avg': np.mean(z_plus) if z_plus.size > 0 else 0,
        'z_minus_avg': np.mean(z_minus) if z_minus.size > 0 else 0,
        'f_plus': np.count_nonzero(spin_positive_indices) / total_runs,
        'f_minus': np.count_nonzero(spin_negative_indices) / total_runs
    }
    
    return results

def process_all_results(simulation_results, burn_in_steps, time_average_proportion):
    """
    Process results for all temperatures using the combined post-processing function.
    """
    temperatures = []
    results_dict = {
        'average_magnetizations': [], 'm_plus_avgs': [], 'm_minus_avgs': [], 
        'g_plus_list': [], 'g_minus_list': [],
        'average_zealot_spins': [], 'z_plus_avgs': [], 'z_minus_avgs': [],
        'f_plus_list': [], 'f_minus_list': []
    }
    
    for temp, data in simulation_results.items():
        # Process both magnetization and zealot data using combined function
        processed = post_process_combined_results(
            data['magnetizations'], 
            data['zealot_spins'], 
            burn_in_steps,
            time_average_proportion
        )
        
        temperatures.append(temp)
        
        # Store all processed results
        results_dict['average_magnetizations'].append(processed['average_magnetization'])
        results_dict['m_plus_avgs'].append(processed['m_plus_avg'])
        results_dict['m_minus_avgs'].append(processed['m_minus_avg'])
        results_dict['g_plus_list'].append(processed['g_plus'])
        results_dict['g_minus_list'].append(processed['g_minus'])
        results_dict['average_zealot_spins'].append(processed['average_zealot_spin'])
        results_dict['z_plus_avgs'].append(processed['z_plus_avg'])
        results_dict['z_minus_avgs'].append(processed['z_minus_avg'])
        results_dict['f_plus_list'].append(processed['f_plus'])
        results_dict['f_minus_list'].append(processed['f_minus'])
    
    results_dict['temperatures'] = temperatures
    return results_dict

def save_plot(plt, filename):
    """Save the current plot and close it."""
    plt.savefig(filename)
    plt.close()
    
def plot_magnetization_over_time(simulation_results, save_path):
    for temp, data in simulation_results.items():       
        all_magnetizations = data['magnetizations']
        plt.figure(figsize=(10, 6))
        for i, magnetization in enumerate(all_magnetizations):
            plt.plot(magnetization, label=f"Seed {i}")
        
        plt.xlabel("Monte Carlo Steps")
        plt.ylabel("Magnetization")
        plt.title(f"Magnetization Over Time (T = {temp})")
        plt.grid(True)
        plt.legend(loc='best', fontsize=8)
        plt.tight_layout()
        save_plot(plt, os.path.join(save_path, f"magnetization_over_time_T={temp}.png"))

def plot_average_magnetizations_vs_temperature(results, save_path):
    # Extract temperatures and average magnetizations
    temperatures = results['temperatures']
    average_magnetizations = results['average_magnetizations']
    standard_errors = results['average_magnetizations_std_errors']
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(temperatures, average_magnetizations, 
                 yerr=standard_errors,
                 marker="o", label="Average Magnetization", color="blue", 
                 capsize=5, ecolor='gray')
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Magnetization", fontsize=12)
    plt.title("Average Magnetization vs. Temperature", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "avg_magnetization_vs_temp.png"))

def plot_m_plus_minus_vs_temperature(results, save_path):
    # Extract data
    temperatures = results['temperatures']
    m_plus_avgs = results['m_plus_avgs']
    m_minus_avgs = results['m_minus_avgs']
    m_plus_errors = results['m_plus_avgs_std_errors']
    m_minus_errors = results['m_minus_avgs_std_errors']
    
    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(temperatures, m_plus_avgs, yerr=m_plus_errors, 
                 marker="o", label="$m_+$ (Fraction Positive)", color="blue", 
                 capsize=5, ecolor='gray')
    plt.errorbar(temperatures, m_minus_avgs, yerr=m_minus_errors, 
                 marker="s", label="$m_-$ (Fraction Negative)", color="orange", 
                 capsize=5, ecolor='gray')
    
    # Add labels, title, and grid
    plt.xlabel("Temperature", fontsize=14)
    plt.ylabel("Average Magnetization", fontsize=14)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "m_plus_minus_vs_temp.png"))

def plot_g_plus_minus_vs_temperature(results, save_path):
    # Extract data
    temperatures = results['temperatures']
    g_plus_list = results['g_plus_list']
    g_minus_list = results['g_minus_list']
    g_plus_errors = results['g_plus_std_errors']
    g_minus_errors = results['g_minus_std_errors']

    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(temperatures, g_plus_list, yerr=g_plus_errors, 
                 marker="o", label="$g_+$ (Fraction Positive)", color="green", 
                 capsize=5, ecolor='gray')
    plt.errorbar(temperatures, g_minus_list, yerr=g_minus_errors, 
                 marker="s", label="$g_-$ (Fraction Negative)", color="red", 
                 capsize=5, ecolor='gray')

    # Add labels, title, and grid
    plt.xlabel("Temperature")
    plt.ylabel("Fraction of Runs")
    plt.title("$g_+$ and $g_-$ vs. Temperature")
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "g_plus_minus_vs_temp.png"))   

def plot_zealot_statistics_vs_temperature(results, save_path):

    temperatures = results['temperatures']
    average_zealot_spins = results['average_zealot_spins']
    zealot_errors = results['average_zealot_spins_std_errors']
    
    # Plot average zealot spin vs temperature
    plt.figure(figsize=(8, 5))
    plt.errorbar(temperatures, average_zealot_spins, 
                 yerr=zealot_errors,
                 marker="o", label="Average Zealot Spin", color="purple", 
                 capsize=5, ecolor='gray')

    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Zealot Spin", fontsize=12)
    plt.title("Average Zealot Spin vs. Temperature", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "avg_zealot_vs_temp.png"))

    z_plus_avgs = results['z_plus_avgs']
    z_minus_avgs = results['z_minus_avgs']
    z_plus_errors = results['z_plus_avgs_std_errors']
    z_minus_errors = results['z_minus_avgs_std_errors']
    
    # Plot z+ and z- vs temperature
    plt.figure(figsize=(8, 5))
    plt.errorbar(temperatures, z_plus_avgs, yerr=z_plus_errors, 
                 marker="o", label="$m_+$ (Fraction Positive)", color="blue", 
                 capsize=5, ecolor='gray')
    plt.errorbar(temperatures, z_minus_avgs, yerr=z_minus_errors, 
                 marker="s", label="$m_-$ (Fraction Negative)", color="red", 
                 capsize=5, ecolor='gray')

    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Zealot Spin", fontsize=12)
    plt.title("$z_+$ and $z_-$ vs. Temperature", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "z_plus_minus_vs_temp.png"))

    f_plus_list = results['f_plus_list']
    f_minus_list = results['f_minus_list']
    f_plus_errors = results['f_plus_std_errors']
    f_minus_errors = results['f_minus_std_errors']

    # Plot f+ and f- vs temperature
    plt.figure(figsize=(8, 5))
    plt.errorbar(temperatures, f_plus_list, yerr=f_plus_errors, 
                 marker="o", label="$f_+$ (Fraction Positive)", color="green", 
                 capsize=5, ecolor='gray')
    plt.errorbar(temperatures, f_minus_list, yerr=f_minus_errors, 
                 marker="s", label="$f_-$ (Fraction Negative)", color="orange", 
                 capsize=5, ecolor='gray')

    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Fraction of Runs", fontsize=12)
    plt.title("$f_+$ and $f_-$ vs. Temperature", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "f_plus_minus_vs_temp.png"))
  
def save_results(simulation_type="fully_connected", results_dir="./simulation_results"):
    """
    Save simulation results to a directory.
    simulation_type: either "fully_connected" or "nearest_neighbour"
    Creates a timestamped subdirectory within the simulation type directory.
    """
    # Create main results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Create simulation type directory if it doesn't exist
    sim_type_dir = os.path.join(results_dir, simulation_type)
    if not os.path.exists(sim_type_dir):
        os.makedirs(sim_type_dir)
    
    # Create timestamped subdirectory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(sim_type_dir, timestamp)
    os.makedirs(run_dir)
    
    # Save parameters
    params = {
        "L": L,
        "N": N,
        "zealot_spin": zealot_spin,
        "k_B": k_B,
        "num_iterations": num_iterations,
        "J_b": J_b,
        "J_s": J_s,
        "h_b": h_b,
        "h_s": h_s,
        "number_of_MC_steps": number_of_MC_steps,
        "seeds": seeds,
        "temperatures": temperatures,
        "burn_in_steps": burn_in_steps,
        "simulation_type": simulation_type
    }
    
    with open(os.path.join(run_dir, "parameters.txt"), "w") as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")
    
    return run_dir


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
seeds = np.linspace(1,30,30).astype(int).tolist()
temperatures = np.linspace(0.1,1.5,30).tolist()
burn_in_steps = int((num_iterations/(number_of_MC_steps*N))*0.5)
results_path = save_results(simulation_type="fully_connected")
time_average_proportion = 0.8


# Run the simulation for all temperatures in parallel
start = time.time()
simulation_results = run_parallel_simulations(temperatures, seeds, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps)
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



n_cores = int(os.environ.get('SLURM_CPUS_PER_TASK', default=1))
print(f"this simulation used {n_cores} cores and took {length} seconds")
