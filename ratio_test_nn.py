import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import itertools
import time
import os
import json
from datetime import datetime



def calculate_energy_change_nn(lattice, L, i, j, J_b, h_b, J_s, zealot_spin):
    """
    Calculate the energy change that would occur if the spin at (i, j) is flipped.
    """
    # Get the spin at the selected site
    spin = lattice[i, j]
    
    # Calculate the sum of products of the selected site spin with each neighboring spin
    neighbor_sum = (
        spin * lattice[(i + 1) % L, j] +  # Right neighbor
        spin * lattice[(i - 1) % L, j] +  # Left neighbor
        spin * lattice[i, (j + 1) % L] +  # Down neighbor
        spin * lattice[i, (j - 1) % L]    # Up neighbor
    )
    
    # Calculate energy change ΔE
    social_influence = -J_b*neighbor_sum
    internal_field = -h_b*spin
    leader_influence= -J_s*zealot_spin*spin
    return -2*(social_influence+internal_field+leader_influence)

def calculate_energy_change_zealot_nn(zealot_spin, magnetization, L, J_s, h_s):
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
   
def metropolis_step_nn(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization):
    """
    Perform one Monte Carlo step on the lattice.
    """

    i, j = np.random.randint(0, L), np.random.randint(0, L)
    delta_E = calculate_energy_change_nn(lattice, L, i, j, J_b, h_b, J_s, zealot_spin)
    
    if np.random.rand() < safe_exp(delta_E, temp, k_B):
        lattice[i, j] *= -1  # Flip the spin
        magnetization += 2*lattice[i, j]

    return magnetization
  
def run_individual_simulation_nn(seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, initial_up_ratio=0.5):
    np.random.seed(seed)
    zealot_spin = np.random.choice([-1, 1])
    p_up = initial_up_ratio
    p_down = 1 - initial_up_ratio
    lattice = np.random.choice([-1, 1], size=(L, L), p=[p_down, p_up])
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
    for interval in range(num_intervals):
        for step in range(number_of_MC_steps):
            for _ in range(N): 
                magnetization = metropolis_step_nn(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

            delta_E_zealot = calculate_energy_change_zealot_nn(zealot_spin, magnetization, L, J_s, h_s)
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
                magnetization = metropolis_step_nn(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

            # Update zealot spin every L^2 steps
            delta_E_zealot = calculate_energy_change_zealot_nn(zealot_spin, magnetization, L, J_s, h_s)
            if np.random.rand() < safe_exp(delta_E_zealot, temp, k_B):
                zealot_spin *= -1

        # Handle any remaining steps without a zealot recalculation
        for _ in range(extra_steps):
            magnetization = metropolis_step_nn(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

        # Final recalculation for the last segment
        magnetization_array[recalculation_index] = magnetization
        zealot_array[recalculation_index] = zealot_spin
        
    all_magnetizations = magnetization_array/N
                   
    return temp, all_magnetizations, zealot_array
  

def run_parallel_simulations(ratios, seeds, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, temp):
    """
    Run simulations for all ratio-seed combinations in parallel.
    """
    combinations = list(itertools.product(ratios, seeds))
    
    # Run all combinations in parallel
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(run_individual_simulation_nn)(
            seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, 
            num_iterations, number_of_MC_steps, ratio
        ) for ratio, seed in combinations
    )
    
    # Organize results by ratio
    results_dict = {}
    for ratio, magnetizations, zealot_spins in results:
        if ratio not in results_dict:
            results_dict[ratio] = {'magnetizations': [], 'zealot_spins': []}
        results_dict[ratio]['magnetizations'].append(magnetizations)
        results_dict[ratio]['zealot_spins'].append(zealot_spins)
    
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
    
    # Calculate g_plus, g_minus and f_plus, f_minus
    g_plus = np.count_nonzero(mag_positive_indices) / total_runs
    g_minus = np.count_nonzero(mag_negative_indices) / total_runs
    f_plus = np.count_nonzero(spin_positive_indices) / total_runs
    f_minus = np.count_nonzero(spin_negative_indices) / total_runs
    
    # Calculate averages
    m_plus_avg = np.mean(m_plus) if m_plus.size > 0 else np.nan
    m_minus_avg = np.mean(m_minus) if m_minus.size > 0 else np.nan
    z_plus_avg = np.mean(z_plus) if z_plus.size > 0 else np.nan
    z_minus_avg = np.mean(z_minus) if z_minus.size > 0 else np.nan
    
    # Calculate standard errors
    m_plus_std_error = np.std(time_avg_magnetizations[mag_positive_indices]) / np.sqrt(np.sum(mag_positive_indices)) if np.sum(mag_positive_indices) > 0 else 0
    m_minus_std_error = np.std(time_avg_magnetizations[mag_negative_indices]) / np.sqrt(np.sum(mag_negative_indices)) if np.sum(mag_negative_indices) > 0 else 0
    z_plus_std_error = np.std(time_avg_spins[spin_positive_indices]) / np.sqrt(np.sum(spin_positive_indices)) if np.sum(spin_positive_indices) > 0 else 0
    z_minus_std_error = np.std(time_avg_spins[spin_negative_indices]) / np.sqrt(np.sum(spin_negative_indices)) if np.sum(spin_negative_indices) > 0 else 0
    
    g_plus_std_error = np.sqrt((g_plus * (1 - g_plus)) / total_runs)
    g_minus_std_error = np.sqrt((g_minus * (1 - g_minus)) / total_runs)
    f_plus_std_error = np.sqrt((f_plus * (1 - f_plus)) / total_runs)
    f_minus_std_error = np.sqrt((f_minus * (1 - f_minus)) / total_runs)
    
    # Calculate weighted averages
    m_plus_term = m_plus_avg * g_plus if not np.isnan(m_plus_avg) else 0
    m_minus_term = m_minus_avg * g_minus if not np.isnan(m_minus_avg) else 0
    average_magnetization = m_plus_term + m_minus_term

    z_plus_term = z_plus_avg * f_plus if not np.isnan(z_plus_avg) else 0
    z_minus_term = z_minus_avg * f_minus if not np.isnan(z_minus_avg) else 0
    average_zealot_spin = z_plus_term + z_minus_term


    # Calculate errors of weighted averages using error propagation
    # For a function f(x,y,z,w) = ax + by, the error is:
    average_magnetization_std_error = np.sqrt(
        (g_plus * m_plus_std_error)**2 if not np.isnan(m_plus_avg) else 0 +     
        (g_minus * m_minus_std_error)**2 if not np.isnan(m_minus_avg) else 0 +    
        (m_plus_avg * g_plus_std_error)**2 if not np.isnan(m_plus_avg) else 0 + 
        (m_minus_avg * g_minus_std_error)**2 if not np.isnan(m_minus_avg) else 0 
    )

    average_zealot_spin_std_error = np.sqrt(
        (f_plus * z_plus_std_error)**2 if not np.isnan(z_plus_avg) else 0 +      
        (f_minus * z_minus_std_error)**2 if not np.isnan(z_minus_avg) else 0 +   
        (z_plus_avg * f_plus_std_error)**2 if not np.isnan(z_plus_avg) else 0 +  
        (z_minus_avg * f_minus_std_error)**2 if not np.isnan(z_minus_avg) else 0 
    )
    
    # Compile all results into a dictionary
    results = {
        # Magnetization results
        'average_magnetization': average_magnetization,
        'average_magnetization_std_error': average_magnetization_std_error,
        
        'm_plus_avg': m_plus_avg,
        'm_plus_avg_std_error': m_plus_std_error,
        
        'm_minus_avg': m_minus_avg,
        'm_minus_avg_std_error': m_minus_std_error,
        
        'g_plus': g_plus,
        'g_plus_std_error': g_plus_std_error,
        
        'g_minus': g_minus,
        'g_minus_std_error': g_minus_std_error,
        
        # Zealot spin results
        'average_zealot_spin': average_zealot_spin,
        'average_zealot_spin_std_error': average_zealot_spin_std_error,
        
        'z_plus_avg': z_plus_avg,
        'z_plus_avg_std_error': z_plus_std_error,
        
        'z_minus_avg': z_minus_avg,
        'z_minus_avg_std_error': z_minus_std_error,
        
        'f_plus': f_plus,
        'f_plus_std_error': f_plus_std_error,
        
        'f_minus': f_minus,
        'f_minus_std_error': f_minus_std_error
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
        'f_plus_list': [], 'f_minus_list': [],
        
        # Add corresponding standard error lists
        'average_magnetizations_std_errors': [], 'm_plus_avgs_std_errors': [], 'm_minus_avgs_std_errors': [],
        'g_plus_std_errors': [], 'g_minus_std_errors': [],
        'average_zealot_spins_std_errors': [], 'z_plus_avgs_std_errors': [], 'z_minus_avgs_std_errors': [],
        'f_plus_std_errors': [], 'f_minus_std_errors': []
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
        results_dict['average_magnetizations_std_errors'].append(processed['average_magnetization_std_error'])
        
        results_dict['m_plus_avgs'].append(processed['m_plus_avg'])
        results_dict['m_plus_avgs_std_errors'].append(processed['m_plus_avg_std_error'])
        
        results_dict['m_minus_avgs'].append(processed['m_minus_avg'])
        results_dict['m_minus_avgs_std_errors'].append(processed['m_minus_avg_std_error'])
        
        results_dict['g_plus_list'].append(processed['g_plus'])
        results_dict['g_plus_std_errors'].append(processed['g_plus_std_error'])
        
        results_dict['g_minus_list'].append(processed['g_minus'])
        results_dict['g_minus_std_errors'].append(processed['g_minus_std_error'])
        
        results_dict['average_zealot_spins'].append(processed['average_zealot_spin'])
        results_dict['average_zealot_spins_std_errors'].append(processed['average_zealot_spin_std_error'])
        
        results_dict['z_plus_avgs'].append(processed['z_plus_avg'])
        results_dict['z_plus_avgs_std_errors'].append(processed['z_plus_avg_std_error'])
        
        results_dict['z_minus_avgs'].append(processed['z_minus_avg'])
        results_dict['z_minus_avgs_std_errors'].append(processed['z_minus_avg_std_error'])
        
        results_dict['f_plus_list'].append(processed['f_plus'])
        results_dict['f_plus_std_errors'].append(processed['f_plus_std_error'])
        
        results_dict['f_minus_list'].append(processed['f_minus'])
        results_dict['f_minus_std_errors'].append(processed['f_minus_std_error'])
    
    results_dict['temperatures'] = temperatures
    return results_dict

def save_plot(plt, filename):
    """Save the current plot and close it."""
    plt.savefig(filename)
    plt.close()
      
def plot_magnetization_over_time(simulation_results, save_path):
    for ratio, data in simulation_results.items():       
        all_magnetizations = data['magnetizations']
        plt.figure(figsize=(10, 6))
        for i, magnetization in enumerate(all_magnetizations):
            plt.plot(magnetization, label=f"Seed {i}")
        
        plt.xlabel("Monte Carlo Steps")
        plt.ylabel("Magnetization")
        plt.title(f"Magnetization Over Time (Ratio = {ratio:.2f})")
        plt.grid(True)
        plt.tight_layout()
        save_plot(plt, os.path.join(save_path, f"magnetization_over_time_ratio={ratio:.2f}.png"))

def plot_average_magnetizations_vs_ratio(results, save_path, temp):
    # Extract ratios and average magnetizations
    ratios = results['ratios']
    average_magnetizations = results['average_magnetizations']
    standard_errors = results['average_magnetizations_std_errors']
    
    plt.figure(figsize=(8, 5))
    plt.errorbar(ratios, average_magnetizations, 
                 yerr=standard_errors,
                 marker="o", label="Average Magnetization", color="blue", 
                 capsize=5, ecolor='gray')
    
    plt.xlabel("Initial Up Ratio", fontsize=12)
    plt.ylabel("Average Magnetization", fontsize=12)
    plt.title(f"Average Magnetization vs. Initial Up Ratio (T = {temp})", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "avg_magnetization_vs_ratio.png"))

def plot_m_plus_minus_vs_ratio(results, save_path, temp):
    # Extract data
    ratios = results['ratios']
    m_plus_avgs = results['m_plus_avgs']
    m_minus_avgs = results['m_minus_avgs']
    m_plus_errors = results['m_plus_avgs_std_errors']
    m_minus_errors = results['m_minus_avgs_std_errors']
    
    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(ratios, m_plus_avgs, yerr=m_plus_errors, 
                 marker="o", label="$m_+$ (Positive)", color="blue", 
                 capsize=5, ecolor='gray')
    plt.errorbar(ratios, m_minus_avgs, yerr=m_minus_errors, 
                 marker="s", label="$m_-$ (Negative)", color="orange", 
                 capsize=5, ecolor='gray')
    
    # Add labels, title, and grid
    plt.xlabel("Initial Up Ratio", fontsize=14)
    plt.ylabel("Average Magnetization", fontsize=14)
    plt.title(f"$m_+$ and $m_-$ vs. Initial Up Ratio (T = {temp})", fontsize=14)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "m_plus_minus_vs_ratio.png"))

def plot_g_plus_minus_vs_ratio(results, save_path, temp):
    # Extract data
    ratios = results['ratios']
    g_plus_list = results['g_plus_list']
    g_minus_list = results['g_minus_list']
    g_plus_errors = results['g_plus_std_errors']
    g_minus_errors = results['g_minus_std_errors']

    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.errorbar(ratios, g_plus_list, yerr=g_plus_errors, 
                 marker="o", label="$g_+$ (Fraction Positive)", color="green", 
                 capsize=5, ecolor='gray')
    plt.errorbar(ratios, g_minus_list, yerr=g_minus_errors, 
                 marker="s", label="$g_-$ (Fraction Negative)", color="red", 
                 capsize=5, ecolor='gray')

    # Add labels, title, and grid
    plt.xlabel("Initial Up Ratio")
    plt.ylabel("Fraction of Runs")
    plt.title(f"$g_+$ and $g_-$ vs. Initial Up Ratio (T = {temp})")
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "g_plus_minus_vs_ratio.png"))   

def plot_zealot_statistics_vs_ratio(results, save_path, temp):
    ratios = results['ratios']
    average_zealot_spins = results['average_zealot_spins']
    zealot_errors = results['average_zealot_spins_std_errors']
    
    # Plot average zealot spin vs ratio
    plt.figure(figsize=(8, 5))
    plt.errorbar(ratios, average_zealot_spins, 
                 yerr=zealot_errors,
                 marker="o", label="Average Zealot Spin", color="purple", 
                 capsize=5, ecolor='gray')

    plt.xlabel("Initial Up Ratio", fontsize=12)
    plt.ylabel("Average Zealot Spin", fontsize=12)
    plt.title(f"Average Zealot Spin vs. Initial Up Ratio (T = {temp})", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "avg_zealot_vs_ratio.png"))

    z_plus_avgs = results['z_plus_avgs']
    z_minus_avgs = results['z_minus_avgs']
    z_plus_errors = results['z_plus_avgs_std_errors']
    z_minus_errors = results['z_minus_avgs_std_errors']
    
    # Plot z+ and z- vs ratio
    plt.figure(figsize=(8, 5))
    plt.errorbar(ratios, z_plus_avgs, yerr=z_plus_errors, 
                 marker="o", label="$z_+$ (Positive)", color="blue", 
                 capsize=5, ecolor='gray')
    plt.errorbar(ratios, z_minus_avgs, yerr=z_minus_errors, 
                 marker="s", label="$z_-$ (Negative)", color="red", 
                 capsize=5, ecolor='gray')

    plt.xlabel("Initial Up Ratio", fontsize=12)
    plt.ylabel("Average Zealot Spin", fontsize=12)
    plt.title(f"$z_+$ and $z_-$ vs. Initial Up Ratio (T = {temp})", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "z_plus_minus_vs_ratio.png"))

    f_plus_list = results['f_plus_list']
    f_minus_list = results['f_minus_list']
    f_plus_errors = results['f_plus_std_errors']
    f_minus_errors = results['f_minus_std_errors']

    # Plot f+ and f- vs ratio
    plt.figure(figsize=(8, 5))
    plt.errorbar(ratios, f_plus_list, yerr=f_plus_errors, 
                 marker="o", label="$f_+$ (Fraction Positive)", color="green", 
                 capsize=5, ecolor='gray')
    plt.errorbar(ratios, f_minus_list, yerr=f_minus_errors, 
                 marker="s", label="$f_-$ (Fraction Negative)", color="orange", 
                 capsize=5, ecolor='gray')

    plt.xlabel("Initial Up Ratio", fontsize=12)
    plt.ylabel("Fraction of Runs", fontsize=12)
    plt.title(f"$f_+$ and $f_-$ vs. Initial Up Ratio (T = {temp})", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    save_plot(plt, os.path.join(save_path, "f_plus_minus_vs_ratio.png"))
  
def save_results(simulation_type="fully_connected", fixed_temp=0.5, J_s=1.01, h_s=10000, results_dir="./simulation_results"):
    """
    Save simulation results to a directory with an organized parameter-based structure.
    
    Parameters:
    - simulation_type: either "fully_connected" or "nearest_neighbour"
    - fixed_temp: the fixed temperature used in the simulation
    - J_s: coupling constant between the zealot and the spins
    - h_s: field strength on the zealot
    - results_dir: base directory for all results
    
    Returns:
    - Path to the specific results directory for this run
    """
    # Create main results directory if it doesn't exist
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # Start with the simulation type folder
    sim_type_dir = os.path.join(results_dir, simulation_type)
    if not os.path.exists(sim_type_dir):
        os.makedirs(sim_type_dir)
    
    # Create the parameter-based directory structure
    temp_str = f"T_{fixed_temp}"
    J_s_str = f"J_s_{J_s}"
    h_s_str = f"h_s_{h_s}"
    
    # Create the three parallel parameter structures
    by_temp_dir = os.path.join(sim_type_dir, "by_temperature", temp_str)
    by_J_s_dir = os.path.join(sim_type_dir, "by_J_s", J_s_str)
    by_h_s_dir = os.path.join(sim_type_dir, "by_h_s", h_s_str)
    
    # Create directories if they don't exist
    for directory in [by_temp_dir, by_J_s_dir, by_h_s_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
    
    # Create timestamp for this specific run
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create the actual run directories with other fixed parameters in the name
    temp_run_dir = os.path.join(by_temp_dir, f"{timestamp}_J_s_{J_s}_h_s_{h_s}")
    J_s_run_dir = os.path.join(by_J_s_dir, f"{timestamp}_T_{fixed_temp}_h_s_{h_s}")
    h_s_run_dir = os.path.join(by_h_s_dir, f"{timestamp}_T_{fixed_temp}_J_s_{J_s}")
    
    # Create all directories
    os.makedirs(temp_run_dir, exist_ok=True)
    os.makedirs(J_s_run_dir, exist_ok=True)
    os.makedirs(h_s_run_dir, exist_ok=True)
    
    # Create a primary run directory (we'll use the temp one as primary)
    run_dir = temp_run_dir
    
    # Save parameters to all three directories
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
        "ratios": ratios,
        "fixed_temperature": fixed_temp,
        "burn_in_steps": burn_in_steps,
        "simulation_type": simulation_type
    }
    
    # Write parameters to all three directories
    for directory in [temp_run_dir, J_s_run_dir, h_s_run_dir]:
        with open(os.path.join(directory, "parameters.txt"), "w") as f:
            for key, value in params.items():
                f.write(f"{key}: {value}\n")
    
    # Return the primary directory for saving plots and other results
    return run_dir
        

# Parameters
L = 100  # Size of the lattice (LxL)
N=L**2
zealot_spin = 1
k_B = 1     # Boltzmann constant
num_iterations = N*1000  # Total number of iterations
J_b = 1/4  # Coupling constant
J_s = 1.01
h_b= -1
h_s = N
number_of_MC_steps = 2
seeds = np.linspace(1,200,200).astype(int).tolist()
fixed_temp = 0.1  # Fixed temperature
ratios = np.linspace(0.45, 0.55, 20).tolist()  # Range of initial up ratios
burn_in_steps = int((num_iterations/(number_of_MC_steps*N))*0.8)
time_average_proportion = 0

results_path = save_results(
    simulation_type="nearest_neighbour", 
    fixed_temp=fixed_temp,
    J_s=J_s,
    h_s=h_s
)

# Run the simulation for all ratios in parallel
start = time.time()
simulation_results = run_parallel_simulations(ratios, seeds, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, fixed_temp)
end = time.time()
length = end - start
processed_results_lists = process_all_results(simulation_results, burn_in_steps, time_average_proportion)

# Create plots
plot_magnetization_over_time(simulation_results, results_path)
plot_average_magnetizations_vs_ratio(processed_results_lists, results_path, fixed_temp)
plot_m_plus_minus_vs_ratio(processed_results_lists, results_path, fixed_temp)
plot_g_plus_minus_vs_ratio(processed_results_lists, results_path, fixed_temp)
plot_zealot_statistics_vs_ratio(processed_results_lists, results_path, fixed_temp)

# Save numerical results
with open(os.path.join(results_path, "numerical_results.json"), "w") as f:
    # Convert numpy values to standard Python types for JSON serialization
    serializable_results = {
        key: [float(x) if isinstance(x, np.float64) else x for x in value]
        for key, value in processed_results_lists.items()
    }
    json.dump(serializable_results, f, indent=4)
    
    
    
np.linspace(0.45, 0.55, 21).tolist()