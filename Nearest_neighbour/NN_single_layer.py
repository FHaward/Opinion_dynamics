import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from joblib import Parallel, delayed
import itertools
import time


def calculate_energy_change(lattice, L, i, j, J_b, h_b, J_s, zealot_spin):
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
   
def create_lookup_table(temp, k_B, J_b, h_b, J_s):
    # Define the possible values of ΔE (based on neighbors and fields)
    possible_neighbor_sums = np.array([-4, -2, 0, 2, 4])  # Neighboring spins sum possibilities
    possible_internal_field= np.array([h_b,-h_b])
    possible_leader_influence= np.array([J_s,-J_s])

    # Combine the two contributions to calculate all ΔE values
    delta_E_values = []
    for neighbor_sum in possible_neighbor_sums:
        for internal_field in possible_internal_field:
            for leader_influence in possible_leader_influence:
                    delta_E = -2 * ((J_b*neighbor_sum)+internal_field+leader_influence)
                    delta_E_values.append(delta_E)

    # Remove duplicates and sort
    delta_E_values = np.unique(delta_E_values)

    # Precompute exponential values for all ΔE
    lookup_table = {delta_E: safe_exp(delta_E, temp, k_B) for delta_E in delta_E_values}

    return lookup_table

def metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table):
    """
    Perform one Monte Carlo step on the lattice.
    """

    i, j = np.random.randint(0, L), np.random.randint(0, L)
    delta_E = calculate_energy_change(lattice, L, i, j, J_b, h_b, J_s, zealot_spin)
    
    if np.random.rand() < lookup_table.get(delta_E, 0):
        lattice[i, j] *= -1  # Flip the spin
  
def run_individual_simulation(seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps):
    np.random.seed(seed)
    zealot_spin = np.random.choice([-1, 1])
    lookup_table = create_lookup_table(temp, k_B, J_b, h_b, J_s)
    lattice = np.random.choice([-1, 1], size=(L, L))
    magnetization_record_interval = number_of_MC_steps * N


    # Preallocate the array for magnetization values
    num_intervals = num_iterations // magnetization_record_interval
    total_recalculations = num_intervals + 1 + (1 if num_iterations % magnetization_record_interval > 0 else 0)
    magnetization_array = np.zeros(total_recalculations, dtype=np.float64)
    zealot_array = np.zeros(total_recalculations, dtype=np.float64)
    magnetization_array[0] = np.sum(lattice)
    zealot_array[0] = zealot_spin
    recalculation_index = 1

    # Loop over each interval
    for interval in tqdm(range(num_intervals), desc=f"Temp {temp}, Seed {seed}"):
        for step in range(number_of_MC_steps):
            for _ in range(N): 
                metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table)

            magnetization = np.sum(lattice)
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
                metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table, magnetization)

            # Update zealot spin every L^2 steps
            magnetization = np.sum(lattice)
            delta_E_zealot = calculate_energy_change_zealot(zealot_spin, magnetization, L, J_s, h_s)
            if np.random.rand() < safe_exp(delta_E_zealot, temp, k_B):
                zealot_spin *= -1

        # Handle any remaining steps without a zealot recalculation
        for _ in range(extra_steps):
            metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table, magnetization)

        magnetization = np.sum(lattice)
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
    results = Parallel(n_jobs=-1)(
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

def plot_magnetization_over_time(simulation_results):
    
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
        plt.show()

def plot_average_magnetizations_vs_temperature(results):
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
    plt.show()
        
def plot_g_plus_minus_vs_temperature(results):
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
    plt.show()

def plot_m_plus_minus_vs_temperature(results):
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
    plt.show()

def plot_zealot_statistics_vs_temperature(results):

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
    plt.show()

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
    plt.show()

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
    plt.show()  
  

# Parameters
L = 50  # Size of the lattice (LxL)
N=L**2
zealot_spin = 1
k_B = 1     #.380649e-23  # Boltzmann constant
num_iterations = N*200  # Total number of iterations
J_b = 1#1/4  # Coupling constant
J_s = 0#1.01
h_b= 0#-1
h_s = 0#N
seeds = list(np.linspace(1,5,5, dtype = int))
number_of_MC_steps = 2
temperatures = list(np.linspace(0.1,1.5,5))
burn_in_steps = int((num_iterations/(number_of_MC_steps*N))*0.5)
time_average_proportion = 0.8

seeds
temperatures
start = time.time()

# Run the simulation for all temperatures in parallel
simulation_results = run_parallel_simulations(temperatures, seeds, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps)
end = time.time()
length = end - start
print("It took", length, "seconds!")



plot_magnetization_over_time(simulation_results)
processed_results_lists = process_all_results(simulation_results, burn_in_steps, time_average_proportion)
plot_average_magnetizations_vs_temperature(processed_results_lists)
plot_m_plus_minus_vs_temperature(processed_results_lists)
plot_g_plus_minus_vs_temperature(processed_results_lists)
plot_zealot_statistics_vs_temperature(processed_results_lists)
  


