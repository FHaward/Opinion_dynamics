import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from joblib import Parallel, delayed

import time
import multiprocessing
m_cores = multiprocessing.cpu_count()
m_cores


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

def metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table):
    """
    Perform one Monte Carlo step on the lattice.
    """

    i, j = np.random.randint(0, L), np.random.randint(0, L)
    delta_E = calculate_energy_change(lattice, L, i, j, J_b, h_b, J_s, zealot_spin)
    
    if np.random.rand() < lookup_table.get(delta_E, 0):
        lattice[i, j] *= -1  # Flip the spin
    
def run_simulation_for_seed(seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, lookup_table):
    np.random.seed(seed)
    lattice = np.random.choice([-1, 1], size=(L, L))
    magnetization_record_interval = number_of_MC_steps * N


    # Preallocate the array for magnetization values
    num_intervals = num_iterations // magnetization_record_interval
    total_recalculations = num_intervals + 1 + (1 if num_iterations % magnetization_record_interval > 0 else 0)
    magnetization_array = np.zeros(total_recalculations, dtype=np.float64)
    magnetization_array[0] = np.sum(lattice)
    recalculation_index = 1

    # Loop over each interval
    for interval in tqdm(range(num_intervals), desc=f"Temp {temp}, Seed {seed}"):
        for step in range(number_of_MC_steps):
            for _ in range(N): 
                metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table)

            magnetization = np.sum(lattice)
            delta_E_zealot = calculate_energy_change_zealot(zealot_spin, magnetization, L, J_s, h_s)
            if np.random.rand() < np.exp(-delta_E_zealot / (k_B * temp)):
                zealot_spin *= -1  # Flip the spin

        magnetization_array[recalculation_index] = magnetization
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
            if np.random.rand() < np.exp(-delta_E_zealot / (k_B * temp)):
                zealot_spin *= -1

        # Handle any remaining steps without a zealot recalculation
        for _ in range(extra_steps):
            metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table, magnetization)

        magnetization = np.sum(lattice)
        magnetization_array[recalculation_index] = magnetization
                   
    return magnetization_array/N

def parallel_run_simulation(L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds, lookup_table):
    # Run each seed in parallel using joblib
    results = Parallel(n_jobs=1)(  # -1 uses all available cores
        delayed(run_simulation_for_seed)(
            seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, lookup_table
        ) for seed in seeds
    )
    return results

def create_lookup_table( temp, k_B, J_b, h_b, h_s, J_s):
    # Define the possible values of ΔE (based on neighbors and fields)
    possible_neighbor_sums = np.array([-4, -2, 0, 2, 4])  # Neighboring spins sum possibilities
    possible_internal_field= np.array([h_b,-h_b,0])
    possible_leader_influence= np.array([J_s,-J_s,0])
    possible_leader_field= np.array([h_s,-h_s,0])

    # Combine the two contributions to calculate all ΔE values
    delta_E_values = []
    for neighbor_sum in possible_neighbor_sums:
        for internal_field in possible_internal_field:
            for leader_influence in possible_leader_influence:
                for leader_field in possible_leader_field:
                    delta_E = -2 * ((J_b*neighbor_sum)+internal_field+leader_influence+leader_field)
                    delta_E_values.append(delta_E)

    # Remove duplicates and sort
    delta_E_values = np.unique(delta_E_values)

    # Precompute exponential values for all ΔE
    lookup_table = {delta_E: np.exp(-delta_E / (k_B * temp)) for delta_E in delta_E_values}

    return lookup_table

def run_simulation_for_temperature(temp, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds):
    if temp == 0:
     temp = 1e-10 
    lookup_table = create_lookup_table(temp, k_B, J_b, h_b, h_s, J_s)
    all_magnetizations = parallel_run_simulation(L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds, lookup_table)
    
    return temp, all_magnetizations

def run_simulation_over_temperatures(temperatures, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds):

    # Use joblib to run the simulation for each temperature in parallel
    results = Parallel(n_jobs=8)(  # -1 uses all available cores
        delayed(run_simulation_for_temperature)(
            temp, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds
        ) for temp in temperatures
    )
    
    # Convert the list of results into a dictionary with temperature as the key
    results_dict = {temp: magnetizations for temp, magnetizations in results}
    
    return results_dict

def post_process_results(all_magnetizations, burn_in=0):
    # Extract the final magnetizations directly using array slicing
    all_magnetizations = np.array(all_magnetizations)
    if burn_in > 0:
        all_magnetizations = all_magnetizations[:, burn_in:]
    final_magnetizations = all_magnetizations[:, -1]

    # Identify indices of runs ending with positive or negative magnetization
    positive_indices = final_magnetizations > 0
    negative_indices = final_magnetizations < 0
    
    # Divide the magnetization time series into two groups based on final magnetization
    m_plus = all_magnetizations[positive_indices]
    m_minus = all_magnetizations[negative_indices]
    
    # Compute the average magnetization across all runs and all times (after burn-in)
    average_magnetization_across_runs_p = np.mean(all_magnetizations)

    # Compute the average magnetization across all runs and all times for each group
    m_plus_avg_p = np.mean(m_plus) if m_plus.size > 0 else 0
    m_minus_avg_p = np.mean(m_minus) if m_minus.size > 0 else 0

    # Compute the fraction of runs ending with positive or negative magnetization
    total_runs = all_magnetizations.shape[0]
    g_plus = np.count_nonzero(positive_indices) / total_runs
    g_minus = np.count_nonzero(negative_indices) / total_runs
      

    return average_magnetization_across_runs_p, m_plus_avg_p, m_minus_avg_p, g_plus, g_minus

def plot_average_magnetization(average_magnetization):
    """
    Plots the average magnetization across multiple simulation runs over time.

    Parameters:
    - average_magnetization: A list or 1D NumPy array of average magnetization over time.
    """
    # Create a plot
    plt.figure(figsize=(10, 5))

    # Plot the average magnetization
    plt.plot(average_magnetization, label="Average Magnetization (across runs)")

    # Labeling the plot
    plt.xlabel("Monte Carlo Step")
    plt.ylabel("Average Magnetization")
    plt.title("Average Magnetization over Time (Averaged over Multiple Runs)")
    plt.legend()
    plt.grid(True)

    # Show the plot
    plt.show()

def plot_magnetization_over_time(all_magnetizations):
    """
    Plots magnetization over time for multiple simulation runs.

    Parameters:
    - all_magnetizations: List of lists or 2D array where each element is a magnetization time series.
    """
    # Convert all_magnetizations into a NumPy array for easier handling (if not already)
    all_magnetizations = np.array(all_magnetizations)

    # Create a plot
    plt.figure(figsize=(10, 6))

    # Loop through each magnetization time series in all_magnetizations
    for i, magnetization in enumerate(all_magnetizations):
        plt.plot(magnetization, label=f"seed {i}")

    # Labeling the plot
    plt.xlabel("Monte Carlo Steps")
    plt.ylabel("Magnetization")
    plt.title("Magnetization Over Time for Multiple Runs")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

def process_simulation_results_to_lists(simulation_results, burn_in=0):

    # Initialize lists for each category
    temperatures = []
    average_magnetizations = []
    m_plus_avgs = []
    m_minus_avgs = []
    g_plus_list = []
    g_minus_list = []

    # Process each temperature
    for temp, all_magnetizations in simulation_results.items():
        # Post-process the magnetization data for this temperature
        average_magnetization, m_plus_avg, m_minus_avg, g_plus, g_minus= post_process_results(
            all_magnetizations, burn_in=burn_in
        )

        # Append results to respective lists
        temperatures.append(temp)
        average_magnetizations.append(average_magnetization)
        m_plus_avgs.append(m_plus_avg)
        m_minus_avgs.append(m_minus_avg)
        g_plus_list.append(g_plus)
        g_minus_list.append(g_minus)

    # Compile the results into a dictionary
    results = {
        'temperatures': temperatures,
        'average_magnetizations': average_magnetizations,
        'm_plus_avgs': m_plus_avgs,
        'm_minus_avgs': m_minus_avgs,
        'g_plus_list': g_plus_list,
        'g_minus_list': g_minus_list,
    }

    return results

def plot_average_magnetizations_vs_temperature(results):
    """
    Plots the average magnetizations against temperatures.

    Parameters:
    - results: Dictionary containing processed results, including:
        - 'temperatures': List of temperatures.
        - 'average_magnetizations': List of average magnetizations corresponding to each temperature.
    """
    # Extract temperatures and average magnetizations
    temperatures = results['temperatures']
    average_magnetizations = results['average_magnetizations']

    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, average_magnetizations, marker="o", label="Average Magnetization", color="blue")

    # Add labels, title, and grid
    plt.xlabel("Temperature")
    plt.ylabel("Average Magnetization")
    plt.title("Average Magnetization vs. Temperature")
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_g_plus_minus_vs_temperature(results):
    """
    Plots g_plus_list and g_minus_list against temperatures on the same graph.

    Parameters:
    - results: Dictionary containing processed results, including:
        - 'temperatures': List of temperatures.
        - 'g_plus_list': List of fractions of runs ending in positive magnetization.
        - 'g_minus_list': List of fractions of runs ending in negative magnetization.
    """
    # Extract data
    temperatures = results['temperatures']
    g_plus_list = results['g_plus_list']
    g_minus_list = results['g_minus_list']

    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, g_plus_list, marker="o", label="$g_+$ (Fraction Positive)", color="green")
    plt.plot(temperatures, g_minus_list, marker="s", label="$g_-$ (Fraction Negative)", color="red")

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
    """
    Plots m_plus_avgs and m_minus_avgs against temperatures on the same graph.

    Parameters:
    - results: Dictionary containing processed results, including:
        - 'temperatures': List of temperatures.
        - 'm_plus_avgs': List of average magnetizations for runs ending in positive magnetization.
        - 'm_minus_avgs': List of average magnetizations for runs ending in negative magnetization.
    """
    # Extract data
    temperatures = results['temperatures']
    m_plus_avgs = results['m_plus_avgs']
    m_minus_avgs = results['m_minus_avgs']

    # Create a plot
    plt.figure(figsize=(8, 5))
    plt.plot(temperatures, m_plus_avgs, marker="o", label="$m_+$ (Average Positive Magnetization)", color="blue")
    plt.plot(temperatures, m_minus_avgs, marker="s", label="$m_-$ (Average Negative Magnetization)", color="orange")

    # Add labels, title, and grid
    plt.xlabel("Temperature")
    plt.ylabel("Average Magnetization")
    plt.title("$m_+$ and $m_-$ vs. Temperature")
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()



import itertools

# [Previous helper functions remain unchanged up to run_simulation_for_seed]

def run_simulation_for_temp_seed(temp_seed_pair, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps):
    """
    Run simulation for a single temperature-seed combination
    """
    temp, seed = temp_seed_pair
    if temp == 0:
        temp = 1e-10
    lookup_table = create_lookup_table(temp, k_B, J_b, h_b, h_s, J_s)
    result = run_simulation_for_seed(seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, 
                                   zealot_spin, num_iterations, number_of_MC_steps, lookup_table)
    return temp, seed, result

def run_simulation_over_temperatures(temperatures, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds):
    # Create all temperature-seed combinations
    temp_seed_combinations = list(itertools.product(temperatures, seeds))
    
    # Run all combinations in parallel
    results = Parallel(n_jobs=-1, backend='loky')(
        delayed(run_simulation_for_temp_seed)(
            temp_seed_pair, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, 
            num_iterations, number_of_MC_steps
        ) for temp_seed_pair in temp_seed_combinations
    )
    
    # Reorganize results into the expected dictionary format
    results_dict = {}
    for temp, seed, result in results:
        if temp not in results_dict:
            results_dict[temp] = []
        results_dict[temp].append(result)
    
    return results_dict





# Parameters
L = 50  # Size of the lattice (LxL)
N=L**2
zealot_spin = 1
k_B = 1     #.380649e-23  # Boltzmann constant
num_iterations = N*10  # Total number of iterations
J_b = 1/4  # Coupling constant
J_s = 1.01
h_b= -1
h_s = N
seeds = list(np.linspace(1,20,20, dtype = int))
number_of_MC_steps = 2
temperatures = list(np.linspace(0.1,1.5,30))
burn_in_steps = int((num_iterations/(number_of_MC_steps*N))*0.5)
seeds
temperatures

start = time.time()

# Run the simulation for all temperatures in parallel
simulation_results = run_simulation_over_temperatures(temperatures, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds)

end = time.time()
length = end - start
print("It took", length, "seconds!")


all_magnetizations_p = simulation_results[1]
plot_magnetization_over_time(all_magnetizations_p)

processed_results_lists = process_simulation_results_to_lists(simulation_results, burn_in=burn_in_steps)
plot_average_magnetizations_vs_temperature(processed_results_lists)
plot_m_plus_minus_vs_temperature(processed_results_lists)
plot_g_plus_minus_vs_temperature(processed_results_lists)




create_lookup_table( 1, k_B, J_b, h_b, h_s, J_s)



