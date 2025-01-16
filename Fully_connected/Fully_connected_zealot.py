import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from joblib import Parallel, delayed
import tracemalloc

tracemalloc.start()


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

def metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization):
    """
    Perform one Metropolis step on the lattice and update the magnetization.
    """
    i, j = np.random.randint(0, L), np.random.randint(0, L)
    delta_E = calculate_energy_change(lattice, L, i, j, J_b, h_b, J_s, zealot_spin, magnetization)
    
    if np.random.rand() < np.exp(-delta_E / (k_B * temp)):
        lattice[i, j] *= -1  # Flip the spin
        magnetization += 2*lattice[i, j]

    return magnetization
    
def run_simulation_for_seed(seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps):
    np.random.seed(seed)
    lattice = np.random.choice([-1, 1], size=(L, L))
    magnetization_record_interval = number_of_MC_steps * N
    magnetization = np.sum(lattice)  # Initial magnetization

    # Preallocate the array for magnetization values
    num_intervals = num_iterations // magnetization_record_interval
    total_recalculations = num_intervals + 1 + (1 if num_iterations % magnetization_record_interval > 0 else 0)
    magnetization_array = np.zeros(total_recalculations, dtype=np.float64)
    magnetization_array[0] = magnetization
    recalculation_index = 1

    # Loop over each interval
    for interval in tqdm(range(num_intervals), desc=f"Temp {temp}, Seed {seed}"):
        for step in range(number_of_MC_steps):
            for _ in range(N): 
                magnetization = metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

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
                magnetization = metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

            # Update zealot spin every L^2 steps
            delta_E_zealot = calculate_energy_change_zealot(zealot_spin, magnetization, L, J_s, h_s)
            if np.random.rand() < np.exp(-delta_E_zealot / (k_B * temp)):
                zealot_spin *= -1

        # Handle any remaining steps without a zealot recalculation
        for _ in range(extra_steps):
            magnetization = metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

        # Final recalculation for the last segment
        magnetization_array[recalculation_index] = magnetization

                   
    return magnetization_array/N

def parallel_run_simulation(L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds):
    # Run each seed in parallel using joblib
    results = Parallel(n_jobs=-1)(  # -1 uses all available cores
        delayed(run_simulation_for_seed)(
            seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps
        ) for seed in seeds
    )
    return results

def run_simulation_for_temperature(temp, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds):
    if temp == 0:
     temp = 1e-3 
    all_magnetizations = parallel_run_simulation(L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds)
    
    return temp, all_magnetizations

def run_simulation_over_temperatures(temperatures, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds):

    # Use joblib to run the simulation for each temperature in parallel
    results = Parallel(n_jobs=-1)(  # -1 uses all available cores
        delayed(run_simulation_for_temperature)(
            temp, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds
        ) for temp in temperatures
    )
    
    # Convert the list of results into a dictionary with temperature as the key
    results_dict = {temp: magnetizations for temp, magnetizations in results}
    
    return results_dict

def post_process_results(all_magnetizations, burn_in_steps):
    # Extract the final magnetizations directly using array slicing
    all_magnetizations = np.array(all_magnetizations)
    if burn_in_steps > 0:
        all_magnetizations = all_magnetizations[:, burn_in_steps:]
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

def process_simulation_results_to_lists(simulation_results, burn_in_steps):

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
            all_magnetizations, burn_in_steps
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
    plt.xlabel("Temperature", fontsize=14)
    plt.ylabel("Average Magnetization", fontsize=14)
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()


# Parameters
L = 50  # Size of the lattice (LxL)
N=L**2
zealot_spin = 1
k_B = 1     #.380649e-23  # Boltzmann constant
num_iterations = N*50  # Total number of iterations
J_b = 1/(N-1)  # Coupling constant
J_s = 1.01
h_b= -1
h_s = N
number_of_MC_steps = 2
seeds = np.linspace(1,4,4).astype(int).tolist()
temperatures = np.linspace(0,1.5,4).tolist()
burn_in_steps = int((num_iterations/(number_of_MC_steps*N))*0.5)

temperatures
# Run the simulation for all temperatures in parallel
simulation_results = run_simulation_over_temperatures(temperatures, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds)
 
all_magnetizations_p = simulation_results[1e-3]
plot_magnetization_over_time(all_magnetizations_p)

processed_results_lists = process_simulation_results_to_lists(simulation_results, burn_in_steps)
plot_average_magnetizations_vs_temperature(processed_results_lists)
plot_m_plus_minus_vs_temperature(processed_results_lists)
plot_g_plus_minus_vs_temperature(processed_results_lists)


# Measure memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.2f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.2f} MB")

tracemalloc.stop()

processed_results_lists = {'temperatures': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.44999999999999996, 0.5, 0.5499999999999999, 0.6, 0.6499999999999999, 0.7, 0.7499999999999999, 0.7999999999999999, 0.8499999999999999, 0.8999999999999999, 0.95, 0.9999999999999999, 1.05, 1.0999999999999999, 1.15, 1.2, 1.25, 1.3, 1.35, 1.4, 1.45, 1.5], 'average_magnetizations': [np.float64(0.3), np.float64(0.09999714285714287), np.float64(0.3999457142857143), np.float64(0.2996257142857143), np.float64(0.29847714285714283), np.float64(0.39551714285714284), np.float64(0.4897142857142857), np.float64(0.6826142857142857), np.float64(0.48063142857142854), np.float64(0.7503314285714286), np.float64(0.7295399999999999), np.float64(0.3539771428571428), np.float64(0.588957142857143), np.float64(0.3977542857142858), np.float64(0.36994285714285713), np.float64(0.46163714285714286), np.float64(0.5059228571428571), np.float64(0.4312000000000001), np.float64(0.29695714285714286), np.float64(0.16223142857142858), np.float64(0.09267714285714285), np.float64(0.062351428571428565), np.float64(0.04607142857142857), np.float64(0.03783999999999999), np.float64(0.02911142857142857), np.float64(0.02169714285714286), np.float64(0.02106857142857143), np.float64(0.024785714285714286), np.float64(0.024414285714285715)], 'm_plus_avgs': [np.float64(1.0), np.float64(0.9999948051948052), np.float64(0.9999224489795919), np.float64(0.9994241758241759), np.float64(0.9976571428571431), np.float64(0.993595918367347), np.float64(0.9859999999999998), np.float64(0.9748033613445378), np.float64(0.9593104761904763), np.float64(0.9374317460317461), np.float64(0.9107492063492064), np.float64(0.8765959183673468), np.float64(0.8381848739495799), np.float64(0.7856380952380951), np.float64(0.723862857142857), np.float64(0.6511932773109242), np.float64(0.5577233082706767), np.float64(0.4312000000000001), np.float64(0.29695714285714286), np.float64(0.16223142857142858), np.float64(0.09827226890756302), np.float64(0.06603609022556392), np.float64(0.04607142857142857), np.float64(0.04280380952380952), np.float64(0.030723809523809528), np.float64(0.02510204081632653), np.float64(0.022706122448979592), np.float64(0.024023529411764707), np.float64(0.02337142857142857)], 'm_minus_avgs': [np.float64(-1.0), np.float64(-1.0), np.float64(-1.0), np.float64(-1.0), np.float64(-1.0), np.float64(-1.0), np.float64(-0.9991428571428571), np.float64(-0.9731238095238094), np.float64(-0.9554057142857145), np.float64(-0.9335714285714285), np.float64(-0.9013428571428571), np.float64(-0.8654666666666666), np.float64(-0.8233333333333335), np.float64(-0.7658971428571427), np.float64(-0.6918171428571428), np.float64(-0.6125142857142857), np.float64(-0.47828571428571426), 0, 0, 0, np.float64(0.06097142857142857), np.float64(-0.007657142857142858), 0, np.float64(0.022948571428571426), np.float64(0.014599999999999997), np.float64(0.01375238095238095), np.float64(0.01724761904761905), np.float64(0.021200000000000004), np.float64(0.030323809523809524)], 'g_plus_list': [0.65, 0.55, 0.7, 0.65, 0.65, 0.7, 0.75, 0.85, 0.75, 0.9, 0.9, 0.7, 0.85, 0.75, 0.75, 0.85, 0.95, 1.0, 1.0, 1.0, 0.85, 0.95, 1.0, 0.75, 0.9, 0.7, 0.7, 0.85, 0.85], 'g_minus_list': [0.35, 0.45, 0.3, 0.35, 0.35, 0.3, 0.25, 0.15, 0.25, 0.1, 0.1, 0.3, 0.15, 0.25, 0.25, 0.15, 0.05, 0.0, 0.0, 0.0, 0.15, 0.05, 0.0, 0.25, 0.1, 0.3, 0.3, 0.1, 0.15]}
