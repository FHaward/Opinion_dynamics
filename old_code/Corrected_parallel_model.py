import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
from joblib import Parallel, delayed


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

def calculate_energy_change_zealot(zealot_spin, magnetization, J_s, h_s):
    """
    Calculate the energy change that would occur if the zealot spin is flipped.
    """

    leader_field = -h_s*zealot_spin
    leader_influence = -J_s*zealot_spin*magnetization/(L**2)  # Should magnetization be normalized somehow here?
    return -2*(leader_influence+leader_field)

def metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table):
    """
    Perform one Monte Carlo step on the lattice.
    """

    i, j = np.random.randint(0, L), np.random.randint(0, L)
    delta_E = calculate_energy_change(lattice, L, i, j, J_b, h_b, J_s, zealot_spin)
    
    if np.random.rand() < lookup_table.get(delta_E, 0):
        lattice[i, j] *= -1  # Flip the spin
    
def run_simulation_for_seed(seed, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, recalculation_interval, lookup_table):
    np.random.seed(seed)
    lattice = np.random.choice([-1, 1], size=(L, L))

    # Preallocate the array for magnetization values
    num_intervals = num_iterations // recalculation_interval
    total_recalculations = num_intervals + 1 + (1 if num_iterations % recalculation_interval > 0 else 0)
    magnetization_array = np.zeros(total_recalculations, dtype=np.float64)
    magnetization_array[0] = np.sum(lattice)
    recalculation_index = 1

    # Loop over each interval
    for interval in tqdm(range(num_intervals), desc=f"Seed {seed}"):
        for step in range(recalculation_interval):
            metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table)


        magnetization = np.sum(lattice)
        magnetization_array[recalculation_index] = magnetization
        recalculation_index += 1
        
        delta_E_zealot = calculate_energy_change_zealot(zealot_spin, magnetization, J_s, h_s)
        if np.random.rand() < np.exp(-delta_E_zealot / (k_B * temp)):
            zealot_spin *= -1  # Flip the spin

    # Handle any remaining steps if num_iterations is not an exact multiple of recalculation_interval
    remaining_steps = num_iterations % recalculation_interval
    if remaining_steps > 0:
        for step in range(remaining_steps):
            metropolis_step(lattice, L, J_b, h_b, h_s, J_s, zealot_spin, lookup_table)
        # Final recalculation for any remaining steps
        magnetization_array[recalculation_index] = np.sum(lattice)

                   
    return magnetization_array / (L * L)

def parallel_run_simulation(L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, recalculation_interval, seeds, lookup_table):
    # Run each seed in parallel using joblib
    results = Parallel(n_jobs=-1)(  # -1 uses all available cores
        delayed(run_simulation_for_seed)(
            seed, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, recalculation_interval, lookup_table
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
    
    # Compute the average magnetization across runs for each group
    m_plus_avg_p = np.mean(m_plus, axis=0) if m_plus.size > 0 else np.zeros(all_magnetizations.shape[1])
    m_minus_avg_p = np.mean(m_minus, axis=0) if m_minus.size > 0 else np.zeros(all_magnetizations.shape[1])
    
    # Compute the fraction of runs ending with positive or negative magnetization
    total_runs = all_magnetizations.shape[0]
    g_plus = np.count_nonzero(positive_indices) / total_runs
    g_minus = np.count_nonzero(negative_indices) / total_runs
      
    average_magnetization_across_runs_p = np.mean(all_magnetizations, axis=0)
    

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
        plt.plot(magnetization, label=f"Run {i+1}")

    # Labeling the plot
    plt.xlabel("Monte Carlo Steps")
    plt.ylabel("Magnetization")
    plt.title("Magnetization Over Time for Multiple Runs")
    plt.legend(loc="upper right", fontsize=8)
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Parameters
L = 50  # Size of the lattice (LxL)
N = L**2
zealot_spin = 1
temp = 1 #6e22  
k_B = 1    #.380649e-23  # Boltzmann constant
num_iterations = N*100  # Total number of iterations
J_b = 1  # Coupling constant
J_s = 1.01
h_b= -1
h_s = 1
seeds = np.linspace(1,8,8).astype(int).tolist()
recalculation_interval = 2*N



lookup_table = create_lookup_table(temp, k_B, J_b, h_b, h_s, J_s,)
# Run simulations in parallel with individual progress bars
all_magnetizations_p = parallel_run_simulation(L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, recalculation_interval, seeds, lookup_table)
# Post-process the results
average_magnetization_across_runs_p, m_plus_avg_p, m_minus_avg_p, g_plus_p, g_minus_p = post_process_results(all_magnetizations_p)

plot_average_magnetization(average_magnetization_across_runs_p)
plot_magnetization_over_time(all_magnetizations_p)

