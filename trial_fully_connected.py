import numpy as np
from joblib import Parallel, delayed


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
    delta_E = -2*(social_influence+internal_field+leader_influence)
    # Prevent overflow by limiting very large values
    return np.clip(delta_E, -700, 700)  # exp(±700) is near the limits of float64

def calculate_energy_change_zealot(zealot_spin, magnetization, L, J_s, h_s):
    """
    Calculate the energy change that would occur if the zealot spin is flipped.
    """

    leader_field = -h_s*zealot_spin
    leader_influence = -J_s*zealot_spin*magnetization 
    delta_E = -2*(leader_influence+leader_field)
    # Prevent overflow by limiting very large values
    return np.clip(delta_E, -700, 700)  # exp(±700) is near the limits of float64

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
    for interval in range(num_intervals):
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
     temp = 1e-10 
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


# Run the simulation for all temperatures in parallel
simulation_results = run_simulation_over_temperatures(temperatures, L, N, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, seeds)
 

processed_results_lists = process_simulation_results_to_lists(simulation_results, burn_in_steps)


processed_results_lists