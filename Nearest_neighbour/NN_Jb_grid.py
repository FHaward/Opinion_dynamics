import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

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

def run_simulation_with_snapshots(seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps, lookup_table):
    np.random.seed(seed)
    lattice = np.random.choice([-1, 1], size=(L, L))
    magnetization_record_interval = number_of_MC_steps * N
    lattice_snapshots = [lattice.copy()]  # Initialize with the starting configuration


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
        
        lattice_snapshots.append(lattice.copy())
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

        lattice_snapshots.append(lattice.copy())
        magnetization = np.sum(lattice)
        magnetization_array[recalculation_index] = magnetization
                   
    return magnetization_array/N, lattice_snapshots


# Parameters
L = 100
N = L**2
zealot_spin = 1
k_B = 1
num_iterations = N*200  # Total number of iterations
J_b = 1
J_s = 0
h_b = 0
h_s = 0
number_of_MC_steps = 2
seed = 10
temp = 0.5


# Define h_b values to test
J_b_values = [0.1, 0.25, 0.5, 0.75, 1]
frame_indices = [0, 5, 10, 15, 20]


# Create a figure with a grid of subplots
fig, axes = plt.subplots(len(J_b_values), len(frame_indices), figsize=(15, 15))

# Run simulations and create visualizations for each J_b value
for i, J_b in enumerate(J_b_values):
    # Create lookup table and run simulation
    lookup_table = create_lookup_table(temp, k_B, J_b, h_b, h_s, J_s)
    magnetization, lattice_snapshots = run_simulation_with_snapshots(
        seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, 
        num_iterations, number_of_MC_steps, lookup_table
    )
    
    # Plot snapshots at specified frames
    for j, frame in enumerate(frame_indices):
        ax = axes[i, j]
        im = ax.imshow(lattice_snapshots[frame], cmap="coolwarm", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add titles only to top row and left column
        if i == 0:
            ax.set_title(f'Step {frame}', fontsize=16)
        if j == 0:
            ax.set_ylabel(f'J = {J_b}', fontsize=16)

# Add a colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Spin')
cbar_ax.tick_params(labelsize=16)

plt.tight_layout()
plt.subplots_adjust(right=0.9)
plt.show()