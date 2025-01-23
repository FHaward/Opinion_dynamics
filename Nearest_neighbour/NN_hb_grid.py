import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm
import os

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
save_dir = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Formative Presentation\vs graphs"


# Define h_b values to test
h_b_values = [-0.05, -0.025, 0.0, 0.025, 0.05]
frame_indices = [0, 2, 5, 10, 20]

# Create a figure with a grid of subplots
fig, axes = plt.subplots(len(h_b_values), len(frame_indices), figsize=(15, 15))

# Run simulations and create visualizations for each h_b value
for i, h_b in enumerate(h_b_values):
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
            ax.set_ylabel(f'h = {h_b}', fontsize=16)

# Add a colorbar
cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
fig.colorbar(im, cax=cbar_ax, label='Spin')
cbar_ax.tick_params(labelsize=16)


plt.tight_layout()
plt.subplots_adjust(right=0.9)
plt.show()



def generate_simulation_data(L=100, N=None, zealot_spin=1, k_B=1, num_iterations=None,
                           J_b=1, J_s=0, h_s=0, number_of_MC_steps=2, seed=10, 
                           temp=0.5, h_b_values=None):
    """
    Generate simulation data for multiple h_b values.
    
    Returns:
    dict: Contains simulation results for each h_b value with structure:
        {h_b: (magnetization_array, lattice_snapshots)}
    """
    if N is None:
        N = L**2
    if num_iterations is None:
        num_iterations = N*200
    if h_b_values is None:
        h_b_values = [-0.05, -0.025, 0.0, 0.025, 0.05]
        
    simulation_results = {}
    
    for h_b in h_b_values:
        # Create lookup table and run simulation
        lookup_table = create_lookup_table(temp, k_B, J_b, h_b, h_s, J_s)
        magnetization, lattice_snapshots = run_simulation_with_snapshots(
            seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, 
            num_iterations, number_of_MC_steps, lookup_table
        )
        simulation_results[h_b] = (magnetization, lattice_snapshots)
    
    return simulation_results
def plot_simulation_results(simulation_results, frame_indices=None, cmap=None):
    """
    Create visualization of simulation results with consistent spacing and adjustments.
    
    Parameters:
    simulation_results : dict
        Output from generate_simulation_data()
    frame_indices : list, optional
        Indices of frames to display (default is DEFAULT_FRAME_INDICES)
    cmap : str, optional
        Colormap for the visualization (default is DEFAULT_CMAP)
    """
    if frame_indices is None:
        frame_indices = DEFAULT_FRAME_INDICES
    if cmap is None:
        cmap = DEFAULT_CMAP

    h_b_values = sorted(simulation_results.keys())

    # Create figure with subplots
    fig, axes = plt.subplots(len(h_b_values), len(frame_indices),
                             figsize=FIGSIZE, constrained_layout=True)
    fig.suptitle('Lattice Evolution for Different h_b Values', 
                 fontsize=TITLE_SIZE, y=TITLE_Y_POS)  # Adjusted title position

    # Plot snapshots for each h_b value and frame
    for i, h_b in enumerate(h_b_values):
        magnetization, lattice_snapshots = simulation_results[h_b]

        for j, frame in enumerate(frame_indices):
            ax = axes[i, j]
            im = ax.imshow(lattice_snapshots[frame], cmap=cmap, 
                          interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])

            # Add titles only to top row and left column
            if i == 0:
                ax.set_title(f'Step {frame}', fontsize=LABEL_SIZE, pad=10)
            if j == 0:
                ax.set_ylabel(f'h_b = {h_b}', fontsize=LABEL_SIZE, labelpad=10)

    # Add colorbar with consistent sizing
    cbar_ax = fig.add_axes(CBAR_AX_POSITION)  # Adjusted for additional white space
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.ax.set_ylabel('Spin', fontsize=LABEL_SIZE, rotation=270, labelpad=20)
    cbar.ax.tick_params(labelsize=LABEL_SIZE)

    # Adjust spacing for uniform subplot layout
    fig.subplots_adjust(wspace=SUBPLOT_WSPACE, hspace=SUBPLOT_HSPACE, 
                        left=SUBPLOT_LEFT, right=SUBPLOT_RIGHT, 
                        top=SUBPLOT_TOP, bottom=SUBPLOT_BOTTOM)

    return fig, axes




# Adjustable parameters
TITLE_SIZE = 16
LABEL_SIZE = 14
FIGSIZE = (16, 16)
TITLE_Y_POS = 0.95
CBAR_AX_POSITION = [0.92, 0.15, 0.02, 0.7]
SUBPLOT_WSPACE = 0.3
SUBPLOT_HSPACE = 0.3
SUBPLOT_LEFT = 0.08
SUBPLOT_RIGHT = 0.9
SUBPLOT_TOP = 0.9
SUBPLOT_BOTTOM = 0.08
DEFAULT_FRAME_INDICES = [0, 2, 5, 10, 20]
DEFAULT_CMAP = "coolwarm"


simulation_results = generate_simulation_data()

fig, axes = plot_simulation_results(simulation_results)
save_path = os.path.join(save_dir, "NN_h_b_grid.png")
plt.savefig(save_path)
plt.show()