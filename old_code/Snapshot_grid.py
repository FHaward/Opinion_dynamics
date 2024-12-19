import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

def calculate_energy_change(lattice, L, i, j, J_b, h_b,J_s, sigma_s):
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
    leader_influence= -J_s*sigma_s*spin
    return -2*(social_influence+internal_field+leader_influence)

def calculate_energy_change_zealot(lattice, L, i, j, J_b, h_s):
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
    leader_field = -h_s*spin
    return -2*(social_influence+leader_field)

def metropolis_step(lattice, L, J_b, h_b, h_s, J_s, temp, k_B, sigma_s):
    """
    Perform one Monte Carlo step on the lattice.
    """
    # Step 2: Choose a random site (i, j)
    i, j = np.random.randint(0, L), np.random.randint(0, L)

    if (i, j) == (L // 2, L // 2):  # Check if the zealot spin is selected
        delta_E = calculate_energy_change_zealot(lattice, L, i, j, J_b, h_s)

        # Step 4: Generate a random number r such that 0 < r < 1
        r = np.random.rand()

        # Step 5: If r < exp(-ΔE / (k_B * T)), flip the spin
        if r < np.exp(-delta_E / (k_B * temp)):
            lattice[i, j] *= -1  # Flip the spin at site (i, j)
            # Update sigma_s only if the zealot spin actually changed
            if lattice[i, j] != sigma_s:
                sigma_s = lattice[i, j]

    else:
        # Step 3: Calculate the energy change ΔE if the spin at (i, j) is flipped
        delta_E = calculate_energy_change(lattice, L, i, j, J_b, h_b, J_s, sigma_s)

        # Step 4: Generate a random number r such that 0 < r < 1
        r = np.random.rand()

        # Step 5: If r < exp(-ΔE / (k_B * T)), flip the spin
        if r < np.exp(-delta_E / (k_B * temp)):
            lattice[i, j] *= -1  # Flip the spin at site (i, j)

    return sigma_s  # Return updated sigma_s if changed, else the original

# Simulation Parameters
np.random.seed(10)
L = 100
temp = 6 * 10**22
k_B = 1.380649e-23
num_iterations = (L**2) * 100
J_b = 1
J_s = 0
h_s = 0
snapshot_intervals = L**2*np.array([0, 10, 20, 30, 40])  # Steps at which we take snapshots

# Range of h_b values
h_b_values = [-0.05, -0.025, 0, 0.025, 0.05]

# Prepare a dictionary to store snapshots for each h_b
snapshot_dict = {h_b: [] for h_b in h_b_values}

# Simulation Function
def run_simulation(h_b):
    lattice = np.random.choice([-1, 1], size=(L, L))
    sigma_s = lattice[L // 2, L // 2]
    snapshots = []

    # Simulation loop with snapshot saving
    for step in tqdm(range(num_iterations)):
        sigma_s = metropolis_step(lattice, L, J_b, h_b, h_s, J_s, temp, k_B, sigma_s)
        
        # Save snapshots at specified intervals
        if step in snapshot_intervals:
            snapshots.append(lattice.copy())
    return snapshots

# Run simulations for each h_b and store the results
for h_b in h_b_values:
    snapshot_dict[h_b] = run_simulation(h_b)

# Plotting the grid of snapshots
fig, axs = plt.subplots(len(h_b_values), len(snapshot_intervals), figsize=(15, 15))
fig.suptitle("Effect of Different $h_b$ Values on Lattice Evolution", fontsize=16)

for row, h_b in enumerate(h_b_values):
    for col, interval in enumerate(snapshot_intervals):
        ax = axs[row, col]
        ax.imshow(snapshot_dict[h_b][col], cmap="coolwarm", interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        if col == 0:
            ax.set_ylabel(f"$h_b$ = {h_b}", fontsize=12)
        if row == 0:
            ax.set_title(f"MC Step = {interval/(L**2)}", fontsize=12)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()


