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

def metropolis_step(lattice, L, J_b, h_b, h_s, J_s, temp, k_B, sigma_s, total_magnetization):
    """
    Perform one Monte Carlo step on the lattice.
    """
    # Step 2: Choose a random site (i, j)
    i, j = np.random.randint(0, L), np.random.randint(0, L)
    previous_spin = lattice[i, j]  # Save the previous spin value

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
    
    # Update total magnetization if the spin at (i, j) was flipped
    if lattice[i, j] != previous_spin:
        total_magnetization += 2 * lattice[i, j]  # Incremental update for magnetization

    return sigma_s, total_magnetization  # Return updated sigma_s if changed, else the original


# Parameters
np.random.seed(46)
L = 50  # Size of the lattice (LxL)
N=L**2
k_B = 1.380649 * 10**-23  # Boltzmann constant
temp = 1  #avoid division by zero error
num_iterations = N*100  # Total number of iterations
J_b = 1  # Coupling constant
J_s = 1.01
h_b = -1
h_s = 1
snapshot_interval = 2*N  # Save lattice every 20 steps for animation



# Initialize the lattice with_brandom spins (-1 or +1)
lattice = np.random.choice([-1, 1], size=(L, L))

# Make Zealot spin start in direction of zealot field without biasing spin distribution
if lattice[(L//2),(L//2)] == -1:
    lattice=-lattice
# Initial setup for sigma_s outside the function
sigma_s = lattice[L // 2, L // 2]  # Initial zealot spin value

# List to store lattice snapshots for animation
lattice_snapshots = []

# Calculate the initial total magnetization
total_magnetization = np.sum(lattice)
average_magnetization = total_magnetization / (L * L)

# List to store average magnetization at each step
average_magnetization_list = [average_magnetization]

# Visualize the lattice
plt.figure(figsize=(6, 6))
plt.imshow(lattice, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Spin")
plt.title("Initial Spin Configuration of Lattice")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.show()

# Main simulation loop with snapshot saving
for step in tqdm(range(num_iterations)):
    sigma_s, total_magnetization = metropolis_step(lattice, L, J_b, h_b, h_s, J_s, temp, k_B, sigma_s, total_magnetization)  # Update sigma_s and total_magnetization

    # Calculate and store the average magnetization
    average_magnetization = total_magnetization / (L * L)
    average_magnetization_list.append(average_magnetization)
        
    # Save the lattice snapshot every 'snapshot_interval' steps
    if step % snapshot_interval == 0:
        lattice_snapshots.append(lattice.copy())
        
       
# Visualize the lattice after the Monte Carlo simulation
plt.figure(figsize=(6, 6))
plt.imshow(lattice, cmap="coolwarm", interpolation="nearest")
plt.colorbar(label="Spin")
plt.title("Final Spin Configuration of Lattice")
plt.xlabel("X position")
plt.ylabel("Y position")
plt.show()


# Set up the figure for animation
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(lattice_snapshots[0], cmap="coolwarm", interpolation="nearest")
ax.set_title("Evolution of Spin Configuration")
ax.set_xlabel("X position")
ax.set_ylabel("Y position")
plt.colorbar(im, label="Spin")


def update(frame):
    """
    Update function for the animation to show each snapshot.
    """
    im.set_array(lattice_snapshots[frame])
    return [im]


# Create the animation using the saved snapshots
ani = animation.FuncAnimation(
    fig, update, frames=len(lattice_snapshots), blit=True, interval=50, repeat=False
)

# Show the animation
plt.show()












# Plot the average magnetization over time
plt.figure(figsize=(10, 5))
plt.plot(average_magnetization_list, label="Average Magnetization")
plt.xlabel("Monte Carlo Step")
plt.ylabel("Average Magnetization")
plt.title("Average Magnetization over Time")
plt.legend()
plt.grid(True)
plt.show()

# Plot the initial and final lattice configurations for comparison
fig, axs = plt.subplots(1, 2, figsize=(12, 6))

# Initial lattice configuration
axs[0].imshow(lattice_snapshots[0], cmap="coolwarm", interpolation="nearest")
axs[0].set_title("Initial Lattice Configuration")
axs[0].set_xlabel("X Position")
axs[0].set_ylabel("Y Position")

# Final lattice configuration
axs[1].imshow(lattice_snapshots[-1], cmap="coolwarm", interpolation="nearest")
axs[1].set_title("Final Lattice Configuration")
axs[1].set_xlabel("X Position")
axs[1].set_ylabel("Y Position")

plt.colorbar(axs[1].images[0], ax=axs, location='right', label="Spin")
plt.tight_layout()
plt.show()
