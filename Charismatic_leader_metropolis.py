import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

def calculate_energy_change(lattice, i, j, J_b, h_b,J_s, sigma_s):
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

def calculate_energy_change_zealot(lattice, i, j, J_b, h_s):
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

def metropolis_step(lattice, L, J_b, temp, k_B, sigma_s):
    """
    Perform one Monte Carlo step on the lattice.
    """
    # Step 2: Choose a random site (i, j)
    i, j = np.random.randint(0, L), np.random.randint(0, L)

    if (i, j) == (L // 2, L // 2):  # Check if the zealot spin is selected
        delta_E = calculate_energy_change_zealot(lattice, i, j, J_b, h_b)

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
        delta_E = calculate_energy_change(lattice, i, j, J_b, h_b, J_s, sigma_s)

        # Step 4: Generate a random number r such that 0 < r < 1
        r = np.random.rand()

        # Step 5: If r < exp(-ΔE / (k_B * T)), flip the spin
        if r < np.exp(-delta_E / (k_B * temp)):
            lattice[i, j] *= -1  # Flip the spin at site (i, j)

    return sigma_s  # Return updated sigma_s if changed, else the original


# Parameters
np.random.seed(10)
L = 100  # Size of the lattice (LxL)
temp = 6*10**22  
k_B = 1.380649 * 10**-23  # Boltzmann constant
num_iterations = (L**2)*100  # Total number of iterations
J_b = 1  # Coupling constant
J_s = 0
h_b= 0
h_s = 0
snapshot_interval = 1000  # Save lattice every 20 steps for animation



# Initialize the lattice with_brandom spins (-1 or +1)
lattice = np.random.choice([-1, 1], size=(L, L))

# Make Zealot spin start in direction of zealot field without biasing spin distribution
if lattice[(L//2),(L//2)] == -1:
    lattice=-lattice
# Initial setup for sigma_s outside the function
sigma_s = lattice[L // 2, L // 2]  # Initial zealot spin value

# List to store lattice snapshots for animation
lattice_snapshots = []

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
    sigma_s = metropolis_step(lattice, L, J_b, temp, k_B, sigma_s)  # Pass and update sigma_s
    
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



calculate_energy_change(lattice, 50, 50, J_b, h_b,J_s, sigma_s)