import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

# Parameters
np.random.seed(10)
L = 500  # Size of the lattice (LxL)
temp = 5*10**24  
k_B = 1.380649 * 10**-23  # Boltzmann constant
num_iterations = 10000000  # Total number of iterations
J = 100  # Coupling constant
snapshot_interval = 10000  # Save lattice every 20 steps for animation

# Initialize the lattice with random spins (-1 or +1)
lattice = np.random.choice([-1, 1], size=(L, L))

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


def calculate_energy_change(lattice, i, j, J):
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
    E_1 = -J/2*neighbor_sum
    return -2*E_1


def metropolis_step(lattice, L, J, temp, k_B):
    """
    Perform one Monte Carlo step on the lattice.
    """
    # Step 2: Choose a random site (i, j)
    i, j = np.random.randint(0, L), np.random.randint(0, L)

    # Step 3: Calculate the energy change ΔE if the spin at (i, j) is flipped
    delta_E = calculate_energy_change(lattice, i, j, J)

    # Step 4: Generate a random number r such that 0 < r < 1
    r = np.random.rand()

    # Step 5: If r < exp(-ΔE / (k_B * T)), flip the spin
    if r < np.exp(-delta_E / (k_B * temp)):
        lattice[i, j] *= -1  # Flip the spin at site (i, j)


# Main simulation loop with snapshot saving
for step in tqdm(range(num_iterations)):
    metropolis_step(lattice, L, J, temp, k_B)
    
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




