import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 100  # Size of the lattice (LxL)
temperature = 1.0  
k_B = 1.380649 * 10**-23  # Boltzmann constant
num_iterations = 100000  # Number of iterations
J = 1.0  # Coupling constant

# Initialize the lattice with random spins (-1 or +1)
lattice = np.random.choice([-1, 1], size=(L, L))

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
    delta_E = -J / 2 * neighbor_sum
    return delta_E

# Example of selecting a random site and calculating energy change
i, j = np.random.randint(0, L), np.random.randint(0, L)  # Random site
delta_E = calculate_energy_change(lattice, i, j, J)
print(f"Energy change ΔE if spin at ({i}, {j}) is flipped: {delta_E}")


