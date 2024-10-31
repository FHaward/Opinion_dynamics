import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Parameters for the lattice
L = 50  # Smaller lattice for faster visualization
max_steps = 200  # Reduced number of steps for animation
ibm = 1  # Random seed
t=10000
# Initialize lattice
is_spins = np.full((L + 2, L + 2), -1)  # Initialize lattice with all spins -1

# Random number generator (equivalent of ibm * 16807)
def random_ibm():
    global ibm
    ibm = (ibm * 16807) % 2147483647
    return ibm / 2147483647

# Precompute the probabilities for the Metropolis algorithm
iex = np.zeros(9)
for ie in range(1, 10):
    ex = math.exp(-2 * (ie - 5) / t)
    iex[ie - 1] = (2.0 * ex / (1.0 + ex) - 1.0)

# Set up lattice with random spin assignment
for i in range(1, L + 1):
    for j in range(1, L + 1):
        if random_ibm() > 0.5:
            is_spins[i, j] = 1

# Function to perform a single Monte Carlo step
def monte_carlo_step():
    for i in range(1, L + 1):
        for j in range(1, L + 1):
            ie = 4 + is_spins[i, j] * (is_spins[i - 1, j] + is_spins[i + 1, j] +
                                        is_spins[i, j - 1] + is_spins[i, j + 1])
            if random_ibm() < iex[ie]:
                is_spins[i, j] = -is_spins[i, j]  # Flip the spin

# Setup figure for animation
fig, ax = plt.subplots()
img = ax.imshow(is_spins[1:L + 1, 1:L + 1], cmap="gray", vmin=-1, vmax=1)

# Animation update function
def update(frame):
    for _ in range(10):  # Run 10 Monte Carlo steps per frame for faster visual effect
        monte_carlo_step()
    img.set_array(is_spins[1:L + 1, 1:L + 1])
    return [img]

# Create the animation
ani = animation.FuncAnimation(fig, update, frames=max_steps // 10, blit=True)

# Display the animation inline
plt.show()





1/2800*(180/np.pi)*3600/1000