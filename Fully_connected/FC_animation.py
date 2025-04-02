import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from tqdm import tqdm

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
    
    return -2*(social_influence+internal_field+leader_influence)

def calculate_energy_change_zealot(zealot_spin, magnetization, L, J_s, h_s):
    """
    Calculate the energy change that would occur if the zealot spin is flipped.
    """

    leader_field = -h_s*zealot_spin
    leader_influence = -J_s*zealot_spin*magnetization 
    return -2*(leader_influence+leader_field)

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

def run_simulation_with_snapshots(seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps):
    np.random.seed(seed)
    lattice = np.random.choice([-1, 1], size=(L, L))
    magnetization_record_interval = number_of_MC_steps * N
    magnetization = np.sum(lattice)  # Initial magnetization
    lattice_snapshots = [lattice.copy()]  # Initialize with the starting configuration

    # Preallocate the array for magnetization values
    num_intervals = num_iterations // magnetization_record_interval
    total_recalculations = num_intervals + 1 + (1 if num_iterations % magnetization_record_interval > 0 else 0)
    magnetization_array = np.zeros(total_recalculations, dtype=np.float64)
    magnetization_array[0] = magnetization
    recalculation_index = 1

    # Loop over each interval
    for interval in tqdm(range(num_intervals), desc=f"Temp {temp}, Seed {seed}"):
        for step in range(number_of_MC_steps):
            for _ in range(N): 
                magnetization = metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

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
                magnetization = metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

            # Update zealot spin every L^2 steps
            delta_E_zealot = calculate_energy_change_zealot(zealot_spin, magnetization, L, J_s, h_s)
            if np.random.rand() < np.exp(-delta_E_zealot / (k_B * temp)):
                zealot_spin *= -1

        # Handle any remaining steps without a zealot recalculation
        for _ in range(extra_steps):
            magnetization = metropolis_step(lattice, L, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, magnetization)

        # Final recalculation for the last segment
        lattice_snapshots.append(lattice.copy())
        magnetization_array[recalculation_index] = magnetization

                   
    return magnetization_array/N, lattice_snapshots


# Parameters
L = 100
N = L**2
zealot_spin = -1
k_B = 1
num_iterations = N*100  # Total number of iterations
J_b = 1.0/(N-1)
J_s = 1.01
h_b = -1
h_s = N
number_of_MC_steps = 2
seed = 13
temp = 0.98286

# Run the simulation
magnetization, lattice_snapshots = run_simulation_with_snapshots(seed, L, N, temp, k_B, J_b, h_b, h_s, J_s, zealot_spin, num_iterations, number_of_MC_steps)



# Set up the figure for animation
fig, ax = plt.subplots(figsize=(6, 6))
im = ax.imshow(lattice_snapshots[0], cmap="coolwarm", interpolation="nearest")
ax.set_title("Lattice Evolution")
ax.set_xlabel("X Position")
ax.set_ylabel("Y Position")
plt.colorbar(im, label="Spin")

# Update function for animation
def update(frame):
    im.set_array(lattice_snapshots[frame])
    return [im]

# Create the animation
ani = animation.FuncAnimation(
    fig, update, frames=len(lattice_snapshots), blit=True, interval=50, repeat=False
)

# Display the animation
plt.show()



time_steps = np.arange(len(magnetization)) * number_of_MC_steps

plt.figure(figsize=(10, 5))
plt.plot(time_steps, magnetization, label="Average Magnetization", color="b")
plt.xlabel("Monte Carlo Steps")
plt.ylabel("Average Magnetization")
plt.title("Magnetization Over Time")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()



import numpy as np

ratios = np.linspace(0.4, 0.6, 50).tolist()  # Range of initial up ratios

filtered_ratios = [r for r in ratios if 0.475 <= r <= 0.525]

print(filtered_ratios)