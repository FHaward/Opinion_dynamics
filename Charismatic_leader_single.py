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

def run_simulation(L, temp, k_B, J_b, h_b, h_s, J_s, num_iterations, seeds, recalculation_interval):
    all_magnetizations = []
    final_magnetizations = []  # To store final magnetization of each run

    for seed in tqdm(seeds, desc="Running repeats"):
        np.random.seed(seed)  # Set the random seed for each run

        # Initialize lattice with random spins based on the current seed
        lattice = np.random.choice([-1, 1], size=(L, L))
        if lattice[(L // 2), (L // 2)] == -1:
            lattice = -lattice
        sigma_s = lattice[L // 2, L // 2]
        
        total_magnetization = np.sum(lattice)
        magnetization_list = [total_magnetization]

        
        # Main simulation loop for this seed
        for step in tqdm(range(num_iterations), desc="Running simulation"):
            sigma_s = metropolis_step(
                lattice, L, J_b, h_b, h_s, J_s, temp, k_B, sigma_s
            )
            
            # Recalculate magnetization every `recalculation_interval` steps
            if step % recalculation_interval == 0:
                total_magnetization = np.sum(lattice)
                magnetization_list.append(total_magnetization)
        
        averaged_magnetizations = np.array(magnetization_list)/(L * L)
        final_magnetizations.append(averaged_magnetizations[-1])  # Divide by L*L at the end for final result
        all_magnetizations.append(averaged_magnetizations)  # Append the entire averaged list
    
    # Separate magnetizations into m_plus and m_minus based on the final magnetizations
    final_magnetizations = np.array(final_magnetizations)  # Convert to a NumPy array
    m_plus = final_magnetizations[final_magnetizations > 0]
    m_minus = final_magnetizations[final_magnetizations < 0]
    
    # Calculate averages of m_plus and m_minus
    m_plus_avg = np.mean(m_plus) if m_plus.size > 0 else 0
    m_minus_avg = np.mean(m_minus) if m_minus.size > 0 else 0

    # Calculate fractions g_plus and g_minus
    total_runs = len(final_magnetizations)
    g_plus = len(m_plus) / total_runs
    g_minus = len(m_minus) / total_runs
    
    return all_magnetizations, m_plus_avg, m_minus_avg, g_plus, g_minus, m_plus, m_minus


# Parameters
L = 100  # Size of the lattice (LxL)
temp = 6e22  
k_B = 1.380649e-23  # Boltzmann constant
num_iterations = (L**2)*100  # Total number of iterations
J_b = 1  # Coupling constant
J_s = 0
h_b= 0
h_s = 0
snapshot_interval = 1000  # Save lattice every 20 steps for animation
seeds = [1,2,3]
recalculation_interval = 10*(L**2)



# Run the modified simulation function
all_magnetizations, m_plus_avg, m_minus_avg, g_plus, g_minus, m_plus, m_minus = run_simulation(
    L, temp, k_B, J_b, h_b, h_s, J_s, num_iterations, seeds, recalculation_interval
)
# Output results
print(f"Average Magnetization of m_plus: {m_plus_avg}")
print(f"Average Magnetization of m_minus: {m_minus_avg}")
print(f"Fraction g_plus (runs with positive magnetization): {g_plus}")
print(f"Fraction g_minus (runs with negative magnetization): {g_minus}")

# Calculate the average magnetization across all simulations at each time step
average_magnetization_across_runs = np.mean(all_magnetizations, axis=0)

# Plot the average magnetization over time
plt.figure(figsize=(10, 5))
plt.plot(average_magnetization_across_runs, label="Average Magnetization (across runs)")
plt.xlabel("Monte Carlo Step")
plt.ylabel("Average Magnetization")
plt.title("Average Magnetization over Time (Averaged over Multiple Runs)")
plt.legend()
plt.grid(True)
plt.show()


all_magnetizations
m_plus_avg
m_minus_avg
g_plus
g_minus
m_plus
m_minus