import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from scipy.optimize import curve_fit

# Data
n_cores = [1, 20, 40, 60, 80, 100, 128]
run_times = [3968.925694465637, 239.18060326576233, 93.63527464866638, 67.5806736946106, 
             64.05664825439453, 55.212268590927124, 46.79753279685974]

# Calculate speedup
speed_up = 3968.925694465637/np.array(run_times)

# Define Amdahl's law function for curve fitting
def amdahls_law(n, p):
    return 1 / ((1-p) + p/n)

# Perform curve fitting to determine optimal P
params, covariance = curve_fit(amdahls_law, n_cores, speed_up, bounds=(0, 1))
optimal_p = params[0]

# Create the figure with an academic style
plt.figure(figsize=(8, 5))

# Create main plot
plt.plot(n_cores, speed_up, 'o-', color='black', markersize=5, label='Measured speedup', linewidth=1.5)

# Add theoretical speedup line
ideal_speedup = np.array(n_cores)
plt.plot(n_cores, ideal_speedup, '--', color='gray', label='Ideal linear scaling', linewidth=1.5)

# Add Amdahl's law curve with optimized P value
x_amdahl = np.linspace(1, max(n_cores) * 1.05, 1000)
amdahl_speedup = amdahls_law(x_amdahl, optimal_p)
plt.plot(x_amdahl, amdahl_speedup, '-', color='red', 
         label=f"Amdahl's law (p = {optimal_p:.4f})", linewidth=1.5)

# Labels with larger font size, no title
plt.xlabel(r'Number of Cores ($N$)', fontsize=16)
plt.ylabel(r'Speedup ($S$)', fontsize=16)

# Set axes limits
plt.xlim(0, max(n_cores) * 1.05)
plt.ylim(0, max(ideal_speedup) * 1.05)

# Remove grid
plt.grid(False)

# Force x-axis to use integer ticks
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))

# Make tick labels larger
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

# Add legend with larger font
plt.legend(frameon=True, loc='upper left', fontsize=14)

plt.tight_layout()

# Save the figure
plt.savefig("core_scaling.png", bbox_inches="tight", dpi=300)

plt.show()

print(f"Optimal parallelizable fraction (P): {optimal_p:.6f}")
print(f"Maximum theoretical speedup: {1/(1-optimal_p):.2f}x")


speed_up[-1]/8
speed_up[-1]

10.6*2.5