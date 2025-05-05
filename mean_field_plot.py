import json
import os
import numpy as np
import matplotlib.pyplot as plt

def solve_m_plus(N, T, J_b, h, Js, m0, tol=1e-8, max_iter=1000):
    # beta = 1/T, so the equation becomes: m = tanh(beta*(h + Js + Jp*m))
    Jp = J_b*(N-1)
    beta = 1.0 / T
    m = m0
    for _ in range(max_iter):
        m_new = np.tanh(beta * (h + Js + Jp * m))
        if abs(m_new - m) < tol:
            return m_new
        m = m_new
    return m  # return the last iteration if convergence criterion isn't met

def solve_m_minus(N, T, J_b, h, Js, m0, tol=1e-8, max_iter=1000):
    # beta = 1/T, so the equation becomes: m = tanh(beta*(h + Js + Jp*m))
    Jp = J_b*(N-1)
    beta = 1.0 / T
    m = m0
    for _ in range(max_iter):
        m_new = np.tanh(beta * (h + Js + Jp * m))
        if abs(m_new - m) < tol:
            return m_new
        m = m_new
    return m  # return the last iteration if convergence criterion isn't met

def load_processed_results(json_path):
    """
    Load processed results from a JSON file into a dictionary format similar to 
    the original processed_results_lists structure.
    
    Args:
        json_path: Path to the JSON file containing numerical results
        
    Returns:
        Dictionary containing all the loaded results with proper numpy arrays
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays for numerical data
    processed_results = {
        'temperatures': np.array(data['temperatures']),
        'average_magnetizations': np.array(data['average_magnetizations']),
        'average_magnetizations_std_errors': np.array(data['average_magnetizations_std_errors']),
        'm_plus_avgs': np.array(data['m_plus_avgs']),
        'm_plus_avgs_std_errors': np.array(data['m_plus_avgs_std_errors']),
        'm_minus_avgs': np.array(data['m_minus_avgs']),
        'm_minus_avgs_std_errors': np.array(data['m_minus_avgs_std_errors']),
        'g_plus_list': np.array(data['g_plus_list']),
        'g_plus_std_errors': np.array(data['g_plus_std_errors']),
        'g_minus_list': np.array(data['g_minus_list']),
        'g_minus_std_errors': np.array(data['g_minus_std_errors']),
        'average_zealot_spins': np.array(data['average_zealot_spins']),
        'average_zealot_spins_std_errors': np.array(data['average_zealot_spins_std_errors']),
        'z_plus_avgs': np.array(data['z_plus_avgs']),
        'z_plus_avgs_std_errors': np.array(data['z_plus_avgs_std_errors']),
        'z_minus_avgs': np.array(data['z_minus_avgs']),
        'z_minus_avgs_std_errors': np.array(data['z_minus_avgs_std_errors']),
        'f_plus_list': np.array(data['f_plus_list']),
        'f_plus_std_errors': np.array(data['f_plus_std_errors']),
        'f_minus_list': np.array(data['f_minus_list']),
        'f_minus_std_errors': np.array(data['f_minus_std_errors'])
    }
    
    return processed_results

def load_and_replace_results(original_json_path, replacement_json_path):
    """
    Load processed results from the original JSON file, replace values at a specific temperature
    with values from the replacement JSON file, and return in the same format as load_processed_results.
    
    Args:
        original_json_path: Path to the original JSON file containing numerical results
        replacement_json_path: Path to the JSON file containing replacement values for a specific temperature
        
    Returns:
        Dictionary containing all the loaded results with proper numpy arrays and replaced values
    """
    # Load both datasets
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    
    with open(replacement_json_path, 'r') as f:
        replacement_data = json.load(f)
    
    # Find the index of the matching temperature in the original data
    target_temp = replacement_data['temperatures'][0]
    
    try:
        index_to_replace = original_data['temperatures'].index(target_temp)
        print(f"Found exact match for temperature {target_temp} at index {index_to_replace}")
    except ValueError:
        # If exact temperature not found, find the closest one
        temperatures = original_data['temperatures']
        index_to_replace = min(range(len(temperatures)), key=lambda i: abs(temperatures[i] - target_temp))
        print(f"No exact match found. Using closest temperature {temperatures[index_to_replace]} at index {index_to_replace}")
    
    # Replace all values at the identified index
    for key in original_data:
        if key != 'temperatures':  # Keep original temperature grid
            if key in replacement_data:
                original_data[key][index_to_replace] = replacement_data[key][0]
            else:
                print(f"Warning: Key '{key}' not found in replacement data, keeping original value")
    
    # Convert lists to numpy arrays for numerical data
    processed_results = {
        'temperatures': np.array(original_data['temperatures']),
        'average_magnetizations': np.array(original_data['average_magnetizations']),
        'average_magnetizations_std_errors': np.array(original_data['average_magnetizations_std_errors']),
        'm_plus_avgs': np.array(original_data['m_plus_avgs']),
        'm_plus_avgs_std_errors': np.array(original_data['m_plus_avgs_std_errors']),
        'm_minus_avgs': np.array(original_data['m_minus_avgs']),
        'm_minus_avgs_std_errors': np.array(original_data['m_minus_avgs_std_errors']),
        'g_plus_list': np.array(original_data['g_plus_list']),
        'g_plus_std_errors': np.array(original_data['g_plus_std_errors']),
        'g_minus_list': np.array(original_data['g_minus_list']),
        'g_minus_std_errors': np.array(original_data['g_minus_std_errors']),
        'average_zealot_spins': np.array(original_data['average_zealot_spins']),
        'average_zealot_spins_std_errors': np.array(original_data['average_zealot_spins_std_errors']),
        'z_plus_avgs': np.array(original_data['z_plus_avgs']),
        'z_plus_avgs_std_errors': np.array(original_data['z_plus_avgs_std_errors']),
        'z_minus_avgs': np.array(original_data['z_minus_avgs']),
        'z_minus_avgs_std_errors': np.array(original_data['z_minus_avgs_std_errors']),
        'f_plus_list': np.array(original_data['f_plus_list']),
        'f_plus_std_errors': np.array(original_data['f_plus_std_errors']),
        'f_minus_list': np.array(original_data['f_minus_list']),
        'f_minus_std_errors': np.array(original_data['f_minus_std_errors'])
    }
    
    return processed_results

def plot_combined_m_plus_minus(results_list, labels, save_path=None):
    """
    Plot m+ and m- from multiple simulations on the same axes.
    
    Args:
        results_list: List of loaded results dictionaries from load_processed_results
        labels: List of labels for each dataset
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
        # Define color pairs for each dataset
    colors = [
        ('navy', 'cornflowerblue'),          # Dark Blue, Light Blue
        ('darkred', 'lightcoral'),           # Dark Red, Light Red
        ('darkgreen', 'yellowgreen'),        # Dark Green, Light Green
        ('rebeccapurple', 'plum'),           # Dark Purple, Light Purple
        ('darkorange', 'gold'),              # Dark Orange, Light Yellow/Gold
        ('teal', 'turquoise'),               # Dark Teal, Light Teal
        ('mediumvioletred', 'hotpink')       # Dark Pink, Light Pink
    ]

    
    # Plot each dataset
    for results, label, (color1, color2) in zip(results_list, labels, colors):
        temps = results['temperatures']
        m_plus = results['m_plus_avgs']
        m_minus = results['m_minus_avgs']
        m_plus_err = results['m_plus_avgs_std_errors']
        m_minus_err = results['m_minus_avgs_std_errors']
        
        if not np.isnan(m_plus).all():
            plt.errorbar(temps, m_plus, yerr=m_plus_err, 
                        marker="o", label=f"{label} ($m_+$)", color=color1,
                        capsize=5, markersize=4)
        if not np.isnan(m_minus).all():
            plt.errorbar(temps, m_minus, yerr=m_minus_err, 
                        marker="s", label=f"{label} ($m_-$)", color=color2,
                        capsize=5, markersize=4)
        
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Magnetization", fontsize=12)
    plt.title("$m_+$ and $m_-$ vs Temperature", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    

    plt.show()

L = 100  
N=L**2
k_B = 1 
J_b = 1/(N-1)
J_s = 1.01
h_b= -1
h_s = N
m0_plus = 1
m0_minus = -1

path_ratio_500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250320_094211_J_s_1.01_h_s_10000\numerical_results.json"
crit_ratio_path_500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250327_181015_J_s_1.01_h_s_10000\numerical_results.json"
combined_results_ratio_500 = load_and_replace_results(path_ratio_500, crit_ratio_path_500)
results = combined_results_ratio_500
temps = results['temperatures']
m_plus = results['m_plus_avgs']
m_minus = results['m_minus_avgs']
m_plus_err = results['m_plus_avgs_std_errors']
m_minus_err = results['m_minus_avgs_std_errors']





# Temperature range for T (avoid T=0 to prevent beta -> infinity)
T_values = np.linspace(0.1, 1.5, 50)
m_plus_mf = np.zeros_like(T_values)
m_minus_mf = np.zeros_like(T_values)

for i, T in enumerate(T_values):
    m_plus_mf[i] = solve_m_plus(N, T, J_b, h_b, J_s, m0_plus, tol=1e-8, max_iter=1000)
    m_minus_mf[i] = solve_m_plus(N, T, J_b, h_b, J_s, m0_minus, tol=1e-8, max_iter=1000)


first_positive_index = np.argmax(m_minus_mf > 0)
first_positive_index
print(T_values[first_positive_index], m_minus_mf[first_positive_index])




plt.figure(figsize=(7, 5))

# Define high-contrast colors
color_plus = 'blue'        # errorbar: dark
color_minus = 'red'          # errorbar: dark
color_mf_plus = 'gold'          # dashed line: bright
color_mf_minus = 'cyan'  

# Plot error bars with lower zorder
if not np.isnan(m_plus).all():
    plt.errorbar(temps, m_plus, yerr=m_plus_err, 
                 marker="o", label=r"$m_+$",
                 capsize=5, markersize=5, color=color_plus, zorder=1, linewidth=5)
if not np.isnan(m_minus).all():
    plt.errorbar(temps, m_minus, yerr=m_minus_err, 
                 marker="s", label=r"$m_-$",
                 capsize=5, markersize=5, color=color_minus, zorder=1, linewidth=5)

# Plot dashed lines with higher zorder (on top)
plt.plot(T_values, m_plus_mf, label=r'$m>0$ mean field',
         linestyle=(0, (4, 2)), color=color_mf_plus, zorder=2, linewidth=3)
plt.plot(T_values[:first_positive_index], m_minus_mf[:first_positive_index],
         label=r'$m<0$ mean field', linestyle=(0, (4, 2)), color=color_mf_minus, zorder=2, linewidth=3)

plt.axhline(y=0, linestyle='--', color='gray', linewidth=1, zorder=0)

# Labels and styling
plt.xlabel("Temperature", fontsize=18)
plt.ylabel("Average Magnetization", fontsize=18)

# Increase tick label size
plt.tick_params(axis='both', which='major', labelsize=18)

# Set legend to center left position
plt.legend(fontsize=18, loc='center left')

plt.savefig("mean_field.png", bbox_inches="tight", dpi=300)

plt.show()



import json
import os
import numpy as np
import matplotlib.pyplot as plt

def load_processed_results(json_path):
    """
    Load processed results from a JSON file into a dictionary format.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Convert lists back to numpy arrays for numerical data
    processed_results = {
        'temperatures': np.array(data['temperatures']),
        'average_magnetizations': np.array(data['average_magnetizations']),
        'average_magnetizations_std_errors': np.array(data['average_magnetizations_std_errors']),
        'm_plus_avgs': np.array(data['m_plus_avgs']),
        'm_plus_avgs_std_errors': np.array(data['m_plus_avgs_std_errors']),
        'm_minus_avgs': np.array(data['m_minus_avgs']),
        'm_minus_avgs_std_errors': np.array(data['m_minus_avgs_std_errors']),
        # other fields remain the same
    }
    
    return processed_results

def load_and_replace_results(original_json_path, replacement_json_path):
    """
    Load processed results from the original JSON file, replace values at a specific temperature
    with values from the replacement JSON file.
    """
    # Load both datasets
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    
    with open(replacement_json_path, 'r') as f:
        replacement_data = json.load(f)
    
    # Find the index of the matching temperature in the original data
    target_temp = replacement_data['temperatures'][0]
    
    try:
        index_to_replace = original_data['temperatures'].index(target_temp)
        print(f"Found exact match for temperature {target_temp} at index {index_to_replace}")
    except ValueError:
        # If exact temperature not found, find the closest one
        temperatures = original_data['temperatures']
        index_to_replace = min(range(len(temperatures)), key=lambda i: abs(temperatures[i] - target_temp))
        print(f"No exact match found. Using closest temperature {temperatures[index_to_replace]} at index {index_to_replace}")
    
    # Replace all values at the identified index
    for key in original_data:
        if key != 'temperatures':  # Keep original temperature grid
            if key in replacement_data:
                original_data[key][index_to_replace] = replacement_data[key][0]
    
    # Convert lists to numpy arrays
    processed_results = {
        'temperatures': np.array(original_data['temperatures']),
        'average_magnetizations': np.array(original_data['average_magnetizations']),
        'average_magnetizations_std_errors': np.array(original_data['average_magnetizations_std_errors']),
        'm_plus_avgs': np.array(original_data['m_plus_avgs']),
        'm_plus_avgs_std_errors': np.array(original_data['m_plus_avgs_std_errors']),
        'm_minus_avgs': np.array(original_data['m_minus_avgs']),
        'm_minus_avgs_std_errors': np.array(original_data['m_minus_avgs_std_errors']),
        # other fields remain the same
    }
    
    return processed_results

def solve_m_plus(N, T, J_b, h, Js, m0, tol=1e-8, max_iter=1000):
    """
    Solve the mean field equation for magnetization with positive initial condition.
    """
    Jp = J_b*(N-1)
    beta = 1.0 / T
    m = m0
    for _ in range(max_iter):
        m_new = np.tanh(beta * (h + Js + Jp * m))
        if abs(m_new - m) < tol:
            return m_new
        m = m_new
    return m  # return the last iteration if convergence criterion isn't met

# Main plotting function
def plot_magnetization_with_mean_field(data_path, crit_data_path, save_path=None):
    # Load data
    combined_results = load_and_replace_results(data_path, crit_data_path)
    results = combined_results
    temps = results['temperatures']
    avg_mag = results['average_magnetizations']
    avg_mag_err = results['average_magnetizations_std_errors']
    m_plus = results['m_plus_avgs']
    m_minus = results['m_minus_avgs']
    m_plus_err = results['m_plus_avgs_std_errors']
    m_minus_err = results['m_minus_avgs_std_errors']

    # Parameters for mean field calculations
    L = 100  
    N = L**2
    J_b = 1/(N-1)
    J_s = 1.01
    h_b = -1
    h_s = N
    m0_plus = 1
    m0_minus = -1

    # Temperature range for mean field theory
    T_values = np.linspace(0.1, 1.5, 50)
    m_plus_mf = np.zeros_like(T_values)
    m_minus_mf = np.zeros_like(T_values)

    # Calculate mean field predictions
    for i, T in enumerate(T_values):
        m_plus_mf[i] = solve_m_plus(N, T, J_b, h_b, J_s, m0_plus, tol=1e-8, max_iter=1000)
        m_minus_mf[i] = solve_m_plus(N, T, J_b, h_b, J_s, m0_minus, tol=1e-8, max_iter=1000)

    # Find where m_minus becomes positive
    first_positive_index = np.argmax(m_minus_mf > 0)
    
    # Create subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [1, 1]})
    
    # Define colors
    color_avg = 'darkblue'
    color_plus = 'blue'
    color_minus = 'red'
    color_mf_plus = 'gold'
    color_mf_minus = 'cyan'

    # Plot 1: Average Magnetization
    ax1.errorbar(temps, avg_mag, yerr=avg_mag_err, marker="o", 
                 label="Average \nMagnetization", capsize=8, markersize=8, 
                 color=color_avg, linewidth=5)
    
    # Add horizontal line at y=0
    ax1.axhline(y=0, linestyle='--', color='gray', linewidth=2, zorder=0)
    
    # Labels and styling for first plot
    ax1.set_xlabel("Temperature", fontsize=22)
    ax1.set_ylabel("Average Magnetization", fontsize=22)
    ax1.tick_params(axis='both', which='major', labelsize=22)

    ax1.legend(fontsize=20, loc='best')

    # Plot 2: m+ and m- with mean field
    ax2.errorbar(temps, m_plus, yerr=m_plus_err, marker="o", label=r"$m_+$",
                 capsize=8, markersize=8, color=color_plus, zorder=1, linewidth=5)
    ax2.errorbar(temps, m_minus, yerr=m_minus_err, marker="s", label=r"$m_-$",
                 capsize=8, markersize=8, color=color_minus, zorder=1, linewidth=5)

    # Add mean field curves
    ax2.plot(T_values, m_plus_mf, label=r'$m>0$ mean field',
             linestyle=(0, (4, 2)), color=color_mf_plus, zorder=2, linewidth=3)
    ax2.plot(T_values[:first_positive_index], m_minus_mf[:first_positive_index],
             label=r'$m<0$ mean field', linestyle=(0, (4, 2)), color=color_mf_minus, zorder=2, linewidth=3)

    # Add horizontal line at y=0
    ax2.axhline(y=0, linestyle='--', color='gray', linewidth=2, zorder=0)

    # Labels and styling for second plot
    ax2.set_xlabel("Temperature", fontsize=22)
    ax2.set_ylabel("Average Magnetization", fontsize=22)
    ax2.tick_params(axis='both', which='major', labelsize=22)

    ax2.legend(fontsize=19, loc='best')

    # Add labels for the subplots
    ax1.text(0.9, 0.9, "(a)", transform=ax1.transAxes, fontsize=20, fontweight='bold')
    ax2.text(0.9, 0.9, "(b)", transform=ax2.transAxes, fontsize=20, fontweight='bold')

    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
    
    plt.show()
    
    return fig, (ax1, ax2)

# Usage example
if __name__ == "__main__":
    path_ratio_500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250320_094211_J_s_1.01_h_s_10000\numerical_results.json"
    crit_ratio_path_500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250327_181015_J_s_1.01_h_s_10000\numerical_results.json"
    
    plot_magnetization_with_mean_field(path_ratio_500, crit_ratio_path_500, "magnetization_mean_field_subplot.png")