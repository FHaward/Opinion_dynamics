import json
import os
import numpy as np
import matplotlib.pyplot as plt

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


def plot_average_magnetizations_vs_temperature(results):
    # Extract temperatures and average magnetizations
    temperatures = results['temperatures']
    average_magnetizations = results['average_magnetizations']
    standard_errors = results['average_magnetizations_std_errors']
    
    plt.figure(figsize=(12, 7))
    plt.errorbar(temperatures, average_magnetizations, 
                 yerr=standard_errors,
                 marker="o", label="Average Magnetization", color="blue", 
                 capsize=5, ecolor='gray')
    
    plt.xlabel("Temperature", fontsize=20)
    plt.ylabel("Average Magnetization", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()
        
def plot_g_plus_minus_vs_temperature(results):
    # Extract data
    temperatures = results['temperatures']
    g_plus_list = results['g_plus_list']
    g_minus_list = results['g_minus_list']
    g_plus_errors = results['g_plus_std_errors']
    g_minus_errors = results['g_minus_std_errors']

    # Create a plot
    plt.figure(figsize=(12, 7))
    plt.errorbar(temperatures, g_plus_list, yerr=g_plus_errors, 
                 marker="o", label="$g_+$ (Fraction Positive)", color="green", 
                 capsize=5, ecolor='gray')
    plt.errorbar(temperatures, g_minus_list, yerr=g_minus_errors, 
                 marker="s", label="$g_-$ (Fraction Negative)", color="red", 
                 capsize=5, ecolor='gray')

    # Add labels and grid
    plt.xlabel("Temperature", fontsize=20)
    plt.ylabel("Fraction of Runs", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_m_plus_minus_vs_temperature(results):
    # Extract data
    temperatures = results['temperatures']
    m_plus_avgs = results['m_plus_avgs']
    m_minus_avgs = results['m_minus_avgs']
    m_plus_errors = results['m_plus_avgs_std_errors']
    m_minus_errors = results['m_minus_avgs_std_errors']
    
    # Create a plot
    plt.figure(figsize=(12, 7))
    plt.errorbar(temperatures, m_plus_avgs, yerr=m_plus_errors, 
                 marker="o", label="$m_+$ (Fraction Positive)", color="blue", 
                 capsize=5, ecolor='gray')
    plt.errorbar(temperatures, m_minus_avgs, yerr=m_minus_errors, 
                 marker="s", label="$m_-$ (Fraction Negative)", color="orange", 
                 capsize=5, ecolor='gray')
    
    # Add labels and grid
    plt.xlabel("Temperature", fontsize=20)
    plt.ylabel("Average Magnetization", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_zealot_statistics_vs_temperature(results):
    temperatures = results['temperatures']
    average_zealot_spins = results['average_zealot_spins']
    zealot_errors = results['average_zealot_spins_std_errors']
    
    # Plot average zealot spin vs temperature
    plt.figure(figsize=(12, 7))
    plt.errorbar(temperatures, average_zealot_spins, 
                 yerr=zealot_errors,
                 marker="o", label="Average Zealot Spin", color="purple", 
                 capsize=5, ecolor='gray')

    plt.xlabel("Temperature", fontsize=20)
    plt.ylabel("Average Zealot Spin", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

    z_plus_avgs = results['z_plus_avgs']
    z_minus_avgs = results['z_minus_avgs']
    z_plus_errors = results['z_plus_avgs_std_errors']
    z_minus_errors = results['z_minus_avgs_std_errors']
    
    # Plot z+ and z- vs temperature
    plt.figure(figsize=(12, 7))
    plt.errorbar(temperatures, z_plus_avgs, yerr=z_plus_errors, 
                 marker="o", label="$z_+$ (Fraction Positive)", color="blue", 
                 capsize=5, ecolor='gray')
    plt.errorbar(temperatures, z_minus_avgs, yerr=z_minus_errors, 
                 marker="s", label="$z_-$ (Fraction Negative)", color="red", 
                 capsize=5, ecolor='gray')

    plt.xlabel("Temperature", fontsize=20)
    plt.ylabel("Average Zealot Spin", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()

    f_plus_list = results['f_plus_list']
    f_minus_list = results['f_minus_list']
    f_plus_errors = results['f_plus_std_errors']
    f_minus_errors = results['f_minus_std_errors']

    # Plot f+ and f- vs temperature
    plt.figure(figsize=(12, 7))
    plt.errorbar(temperatures, f_plus_list, yerr=f_plus_errors, 
                 marker="o", label="$f_+$ (Fraction Positive)", color="green", 
                 capsize=5, ecolor='gray')
    plt.errorbar(temperatures, f_minus_list, yerr=f_minus_errors, 
                 marker="s", label="$f_-$ (Fraction Negative)", color="orange", 
                 capsize=5, ecolor='gray')

    plt.xlabel("Temperature", fontsize=20)
    plt.ylabel("Fraction of Runs", fontsize=20)
    plt.tick_params(axis='both', which='major', labelsize=18)
    plt.grid(True)
    plt.legend(fontsize=18)
    plt.tight_layout()
    plt.show()


path_benchmark = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250219_224558\numerical_results.json"
results_benchmark = load_processed_results(path_benchmark)











plot_average_magnetizations_vs_temperature(results_benchmark)
plot_m_plus_minus_vs_temperature(results_benchmark)
plot_g_plus_minus_vs_temperature(results_benchmark)
plot_zealot_statistics_vs_temperature(results_benchmark)
  
