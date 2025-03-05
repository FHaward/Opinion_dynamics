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
    colors = [('blue', 'orange'), ('green', 'red'), ('purple', 'brown'), 
              ('pink', 'gray'), ('cyan', 'magenta')]
    
    # Plot each dataset
    for results, label, (color1, color2) in zip(results_list, labels, colors):
        temps = results['temperatures']
        m_plus = results['m_plus_avgs']
        m_minus = results['m_minus_avgs']
        m_plus_err = results['m_plus_avgs_std_errors']
        m_minus_err = results['m_minus_avgs_std_errors']
        
        plt.errorbar(temps, m_plus, yerr=m_plus_err, 
                    marker="o", label=f"{label} ($m_+$)", color=color1,
                    capsize=5, markersize=4)
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

def plot_combined_average_magnetizations(results_list, labels, save_path=None):
    """
    Plot average magnetizations from multiple simulations on the same axes.
    
    Args:
        results_list: List of loaded results dictionaries from load_processed_results
        labels: List of labels for each dataset
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Define colors for each dataset
    colors = ['blue', 'green', 'purple', 'pink', 'cyan', 'orange', 'red', 'brown']
    
    # Plot each dataset
    for results, label, color in zip(results_list, labels, colors):
        temps = results['temperatures']
        avg_mag = results['average_magnetizations']
        avg_mag_err = results['average_magnetizations_std_errors']
        
        plt.errorbar(temps, avg_mag, yerr=avg_mag_err, 
                    marker="o", label=label, color=color,
                    capsize=5, markersize=4)
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Magnetization", fontsize=12)
    plt.title("Average Magnetization vs Temperature", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    

    plt.show()

def plot_combined_g_plus_minus(results_list, labels, save_path=None):
    """
    Plot g+ and g- from multiple simulations on the same axes.
    
    Args:
        results_list: List of loaded results dictionaries from load_processed_results
        labels: List of labels for each dataset
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Define color pairs for each dataset
    colors = [('blue', 'orange'), ('green', 'red'), ('purple', 'brown'), 
              ('pink', 'gray'), ('cyan', 'magenta')]
    
    # Plot each dataset
    for results, label, (color1, color2) in zip(results_list, labels, colors):
        temps = results['temperatures']
        g_plus = results['g_plus_list']
        g_minus = results['g_minus_list']
        g_plus_err = results['g_plus_std_errors']
        g_minus_err = results['g_minus_std_errors']
        
        plt.errorbar(temps, g_plus, yerr=g_plus_err, 
                    marker="o", label=f"{label} ($g_+$)", color=color1,
                    capsize=5, markersize=4)
        '''plt.errorbar(temps, g_minus, yerr=g_minus_err, 
                    marker="s", label=f"{label} ($g_-$)", color=color2,
                    capsize=5, markersize=4)'''
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Fraction of Runs", fontsize=12)
    plt.title("$g_+$ and $g_-$ vs Temperature", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    plt.show()

def plot_combined_average_zealot_spins(results_list, labels, save_path=None):
    """
    Plot average zealot spins from multiple simulations on the same axes.
    
    Args:
        results_list: List of loaded results dictionaries from load_processed_results
        labels: List of labels for each dataset
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Define colors for each dataset
    colors = ['blue', 'green', 'purple', 'pink', 'cyan', 'orange', 'red', 'brown']
    
    # Plot each dataset
    for results, label, color in zip(results_list, labels, colors):
        temps = results['temperatures']
        avg_zealot = results['average_zealot_spins']
        avg_zealot_err = results['average_zealot_spins_std_errors']
        
        plt.errorbar(temps, avg_zealot, yerr=avg_zealot_err, 
                    marker="o", label=label, color=color,
                    capsize=5, markersize=4)
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Zealot Spin", fontsize=12)
    plt.title("Average Zealot Spin vs Temperature", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_combined_z_plus_minus(results_list, labels, save_path=None):
    """
    Plot z+ and z- from multiple simulations on the same axes.
    
    Args:
        results_list: List of loaded results dictionaries from load_processed_results
        labels: List of labels for each dataset
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Define color pairs for each dataset
    colors = [('blue', 'orange'), ('green', 'red'), ('purple', 'brown'), 
              ('pink', 'gray'), ('cyan', 'magenta')]
    
    # Plot each dataset
    for results, label, (color1, color2) in zip(results_list, labels, colors):
        temps = results['temperatures']
        z_plus = results['z_plus_avgs']
        z_minus = results['z_minus_avgs']
        z_plus_err = results['z_plus_avgs_std_errors']
        z_minus_err = results['z_minus_avgs_std_errors']
        
        plt.errorbar(temps, z_plus, yerr=z_plus_err, 
                    marker="o", label=f"{label} ($z_+$)", color=color1,
                    capsize=5, markersize=4)
        plt.errorbar(temps, z_minus, yerr=z_minus_err, 
                    marker="s", label=f"{label} ($z_-$)", color=color2,
                    capsize=5, markersize=4)
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Zealot Spin", fontsize=12)
    plt.title("$z_+$ and $z_-$ vs Temperature", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_combined_f_plus_minus(results_list, labels, save_path=None):
    """
    Plot f+ and f- from multiple simulations on the same axes.
    
    Args:
        results_list: List of loaded results dictionaries from load_processed_results
        labels: List of labels for each dataset
        save_path: Optional path to save the plot
    """
    plt.figure(figsize=(10, 6))
    
    # Define color pairs for each dataset
    colors = [('blue', 'orange'), ('green', 'red'), ('purple', 'brown'), 
              ('pink', 'gray'), ('cyan', 'magenta')]
    
    # Plot each dataset
    for results, label, (color1, color2) in zip(results_list, labels, colors):
        temps = results['temperatures']
        f_plus = results['f_plus_list']
        f_minus = results['f_minus_list']
        f_plus_err = results['f_plus_std_errors']
        f_minus_err = results['f_minus_std_errors']
        
        plt.errorbar(temps, f_plus, yerr=f_plus_err, 
                    marker="o", label=f"{label} ($f_+$)", color=color1,
                    capsize=5, markersize=4)
        '''plt.errorbar(temps, f_minus, yerr=f_minus_err, 
                    marker="s", label=f"{label} ($f_-$)", color=color2,
                    capsize=5, markersize=4)'''
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Fraction of Runs", fontsize=12)
    plt.title("$f_+$ and $f_-$ vs Temperature", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_all_combined_results(results_list, labels, save_dir=None):
    """
    Generate all combined plots for the provided results list.
    
    Args:
        results_list: List of loaded results dictionaries
        labels: List of labels for each dataset
        save_dir: Optional directory to save the plots
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Generate each type of plot
    plot_combined_average_magnetizations(results_list, labels, 
                                        save_path=os.path.join(save_dir, 'combined_avg_magnetization.png') if save_dir else None)
    
    plot_combined_m_plus_minus(results_list, labels,
                              save_path=os.path.join(save_dir, 'combined_m_plus_minus.png') if save_dir else None)
    
    plot_combined_g_plus_minus(results_list, labels,
                              save_path=os.path.join(save_dir, 'combined_g_plus_minus.png') if save_dir else None)
    
    plot_combined_average_zealot_spins(results_list, labels,
                                      save_path=os.path.join(save_dir, 'combined_avg_zealot_spins.png') if save_dir else None)
    
    plot_combined_z_plus_minus(results_list, labels,
                              save_path=os.path.join(save_dir, 'combined_z_plus_minus.png') if save_dir else None)
    
    plot_combined_f_plus_minus(results_list, labels,
                              save_path=os.path.join(save_dir, 'combined_f_plus_minus.png') if save_dir else None)



# Load multiple results
path1 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Initial results\simulation_results\fully_connected\ratio_80_to_20\zealot_field_10000\20250212_132410\numerical_results.json"
path2 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Initial results\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250212_132349\numerical_results.json"
path3 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Initial results\simulation_results\fully_connected\ratio_20_to_80\zealot_field_10000\20250217_145915\numerical_results.json"
path_40_60 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Initial results\simulation_results\fully_connected\ratio_40_to_60\zealot_field_10000\20250217_200050\numerical_results.json"
path_50_50 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Initial results\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250218_012653\numerical_results.json"
path_60_40 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Initial results\simulation_results\fully_connected\ratio_60_to_40\zealot_field_10000\20250217_201038\numerical_results.json"

results1 = load_processed_results(path1)
results2 = load_processed_results(path2)
results3 = load_processed_results(path3)
results_40_60 = load_processed_results(path_40_60)
results_50_50 = load_processed_results(path_50_50)
results_60_40 = load_processed_results(path_60_40)



path_benchmark = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250219_224558\numerical_results.json"
path_60_up = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_60_to_40\zealot_field_10000\20250224_182947\numerical_results.json"
path_40_up = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_40_to_60\zealot_field_10000\20250224_183021\numerical_results.json"
path_lower_z_field =r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_8000.0\20250224_225357\numerical_results.json"
path_low_z_field = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_9000.0\20250224_110453\numerical_results.json"
path_high_z_field = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_11000.0\20250224_110447\numerical_results.json"
path_low_interaction = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250224_230658\numerical_results.json"
path_high_interaction = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250224_230647\numerical_results.json"
path_negative_interaction = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250225_090911\numerical_results.json"

results_benchmark = load_processed_results(path_benchmark)
results_60_up = load_processed_results(path_60_up)
results_40_up = load_processed_results(path_40_up)
results_lower_z_field = load_processed_results(path_lower_z_field)
results_low_z_field = load_processed_results(path_low_z_field)
results_high_z_field = load_processed_results(path_high_z_field)
results_low_interaction = load_processed_results(path_low_interaction)
results_high_interaction = load_processed_results(path_high_interaction)
results_negative_interaction = load_processed_results(path_negative_interaction)


# Create labels for each dataset
labels0 = ["40:60", "50:50", "60:40"]
#plot_all_combined_results([results_40_60, results_50_50, results_60_40], labels0)

labels1 = [r"40% up initial", r"50% up initial", r"60% up initial"]
plot_all_combined_results([results_40_up, results_benchmark, results_60_up], labels1)

labels2 = ["zealot field = 0.8", "zealot field = 0.9", "zealot field = 1", "zealot field= 1.1"]
plot_all_combined_results([results_lower_z_field, results_low_z_field, results_benchmark, results_high_z_field], labels2)

labels3 = ["zealot interaction= 0.99", "zealot interaction= 1.00", "zealot interaction= 1.01", "zealot interaction= 1.02"]
plot_all_combined_results([results_negative_interaction, results_low_interaction, results_benchmark, results_high_interaction], labels3)
