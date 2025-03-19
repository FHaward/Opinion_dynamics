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
          ('pink', 'gray'), ('cyan', 'magenta'), ('navy', 'gold'),
          ('darkgreen', 'salmon'), ('indigo', 'olive'), ('teal', 'maroon')]
    
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
    colors = ['blue', 'green', 'purple', 'pink', 'cyan', 'orange', 'red', 'brown', 'darkgreen', 'navy']
    
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
          ('pink', 'gray'), ('cyan', 'magenta'), ('navy', 'gold'),
          ('darkgreen', 'salmon'), ('indigo', 'olive'), ('teal', 'maroon')]
    
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
    colors = ['blue', 'green', 'purple', 'pink', 'cyan', 'orange', 'red', 'brown', 'darkgreen', 'navy']
    
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
          ('pink', 'gray'), ('cyan', 'magenta'), ('navy', 'gold'),
          ('darkgreen', 'salmon'), ('indigo', 'olive'), ('teal', 'maroon')]
    
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
          ('pink', 'gray'), ('cyan', 'magenta'), ('navy', 'gold'),
          ('darkgreen', 'salmon'), ('indigo', 'olive'), ('teal', 'maroon')]
    
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



#create paths 
path_benchmark = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250219_224558\numerical_results.json"

path_80_up = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_80_to_20\zealot_field_10000\20250312_232651\numerical_results.json"
path_60_up = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_60_to_40\zealot_field_10000\20250224_182947\numerical_results.json"
path_45_up = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_45_to_55\zealot_field_10000\20250312_232507\numerical_results.json"
path_40_up = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_40_to_60\zealot_field_10000\20250224_183021\numerical_results.json"
path_01_z_field =r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_1000.0\20250312_232633\numerical_results.json"
path_06_z_field =r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_6000.0\20250312_232601\numerical_results.json"
path_08_z_field =r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_8000.0\20250224_225357\numerical_results.json"
path_09_z_field = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_9000.0\20250224_110453\numerical_results.json"
path_11_z_field = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_11000.0\20250224_110447\numerical_results.json"
path_095_interaction = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250312_232621\numerical_results.json"
path_099_interaction = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250225_090911\numerical_results.json"
path_100_interaction = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250224_230658\numerical_results.json"
path_102_interaction = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250224_230647\numerical_results.json"
path_105_interaction = r"C:\Users\frase\Documents\Durham\4th Year\1Project\thousand_seed_runs\simulation_results\fully_connected\ratio_50_to_50\zealot_field_10000\20250312_232537\numerical_results.json"




#new paths

path_ratio_475 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\ratio\ratio_47\20250316_153828_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_480 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\ratio\ratio_48\20250316_153833_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_485 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\ratio\ratio_48\20250316_153838_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_490 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\ratio\ratio_49\20250316_153843_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_495 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\ratio\ratio_49\20250316_153848_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\ratio\ratio_50\20250316_153854_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_505 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\ratio\ratio_50\20250316_153857_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_510 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\ratio\ratio_51\20250316_153902_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_515 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\ratio\ratio_51\20250316_153905_J_s_1.01_h_s_10000\numerical_results.json"

path_hs_085 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\h_s\h_s_8500.0\20250318_003453_ratio_50_J_s_1.01\numerical_results.json"
path_hs_075 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\h_s\h_s_7500.0\20250318_003500_ratio_50_J_s_1.01\numerical_results.json"
path_hs_070 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\h_s\h_s_7000.0\20250318_003506_ratio_50_J_s_1.01\numerical_results.json"
path_hs_065 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\h_s\h_s_6500.0\20250318_003512_ratio_50_J_s_1.01\numerical_results.json"
path_hs_055 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\h_s\h_s_5500.0\20250318_003515_ratio_50_J_s_1.01\numerical_results.json"
path_hs_050 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\h_s\h_s_5000.0\20250318_003518_ratio_50_J_s_1.01\numerical_results.json"
path_hs_045 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\h_s\h_s_4500.0\20250318_003522_ratio_50_J_s_1.01\numerical_results.json"
path_hs_040 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results\simulation_results\fully_connected\h_s\h_s_4000.0\20250318_003524_ratio_50_J_s_1.01\numerical_results.json"


#load results
results_benchmark = load_processed_results(path_benchmark)

results_80_up = load_processed_results(path_80_up)
results_60_up = load_processed_results(path_60_up)
results_45_up = load_processed_results(path_45_up)
results_40_up = load_processed_results(path_40_up)
results_01_z_field = load_processed_results(path_01_z_field)
results_06_z_field = load_processed_results(path_06_z_field)
results_08_z_field = load_processed_results(path_08_z_field)
results_09_z_field = load_processed_results(path_09_z_field)
results_11_z_field = load_processed_results(path_11_z_field)
results_095_interaction = load_processed_results(path_095_interaction)
results_099_interaction = load_processed_results(path_099_interaction)
results_100_interaction = load_processed_results(path_100_interaction)
results_102_interaction = load_processed_results(path_102_interaction)
results_105_interaction = load_processed_results(path_105_interaction)

#new results

results_ratio_475 = load_processed_results(path_ratio_475)
results_ratio_480 = load_processed_results(path_ratio_480)
results_ratio_485 = load_processed_results(path_ratio_485)
results_ratio_490 = load_processed_results(path_ratio_490)
results_ratio_495 = load_processed_results(path_ratio_495)
results_ratio_500 = load_processed_results(path_ratio_500)
results_ratio_505 = load_processed_results(path_ratio_505)
results_ratio_510 = load_processed_results(path_ratio_510)
results_ratio_515 = load_processed_results(path_ratio_515)

results_hs_085 = load_processed_results(path_hs_085)
results_hs_075 = load_processed_results(path_hs_075)
results_hs_070 = load_processed_results(path_hs_070)
results_hs_065 = load_processed_results(path_hs_065)
results_hs_055 = load_processed_results(path_hs_055)
results_hs_050 = load_processed_results(path_hs_050)
results_hs_045 = load_processed_results(path_hs_045)
results_hs_040 = load_processed_results(path_hs_040)






labels1 = [r"40% up initial", r"45% up initial", r"50% up initial", r"60% up initial", r"80% up initial"]
plot_all_combined_results([results_40_up, results_45_up, results_benchmark, results_60_up, results_80_up], labels1)

labels2 = ["zealot field = 0.1", "zealot field = 0.6", "zealot field = 0.8", "zealot field = 0.9", "zealot field = 1", "zealot field= 1.1"]
plot_all_combined_results([results_01_z_field, results_06_z_field, results_08_z_field, results_09_z_field, results_benchmark, results_11_z_field], labels2)

labels3 = ["zealot interaction= 0.95", "zealot interaction= 0.99", "zealot interaction= 1.00", "zealot interaction= 1.01", "zealot interaction= 1.02", "zealot interaction= 1.05"]
plot_all_combined_results([results_095_interaction, results_099_interaction, results_100_interaction, results_benchmark, results_102_interaction, results_105_interaction], labels3)



labels_ratio = [r"47.5% up", r"48% up", r"48.5% up", r"49% up", r"49.5% up", r"50% up", r"50.5% up", r"51% up", r"51.5% up"]
plot_all_combined_results([results_ratio_475, results_ratio_480, results_ratio_485, results_ratio_490, results_ratio_495, results_ratio_500, results_ratio_505, results_ratio_510, results_ratio_515], labels_ratio)


labels_hs = ["zealot field = 1.1N", "zealot field = N", "zealot field = 0.9N", "zealot field = 0.85N", "zealot field = 0.8N", "zealot field = 0.75N", "zealot field = 0.7N", "zealot field = 0.65N", 
             "zealot field = 0.6N", "zealot field = 0.55N", "zealot field = 0.5N", "zealot field = 0.45N", "zealot field = 0.4N", "zealot field = 0.1N"]
plot_all_combined_results([results_11_z_field, results_benchmark, results_09_z_field, results_hs_085, results_08_z_field, results_hs_075, results_hs_070, results_hs_065,
                           results_06_z_field, results_hs_055, results_hs_050, results_hs_045, results_hs_040, results_01_z_field], labels_hs)

