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
    colors = ['navy', 'darkred', 'darkgreen', 'rebeccapurple', 'darkorange', 'teal', 'mediumvioletred']
    
    # Plot each dataset
    for results, label, color in zip(results_list, labels, colors):
        temps = results['temperatures']
        avg_mag = results['average_magnetizations']
        avg_mag_err = results['average_magnetizations_std_errors']

        if not np.isnan(avg_mag).all():
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
        g_plus = results['g_plus_list']
        g_minus = results['g_minus_list']
        g_plus_err = results['g_plus_std_errors']
        g_minus_err = results['g_minus_std_errors']
        
        if not np.isnan(g_plus).all():
            plt.errorbar(temps, g_plus, yerr=g_plus_err, 
                    marker="o", label=f"{label} ($g_+$)", color=color1,
                    capsize=5, markersize=4)
    
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
    colors = ['navy', 'darkred', 'darkgreen', 'rebeccapurple', 'darkorange', 'teal', 'mediumvioletred']
    
    # Plot each dataset
    for results, label, color in zip(results_list, labels, colors):
        temps = results['temperatures']
        avg_zealot = results['average_zealot_spins']
        avg_zealot_err = results['average_zealot_spins_std_errors']
        
        if not np.isnan(avg_zealot).all():
            plt.errorbar(temps, avg_zealot, yerr=avg_zealot_err, 
                    marker="o", label=label, color=color,
                    capsize=5, markersize=4)
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Zealot Spin", fontsize=12)
    plt.title("Average Zealot Spin vs Temperature", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    

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
        z_plus = results['z_plus_avgs']
        z_minus = results['z_minus_avgs']
        z_plus_err = results['z_plus_avgs_std_errors']
        z_minus_err = results['z_minus_avgs_std_errors']
        
        if not np.isnan(z_plus).all():
            plt.errorbar(temps, z_plus, yerr=z_plus_err, 
                    marker="o", label=f"{label} ($z_+$)", color=color1,
                    capsize=5, markersize=4)
        if not np.isnan(z_minus).all():
            plt.errorbar(temps, z_minus, yerr=z_minus_err, 
                    marker="s", label=f"{label} ($z_-$)", color=color2,
                    capsize=5, markersize=4)
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Average Zealot Spin", fontsize=12)
    plt.title("$z_+$ and $z_-$ vs Temperature", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
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
        f_plus = results['f_plus_list']
        f_minus = results['f_minus_list']
        f_plus_err = results['f_plus_std_errors']
        f_minus_err = results['f_minus_std_errors']
        
        if not np.isnan(f_plus).all():
            plt.errorbar(temps, f_plus, yerr=f_plus_err, 
                    marker="o", label=f"{label} ($f_+$)", color=color1,
                    capsize=5, markersize=4)
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Fraction of Runs", fontsize=12)
    plt.title("$f_+$ and $f_-$ vs Temperature", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
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

def plot_combined_results_subplot(results_list, labels, save_path=None, figsize=(18, 12)):
    """
    Generate a 2x3 subplot layout with non-zealot plots on the top row
    and zealot plots on the bottom row.
    
    Args:
        results_list: List of loaded results dictionaries from load_processed_results
        labels: List of labels for each dataset
        save_path: Optional path to save the plot
        figsize: Size of the figure (width, height) in inches
    """
    # Create a 2x3 subplot layout
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    
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
    
    # Define a single color set for average plots
    single_colors = ['navy', 'darkred', 'darkgreen', 'rebeccapurple', 'darkorange', 'teal', 'mediumvioletred']
    
    # TOP ROW: Non-zealot plots
    
    # 1. Average Magnetization (top left)
    for results, label, color in zip(results_list, labels, single_colors):
        temps = results['temperatures']
        avg_mag = results['average_magnetizations']
        avg_mag_err = results['average_magnetizations_std_errors']
        
        if not np.isnan(avg_mag).all():
            axes[0, 0].errorbar(temps, avg_mag, yerr=avg_mag_err, 
                            marker="o", label=label, color=color,
                            capsize=3, markersize=4)
    
    axes[0, 0].set_xlabel("Temperature", fontsize=10)
    axes[0, 0].set_ylabel("Average Magnetization", fontsize=10)
    axes[0, 0].set_title("Average Magnetization vs Temperature", fontsize=12)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    
    # 2. M+ and M- (top middle)
    for results, label, (color1, color2) in zip(results_list, labels, colors):
        temps = results['temperatures']
        m_plus = results['m_plus_avgs']
        m_minus = results['m_minus_avgs']
        m_plus_err = results['m_plus_avgs_std_errors']
        m_minus_err = results['m_minus_avgs_std_errors']
        
        if not np.isnan(m_plus).all():
            axes[0, 1].errorbar(temps, m_plus, yerr=m_plus_err, 
                            marker="o", label=f"{label} ($m_+$)", color=color1,
                            capsize=3, markersize=4)
        if not np.isnan(m_minus).all():
            axes[0, 1].errorbar(temps, m_minus, yerr=m_minus_err, 
                            marker="s", label=f"{label} ($m_-$)", color=color2,
                            capsize=3, markersize=4)
    
    axes[0, 1].set_xlabel("Temperature", fontsize=10)
    axes[0, 1].set_ylabel("Average Magnetization", fontsize=10)
    axes[0, 1].set_title("$m_+$ and $m_-$ vs Temperature", fontsize=12)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend(fontsize=8)
    
    # 3. g+ and g- (top right)
    for results, label, (color1, color2) in zip(results_list, labels, colors):
        temps = results['temperatures']
        g_plus = results['g_plus_list']
        g_minus = results['g_minus_list']
        g_plus_err = results['g_plus_std_errors']
        g_minus_err = results['g_minus_std_errors']
        
        if not np.isnan(g_plus).all():
            axes[0, 2].errorbar(temps, g_plus, yerr=g_plus_err, 
                          marker="o", label=f"{label} ($g_+$)", color=color1,
                          capsize=3, markersize=4)
        if not np.isnan(g_minus).all():
            axes[0, 2].errorbar(temps, g_minus, yerr=g_minus_err, 
                          marker="s", label=f"{label} ($g_-$)", color=color2,
                          capsize=3, markersize=4)
    
    axes[0, 2].set_xlabel("Temperature", fontsize=10)
    axes[0, 2].set_ylabel("Fraction of Runs", fontsize=10)
    axes[0, 2].set_title("$g_+$ and $g_-$ vs Temperature", fontsize=12)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].legend(fontsize=8)
    
    # BOTTOM ROW: Zealot plots
    
    # 4. Average Zealot Spins (bottom left)
    for results, label, color in zip(results_list, labels, single_colors):
        temps = results['temperatures']
        avg_zealot = results['average_zealot_spins']
        avg_zealot_err = results['average_zealot_spins_std_errors']
        
        if not np.isnan(avg_zealot).all():
            axes[1, 0].errorbar(temps, avg_zealot, yerr=avg_zealot_err, 
                          marker="o", label=label, color=color,
                          capsize=3, markersize=4)
    
    axes[1, 0].set_xlabel("Temperature", fontsize=10)
    axes[1, 0].set_ylabel("Average Zealot Spin", fontsize=10)
    axes[1, 0].set_title("Average Zealot Spin vs Temperature", fontsize=12)
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend(fontsize=8)
    
    # 5. z+ and z- (bottom middle)
    for results, label, (color1, color2) in zip(results_list, labels, colors):
        temps = results['temperatures']
        z_plus = results['z_plus_avgs']
        z_minus = results['z_minus_avgs']
        z_plus_err = results['z_plus_avgs_std_errors']
        z_minus_err = results['z_minus_avgs_std_errors']
        
        if not np.isnan(z_plus).all():
            axes[1, 1].errorbar(temps, z_plus, yerr=z_plus_err, 
                          marker="o", label=f"{label} ($z_+$)", color=color1,
                          capsize=3, markersize=4)
        if not np.isnan(z_minus).all():
            axes[1, 1].errorbar(temps, z_minus, yerr=z_minus_err, 
                          marker="s", label=f"{label} ($z_-$)", color=color2,
                          capsize=3, markersize=4)
    
    axes[1, 1].set_xlabel("Temperature", fontsize=10)
    axes[1, 1].set_ylabel("Average Zealot Spin", fontsize=10)
    axes[1, 1].set_title("$z_+$ and $z_-$ vs Temperature", fontsize=12)
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend(fontsize=8)
    
    # 6. f+ and f- (bottom right)
    for results, label, (color1, color2) in zip(results_list, labels, colors):
        temps = results['temperatures']
        f_plus = results['f_plus_list']
        f_minus = results['f_minus_list']
        f_plus_err = results['f_plus_std_errors']
        f_minus_err = results['f_minus_std_errors']
        
        if not np.isnan(f_plus).all():
            axes[1, 2].errorbar(temps, f_plus, yerr=f_plus_err, 
                          marker="o", label=f"{label} ($f_+$)", color=color1,
                          capsize=3, markersize=4)
        if not np.isnan(f_minus).all():
            axes[1, 2].errorbar(temps, f_minus, yerr=f_minus_err, 
                          marker="s", label=f"{label} ($f_-$)", color=color2,
                          capsize=3, markersize=4)
    
    axes[1, 2].set_xlabel("Temperature", fontsize=10)
    axes[1, 2].set_ylabel("Fraction of Runs", fontsize=10)
    axes[1, 2].set_title("$f_+$ and $f_-$ vs Temperature", fontsize=12)
    axes[1, 2].grid(True, alpha=0.3)
    axes[1, 2].legend(fontsize=8)
    
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    return fig, axes

#new paths
#ratio
path_ratio_480 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_48.0\20250320_094156_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_485 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_48.5\20250320_094203_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_490 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_49.0\20250320_094206_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_495 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_49.5\20250320_094209_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250320_094211_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_510 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_51.0\20250320_094217_J_s_1.01_h_s_10000\numerical_results.json"

#h_s
path_hs_11000 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_11000.0\20250326_162840_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_09000 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_9000.0\20250326_162840_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_08500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_8500.0\20250326_164710_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_08000 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_8000.0\20250326_165318_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_07500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_7500.0\20250326_165744_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_07000 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_7000.0\20250326_170750_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_06000 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_6000.0\20250324_155117_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_06125 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_6125.0\20250324_155117_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_06250 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_6250.0\20250324_155117_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_06375 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_6375.0\20250324_155224_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_06500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\h_s\h_s_6500.0\20250324_155939_ratio_50.0_J_s_1.01\numerical_results.json"

#J_s
path_Js_095 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\J_s\J_s_0.95\20250327_183452_ratio_50.0_h_s_10000\numerical_results.json"
path_Js_099 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\J_s\J_s_0.99\20250327_183452_ratio_50.0_h_s_10000\numerical_results.json"
path_Js_100 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\J_s\J_s_1\20250327_183523_ratio_50.0_h_s_10000\numerical_results.json"
path_Js_101 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\J_s\J_s_1.01\20250327_183523_ratio_50.0_h_s_10000\numerical_results.json"
path_Js_102 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\J_s\J_s_1.02\20250327_183523_ratio_50.0_h_s_10000\numerical_results.json"
path_Js_105 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\J_s\J_s_1.05\20250327_183523_ratio_50.0_h_s_10000\numerical_results.json"


#critical temperature paths 
crit_ratio_path_480 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_48.0\20250327_181031_J_s_1.01_h_s_10000\numerical_results.json"
crit_ratio_path_485 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_48.5\20250327_181033_J_s_1.01_h_s_10000\numerical_results.json"
crit_ratio_path_490 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_49.0\20250327_181024_J_s_1.01_h_s_10000\numerical_results.json"
crit_ratio_path_495 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_49.5\20250327_181022_J_s_1.01_h_s_10000\numerical_results.json"
crit_ratio_path_500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250327_181015_J_s_1.01_h_s_10000\numerical_results.json"
crit_ratio_path_510 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_51.0\20250327_181019_J_s_1.01_h_s_10000\numerical_results.json"

crit_hs_path_11000 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250331_105536_J_s_1.01_h_s_11000.0\numerical_results.json"
crit_hs_path_09000 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250331_111741_J_s_1.01_h_s_9000.0\numerical_results.json"
crit_hs_path_08500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250331_105838_J_s_1.01_h_s_8500.0\numerical_results.json"
crit_hs_path_08000 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250331_110823_J_s_1.01_h_s_8000.0\numerical_results.json"
crit_hs_path_07500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250331_113914_J_s_1.01_h_s_7500.0\numerical_results.json"
crit_hs_path_07000 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250331_111119_J_s_1.01_h_s_7000.0\numerical_results.json"
crit_hs_path_06500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250331_121547_J_s_1.01_h_s_6500.0\numerical_results.json"

crit_Js_path_102 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\fully_connected\ratio\ratio_50.0\20250331_125941_J_s_1.02_h_s_10000\numerical_results.json"


#new results

results_ratio_480 = load_processed_results(path_ratio_480)
results_ratio_485 = load_processed_results(path_ratio_485)
results_ratio_490 = load_processed_results(path_ratio_490)
results_ratio_495 = load_processed_results(path_ratio_495)
results_ratio_500 = load_processed_results(path_ratio_500)
results_ratio_510 = load_processed_results(path_ratio_510)

results_hs_06000 = load_processed_results(path_hs_06000)
results_hs_06125 = load_processed_results(path_hs_06125)
results_hs_06250 = load_processed_results(path_hs_06250)
results_hs_06375 = load_processed_results(path_hs_06375)
results_hs_06500 = load_processed_results(path_hs_06500)


#combined results 
combined_results_ratio_480 = load_and_replace_results(path_ratio_480, crit_ratio_path_480)
combined_results_ratio_485 = load_and_replace_results(path_ratio_485, crit_ratio_path_485)
combined_results_ratio_490 = load_and_replace_results(path_ratio_490, crit_ratio_path_490)
combined_results_ratio_495 = load_and_replace_results(path_ratio_495, crit_ratio_path_495)
combined_results_ratio_500 = load_and_replace_results(path_ratio_500, crit_ratio_path_500)
combined_results_ratio_510 = load_and_replace_results(path_ratio_510, crit_ratio_path_510)

combined_results_hs_11000 = load_and_replace_results(path_hs_11000, crit_hs_path_11000)
combined_results_hs_09000 = load_and_replace_results(path_hs_09000, crit_hs_path_09000)
combined_results_hs_08500 = load_and_replace_results(path_hs_08500, crit_hs_path_08500)
combined_results_hs_08000 = load_and_replace_results(path_hs_08000, crit_hs_path_08000)
combined_results_hs_07500 = load_and_replace_results(path_hs_07500, crit_hs_path_07500)
combined_results_hs_07000 = load_and_replace_results(path_hs_07000, crit_hs_path_07000)
combined_results_hs_06500 = load_and_replace_results(path_hs_06500, crit_hs_path_06500)

results_Js_095 = load_processed_results(path_Js_095)
results_Js_099 = load_processed_results(path_Js_099)
results_Js_100 = load_processed_results(path_Js_100)
combined_results_Js_101 = load_processed_results(path_Js_101)
combined_results_Js_102 = load_and_replace_results(path_Js_102, crit_Js_path_102)
combined_results_Js_105 = load_processed_results(path_Js_105)
 

labels_ratio = [r"48% up", r"48.5% up", r"49% up", r"49.5% up", r"50% up", r"51% up"]
results_ratio = [combined_results_ratio_480, combined_results_ratio_485, combined_results_ratio_490, combined_results_ratio_495, combined_results_ratio_500, combined_results_ratio_510]
plot_all_combined_results([combined_results_ratio_480, combined_results_ratio_485, combined_results_ratio_490, combined_results_ratio_495, combined_results_ratio_500, combined_results_ratio_510], labels_ratio)

labels_hs = ["zealot field = 1.1N", "zealot field = N", "zealot field = 0.9N", "zealot field = 0.85N", "zealot field = 0.8N", "zealot field = 0.75N", "zealot field = 0.7N"]
plot_all_combined_results([combined_results_hs_11000, combined_results_ratio_500, combined_results_hs_09000, combined_results_hs_08500, combined_results_hs_08000, combined_results_hs_07500, combined_results_hs_07000], labels_hs)

labels_crititcal_hs = ["0.6", "0.6125", "0.625", "0.6375", "0.65"]
plot_all_combined_results([results_hs_06000, results_hs_06125, results_hs_06250, results_hs_06375, results_hs_06500], labels_crititcal_hs)

labels_Js = ["J_s = 0.95", "J_s = 0.99", "J_s = 1", "J_s = 1.01", "J_s = 1.02", "J_s = 1.05"]
plot_all_combined_results([results_Js_095, results_Js_099, results_Js_100, combined_results_Js_101, combined_results_Js_102, combined_results_Js_105], labels_Js)



fig, axes = plot_combined_results_subplot(results_ratio, labels_ratio, save_path="combined_plots.png")
