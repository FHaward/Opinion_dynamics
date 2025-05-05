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

def plot_combined_results_subplot_report(results_list, labels, save_path=None, figsize=(24, 16),
                                         label_positions=None, legend_column_positions=None, 
                                         legend_y_positions=None, subplot_adjustments=None):
    """
    Generate a 2x3 subplot layout with non-zealot plots on the top row
    and zealot plots on the bottom row, styled for reports with larger text.
    Plots only g+ and f+ (not g- and f-).
    
    Features:
    - Shared x-axes between top and bottom rows
    - Shared y-axes between first two plots in each row (magnetization plots)
    - Plot labels (a-f) positioned in bottom right (except c in top left)
    - Dashed horizontal lines at y=0 for magnetization plots
    - No titles, maximizing plot area
    - Larger text and numbers
    - Two-column legends with improved positioning
    
    Args:
        results_list: List of loaded results dictionaries from load_processed_results
        labels: List of labels for each dataset
        save_path: Optional path to save the plot
        figsize: Size of the figure (width, height) in inches
        label_positions: List of (x, y) tuples for positioning plot labels (a-f)
                         Default positions have a-b-d-e-f in bottom right, c in top left
        legend_column_positions: List of x positions for the three legend columns
                                Default: [0.24, 0.5, 0.79] (centers of columns)
        legend_y_positions: List of y positions for the three legends
                           Default: [0.18, 0.12, 0.18]
        subplot_adjustments: Dictionary with subplot adjustment parameters
                            Default: {'bottom': 0.36, 'wspace': 0.12, 'hspace': 0.15}
    """
    # Set default values for new parameters if not provided
    if label_positions is None:
        label_positions = [
            (0.95, 0.05),  # a - bottom right
            (0.95, 0.1),   # b - bottom right
            (0.03, 0.92),  # c - top left (special case)
            (0.95, 0.05),  # d - bottom right
            (0.95, 0.1),   # e - bottom right
            (0.95, 0.15),  # f - bottom right
        ]
    
    if legend_column_positions is None:
        legend_column_positions = [0.24, 0.5, 0.79]  # Centers of columns
    
    if legend_y_positions is None:
        legend_y_positions = [0.18, 0.12, 0.18]  # Y positions for each legend
    
    if subplot_adjustments is None:
        subplot_adjustments = {
            'bottom': 0.36,
            'wspace': 0.12,
            'hspace': 0.15
        }
    
    # Create a figure with increased width
    fig = plt.figure(figsize=figsize)
    
    # Create subplot grid with shared axes and more horizontal space
    gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.20)
    
    # Create axes with shared x and y axes where appropriate
    # Top row
    ax00 = fig.add_subplot(gs[0, 0])  # top left
    ax01 = fig.add_subplot(gs[0, 1], sharey=ax00)  # top middle - shares y with top left
    ax02 = fig.add_subplot(gs[0, 2])  # top right
    
    # Bottom row - share x-axes with top row
    ax10 = fig.add_subplot(gs[1, 0], sharex=ax00)  # bottom left - shares x with top left
    ax11 = fig.add_subplot(gs[1, 1], sharex=ax01, sharey=ax10)  # bottom middle - shares x with top middle, y with bottom left
    ax12 = fig.add_subplot(gs[1, 2], sharex=ax02)  # bottom right - shares x with top right
    
    # Create a list of all axes for easier iteration
    axes = [[ax00, ax01, ax02], [ax10, ax11, ax12]]
    
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
    
    # Set larger font sizes
    LABEL_SIZE = 26
    TICK_SIZE = 24
    LEGEND_SIZE = 24
    
    # Create empty lists to store handles and labels for shared legends
    handles_col0 = []
    labels_col0 = []
    handles_col1 = []
    labels_col1 = []
    handles_col2 = []
    labels_col2 = []
    
    # Add plot labels (a-f) with improved positioning
    label_texts = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    for row in range(2):
        for col in range(3):
            idx = row*3 + col
            pos_x, pos_y = label_positions[idx]
            
            # For most labels, use bottom right alignment
            ha = 'center'
            va = 'center'
            
            axes[row][col].text(pos_x, pos_y, label_texts[idx], 
                              transform=axes[row][col].transAxes, 
                              fontsize=LABEL_SIZE+2, fontweight='bold',
                              ha=ha, va=va)
    
    # TOP ROW: Non-zealot plots
    
    # 1. Average Magnetization (top left)
    for results, label, color in zip(results_list, labels, single_colors):
        temps = results['temperatures']
        avg_mag = results['average_magnetizations']
        avg_mag_err = results['average_magnetizations_std_errors']
        
        if not np.isnan(avg_mag).all():
            line = ax00.errorbar(temps, avg_mag, yerr=avg_mag_err, 
                            marker="o", label=label, color=color,
                            capsize=5, markersize=10, linewidth=3, elinewidth=2)
            # Store handles and labels for shared legend
            handles_col0.append(line)
            labels_col0.append(label)
    
    # Add dashed horizontal line at y=0
    ax00.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax00.set_ylabel("Average Magnetization", fontsize=LABEL_SIZE)
    ax00.grid(True, alpha=0.3, linewidth=1.5)
    ax00.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax00.spines.values():
        spine.set_linewidth(2)
    
    # 2. M+ and M- (top middle)
    for i, (results, label, (color1, color2)) in enumerate(zip(results_list, labels, colors)):
        temps = results['temperatures']
        m_plus = results['m_plus_avgs']
        m_minus = results['m_minus_avgs']
        m_plus_err = results['m_plus_avgs_std_errors']
        m_minus_err = results['m_minus_avgs_std_errors']
        
        if not np.isnan(m_plus).all():
            line1 = ax01.errorbar(temps, m_plus, yerr=m_plus_err, 
                            marker="o", label=f"{label} $m_+$", color=color1,
                            capsize=5, markersize=10, linewidth=3, elinewidth=2)
            handles_col1.append(line1)
            labels_col1.append(f"{label} $m_+$")
            
        if not np.isnan(m_minus).all():
            line2 = ax01.errorbar(temps, m_minus, yerr=m_minus_err, 
                            marker="s", label=f"{label} $m_-$", color=color2,
                            capsize=5, markersize=10, linewidth=3, elinewidth=2)
            handles_col1.append(line2)
            labels_col1.append(f"{label} $m_-$")
    
    # Add dashed horizontal line at y=0
    ax01.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax01.set_ylabel("")  # Hide y-axis label for middle plot since it shares with left plot
    ax01.grid(True, alpha=0.3, linewidth=1.5)
    ax01.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax01.spines.values():
        spine.set_linewidth(2)
    
    # 3. g+ only (top right)
    for results, label, (color1, _) in zip(results_list, labels, colors):
        temps = results['temperatures']
        g_plus = results['g_plus_list']
        g_plus_err = results['g_plus_std_errors']
        
        if not np.isnan(g_plus).all():
            line = ax02.errorbar(temps, g_plus, yerr=g_plus_err, 
                          marker="o", label=f"{label} $g_+$", color=color1,
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
            handles_col2.append(line)
            labels_col2.append(f"{label} $g_+$")
    
    ax02.set_ylabel("Positive Fraction", fontsize=LABEL_SIZE)
    ax02.grid(True, alpha=0.3, linewidth=1.5)
    ax02.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax02.spines.values():
        spine.set_linewidth(2)
    
    # BOTTOM ROW: Zealot plots
    
    # 4. Average Zealot Spins (bottom left)
    for results, label, color in zip(results_list, labels, single_colors):
        temps = results['temperatures']
        avg_zealot = results['average_zealot_spins']
        avg_zealot_err = results['average_zealot_spins_std_errors']
        
        if not np.isnan(avg_zealot).all():
            ax10.errorbar(temps, avg_zealot, yerr=avg_zealot_err, 
                          marker="o", color=color,  # No label for zealot plots
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
    
    # Add dashed horizontal line at y=0
    ax10.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax10.set_xlabel("Temperature", fontsize=LABEL_SIZE)
    ax10.set_ylabel("Average Zealot Spin", fontsize=LABEL_SIZE)
    ax10.grid(True, alpha=0.3, linewidth=1.5)
    ax10.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax10.spines.values():
        spine.set_linewidth(2)
    
    # 5. z+ and z- (bottom middle)
    for i, (results, label, (color1, color2)) in enumerate(zip(results_list, labels, colors)):
        temps = results['temperatures']
        z_plus = results['z_plus_avgs']
        z_minus = results['z_minus_avgs']
        z_plus_err = results['z_plus_avgs_std_errors']
        z_minus_err = results['z_minus_avgs_std_errors']
        
        if not np.isnan(z_plus).all():
            ax11.errorbar(temps, z_plus, yerr=z_plus_err, 
                          marker="o", color=color1,  # No label for zealot plots
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
            
        if not np.isnan(z_minus).all():
            ax11.errorbar(temps, z_minus, yerr=z_minus_err, 
                          marker="s", color=color2,  # No label for zealot plots
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
    
    # Add dashed horizontal line at y=0
    ax11.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax11.set_xlabel("Temperature", fontsize=LABEL_SIZE)
    ax11.set_ylabel("")  # Hide y-axis label for middle plot since it shares with left plot
    ax11.grid(True, alpha=0.3, linewidth=1.5)
    ax11.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax11.spines.values():
        spine.set_linewidth(2)
    
    # 6. f+ only (bottom right)
    for results, label, (color1, _) in zip(results_list, labels, colors):
        temps = results['temperatures']
        f_plus = results['f_plus_list']
        f_plus_err = results['f_plus_std_errors']
        
        if not np.isnan(f_plus).all():
            ax12.errorbar(temps, f_plus, yerr=f_plus_err, 
                          marker="o", color=color1,  # No label for zealot plots
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
    
    ax12.set_xlabel("Temperature", fontsize=LABEL_SIZE)
    ax12.set_ylabel("Positive Fraction", fontsize=LABEL_SIZE)
    ax12.grid(True, alpha=0.3, linewidth=1.5)
    ax12.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax12.spines.values():
        spine.set_linewidth(2)
    
    # Adjust layout with shared axes and more horizontal space
    plt.tight_layout()
    
    # Adjust subplots to make much more room for legends
    plt.subplots_adjust(bottom=subplot_adjustments['bottom'], 
                        wspace=subplot_adjustments['wspace'], 
                        hspace=subplot_adjustments['hspace'],
                        top=subplot_adjustments['top'])

    
    # Adjust position of the third column to create more space for axis labels
    for i in range(2):  # For both rows
        pos = axes[i][2].get_position()
        # Move the third column plots slightly to the right
        # Increase the x0 value to add more space
        axes[i][2].set_position([pos.x0 + 0.02, pos.y0, pos.width, pos.height])
        
    # Add legends for each column, properly centered
    # First column legend (left)
    leg1 = fig.legend(handles_col0, labels_col0, loc='lower center', 
               bbox_to_anchor=(legend_column_positions[0], legend_y_positions[0]), fontsize=LEGEND_SIZE,
               ncol=min(2, len(labels_col0)), frameon=True)
    leg1.get_frame().set_linewidth(2)
    
    # Second column legend (middle)
    # Group all m+ entries first, then all m- entries
    m_plus_handles = []
    m_plus_labels = []
    m_minus_handles = []
    m_minus_labels = []
    
    # Split into two groups for clearer organization
    for i in range(0, len(handles_col1), 2):
        if i < len(handles_col1):
            m_plus_handles.append(handles_col1[i])
            m_plus_labels.append(labels_col1[i])
        if i+1 < len(handles_col1):
            m_minus_handles.append(handles_col1[i+1])
            m_minus_labels.append(labels_col1[i+1])
    
    combined_handles = m_plus_handles + m_minus_handles
    combined_labels = m_plus_labels + m_minus_labels
    
    # Calculate number of columns for legend - changed to 2 columns max
    n_datasets = len(labels)
    n_cols = min(n_datasets, 2)  # Maximum 2 columns for clarity
    
    leg2 = fig.legend(combined_handles, combined_labels, loc='lower center', 
               bbox_to_anchor=(legend_column_positions[1], legend_y_positions[1]), fontsize=LEGEND_SIZE,
               ncol=n_cols, frameon=True)
    leg2.get_frame().set_linewidth(2)
    
    # Third column legend (right)
    leg3 = fig.legend(handles_col2, labels_col2, loc='lower center', 
               bbox_to_anchor=(legend_column_positions[2], legend_y_positions[2]), fontsize=LEGEND_SIZE,
               ncol=min(2, len(labels_col2)), frameon=True)
    leg3.get_frame().set_linewidth(2)
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    # Show the plot
    plt.show()
    
    return fig, axes

def plot_combined_results_subplot_report_2(results_list, labels, save_path=None, figsize=(24, 16),
                                         label_positions=None, legend_column_positions=None, 
                                         legend_y_positions=None, subplot_adjustments=None):
    """
    Generate a 2x3 subplot layout with non-zealot plots on the top row
    and zealot plots on the bottom row, styled for reports with larger text.
    Plots only g+ and f+ (not g- and f-).
    
    Features:
    - Shared x-axes between top and bottom rows
    - Independent y-axes for all plots for maximum detail
    - Plot labels (a-f) positioned in bottom right (except c in top left)
    - Dashed horizontal lines at y=0 for magnetization plots
    - No titles, maximizing plot area
    - Larger text and numbers
    - Two-column legends with improved positioning
    - Adjusted column positions to make room for y-axis labels
    
    Args:
        results_list: List of loaded results dictionaries from load_processed_results
        labels: List of labels for each dataset
        save_path: Optional path to save the plot
        figsize: Size of the figure (width, height) in inches
        label_positions: List of (x, y) tuples for positioning plot labels (a-f)
                         Default positions have a-b-d-e-f in bottom right, c in top left
        legend_column_positions: List of x positions for the three legend columns
                                Default: [0.24, 0.5, 0.79] (centers of columns)
        legend_y_positions: List of y positions for the three legends
                           Default: [0.18, 0.12, 0.18]
        subplot_adjustments: Dictionary with subplot adjustment parameters
                            Default: {'bottom': 0.36, 'wspace': 0.12, 'hspace': 0.15}
    """
    # Set default values for new parameters if not provided
    if label_positions is None:
        label_positions = [
            (0.95, 0.05),  # a - bottom right
            (0.95, 0.1),   # b - bottom right
            (0.03, 0.92),  # c - top left (special case)
            (0.95, 0.05),  # d - bottom right
            (0.95, 0.1),   # e - bottom right
            (0.95, 0.15),  # f - bottom right
        ]
    
    if legend_column_positions is None:
        legend_column_positions = [0.24, 0.5, 0.79]  # Centers of columns
    
    if legend_y_positions is None:
        legend_y_positions = [0.18, 0.12, 0.18]  # Y positions for each legend
    
    if subplot_adjustments is None:
        subplot_adjustments = {
            'bottom': 0.36,
            'wspace': 0.12,
            'hspace': 0.15,
            'top': 0.95
        }
    
    # Create a figure with increased width
    fig = plt.figure(figsize=figsize)
    
    # Create subplot grid with more horizontal space
    gs = fig.add_gridspec(2, 3, hspace=0.15, wspace=0.20)
    
    # Create axes with shared x axes but independent y axes
    # Top row
    ax00 = fig.add_subplot(gs[0, 0])  # top left
    ax01 = fig.add_subplot(gs[0, 1])  # top middle - NO shared y with top left
    ax02 = fig.add_subplot(gs[0, 2])  # top right
    
    # Bottom row - share x-axes with top row
    ax10 = fig.add_subplot(gs[1, 0], sharex=ax00)  # bottom left - shares x with top left
    ax11 = fig.add_subplot(gs[1, 1], sharex=ax01)  # bottom middle - shares x with top middle but NO shared y
    ax12 = fig.add_subplot(gs[1, 2], sharex=ax02)  # bottom right - shares x with top right
    
    # Create a list of all axes for easier iteration
    axes = [[ax00, ax01, ax02], [ax10, ax11, ax12]]
    
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
    
    # Set larger font sizes
    LABEL_SIZE = 26
    TICK_SIZE = 24
    LEGEND_SIZE = 24
    
    # Create empty lists to store handles and labels for shared legends
    handles_col0 = []
    labels_col0 = []
    handles_col1 = []
    labels_col1 = []
    handles_col2 = []
    labels_col2 = []
    
    # Add plot labels (a-f) with improved positioning
    label_texts = ['(a)', '(b)', '(c)', '(d)', '(e)', '(f)']
    
    for row in range(2):
        for col in range(3):
            idx = row*3 + col
            pos_x, pos_y = label_positions[idx]
            
            # For most labels, use bottom right alignment
            ha = 'center'
            va = 'center'
            
            axes[row][col].text(pos_x, pos_y, label_texts[idx], 
                              transform=axes[row][col].transAxes, 
                              fontsize=LABEL_SIZE+2, fontweight='bold',
                              ha=ha, va=va)
    
    # TOP ROW: Non-zealot plots
    
    # 1. Average Magnetization (top left)
    for results, label, color in zip(results_list, labels, single_colors):
        temps = results['temperatures']
        avg_mag = results['average_magnetizations']
        avg_mag_err = results['average_magnetizations_std_errors']
        
        if not np.isnan(avg_mag).all():
            line = ax00.errorbar(temps, avg_mag, yerr=avg_mag_err, 
                            marker="o", label=label, color=color,
                            capsize=5, markersize=10, linewidth=3, elinewidth=2)
            # Store handles and labels for shared legend
            handles_col0.append(line)
            labels_col0.append(label)
    
    # Add dashed horizontal line at y=0
    ax00.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax00.set_ylabel("Average Magnetization", fontsize=LABEL_SIZE)
    ax00.grid(True, alpha=0.3, linewidth=1.5)
    ax00.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax00.spines.values():
        spine.set_linewidth(2)
    
    # 2. M+ and M- (top middle)
    for i, (results, label, (color1, color2)) in enumerate(zip(results_list, labels, colors)):
        temps = results['temperatures']
        m_plus = results['m_plus_avgs']
        m_minus = results['m_minus_avgs']
        m_plus_err = results['m_plus_avgs_std_errors']
        m_minus_err = results['m_minus_avgs_std_errors']
        
        if not np.isnan(m_plus).all():
            line1 = ax01.errorbar(temps, m_plus, yerr=m_plus_err, 
                            marker="o", label=f"{label} $m_+$", color=color1,
                            capsize=5, markersize=10, linewidth=3, elinewidth=2)
            handles_col1.append(line1)
            labels_col1.append(f"{label} $m_+$")
            
        if not np.isnan(m_minus).all():
            line2 = ax01.errorbar(temps, m_minus, yerr=m_minus_err, 
                            marker="s", label=f"{label} $m_-$", color=color2,
                            capsize=5, markersize=10, linewidth=3, elinewidth=2)
            handles_col1.append(line2)
            labels_col1.append(f"{label} $m_-$")
    
    # Add dashed horizontal line at y=0
    ax01.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Add y-axis label for middle plot since it now has its own scale
    ax01.set_ylabel("Average $m_+$ and $m_-$", fontsize=LABEL_SIZE)
    ax01.grid(True, alpha=0.3, linewidth=1.5)
    ax01.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax01.spines.values():
        spine.set_linewidth(2)
    
    # 3. g+ only (top right)
    for results, label, (color1, _) in zip(results_list, labels, colors):
        temps = results['temperatures']
        g_plus = results['g_plus_list']
        g_plus_err = results['g_plus_std_errors']
        
        if not np.isnan(g_plus).all():
            line = ax02.errorbar(temps, g_plus, yerr=g_plus_err, 
                          marker="o", label=f"{label} $g_+$", color=color1,
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
            handles_col2.append(line)
            labels_col2.append(f"{label} $g_+$")
    
    ax02.set_ylabel("Positive Fraction", fontsize=LABEL_SIZE)
    ax02.grid(True, alpha=0.3, linewidth=1.5)
    ax02.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax02.spines.values():
        spine.set_linewidth(2)
    
    # BOTTOM ROW: Zealot plots
    
    # 4. Average Zealot Spins (bottom left)
    for results, label, color in zip(results_list, labels, single_colors):
        temps = results['temperatures']
        avg_zealot = results['average_zealot_spins']
        avg_zealot_err = results['average_zealot_spins_std_errors']
        
        if not np.isnan(avg_zealot).all():
            ax10.errorbar(temps, avg_zealot, yerr=avg_zealot_err, 
                          marker="o", color=color,  # No label for zealot plots
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
    
    # Add dashed horizontal line at y=0
    ax10.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax10.set_xlabel("Temperature", fontsize=LABEL_SIZE)
    ax10.set_ylabel("Average Zealot Spin", fontsize=LABEL_SIZE)
    ax10.grid(True, alpha=0.3, linewidth=1.5)
    ax10.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax10.spines.values():
        spine.set_linewidth(2)
    
    # 5. z+ and z- (bottom middle)
    for i, (results, label, (color1, color2)) in enumerate(zip(results_list, labels, colors)):
        temps = results['temperatures']
        z_plus = results['z_plus_avgs']
        z_minus = results['z_minus_avgs']
        z_plus_err = results['z_plus_avgs_std_errors']
        z_minus_err = results['z_minus_avgs_std_errors']
        
        if not np.isnan(z_plus).all():
            ax11.errorbar(temps, z_plus, yerr=z_plus_err, 
                          marker="o", color=color1,  # No label for zealot plots
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
            
        if not np.isnan(z_minus).all():
            ax11.errorbar(temps, z_minus, yerr=z_minus_err, 
                          marker="s", color=color2,  # No label for zealot plots
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
    
    # Add dashed horizontal line at y=0
    ax11.axhline(y=0, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    
    ax11.set_xlabel("Temperature", fontsize=LABEL_SIZE)
    # Add y-axis label for middle plot since it now has its own scale
    ax11.set_ylabel("Average $Z_+$ and $Z_-$", fontsize=LABEL_SIZE)
    ax11.grid(True, alpha=0.3, linewidth=1.5)
    ax11.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax11.spines.values():
        spine.set_linewidth(2)
    
    # 6. f+ only (bottom right)
    for results, label, (color1, _) in zip(results_list, labels, colors):
        temps = results['temperatures']
        f_plus = results['f_plus_list']
        f_plus_err = results['f_plus_std_errors']
        
        if not np.isnan(f_plus).all():
            ax12.errorbar(temps, f_plus, yerr=f_plus_err, 
                          marker="o", color=color1,  # No label for zealot plots
                          capsize=5, markersize=10, linewidth=3, elinewidth=2)
    
    ax12.set_xlabel("Temperature", fontsize=LABEL_SIZE)
    ax12.set_ylabel("Positive Fraction", fontsize=LABEL_SIZE)
    ax12.grid(True, alpha=0.3, linewidth=1.5)
    ax12.tick_params(axis='both', which='major', labelsize=TICK_SIZE, width=2, length=8)
    for spine in ax12.spines.values():
        spine.set_linewidth(2)
    
    # Adjust layout with shared axes and more horizontal space
    plt.tight_layout()
    
    # Adjust subplots to make much more room for legends
    plt.subplots_adjust(bottom=subplot_adjustments['bottom'], 
                        wspace=subplot_adjustments['wspace'], 
                        hspace=subplot_adjustments['hspace'],
                        top=subplot_adjustments['top'])
    
    # Adjust position of the columns to create more space for axis labels
    for i in range(2):  # For both rows
        # Adjust second column position to make room for first column y-axis label
        pos_mid = axes[i][1].get_position()
        axes[i][1].set_position([pos_mid.x0 + 0.02, pos_mid.y0, pos_mid.width, pos_mid.height])
        
        # Adjust third column position to make room for second column y-axis label
        pos_right = axes[i][2].get_position()
        axes[i][2].set_position([pos_right.x0 + 0.04, pos_right.y0, pos_right.width, pos_right.height])
        
    # Add legends for each column, properly centered
    # First column legend (left)
    leg1 = fig.legend(handles_col0, labels_col0, loc='lower center', 
               bbox_to_anchor=(legend_column_positions[0], legend_y_positions[0]), fontsize=LEGEND_SIZE,
               ncol=min(2, len(labels_col0)), frameon=True)
    leg1.get_frame().set_linewidth(2)
    
    # Second column legend (middle)
    # Group all m+ entries first, then all m- entries
    m_plus_handles = []
    m_plus_labels = []
    m_minus_handles = []
    m_minus_labels = []
    
    # Split into two groups for clearer organization
    for i in range(0, len(handles_col1), 2):
        if i < len(handles_col1):
            m_plus_handles.append(handles_col1[i])
            m_plus_labels.append(labels_col1[i])
        if i+1 < len(handles_col1):
            m_minus_handles.append(handles_col1[i+1])
            m_minus_labels.append(labels_col1[i+1])
    
    combined_handles = m_plus_handles + m_minus_handles
    combined_labels = m_plus_labels + m_minus_labels
    
    # Calculate number of columns for legend - changed to 2 columns max
    n_datasets = len(labels)
    n_cols = min(n_datasets, 2)  # Maximum 2 columns for clarity
    
    leg2 = fig.legend(combined_handles, combined_labels, loc='lower center', 
               bbox_to_anchor=(legend_column_positions[1], legend_y_positions[1]), fontsize=LEGEND_SIZE,
               ncol=n_cols, frameon=True)
    leg2.get_frame().set_linewidth(2)
    
    # Third column legend (right)
    leg3 = fig.legend(handles_col2, labels_col2, loc='lower center', 
               bbox_to_anchor=(legend_column_positions[2], legend_y_positions[2]), fontsize=LEGEND_SIZE,
               ncol=min(2, len(labels_col2)), frameon=True)
    leg3.get_frame().set_linewidth(2)
    
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
 

#Ratio plotting
labels_ratio =  [r"$r_+$=48", r"$r_+$=48.5%", r"$r_+$=49%", r"$r_+$=49.5%", r"$r_+$=50%", r"$r_+$=51%"]
results_ratio = [combined_results_ratio_480, combined_results_ratio_485, combined_results_ratio_490, combined_results_ratio_495, combined_results_ratio_500, combined_results_ratio_510]
fig, axes = plot_combined_results_subplot_report(
    results_ratio, 
    labels_ratio, 
    save_path="fc_ratio_combined_plots.png",
    figsize=(24, 16),
    label_positions=[
        (0.9, 0.1),  # a - bottom right
        (0.9, 0.1),  # b - bottom right
        (0.9, 0.1),  # c - bottom right
        (0.9, 0.1),  # d - bottom right
        (0.9, 0.1),  # e - bottom right
        (0.9, 0.1)   # f - bottom right
    ],
    legend_column_positions=[0.18, 0.475, 0.79],
    legend_y_positions=[0.21, 0.12, 0.21],
    subplot_adjustments={
        'bottom': 0.39,   
        'wspace': 0.12,  
        'hspace': 0.15,
        'top': 0.95      
    }
)



#hs plotting
labels_hs = ["$h_s$=1.1N", "$h_s$=N", "$h_s$=0.9N", "$h_s$=0.8N","$h_s$=0.7N"]
results_hs = [combined_results_hs_11000, combined_results_ratio_500, combined_results_hs_09000, combined_results_hs_08000, combined_results_hs_07000]
fig, axes = plot_combined_results_subplot_report_2(
    results_hs, 
    labels_hs, 
    save_path="fc_hs_combined_plots.png",
    figsize=(24, 16),
    label_positions=[
        (0.9, 0.1),  # a - bottom right
        (0.9, 0.9),  # b - bottom right
        (0.1, 0.9),  # c - bottom right
        (0.9, 0.1),  # d - bottom right
        (0.9, 0.15),  # e - bottom right
        (0.9, 0.25)   # f - bottom right
    ],
    legend_column_positions=[0.19, 0.48, 0.79],
    legend_y_positions=[0.18, 0.12, 0.18],
    subplot_adjustments={
        'bottom': 0.36,   
        'wspace': 0.12,  
        'hspace': 0.15,
        'top': 0.95    
    }
)

#hs limit plotting
labels_critical_hs = ["$h_s$=0.6N", "$h_s$=0.6125N", "$h_s$=0.625N", "$h_s$=0.6375N","$h_s$=0.65N"]
results_critical_hs = [results_hs_06000, results_hs_06125, results_hs_06250, results_hs_06375, results_hs_06500]
fig, axes = plot_combined_results_subplot_report_2(
    results_critical_hs, 
    labels_critical_hs, 
    save_path="fc_hs_critical_combined_plots.png",
    figsize=(24, 16),
    label_positions=[
        (0.9, 0.1),  # a - bottom right
        (0.9, 0.9),  # b - bottom right
        (0.1, 0.9),  # c - bottom right
        (0.9, 0.1),  # d - bottom right
        (0.9, 0.15),  # e - bottom right
        (0.9, 0.25)   # f - bottom right
    ],
    legend_column_positions=[0.19, 0.48, 0.79],
    legend_y_positions=[0.21, 0.08, 0.21],
    subplot_adjustments={
        'bottom': 0.38,   
        'wspace': 0.12,  
        'hspace': 0.15,
        'top': 0.95    
    }
)



#JS plotting
labels_Js = ["$J_s$=0.95", "$J_s$=0.99", "$J_s$=1.00", "$J_s$=1.01", "$J_s$=1.02", "$J_s$=1.05"]
results_Js = [results_Js_095, results_Js_099, results_Js_100, combined_results_Js_101, combined_results_Js_102, combined_results_Js_105]
fig, axes = plot_combined_results_subplot_report(
    results_Js, 
    labels_Js, 
    save_path="fc_Js_combined_plots.png",
    figsize=(24, 16),
    label_positions=[
        (0.9, 0.1),  # a - bottom right
        (0.9, 0.1),  # b - bottom right
        (0.1, 0.9),  # c - bottom right
        (0.9, 0.1),  # d - bottom right
        (0.9, 0.1),  # e - bottom right
        (0.9, 0.1)   # f - bottom right
    ],
    legend_column_positions=[0.23, 0.5, 0.79],
    legend_y_positions=[0.21, 0.12, 0.21],
    subplot_adjustments={
        'bottom': 0.39,   
        'wspace': 0.12,  
        'hspace': 0.15,
        'top': 0.95    
    }
)







#hs plotting
labels_hs = ["$h_s$=0.6N", "$h_s$=0.625N", "$h_s$=0.7N", "$h_s$=0.8N", "$h_s$=N", "$h_s$=1.1N"]
results_hs = [results_hs_06000, results_hs_06250, combined_results_hs_07000, combined_results_hs_08000, combined_results_ratio_500, combined_results_hs_11000]
fig, axes = plot_combined_results_subplot_report_2(
    results_hs, 
    labels_hs, 
    save_path="fc_hs_new_combined_plots.png",
    figsize=(24, 16),
    label_positions=[
        (0.1, 0.9),  # a - bottom right
        (0.9, 0.9),  # b - bottom right
        (0.1, 0.9),  # c - bottom right
        (0.9, 0.1),  # d - bottom right
        (0.9, 0.15),  # e - bottom right
        (0.9, 0.1)   # f - bottom right
    ],
    legend_column_positions=[0.19, 0.48, 0.79],
    legend_y_positions=[0.21, 0.12, 0.21],
    subplot_adjustments={
        'bottom': 0.39,   
        'wspace': 0.12,  
        'hspace': 0.15,
        'top': 0.95    
    }
)