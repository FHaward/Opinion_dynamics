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

def load_and_replace_results(original_json_path, *replacement_json_paths):
    """
    Load processed results from the original JSON file, replace values for all temperatures
    found in multiple replacement JSON files, and return in the same format as load_processed_results.
    
    Args:
        original_json_path: Path to the original JSON file containing numerical results
        *replacement_json_paths: Variable number of paths to JSON files containing replacement values
        
    Returns:
        Dictionary containing all the loaded results with proper numpy arrays and replaced values
    """
    # Load original dataset
    with open(original_json_path, 'r') as f:
        original_data = json.load(f)
    
    # Process each replacement file
    for replacement_path in replacement_json_paths:
        print(f"Processing replacement file: {replacement_path}")
        
        # Load replacement data
        with open(replacement_path, 'r') as f:
            replacement_data = json.load(f)
        
        # Get all temperatures from replacement data
        replacement_temps = replacement_data['temperatures']
        
        # For each temperature in the replacement data
        for i, target_temp in enumerate(replacement_temps):
            try:
                # Try to find exact match
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
                        # Make sure the replacement data has enough elements
                        if i < len(replacement_data[key]):
                            original_data[key][index_to_replace] = replacement_data[key][i]
                        else:
                            print(f"Warning: Replacement data for '{key}' doesn't have enough elements for temperature {target_temp}")
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
                    marker="o", label=f"{label}", color=color1,
                    capsize=5, markersize=4)
    
    plt.xlabel("Temperature", fontsize=12)
    plt.ylabel("Positive Fraction", fontsize=12)
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


#file paths

#ratio
path_ratio_55 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\ratio\ratio_55.00000000000001\20250413_203826_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_50 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\ratio\ratio_50.0\20250413_200043_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_45 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\ratio\ratio_45.0\20250413_204038_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_425 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\ratio\ratio_42.5\20250416_124336_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_40 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\ratio\ratio_40.0\20250413_210024_J_s_1.01_h_s_10000\numerical_results.json"
path_ratio_35 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\ratio\ratio_35.0\20250413_215552_J_s_1.01_h_s_10000\numerical_results.json"

crit_path_ratio_45 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\ratio\ratio_40.0\20250416_124545_J_s_1.01_h_s_10000\numerical_results.json"
crit_path_500 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\ratio\ratio_35.0\20250416_114103_J_s_1.01_h_s_10000\numerical_results.json"

#h_s
path_hs_110 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_11000.0\20250415_220401_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_095 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_9500.0\20250415_221623_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_090 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_9000.0\20250414_172924_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_085 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_8500.0\20250415_234958_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_080 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_8000.0\20250414_175106_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_075 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_7500.0\20250416_002523_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_070 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_7000.0\20250414_182602_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_065 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_6500.0\20250416_002558_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_060 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_6000.0\20250414_205538_ratio_50.0_J_s_1.01\numerical_results.json"
path_hs_0625= r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_6250.0\20250422_175142_ratio_50.0_J_s_1.01\numerical_results.json"

#J_s
path_Js_105 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_10000\20250415_195213_ratio_50.0_J_s_1.05\numerical_results.json"
path_Js_102 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_10000\20250415_194824_ratio_50.0_J_s_1.02\numerical_results.json"
path_Js_100 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_10000\20250415_190523_ratio_50.0_J_s_1.0\numerical_results.json"
path_Js_099 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_10000\20250415_194906_ratio_50.0_J_s_0.99\numerical_results.json"
path_Js_095 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\h_s\h_s_10000\20250415_201147_ratio_50.0_J_s_0.95\numerical_results.json"

crit_path_Js_100 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\J_s\J_s_1.0\20250416_124645_ratio_50.0_h_s_10000\numerical_results.json"
crit_path_Js_102 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\J_s\J_s_1.02\20250416_115353_ratio_50.0_h_s_10000\numerical_results.json"
crit_path_Js_105 = r"C:\Users\frase\Documents\Durham\4th Year\1Project\Results_2\simulation_results\nearest_neighbour\J_s\J_s_1.05\20250416_124649_ratio_50.0_h_s_10000\numerical_results.json"


#ratio results
results_ratio_35 = load_and_replace_results(path_ratio_35, crit_path_500)
results_ratio_40 = load_and_replace_results(path_ratio_40, crit_path_ratio_45, crit_path_500)
results_ratio_425 = load_and_replace_results(path_ratio_425, crit_path_500)
results_ratio_45 = load_and_replace_results(path_ratio_45, crit_path_500)
results_ratio_50 = load_and_replace_results(path_ratio_50, crit_path_500)
results_ratio_55 = load_and_replace_results(path_ratio_55, crit_path_500)

#hs results
results_hs_110 = load_and_replace_results(path_hs_110, crit_path_500)
results_hs_095 = load_and_replace_results(path_hs_095, crit_path_500)
results_hs_090 = load_processed_results(path_hs_090)
results_hs_085 = load_processed_results(path_hs_085)
results_hs_080 = load_processed_results(path_hs_080)
results_hs_075 = load_processed_results(path_hs_075)
results_hs_070 = load_processed_results(path_hs_070)
results_hs_065 = load_processed_results(path_hs_065)
results_hs_060 = load_processed_results(path_hs_060)
results_hs_0625 = load_processed_results(path_hs_0625)

#Js results
results_Js_105 = load_and_replace_results(path_Js_105, crit_path_Js_105)
results_Js_102 = load_and_replace_results(path_Js_102, crit_path_Js_102)
results_Js_100 = load_and_replace_results(path_Js_100, crit_path_Js_100)
results_Js_099 = load_processed_results(path_Js_099)
results_Js_095 = load_processed_results(path_Js_095)


#Ratio plotting
labels_ratio =  [r"$r_+$=35%", r"$r_+$=40%", r"$r_+$=42.5%", r"$r_+$=45%", r"$r_+$=50%", r"$r_+$=55%"]
results_ratio = [results_ratio_35, results_ratio_40, results_ratio_425, results_ratio_45, results_ratio_50, results_ratio_55]
fig, axes = plot_combined_results_subplot_report(
    results_ratio, 
    labels_ratio, 
    save_path="nn_ratio_combined_plots.png",
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


["$h_s$=1.1N", "$h_s$=N", "$h_s$=0.8N", "$h_s$=0.7N","$h_s$=0.625N", "$h_s$=0.6N"]
[results_hs_110, results_ratio_50, results_hs_080, results_hs_070, results_hs_0625, results_hs_060]

#hs plotting
labels_hs = ["$h_s$=0.6N", "$h_s$=0.625N", "$h_s$=0.7N", "$h_s$=0.8N", "$h_s$=N", "$h_s$=1.1N"]
results_hs = [results_hs_060, results_hs_0625, results_hs_070, results_hs_080, results_ratio_50, results_hs_110]
fig, axes = plot_combined_results_subplot_report(
    results_hs, 
    labels_hs, 
    save_path="nn_hs_combined_plots.png",
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
    legend_y_positions=[0.21, 0.12, 0.21],
    subplot_adjustments={
        'bottom': 0.39,   
        'wspace': 0.12,  
        'hspace': 0.15,
        'top': 0.95      
    }
)


#JS plotting
labels_Js = ["$J_s$=0.95", "$J_s$=0.99", "$J_s$=1.00", "$J_s$=1.01", "$J_s$=1.02", "$J_s$=1.05"]
results_Js = [results_Js_095, results_Js_099, results_Js_100, results_ratio_50, results_Js_102, results_Js_105]
fig, axes = plot_combined_results_subplot_report(
    results_Js, 
    labels_Js, 
    save_path="nn_Js_combined_plots.png",
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
    legend_y_positions=[0.18, 0.12, 0.18],
    subplot_adjustments={
        'bottom': 0.36,   
        'wspace': 0.12,  
        'hspace': 0.15,
        'top': 0.95      
    }
)
