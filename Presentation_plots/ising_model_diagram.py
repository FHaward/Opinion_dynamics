import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import random

def visualize_spin_lattice():
    """
    Visualize a simple 5x5 lattice with random up/down spins
    Arrows are placed at grid intersections and the grid is angled
    """
    # Fixed parameters
    size = 5  # 5x5 grid
    angle = 30  # degrees
    angle_rad = np.radians(angle)
    
    # Set random seed for reproducibility
    random.seed(1)
    np.random.seed(1)
    
    # Generate random spins (roughly 50:50 distribution)
    spins = np.random.choice([-1, 1], size=(size, size))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create horizontal grid lines (angled)
    for i in range(size):
        x_points = np.array([j for j in range(size)])
        y_points = np.ones_like(x_points) * i
        
        # Apply rotation to make grid appear angled
        x_rotated = x_points + y_points * np.sin(angle_rad)
        y_rotated = y_points * np.cos(angle_rad)
        
        ax.plot(x_rotated, y_rotated, color='black', linewidth=0.5)
    
    # Create vertical grid lines (angled)
    for j in range(size):
        y_points = np.array([i for i in range(size)])
        x_points = np.ones_like(y_points) * j
        
        # Apply rotation to make grid appear angled
        x_rotated = x_points + y_points * np.sin(angle_rad)
        y_rotated = y_points * np.cos(angle_rad)
        
        ax.plot(x_rotated, y_rotated, color='black', linewidth=0.5)
    
    # Bigger arrows
    arrow_length = 0.8
    
    # Draw arrows at each grid intersection
    for i in range(size):
        for j in range(size):
            # Calculate the center position with rotation
            x_center = j + i * np.sin(angle_rad)
            y_center = i * np.cos(angle_rad)
            
            # Vertical arrows (regardless of grid angle)
            if spins[i, j] == 1:  # Up spin
                arrow = FancyArrowPatch((x_center, y_center - arrow_length/2), 
                                       (x_center, y_center + arrow_length/2), 
                                       arrowstyle='->', color='red', 
                                       mutation_scale=20, linewidth=3)
            else:  # Down spin
                arrow = FancyArrowPatch((x_center, y_center + arrow_length/2), 
                                       (x_center, y_center - arrow_length/2), 
                                       arrowstyle='->', color='blue', 
                                       mutation_scale=20, linewidth=3)
            ax.add_patch(arrow)
    
    # Set limits precisely to the grid size
    ax.set_xlim(-0.5, size-0.5 + (size-1) * np.sin(angle_rad))
    ax.set_ylim(-0.5, (size-1) * np.cos(angle_rad) + 0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    
   
    # Add legend
    up_arrow = FancyArrowPatch((0, 0), (0, arrow_length), arrowstyle='->', color='red', 
                              mutation_scale=20, linewidth=3)
    down_arrow = FancyArrowPatch((0, arrow_length), (0, 0), arrowstyle='->', color='blue', 
                                mutation_scale=20, linewidth=3)
    
    ax.legend([up_arrow, down_arrow], ['Spin Up (+1)', 'Spin Down (-1)'],  prop={'size': 20})
    
    plt.tight_layout()
    return fig

# Run the visualization
if __name__ == "__main__":
    fig = visualize_spin_lattice()
    
    # Save the figure
    plt.savefig('spin_lattice_5x5.png', dpi=300, bbox_inches='tight')
    
    # Show the figure
    plt.show()