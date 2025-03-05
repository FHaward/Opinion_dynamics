import networkx as nx
import matplotlib.pyplot as plt
import math

def create_complete_graph(n):
    """Create a complete graph with n nodes"""
    return nx.complete_graph(n)

def circular_layout(G):
    """
    Create a circular layout for the graph nodes
    Ensures nodes are evenly spaced around a circle
    """
    n = len(G.nodes())
    pos = {}
    for i, node in enumerate(G.nodes()):
        angle = 2 * math.pi * i / n
        pos[node] = (math.cos(angle), math.sin(angle))
    return pos

# Create a grid of subplots
plt.figure(figsize=(16, 10))

# Generate and plot graphs from K3 to K10
for k in range(3, 11):
    # Create subplot
    ax = plt.subplot(2, 4, k-2)
    
    # Create the complete graph
    G = create_complete_graph(k)
    
    # Draw the graph with circular layout
    pos = circular_layout(G)
    nx.draw(G, pos, ax=ax, 
            with_labels=False,  # No node labels 
            node_color='lightblue', 
            node_size=100,  # Slightly larger nodes
            edge_color='gray',
            alpha=0.7)  # Slight transparency for better visibility
    
    # Set title for each subplot with large, bold font
    ax.set_title(f'K{k}', fontsize=20, fontweight='bold', pad=20)
    
    # Remove axis for cleaner look
    ax.set_axis_off()

# Adjust layout and save
plt.tight_layout(pad=4.0)
plt.savefig('complete_graphs_presentation.png', dpi=300, bbox_inches='tight')
plt.show()