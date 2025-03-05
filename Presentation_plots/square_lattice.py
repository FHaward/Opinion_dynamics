import networkx as nx
import matplotlib.pyplot as plt

def create_square_lattice(m, n):
    """
    Create a square lattice graph
    
    Parameters:
    m (int): Number of rows
    n (int): Number of columns
    
    Returns:
    NetworkX graph representing the square lattice
    """
    # Create a grid graph
    G = nx.grid_2d_graph(m, n)
    return G

# Create the graph
rows, cols = 6, 8  # 6 rows, 8 columns
G = create_square_lattice(rows, cols)

# Create the plot
plt.figure(figsize=(10, 8))

# Draw the graph
pos = {(x,y):(y,-x) for x,y in G.nodes()}  # Adjust layout to make it more intuitive
nx.draw(G, pos, 
        with_labels=False,  # No node labels
        node_color='lightblue', 
        node_size=100,  # Node size
        edge_color='gray')

plt.title(f'Square Lattice ({rows} × {cols})', fontsize=20, fontweight='bold')
plt.axis('off')  # Turn off axis
plt.tight_layout()

# Save the figure
plt.savefig('square_lattice.png', dpi=300, bbox_inches='tight')
plt.show()