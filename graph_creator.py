import numpy as np
import os
import json
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def process_node_name(name):
    name = name.replace("_words.json", "")
    name = name.replace("(video_game)", "")
    name = name.replace("_", " ")
    return name

def create_graph(distance_matrix, node_names):
    G = nx.Graph()
    num_nodes = len(node_names)
    processed_names = [process_node_name(name) for name in node_names]
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if distance_matrix[i, j] > 0:
                G.add_edge(processed_names[i], processed_names[j], weight=distance_matrix[i, j])
    return G, processed_names


def draw_graph(G, processed_names, relationship_values, graph_title="Graph Title", edge_label="Edge Weight", relationship_label="Node Relationship Value"):
    pos = nx.spring_layout(G, seed=42)
    edges = G.edges(data=True)
    weights = [edge[2]['weight'] for edge in edges]
    
    # Normalize weights for color mapping
    norm = mcolors.Normalize(vmin=0, vmax=100)
    cmap = mcolors.LinearSegmentedColormap.from_list('alpha_cmap', [(0, 0, 1), (0, 1, 1)])
    
    edge_colors = [cmap(norm(weight)) for weight in weights]
    
    # Normalize relationship values for node color mapping
    node_norm = mcolors.Normalize(vmin=0, vmax=100)
    node_cmap = mcolors.LinearSegmentedColormap.from_list('node_cmap', [(0, 1, 1), (1, 1, 0)])
    node_colors = [node_cmap(node_norm(value)) for value in relationship_values]
    
    # Use numbers for nodes and add labels
    labels = {name: str(i + 1) for i, name in enumerate(G.nodes())}
    
    # Draw the graph
    fig, ax = plt.subplots()
    nx.draw(G, pos, with_labels=True, labels=labels, node_size=700, node_color=node_colors[:len(G.nodes)], edge_color=edge_colors, edge_cmap=cmap, width=2, ax=ax)
    
    # Add legend for node labels
    legend_text = "\n".join([f"{i + 1}: {label}" for i, label in enumerate(processed_names)])
    plt.gcf().text(0.02, 0.98, legend_text, fontsize=10, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.5, edgecolor='black'))
    
    # Add colorbars for edge weights and node relationship values
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation='vertical', fraction=0.046, pad=0.12)
    cbar.set_label(edge_label)
    
    sm_nodes = plt.cm.ScalarMappable(cmap=node_cmap, norm=node_norm)
    sm_nodes.set_array([])
    cbar_nodes = plt.colorbar(sm_nodes, ax=ax, orientation='vertical', fraction=0.046, pad=0.12)  # Increased pad to give more space
    cbar_nodes.set_label(relationship_label)
    
    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    # Add graph title
    plt.gcf().text(0.5, 0.02, graph_title, fontsize=12, horizontalalignment='center')
    
    plt.show()


def main():
    # Get rounded_distance from the data_distance_matrix.json file
    with open("Data/data_distance_matrix.json", 'r') as file:
        data = json.load(file)
        rounded_distance = np.array(data["distance_matrix"])
        row_names = data["node_names"]
        relationship_values = data["relationship_values"]
    
    # Calculate the relationship matrix
    G, processed_names = create_graph(rounded_distance, row_names)
    draw_graph(G, processed_names, relationship_values, "Gráfica de relación entre los videojuegos más vendidos", "Relación entre videojuegos", "Valor de relación entre videojuegos")

if __name__ == "__main__":
    main()