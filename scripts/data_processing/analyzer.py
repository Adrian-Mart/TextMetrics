import numpy as np
import os
import json
import argparse

def normalize_vector(alpha=1, beta=1, gamma=1):
    if not (0 <= alpha <= 1 and 0 <= beta <= 1 and 0 <= gamma <= 1):
        raise ValueError("All parameters must be in the range [0, 1]")
    
    vector = np.array([alpha, beta, gamma])
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    normalized_vector = vector / norm
    
    return normalized_vector

def calculate_distance(m1, m2, m3, alpha=1, beta=1, gamma=1):
    weights = normalize_vector(alpha, beta, gamma)
    distance = m1 * weights[0] + m2 * weights[1] + m3 * weights[2]
    
    return distance

def load_distance_matrices(directory):
    matrices = []
    row_names = []
    col_names = []
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r') as file:
                data = json.load(file)
                if "distance_matrix" in data:
                    matrices.append(np.array(data["distance_matrix"]))
                if "filenames" in data:
                    row_names = data["filenames"]
                    col_names = data["filenames"]
    return matrices, row_names, col_names

def interpolate_matrix(matrix):
    min_val = np.min(matrix[np.nonzero(matrix)])
    max_val = np.max(matrix)
    if min_val == max_val:
        return matrix
    interpolated_matrix = (matrix - min_val) / (max_val - min_val) * 100
    interpolated_matrix[matrix == 0] = 0
    return interpolated_matrix

def apply_threshold(matrix, threshold):
    thresholded_matrix = matrix.copy()
    thresholded_matrix[thresholded_matrix > threshold] = 100
    return thresholded_matrix

def calculate_relationship_values(matrix):
    column_sums = np.sum(matrix, axis=0)
    min_val = np.min(column_sums)
    max_val = np.max(column_sums)
    
    if min_val == max_val:
        return np.zeros_like(column_sums)
    
    interpolated_values = column_sums / max_val * 100
    return interpolated_values

def save_extreme_values(distance_matrix, node_names, alpha, beta, gamma, output_file):
    non_zero_matrix = distance_matrix[distance_matrix > 0]
    if non_zero_matrix.size == 0:
        raise ValueError("The distance matrix contains only zero values")
    
    min_val = np.min(non_zero_matrix)
    max_val = np.max(distance_matrix)
    min_indices = np.where(distance_matrix == min_val)
    max_indices = np.where(distance_matrix == max_val)
    
    min_nodes = [(node_names[i], node_names[j]) for i, j in zip(min_indices[0], min_indices[1])]
    max_nodes = [(node_names[i], node_names[j]) for i, j in zip(max_indices[0], max_indices[1])]
    
    extreme_values = {
        "alpha": alpha,
        "beta": beta,
        "gamma": gamma,
        "min_value": min_val,
        "min_nodes": min_nodes,
        "max_value": max_val,
        "max_nodes": max_nodes
    }
    
    with open(output_file, 'w') as file:
        json.dump(extreme_values, file, indent=4)

def main(directory, alpha, beta, gamma, threshold, output_dir):
    distance_matrices, row_names, _ = load_distance_matrices(directory)

    if len(distance_matrices) < 3:
        raise ValueError("There must be at least three distance matrices in the directory")

    m1, m2, m3 = distance_matrices[:3]
    distance = calculate_distance(m1, m2, m3, alpha, beta, gamma)
    
    if distance.size == 0:
        raise ValueError("The distance matrix is empty")
    interpolated_distance = interpolate_matrix(distance)
    relationship_values = calculate_relationship_values(interpolated_distance)
    interpolated_distance = apply_threshold(interpolated_distance, threshold)
    rounded_distance = np.round(interpolated_distance, 2)
    
    # Calculate the relationship values
    relationship_values = 100 - relationship_values

    # Save the rounded distance matrix and node names to a JSON file
    output_file = os.path.join(output_dir, 'data_distance_matrix.json')
    output_data = {
        "distance_matrix": rounded_distance.tolist(),
        "raw_distance_matrix": distance.tolist(),
        "node_names": row_names,
        "relationship_values": relationship_values.tolist()
    }
    with open(output_file, 'w') as file:
        json.dump(output_data, file, indent=4)

    # Save the extreme values to a JSON file
    extreme_values_file = os.path.join(output_dir, 'extreme_values.json')
    save_extreme_values(distance, row_names, alpha, beta, gamma, extreme_values_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process distance matrices.")
    parser.add_argument('-i', '--input', type=str, default='Data/Distances', help='Input directory containing distance matrices')
    parser.add_argument('-o', '--output', type=str, default='Data', help='Output directory for results')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='Alpha value for normalization')
    parser.add_argument('-b', '--beta', type=float, default=1.0, help='Beta value for normalization')
    parser.add_argument('-g', '--gamma', type=float, default=1.0, help='Gamma value for normalization')
    parser.add_argument('-t', '--threshold', type=float, default=50, help='Threshold value for distance matrix')

    args = parser.parse_args()

    main(args.input, args.alpha, args.beta, args.gamma, args.threshold, args.output)
