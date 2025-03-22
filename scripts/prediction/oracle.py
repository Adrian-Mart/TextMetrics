import json
import os
import argparse
from ..data_processing import analyzer

def main():
    parser = argparse.ArgumentParser(description='Text processing with oracle.')
    parser.add_argument('-n', '--name', required=True, help='Name of the element to be processed')
    parser.add_argument('-i', '--input_dir', required=True, help='Path to the data directory with the distance matrix')
    parser.add_argument('-o', '--output_log', required=True, help='Path to the output log JSON file')
    parser.add_argument('-e', '--extreme_values', required=True, default='data/extreme_values.json', help='Path to the extreme values JSON file')
    parser.add_argument('-a', '--alpha', required=True, type=float, help='Alpha value for normalization')
    parser.add_argument('-b', '--beta', required=True, type=float, help='Beta value for normalization')
    parser.add_argument('-g', '--gamma', required=True, type=float, help='Gamma value for normalization')
    parser.add_argument('-t', '--threshold', type=float, default=50, help='Threshold value for distance matrix')
    parser.add_argument('-bd', '--base_dir', help='Base directory for the output files')
    args = parser.parse_args()

    # If base directory is not provided, use the base directory of the input directory
    if args.base_dir is None:
        args.base_dir = args.input_dir.split(os.sep)[0]

    run_oracle(args.name, args.alpha, args.beta, args.gamma, args.threshold, args.input_dir, args.output_log, args.extreme_values, args.base_dir)

def run_oracle(name, alpha, beta, gamma, threshold, input_dir, output_log, extreme_values, output_dir):
    """
    Runs the oracle prediction process.
    Parameters:
    name (str): The name of the node to search for in the distance matrix.
    alpha (float): A parameter for the oracle process (not used in the current implementation).
    beta (float): A parameter for the oracle process (not used in the current implementation).
    gamma (float): A parameter for the oracle process (not used in the current implementation).
    threshold (float): A threshold value for the oracle process (not used in the current implementation).
    input_dir (str): The directory where the distance matrix file is located.
    output_log (str): The file path where the output log will be saved.
    Raises:
    FileNotFoundError: If the distance matrix file is not found in the input directory.
    Returns:
    None
    """

    # Run the analyzer
    analyzer.main(input_dir, alpha, beta, gamma, threshold, output_dir)

    distance_matrix_path = os.path.join(output_dir, 'data_distance_matrix.json')
    if not os.path.exists(distance_matrix_path):
        raise FileNotFoundError(f"Distance matrix file not found in {output_dir}")

    # Get extreme values from extreme values file
    with open(extreme_values, 'r') as file:
        extreme_values = json.load(file)

    with open(distance_matrix_path, 'r') as f:
        # Load the distance matrix
        data_distance_matrix = json.load(f)
        # Search n value in node names (or the closest to it) from the data distance file
        node_names = data_distance_matrix['node_names']
        n_value_index = node_names.index(f'{name.replace('_', ' ')}_words.json')
        # Get the relationship value for the n value
        relationship_value = data_distance_matrix['relationship_values'][n_value_index]
        # Get the minimum distance value for the n value excluding 0
        min_distance = min([dist for dist in data_distance_matrix['raw_distance_matrix'][n_value_index] if dist > 0])
        # Create the result object
        result = {
            'name': name,
            'relationship': relationship_value,
            'min_distance': min_distance,
            'extreme_values': extreme_values,
            'pass': min_distance < extreme_values['min_value']
        }

    # Append the result to the output log
    if os.path.exists(output_log):
        with open(output_log, 'r') as file:
            output_data = json.load(file)
    else:
        output_data = []

    output_data.append(result)

    with open(output_log, 'w') as file:
        json.dump(output_data, file)

if __name__ == '__main__':
    main()