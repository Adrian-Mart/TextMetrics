import json
import os
import shutil
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Text processing with Oracle.')
    parser.add_argument('-n', required=True, help='Value for the -n argument')
    parser.add_argument('-a', '--alpha', type=float, default=1.0, help='Alpha value for normalization')
    parser.add_argument('-b', '--beta', type=float, default=1.0, help='Beta value for normalization')
    parser.add_argument('-g', '--gamma', type=float, default=1.0, help='Gamma value for normalization')
    parser.add_argument('-t', '--threshold', type=float, default=50, help='Threshold value for distance matrix')
    parser.add_argument('-s', '--skip', help='Skips the download process', action='store_true')
    args = parser.parse_args()

    # Delete the Oracle folder
    oracle_dir = 'Oracle'
    
    
    n_value = args.n
    
    if not args.skip:
        if os.path.exists(oracle_dir):
            shutil.rmtree(oracle_dir)
        # Run downloader with parameters -o 'Oracle/Raw' and -n
        subprocess.run(['python', 'downloader.py', '-ns', '-y', '-o', 'Oracle/Raw', '-n', n_value])
    
        # Run processor with 'Oracle' as base path
        subprocess.run(['python', 'processor.py', '-i', 'Oracle/Raw', '-o', 'Oracle/Words'])
    
        # Run distance matrix generator with 'Oracle' as base path
        subprocess.run(['python', 'distance_matrix_generator.py', '-i', 'Oracle/Words', '-o', 'Oracle/Distances'])
    
    # Run analyzer with default as base path
    subprocess.run(['python', 'analyzer.py', '-a', str(args.alpha), '-b', str(args.beta), '-g', str(args.gamma), '-t', str(args.threshold)])

    # Run analyzer with 'Oracle' as base path
    subprocess.run(['python', 'analyzer.py', '-i', 'Oracle/Distances', '-o', 'Oracle', '-a', str(args.alpha), '-b', str(args.beta), '-g', str(args.gamma), '-t', str(args.threshold)])

    # Get extreme values from 'Data/extreme_values.json'
    with open('Data/extreme_values.json', 'r') as file:
        extreme_values = json.load(file)
        
    # Print the extreme values
    print(extreme_values)

    # Get the data distance matrix from 'Oracle/data_distance_matrix.json'
    with open('Oracle/data_distance_matrix.json', 'r') as file:
        data_distance_matrix = json.load(file)
        # Search n value in node names (or the closest to it) from the data distance file
        node_names = data_distance_matrix['node_names']
        n_value_index = node_names.index(f'{n_value.replace('_', ' ')}_words.json')
        # Get the relationship value for the n value
        relationship_value = data_distance_matrix['relationship_values'][n_value_index]
        # Print the relationship value
        print(relationship_value)
        # Get the minimum distance value for the n value excluding 0
        min_distance = min([dist for dist in data_distance_matrix['raw_distance_matrix'][n_value_index] if dist > 0])
        # Print the minimum distance value
        print(min_distance)
        # Check if the minimum distance is less than the minimum value from the extreme values
        if min_distance < extreme_values['min_value']:
            print('The minimum distance is less than the minimum value')
        else:
            print('The minimum distance is not less than the minimum value')


if __name__ == "__main__":
    main()
