import os
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import euclidean_distances
import tqdm
import json
import argparse

def load_texts_from_folder(folder_path):
    texts = []
    filenames = []
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            with open(file_path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
            filenames.append(file)
    return texts, filenames

def load_texts_from_array(paths):
    texts = []
    filenames = []
    for path in paths:
        if os.path.isfile(path):
            with open(path, 'r', encoding='utf-8') as f:
                texts.append(f.read())
            filenames.append(os.path.basename(path))
    return texts, filenames

def calculate_distance_matrix(texts):
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(texts)
    distance_matrix = euclidean_distances(X)
    return distance_matrix

def save_distance_matrix(matrix, filenames, output_file):
    data = {
        'filenames': filenames,
        'distance_matrix': matrix.tolist()
    }
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f)

def get_distance_matrix(path):
    texts, filenames = load_texts_from_array(path)
    data = {
        'filenames': filenames,
        'distance_matrix': calculate_distance_matrix(texts).tolist()
    }
    return data

def calculate_distance_matrices(info_paths, link_paths, text_paths):
    info_matrix_data = get_distance_matrix(info_paths)
    link_matrix_data = get_distance_matrix(link_paths)
    text_matrix_data = get_distance_matrix(text_paths)
    return info_matrix_data, link_matrix_data, text_matrix_data

def main(input_path, output_path):
    if not output_path:
        output_path = os.path.join(args.base_dir, 'data/distances')

    os.makedirs(output_path, exist_ok=True)
    
    subfolders = [f.path for f in os.scandir(input_path) if f.is_dir()]

    with tqdm.tqdm(total=len(subfolders), desc="Processing") as pbar:
        for subfolder in subfolders:
            texts, filenames = load_texts_from_folder(subfolder)
            if input_path:
                tqdm.tqdm.write(f"Processing {os.path.basename(subfolder)}")
                input_texts, input_filenames = load_texts_from_folder(os.path.join(input_path, os.path.basename(subfolder)))
                texts.extend(input_texts)
                filenames.extend(input_filenames)
            if texts:
                distance_matrix = calculate_distance_matrix(texts)
                subfolder_name = os.path.basename(subfolder)
                output_file = os.path.join(output_path, f'{subfolder_name}_distances.json')
                save_distance_matrix(distance_matrix, filenames, output_file)
            pbar.update(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate distance matrices for text files.")
    parser.add_argument('-i', '--input', type=str, help='Input folder path')
    parser.add_argument('-o', '--output', type=str, help='Output folder path (required if -i is used)')
    parser.add_argument('-b', '--base-dir', default='.\\', type=str, help="Base directory for the output files")
    args = parser.parse_args()
    
    main(args.input, args.output)
