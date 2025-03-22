import argparse
import json
import os
from ..data_processing import downloader
from ..data_processing import processor
from ..data_processing import distance_matrix_generator as dmg
from ..data_processing import analyzer
from ..prediction import oracle

def verify_cache(output_dir, cache_file_path):
    print(f"Verifying cache file in {output_dir}")

    # if output_dir does not exist, pass
    if not os.path.exists(output_dir):
        return

    # Get a list of all the subdirs in the output dir
    subdirs = [name for name in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir, name))]
    # Get a list of all the files in each subdir
    files = {}
    for subdir in subdirs:
        files[subdir] = os.listdir(os.path.join(output_dir, subdir))
    # Check if the cache file exists
    if not os.path.exists(cache_file_path):
        raise FileNotFoundError(f"Cache file not found: {cache_file_path}")
    
    log = []
    cached = []
    cache_fix = []
    empty_files = []
    non_txt_files = []

    # Check that each file in the cache file exists in each subdir and the file is not empty
    with open(cache_file_path, 'r', encoding='utf-8') as cache_file:
        cached_files = set(json.load(cache_file))
    for subdir, subdir_files in files.items():
        for file in subdir_files:
            file_path = os.path.join(output_dir, subdir, file)
            if '.' in file.replace('.txt', ''):
                non_txt_files.append(file_path)
                continue
            if not file.endswith('.txt'):
                non_txt_files.append(file_path)
                continue
            if file.replace('.txt', '') not in cached_files:
                log.append(f'FileNotFoundError => File not found in cache: {subdir}/{file}')
                if file not in cached:
                    cached.append(file)
                    cache_fix.append(f'"{file.replace(".txt", "")}"')
            elif os.path.getsize(file_path) == 0:
                log.append(f'ValueError => File is empty: {subdir}/{file}')
                empty_files.append(file_path)
            else:
                if file not in cached:
                    cached.append(file)
                    cache_fix.append(f'"{file.replace(".txt", "")}"')

    if len(log) > 0:
        print("Errors found:")
        print("\t" + "\n\t".join(log))
        print("Try the following cache fix:")
        print("\t[" + ",".join(cache_fix) + "]")
        if len(empty_files) > 0:
            for file in empty_files:
                os.remove(file)
            print("Empty files removed")
        raise ValueError("Errors found in cache file")

    if len(non_txt_files) > 0:
        for file in non_txt_files:
            os.remove(file)
        print("Non-txt files removed")

def verify_files(set_a_dir, set_b_dir, base_dir):
    set_a_dir = os.path.join(base_dir, set_a_dir)
    set_b_dir = os.path.join(base_dir, set_b_dir)

    # Verify that the files exist
    if not os.path.exists(set_a_dir):
        raise FileNotFoundError(f"Set A directory not found: {set_a_dir}")
    if not os.path.exists(set_b_dir):
        raise FileNotFoundError(f"Set B directory not found: {set_b_dir}")
    # Verify that the files does not have the same name
    if set_a_dir == set_b_dir:
        raise ValueError("Set A and Set B directories must be different")
    # Verify that the files does not have any common elements
    with open(set_a_dir, 'r', encoding='utf-8') as file_a:
        set_a = set(file_a.readlines())
    with open(set_b_dir, 'r', encoding='utf-8') as file_b:
        set_b = set(file_b.readlines())
    common_elements = set_a.intersection(set_b)
    if common_elements:
        raise ValueError(f"Set A and Set B have common elements: {common_elements}")

def download_files(set_dir, output_dir, base_dir, auto_download = True):
    print(f"Downloading files from {os.path.join(base_dir, set_dir)} to {os.path.join(base_dir, output_dir, 'raw')}")
    downloader.download(True, os.path.join(output_dir, 'raw'), None, auto_download, base_dir, set_dir)

def process_files(input_dir, output_dir, cache_file_path, base_dir):
    print(f"Processing files in {output_dir}")
    processor.process_text_files(input_dir, output_dir, base_dir, cache_file_path)

def main():
    parser = argparse.ArgumentParser(description="Set Builder Script")
    parser.add_argument('-sad', '--set-a-dir', type=str, required=True, help='Path to the list of elements for set A')
    parser.add_argument('-sbd', '--set-b-dir', type=str, required=True, help='Path to the list of elements for set B')
    parser.add_argument('-oa', '--output-a', type=str, required=True, help='Output directory path for set A')
    parser.add_argument('-ob', '--output-b', type=str, required=True, help='Output directory path for set B')
    parser.add_argument('-b', '--base-dir', default='.\\', type=str, help="Base directory")

    args = parser.parse_args()

    # Verify files
    verify_files(args.set_a_dir, args.set_b_dir, args.base_dir)
    verify_cache(os.path.join(args.base_dir, args.output_a, 'raw'), os.path.join(args.base_dir, args.output_a, 'raw', 'raw.cache'))
    verify_cache(os.path.join(args.base_dir, args.output_b, 'raw'), os.path.join(args.base_dir, args.output_b, 'raw', 'raw.cache'))

    # Download files
    download_files(args.set_a_dir, args.output_a, args.base_dir)
    download_files(args.set_b_dir, args.output_b, args.base_dir)

    # Process files
    process_a_set_input_dir = os.path.join(args.base_dir, args.output_a, 'raw')
    process_a_set_output_dir = os.path.join(args.base_dir, args.output_a, 'words')
    process_a_set_cache_file_path = os.path.join(args.base_dir, args.output_a, 'raw', 'raw.cache')
    process_files(process_a_set_input_dir, process_a_set_output_dir, process_a_set_cache_file_path, args.base_dir)

    process_b_set_input_dir = os.path.join(args.base_dir, args.output_b, 'raw')
    process_b_set_output_dir = os.path.join(args.base_dir, args.output_b, 'words')
    process_b_set_cache_file_path = os.path.join(args.base_dir, args.output_b, 'raw', 'raw.cache')
    process_files(process_b_set_input_dir, process_b_set_output_dir, process_b_set_cache_file_path, args.base_dir)

    # Get distance matrix
    distance_matrix_a_output_dir = os.path.join(args.base_dir, args.output_a, 'distances')
    dmg.main(process_a_set_output_dir, distance_matrix_a_output_dir)
    distance_matrix_b_output_dir = os.path.join(args.base_dir, args.output_b, 'distances')
    dmg.main(process_b_set_output_dir, distance_matrix_b_output_dir)

if __name__ == "__main__":
    main()