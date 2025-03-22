import os
import json
from collections import Counter
import nltk
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import re
import argparse

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z0-9\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def process_text_files(input_dir, output_dir, base_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    cache_file_path = os.path.join(base_dir, 'data', 'raw.cache')
    with open(cache_file_path, 'r', encoding='utf-8') as cache_file:
        cached_files = set(json.load(cache_file))

    text_files = []
    for subdir in ['Infoboxes', 'Links', 'Texts']:
        subdir_path = os.path.join(input_dir, subdir)
        for file_name in cached_files:
            file_path = os.path.join(subdir_path, f'{file_name}.txt')
            if os.path.exists(file_path) and file_path.endswith('.txt'):
                text_files.append(file_path)

    for file_path in tqdm(text_files, desc="Processing files"):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        cleaned_text = clean_text(text)
        
        words = word_tokenize(cleaned_text)
        word_counts = Counter(words)
        
        relative_path = os.path.relpath(os.path.dirname(file_path), input_dir)
        output_subdir = os.path.join(output_dir, relative_path)
        if not os.path.exists(output_subdir):
            os.makedirs(output_subdir)
        
        output_file_path = os.path.join(output_subdir, f"{os.path.splitext(os.path.basename(file_path))[0]}_words.json")
        with open(output_file_path, 'w', encoding='utf-8') as out_f:
            json.dump(word_counts, out_f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process text files.")
    parser.add_argument('-i', '--input', type=str, help='Input directory')
    parser.add_argument('-o', '--output', type=str, help='Output directory')
    parser.add_argument('-b', '--base-dir', default='.\\', type=str, help="Base directory for the output files")
    args = parser.parse_args()

    if not args.input:
        args.input = os.path.join(args.base_dir, 'data', 'raw')

    if not args.output:
        args.output = os.path.join(args.base_dir, 'data', 'words')

    process_text_files(args.input, args.output, args.base_dir)