import os
import wikipedia
from tqdm import tqdm
from bs4 import BeautifulSoup
import argparse
import json

# Parse command line arguments
parser = argparse.ArgumentParser(description="Download Wikipedia pages and infoboxes.")
parser.add_argument('-s', '--skip', action='store_true', help="Skip already downloaded pages")
parser.add_argument('-ns', '--not_skip', action='store_false', help="Does not skip already downloaded pages")
parser.add_argument('-o', '--output', type=str, default='Data/Raw', help="Directory to store raw data")
parser.add_argument('-n', '--name', type=str, help="Specific Wikipedia page name to download")
parser.add_argument('-y', '--yes', action='store_true', help="Automatically confirm downloading content")
args = parser.parse_args()

# Create the Raw directory and subdirectories if they don't exist
raw_dir = os.path.join(os.path.dirname(__file__), args.output)
skip = args.not_skip is not None
texts_dir = os.path.join(raw_dir, 'Texts')
links_dir = os.path.join(raw_dir, 'Links')
infoboxes_dir = os.path.join(raw_dir, 'Infoboxes')
os.makedirs(texts_dir, exist_ok=True)
os.makedirs(links_dir, exist_ok=True)
os.makedirs(infoboxes_dir, exist_ok=True)

# Initialize Wikipedia API
wikipedia.set_lang("en")

# Determine the titles to process
if args.name:
    titles = [args.name]
else:
    with open(os.path.join(os.path.dirname(__file__), 'Data', 'titles.txt'), 'r') as file:
        titles = file.readlines()

# Load the cache of already downloaded pages if --skip is not specified or is the default
if skip:
    cache_file_path = os.path.join(os.path.dirname(__file__), 'Data', 'raw.cache')
    if os.path.exists(cache_file_path):
        with open(cache_file_path, 'r', encoding='utf-8') as cache_file:
            downloaded_pages = set(json.load(cache_file))
    else:
        downloaded_pages = set()
else:
    downloaded_pages = set()

# Process each title with a progress bar
for title in tqdm(titles, desc="Downloading content"):
    title = title.strip()
    if title:
        # Check if the page is already downloaded
        if skip and title in downloaded_pages:
            tqdm.write(f"Skipping already downloaded content for: {title}")
            continue

        # Print the page title and ask for confirmation if --yes is not specified
        if not args.yes:
            confirmation = input(f"\nDo you want to download the content for: {title}? (y/n) ").strip().lower()
        else:
            confirmation = 'y'

        if confirmation in ['y', 'yes', 'si']:
            try:
                # Fetch the page content
                page = wikipedia.page(title)
            except wikipedia.exceptions.DisambiguationError as e:
                tqdm.write(f"Disambiguation error for title: {title}. Options: {e.options}")
                continue
            except wikipedia.exceptions.PageError:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                except wikipedia.exceptions.PageError:
                    tqdm.write(f"Page does not exist for title: {title}")
                    continue

            with tqdm(total=3, desc=f"Saving files for {title}", leave=False) as sub_pbar:
                # Save the content to a file in the Texts directory
                content_file_path = os.path.join(texts_dir, f"{title}.txt")
                with open(content_file_path, 'w', encoding='utf-8') as output_file:
                    output_file.write(page.content)
                sub_pbar.update(1)
                tqdm.write(f"Downloaded and saved content for: {title}")

                # Parse the HTML content to extract the infobox
                soup = BeautifulSoup(page.html(), 'html.parser')
                infobox = soup.find('table', class_='infobox')
                if (infobox):
                    infobox_text = infobox.get_text(separator='\n').strip()
                    # Save the infobox content to a file in the Infoboxes directory
                    infobox_file_path = os.path.join(infoboxes_dir, f"{title}.txt")
                    with open(infobox_file_path, 'w', encoding='utf-8') as infobox_file:
                        infobox_file.write(infobox_text)
                    tqdm.write(f"Downloaded and saved infobox for: {title}")
                else:
                    tqdm.write(f"No infobox found for: {title}")
                sub_pbar.update(1)

                # Extract and save the links to a file in the Links directory using Wikipedia API
                links = page.links
                links_file_path = os.path.join(links_dir, f"{title}.txt")
                with open(links_file_path, 'w', encoding='utf-8') as links_file:
                    links_file.write('\n'.join(links))
                tqdm.write(f"Downloaded and saved links for: {title}")
                sub_pbar.update(1)

            # Add the page title to the cache
            downloaded_pages.add(title)
            with open(cache_file_path, 'w', encoding='utf-8') as cache_file:
                json.dump(list(downloaded_pages), cache_file)
        else:
            tqdm.write(f"Skipped downloading content for: {title}")