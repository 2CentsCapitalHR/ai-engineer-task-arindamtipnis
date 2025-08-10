import os
import requests
from bs4 import BeautifulSoup
import urllib.parse

# --- Configuration ---
DATA_DIR = "data"
URLS = [
    "https://www.adgm.com/registration-authority/registration-and-incorporation",
    "https://assets.adgm.com/download/assets/adgm-ra-resolution-multiple-incorporate-shareholders-LTD-incorporation-v2.docx/186a12846c3911efa4e6c6223862cd87",
    "https://www.adgm.com/setting-up",
    "https://www.adgm.com/legal-framework/guidance-and-policy-statements",
    "https://www.adgm.com/documents/registration-authority/registration-and-incorporation/checklist/branch-non-financial-services-20231228.pdf",
    "https://www.adgm.com/documents/registration-authority/registration-and-incorporation/checklist/private-company-limited-by-guarantee-non-financial-services-20231228.pdf",
    "https://assets.adgm.com/download/assets/ADGM+Standard+Employment+Contract+Template+-+ER+2024+(Feb+2025).docx/ee14b252edbe11efa63b12b3a30e5e3a",
    "https://assets.adgm.com/download/assets/ADGM+Standard+Employment+Contract+-+ER+2019+-+Short+Version+(May+2024).docx/33b57a92ecfe11ef97a536cc36767ef8",
    "https://www.adgm.com/documents/office-of-data-protection/templates/adgm-dpr-2021-appropriate-policy-document.pdf",
    "https://www.adgm.com/operating-in-adgm/obligations-of-adgm-registered-entities/annual-filings/annual-accounts",
    "https://www.adgm.com/operating-in-adgm/post-registration-services/letters-and-permits",
    "https://en.adgm.thomsonreuters.com/rulebook/7-company-incorporation-package",
    "https://assets.adgm.com/download/assets/Templates_SHReso_AmendmentArticles-v1-20220107.docx/97120d7c5af911efae4b1e183375c0b2?forcedownload=1"
]

def download_and_save(url, save_path):
    """Downloads content from a URL and saves it."""
    try:
        response = requests.get(url, allow_redirects=True)
        response.raise_for_status()  # Raise an exception for bad status codes
        with open(save_path, 'wb') as f:
            f.write(response.content)
        print(f"Successfully downloaded {url} to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {url}: {e}")

def scrape_and_save(url, save_path):
    """Scrapes text from a webpage and saves it."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        # Extract text and clean it up
        text = ' '.join(soup.stripped_strings)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Successfully scraped {url} to {save_path}")
    except requests.exceptions.RequestException as e:
        print(f"Error scraping {url}: {e}")

def main():
    """Main function to ingest data from all sources."""
    os.makedirs(DATA_DIR, exist_ok=True)

    for url in URLS:
        try:
            # Create a safe filename from the URL
            parsed_url = urllib.parse.urlparse(url)
            
            # Heuristic to get a descriptive name
            path_segments = [segment for segment in parsed_url.path.split('/') if segment]
            if len(path_segments) > 1:
                # e.g. ['registration-authority', 'registration-and-incorporation']
                filename_base = '_'.join(path_segments[-2:])
            elif path_segments:
                 # e.g. ['setting-up']
                filename_base = path_segments[-1]
            else:
                # Use domain if path is empty
                filename_base = parsed_url.netloc.replace('.', '_')

            # Clean and create final filename
            filename_base = urllib.parse.unquote(filename_base).replace('?forcedownload=1', '')

            if '.pdf' in url.lower():
                filename = f"{filename_base}.pdf" if not filename_base.lower().endswith('.pdf') else filename_base
                save_path = os.path.join(DATA_DIR, filename)
                download_and_save(url, save_path)
            elif '.docx' in url.lower():
                filename = f"{filename_base}.docx" if not filename_base.lower().endswith('.docx') else filename_base
                save_path = os.path.join(DATA_DIR, filename)
                download_and_save(url, save_path)
            else: # Assume HTML page
                filename = f"{filename_base}.txt"
                save_path = os.path.join(DATA_DIR, filename)
                scrape_and_save(url, save_path)
        except Exception as e:
            print(f"Failed to process URL {url}: {e}")


if __name__ == "__main__":
    main()
