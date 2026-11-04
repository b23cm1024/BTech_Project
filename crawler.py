import os
import re
import time
import json
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
MANIFEST_PATH = os.path.join(DATA_DIR, "pdf_manifest.json")
os.makedirs(PDF_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

START_HUBS = [
    "https://www.iitj.ac.in/",
    "https://www.iitj.ac.in/Office-of-Academics/en/academics",
    "https://www.iitj.ac.in/Office-of-Academics/en/curriculum-for-programs-from-2019-to-2025",
    "https://www.iitj.ac.in/Office-of-Academics/en/Download-Forms",
    "https://www.iitj.ac.in/Office-of-Academics/en/academic-calendar",
    "https://www.iitj.ac.in/Office-of-Academics/en/list-of-academic-programs",
    "https://www.iitj.ac.in/Office-of-Academics/en/Academic-Regulations",
    "https://www.iitj.ac.in/Office-of-Academics/en/ug-registration-guidelines",
    "https://www.iitj.ac.in/Admission/en/admission",
    "https://www.iitj.ac.in/People?dept=Office-of-Academics",
]

def sanitize_filename(name: str) -> str:
    """Sanitize string to be safe for Windows filenames."""
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name.strip('._')[:80]

def discover_pdfs(max_pages=150):
    """Crawl iitj.ac.in starting from major academic and administrative hubs."""
    visited = set()
    queue = list(START_HUBS)
    found_pdfs = {}

    # Load existing manifest if any
    if os.path.exists(MANIFEST_PATH):
        try:
            with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
                found_pdfs = json.load(f)
        except Exception:
            pass

    print(f"Starting crawl across IIT Jodhpur website (max {max_pages} pages)...")
    
    while queue and len(visited) < max_pages:
        url = queue.pop(0)
        if url in visited:
            continue
        visited.add(url)
        
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                content_type = resp.headers.get('Content-Type', '').lower()
                if 'html' not in content_type:
                    continue
                html = resp.read().decode('utf-8', errors='ignore')
                
            soup = BeautifulSoup(html, 'html.parser')
            
            for a in soup.find_all('a', href=True):
                href = a['href'].strip()
                if not href or href.startswith(('mailto:', 'tel:', 'javascript:', '#')):
                    continue
                    
                full_url = urllib.parse.urljoin(url, href)
                parsed = urllib.parse.urlparse(full_url)
                
                # Stay within iitj.ac.in
                if 'iitj.ac.in' in parsed.netloc:
                    # Check if it's a PDF
                    if parsed.path.lower().endswith('.pdf'):
                        title = a.get_text(strip=True) or os.path.basename(parsed.path)
                        title = sanitize_filename(title)
                        if full_url not in found_pdfs:
                            found_pdfs[full_url] = {
                                "title": title,
                                "url": full_url,
                                "source_page": url
                            }
                            print(f"  [PDF Discovered] {title} -> {full_url}")
                    elif full_url not in visited and parsed.scheme in ['http', 'https']:
                        # Avoid crawling search or login loops
                        if not any(bad in full_url.lower() for bad in ['login', 'search', 'logout', 'redirect', 'mailto']):
                            queue.append(full_url)
                            
        except Exception as e:
            # print(f"Error crawling {url}: {e}")
            pass

    print(f"\nCrawl complete. Total unique PDFs discovered: {len(found_pdfs)}")
    with open(MANIFEST_PATH, 'w', encoding='utf-8') as f:
        json.dump(found_pdfs, f, indent=2, ensure_ascii=False)
        
    return found_pdfs

def download_file(item):
    """Download a single PDF and save locally."""
    full_url, info = item
    title = info["title"]
    parsed = urllib.parse.urlparse(full_url)
    ext = os.path.splitext(parsed.path)[1] or ".pdf"
    clean_name = f"{title}_{os.path.basename(parsed.path)}"
    clean_name = sanitize_filename(clean_name) + ext
    target_path = os.path.join(PDF_DIR, clean_name)
    
    if os.path.exists(target_path) and os.path.getsize(target_path) > 3000:
        return (clean_name, True, "Already downloaded")
        
    try:
        req = urllib.request.Request(full_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=25) as resp:
            data = resp.read()
            if len(data) > 3000 and data.startswith(b'%PDF'):
                with open(target_path, 'wb') as f:
                    f.write(data)
                return (clean_name, True, f"{len(data)} bytes")
            else:
                return (clean_name, False, "Not a valid PDF")
    except Exception as e:
        return (clean_name, False, str(e))

def download_all_discovered(max_workers=5):
    """Downloads all discovered PDFs concurrently."""
    if not os.path.exists(MANIFEST_PATH):
        print("Manifest not found, discovering PDFs first...")
        discover_pdfs()
        
    with open(MANIFEST_PATH, 'r', encoding='utf-8') as f:
        manifest = json.load(f)
        
    print(f"\n=== Downloading {len(manifest)} discovered PDFs into {PDF_DIR} ===")
    success_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(download_file, item): item for item in manifest.items()}
        for fut in as_completed(futures):
            name, ok, msg = fut.result()
            if ok:
                success_count += 1
                print(f"  [OK] {name} ({msg})")
            else:
                # print(f"  [Failed] {name}: {msg}")
                pass
                
    print(f"\nDownload finished: {success_count}/{len(manifest)} PDFs available in {PDF_DIR}")
    return success_count

if __name__ == "__main__":
    discover_pdfs(max_pages=120)
    download_all_discovered()
