import os
import re
import json
import time
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import subprocess

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WEB_DIR = os.path.join(DATA_DIR, "web_pages")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")

os.makedirs(WEB_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
}

# Subdomain roots and major entry seeds across the entire institute
CORE_SEEDS = [
    "https://www.iitj.ac.in/main/en/iitj",
    "https://www.iitj.ac.in/AtoZ?lg=en",
    "https://www.iitj.ac.in/sitemap.xml",
    "https://spc.iitj.ac.in/",
    "https://library.iitj.ac.in/",
    "https://www.iitj.ac.in/computer-science-engineering/",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/aids",
    "https://www.iitj.ac.in/electrical-engineering/",
    "https://www.iitj.ac.in/mechanical-engineering/",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/",
    "https://www.iitj.ac.in/bioscience-bioengineering/",
    "https://www.iitj.ac.in/chemical-engineering/",
    "https://www.iitj.ac.in/materials-engineering/en/materials-engineering",
    "https://www.iitj.ac.in/chemistry/en/chemistry",
    "https://www.iitj.ac.in/physics/",
    "https://www.iitj.ac.in/mathematics/",
    "https://www.iitj.ac.in/humanities-and-social-sciences/",
    "https://www.iitj.ac.in/schools/en/sme",
    "https://www.iitj.ac.in/school-of-design/en/school-of-design",
    "https://www.iitj.ac.in/office-of-academics/en/academics",
    "https://www.iitj.ac.in/Office-of-Academics/en/Academic-Regulations",
    "https://www.iitj.ac.in/office-of-students/en/office-of-students",
    "https://www.iitj.ac.in/office-of-training-and-placement/en/office-of-training-and-placement",
    "https://www.iitj.ac.in/office-of-research-development/en/office-of-research-and-development",
    "https://www.iitj.ac.in/office-of-international-relations/en/office-of-international-relations",
    "https://www.iitj.ac.in/office-of-director/en/office-of-director",
    "https://www.iitj.ac.in/office-of-administration/en/office-of-administration",
    "https://www.iitj.ac.in/office-of-stores-purchase/en/tender-details",
    "https://www.iitj.ac.in/main/en/all-announcements",
    "https://www.iitj.ac.in/main/en/news",
    "https://www.iitj.ac.in/main/en/events",
    "https://www.iitj.ac.in/main/en/research-highlight",
    "https://www.iitj.ac.in/main/en/faculty",
    "https://www.iitj.ac.in/bachelor-of-technology/en/bachelor-of-technology",
    "https://www.iitj.ac.in/master-of-technology/en/master-of-technology",
    "https://www.iitj.ac.in/doctor-of-philosophy/en/doctor-of-philosophy",
    "https://www.iitj.ac.in/master-of-science/en/master-of-science",
    "https://www.iitj.ac.in/Executive-Programs/en/executive-programs",
    "https://intranet.iitj.ac.in/acad_website/intra/2025_onwords/06_Revised-UG-Curriculum-AIDE.pdf",
    "https://intranet.iitj.ac.in/acad_website/intra/5. B.Tech AIDS.pdf"
]

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name.strip('._')[:85]

def get_visited_urls():
    visited = set()
    for fname in os.listdir(WEB_DIR):
        if fname.endswith(".json"):
            try:
                with open(os.path.join(WEB_DIR, fname), 'r', encoding='utf-8') as f:
                    d = json.load(f)
                    u = d.get('url', '')
                    if u:
                        visited.add(u)
            except Exception:
                pass
    return visited

def get_existing_pdfs():
    pdfs = set()
    for fname in os.listdir(PDF_DIR):
        if fname.endswith(".pdf"):
            pdfs.add(fname.lower())
    return pdfs

def normalize_url(url: str) -> str:
    url = url.split('#')[0].strip()
    parsed = urllib.parse.urlparse(url)
    # clean trailing slash if not root
    path = parsed.path
    if path != '/' and path.endswith('/'):
        path = path[:-1]
    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, parsed.params, parsed.query, ''))

def is_valid_iitj_url(url: str) -> bool:
    parsed = urllib.parse.urlparse(url)
    netloc = parsed.netloc.lower()
    if 'iitj.ac.in' not in netloc:
        return False
    # Skip non-English language duplications to prevent double crawling
    if '?lg=hi' in url or '/hi/' in url:
        return False
    # Skip obvious binary media or logins
    lower_path = parsed.path.lower()
    skip_exts = ('.jpg', '.jpeg', '.png', '.gif', '.svg', '.mp4', '.avi', '.mov', '.zip', '.tar', '.gz', '.rar', '.exe')
    if any(lower_path.endswith(ext) for ext in skip_exts):
        return False
    if any(term in lower_path for term in ['login', 'logout', 'webmail', 'print']):
        return False
    return True

def fetch_and_process(url: str, visited_urls: set, existing_pdfs: set):
    """Downloads PDF or parses webpage, extracting all internal links and text."""
    norm_url = normalize_url(url)
    parsed = urllib.parse.urlparse(norm_url)
    discovered_links = []
    
    # Case 1: PDF Document
    if parsed.path.lower().endswith('.pdf'):
        pdf_basename = os.path.basename(parsed.path)
        clean_name = sanitize_filename(pdf_basename) + ".pdf"
        target_path = os.path.join(PDF_DIR, clean_name)
        
        if clean_name.lower() in existing_pdfs or os.path.exists(target_path):
            return ("PDF_EXISTS", norm_url, 0, [])
            
        try:
            req = urllib.request.Request(norm_url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=12) as resp:
                data = resp.read()
                if len(data) > 1000 and data.startswith(b'%PDF'):
                    with open(target_path, 'wb') as pf:
                        pf.write(data)
                    return ("PDF_SAVED", norm_url, len(data), [])
        except Exception:
            return ("PDF_ERROR", norm_url, 0, [])
            
        return ("PDF_INVALID", norm_url, 0, [])

    # Case 2: HTML Webpage
    try:
        req = urllib.request.Request(norm_url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get('Content-Type', '').lower()
            if 'html' not in content_type and 'text' not in content_type:
                return ("SKIP_TYPE", norm_url, 0, [])
            html = resp.read().decode('utf-8', errors='ignore')
    except Exception:
        return ("FETCH_ERROR", norm_url, 0, [])

    soup = BeautifulSoup(html, 'html.parser')

    # Extract all internal links for recursive frontier
    for a in soup.find_all('a', href=True):
        href = a['href'].strip()
        if href and not href.startswith(('javascript:', 'mailto:', 'tel:', '#')):
            full = urllib.parse.urljoin(norm_url, href)
            full_norm = normalize_url(full)
            if is_valid_iitj_url(full_norm):
                discovered_links.append(full_norm)

    # If this page was already indexed, only return its discovered links
    if norm_url in visited_urls:
        return ("ALREADY_INDEXED", norm_url, 0, discovered_links)

    # Extract clean readable text
    title_tag = soup.find('title')
    title = title_tag.get_text(strip=True) if title_tag else "IIT Jodhpur Portal"
    title = title.replace(" - IIT Jodhpur", "").replace(" | IIT Jodhpur", "").strip()

    main = soup.find('div', id='maincontent') or soup.find('div', class_='entry-content') or soup.find('body')
    if not main:
        return ("NO_BODY", norm_url, 0, discovered_links)

    for tag in main.find_all(['script', 'style', 'header', 'nav', 'footer', 'noscript', 'svg']):
        tag.decompose()

    content_lines = []
    for elem in main.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'tr', 'td', 'th']):
        txt = ' '.join(elem.get_text().split())
        if len(txt) > 15:
            content_lines.append(txt)

    page_text = "\n".join(content_lines)
    if len(page_text) < 40:
        return ("TOO_SHORT", norm_url, 0, discovered_links)

    slug = sanitize_filename(f"{title}_{parsed.netloc}_{parsed.path.replace('/', '_')}")
    out_json = os.path.join(WEB_DIR, f"{slug}.json")

    data = {
        "title": title,
        "url": norm_url,
        "text": page_text
    }
    with open(out_json, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    visited_urls.add(norm_url)
    return ("PAGE_SAVED", norm_url, len(page_text), discovered_links)

HUBS = [
    "https://www.iitj.ac.in/main/en/iitj",
    "https://www.iitj.ac.in/AtoZ?lg=en",
    "https://www.iitj.ac.in/sitemap.xml",
    "https://www.iitj.ac.in/main/en/all-announcements",
    "https://www.iitj.ac.in/main/en/news",
    "https://www.iitj.ac.in/main/en/events",
    "https://www.iitj.ac.in/main/en/research-highlight",
    "https://www.iitj.ac.in/main/en/faculty",
    "https://www.iitj.ac.in/office-of-stores-purchase/en/tender-details",
    "https://spc.iitj.ac.in/",
    "https://library.iitj.ac.in/",
    "https://www.iitj.ac.in/office-of-academics/en/curriculum",
    "https://www.iitj.ac.in/office-of-academics/en/curriculum-for-programs-from-2019-to-2025",
    "https://www.iitj.ac.in/office-of-academics/en/Curriculum-for-Programs-Before-2019",
    "https://www.iitj.ac.in/Office-of-Academics/en/Academic-Regulations",
    "https://www.iitj.ac.in/office-of-students/en/office-of-students",
    "https://www.iitj.ac.in/office-of-training-and-placement/en/office-of-training-and-placement",
    "https://www.iitj.ac.in/schools/en/sme",
    "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/aids",
    "https://www.iitj.ac.in/computer-science-engineering/",
    "https://www.iitj.ac.in/electrical-engineering/",
    "https://www.iitj.ac.in/mechanical-engineering/",
    "https://www.iitj.ac.in/civil-and-infrastructure-engineering/",
    "https://www.iitj.ac.in/bioscience-bioengineering/",
    "https://www.iitj.ac.in/materials-engineering/en/materials-engineering",
    "https://www.iitj.ac.in/chemical-engineering/",
    "https://www.iitj.ac.in/chemistry/en/chemistry",
    "https://www.iitj.ac.in/physics/",
    "https://www.iitj.ac.in/mathematics/",
    "https://www.iitj.ac.in/bachelor-of-technology/en/bachelor-of-technology",
    "https://www.iitj.ac.in/master-of-technology/en/master-of-technology",
    "https://www.iitj.ac.in/doctor-of-philosophy/en/doctor-of-philosophy"
]

def crawl_entire_domain(max_pages=25000, max_workers=24):
    print("=== Launching Exhaustive Domain Crawler for IIT JODHPUR (iitj.ac.in) ===", flush=True)
    visited_urls = get_visited_urls()
    existing_pdfs = get_existing_pdfs()
    print(f"Pre-existing in database: {len(visited_urls)} web pages, {len(existing_pdfs)} PDFs", flush=True)

    frontier = set()

    # 1. Harvest seed links directly from all 32 major hubs
    print(f"Harvesting live links from {len(HUBS)} institutional hub portals...", flush=True)
    for h in HUBS:
        norm = normalize_url(h)
        if norm not in visited_urls:
            frontier.add(norm)
        try:
            req = urllib.request.Request(h, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=8) as r:
                soup = BeautifulSoup(r.read().decode('utf-8', errors='ignore'), 'html.parser')
                for a in soup.find_all('a', href=True):
                    full = urllib.parse.urljoin(h, a['href'])
                    norm_full = normalize_url(full)
                    if is_valid_iitj_url(norm_full) and norm_full not in visited_urls:
                        frontier.add(norm_full)
        except Exception:
            pass

    print(f"Initial frontier established with {len(frontier)} unvisited targets.", flush=True)

    new_pages_saved = 0
    new_pdfs_saved = 0
    processed_count = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while frontier and processed_count < max_pages:
            batch = list(frontier)[:500]
            for u in batch:
                frontier.remove(u)

            future_to_url = {executor.submit(fetch_and_process, u, visited_urls, existing_pdfs): u for u in batch}
            for fut in as_completed(future_to_url):
                processed_count += 1
                try:
                    status, url, size, discovered = fut.result()
                    if status == "PAGE_SAVED":
                        new_pages_saved += 1
                    elif status == "PDF_SAVED":
                        new_pdfs_saved += 1

                    for link in discovered:
                        if link not in visited_urls and link not in frontier:
                            frontier.add(link)
                except Exception:
                    pass

                if processed_count % 50 == 0:
                    print(f"  [Progress] Processed: {processed_count} | Queue: {len(frontier)} | New Pages: {new_pages_saved} | New PDFs: {new_pdfs_saved}", flush=True)

    print(f"\nExhaustive Crawl Finished! Total New Pages Saved: {new_pages_saved}, Total New PDFs: {new_pdfs_saved}", flush=True)
    
    # Automatically trigger ingestion to rebuild index
    print("\nRe-indexing complete institutional knowledge base...", flush=True)
    try:
        import ingest
        ingest.build_index()
    except Exception as e:
        print(f"Error during automatic ingestion: {e}", flush=True)

if __name__ == "__main__":
    crawl_entire_domain(max_pages=25000, max_workers=24)
