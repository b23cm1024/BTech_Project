import os
import re
import json
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
WEB_DIR = os.path.join(DATA_DIR, "web_pages")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")

os.makedirs(WEB_DIR, exist_ok=True)
os.makedirs(PDF_DIR, exist_ok=True)

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

ATOZ_URL = "https://www.iitj.ac.in/AtoZ?lg=en"

# Core seeds covering every major section
CORE_SEEDS = [
    ("Home Page", "https://www.iitj.ac.in/main/en/iitj"),
    ("About Institute", "https://www.iitj.ac.in/m/Index/main-institute?lg=en"),
    ("Statutory Bodies", "https://www.iitj.ac.in/m/Index/main-statutory%20bodies?lg=en"),
    ("Key Functionaries", "https://www.iitj.ac.in/m/Index/main-key%20functionaries?lg=en"),
    ("Infrastructure", "https://www.iitj.ac.in/m/Index/main-infrastructure?lg=en"),
    ("Outreach", "https://www.iitj.ac.in/m/Index/main-outreach?lg=en"),
    ("Director Office", "https://www.iitj.ac.in/office-of-director/en/office-of-director"),
    ("About Director", "https://www.iitj.ac.in/office-of-director/en/about-director"),
    ("Director Profile", "https://www.iitj.ac.in/main/en/director"),
    ("Deputy Director Office", "https://www.iitj.ac.in/office-of-deputy-director/en/office-of-deputy-director"),
    ("Registrar Office", "https://www.iitj.ac.in/office-of-registrar/en/office-of-registrar"),
    ("Administration", "https://www.iitj.ac.in/office-of-administration/en/office-of-administration"),
    ("Administrative Contacts", "https://www.iitj.ac.in/main/en/administrative-contact"),
    ("Chairman BoG", "https://www.iitj.ac.in/main/en/chairman"),
    ("Deans & Associate Deans", "https://www.iitj.ac.in/main/en/deans-associate-deans"),
    ("Professors In-Charge", "https://www.iitj.ac.in/main/en/professor-in-charge"),
    ("Chairpersons of Committees", "https://www.iitj.ac.in/main/en/chairpersons-coordinators-of-committees"),
    ("Officers", "https://www.iitj.ac.in/main/en/officers"),
    ("Hostel Wardens", "https://www.iitj.ac.in/main/en/wardens"),
    ("Academic Affairs", "https://www.iitj.ac.in/office-of-academics/en/academics"),
    ("Academic Regulations", "https://www.iitj.ac.in/Office-of-Academics/en/Academic-Regulations"),
    ("Training and Placement Office", "https://www.iitj.ac.in/office-of-training-and-placement/en/office-of-training-and-placement"),
    ("Student Placement Cell Portal", "https://spc.iitj.ac.in/"),
    ("Placement Process", "https://www.iitj.ac.in/schools/en/placement-process"),
    ("Placement Brochure Portal", "https://www.iitj.ac.in/schools/en/placement-brochure"),
    ("B.Tech Internships & Placements", "https://www.iitj.ac.in/bachelor-of-technology/en/internships-placements"),
    ("Career Development Centre CDC", "https://www.iitj.ac.in/cdc"),
    ("Office of Career Services", "https://www.iitj.ac.in/office-of-career-services"),
    ("Scholarships and Placements", "https://www.iitj.ac.in/office-of-students/en/Scholarships-and-Placements"),
    ("Departments List", "https://www.iitj.ac.in/m/Index/main-departments?lg=en"),
    ("Schools", "https://www.iitj.ac.in/m/Index/main-schools?lg=en"),
    ("Centers", "https://www.iitj.ac.in/m/Index/main-centers?lg=en"),
    ("Health Center", "https://www.iitj.ac.in/health-center/en/health-center"),
    ("Digital Infrastructure DIA", "https://www.iitj.ac.in/dia/en/dia"),
    ("Central Research Facility", "https://www.iitj.ac.in/crf/en/crf"),
    ("Research & Development", "https://www.iitj.ac.in/office-of-research-development/en/office-of-research-and-development"),
    ("Computer Science & Engineering", "https://www.iitj.ac.in/computer-science-engineering/"),
    ("Electrical Engineering", "https://www.iitj.ac.in/electrical-engineering/"),
    ("Mechanical Engineering", "https://www.iitj.ac.in/mechanical-engineering/"),
    ("Civil Engineering", "https://www.iitj.ac.in/civil-and-infrastructure-engineering/"),
    ("Bioscience & Bioengineering", "https://www.iitj.ac.in/bioscience-bioengineering"),
    ("Chemical Engineering", "https://www.iitj.ac.in/chemical-engineering/"),
    ("Materials Engineering", "https://www.iitj.ac.in/materials-engineering/en/materials-engineering"),
    ("Chemistry", "https://www.iitj.ac.in/chemistry/en/chemistry"),
    ("Physics", "https://www.iitj.ac.in/physics/"),
    ("Mathematics", "https://www.iitj.ac.in/mathematics/"),
    ("Economics", "https://www.iitj.ac.in/economics"),
    ("School of Management & Entrepreneurship", "https://www.iitj.ac.in/schools/en/sme"),
    ("School of AI and Data Science", "https://www.iitj.ac.in/school-of-artificial-intelligence-data-science/en/aids"),
    ("Central Library", "https://library.iitj.ac.in/"),
]

def sanitize_filename(name: str) -> str:
    name = re.sub(r'[\\/*?:"<>|]', '_', name)
    name = re.sub(r'\s+', '_', name)
    return name.strip('._')[:80]

def collect_all_urls():
    """Scrape AtoZ sitemap and combine with core seeds to build master target list."""
    print("Collecting master URL index from IIT Jodhpur A-to-Z sitemap...", flush=True)
    all_targets = {}
    
    for title, url in CORE_SEEDS:
        all_targets[url] = title

    try:
        req = urllib.request.Request(ATOZ_URL, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
        soup = BeautifulSoup(html, 'html.parser')
        for a in soup.find_all('a', href=True):
            href = a['href'].strip()
            text = a.get_text(strip=True)
            if href and not href.startswith(('#', 'javascript:', 'mailto:', 'tel:')):
                full_url = urllib.parse.urljoin(ATOZ_URL, href)
                parsed = urllib.parse.urlparse(full_url)
                if 'iitj.ac.in' in parsed.netloc and text and len(text) > 2:
                    if '?lg=hi' not in full_url and '/hi/' not in full_url:
                        if full_url not in all_targets:
                            all_targets[full_url] = text
    except Exception as e:
        print(f"Error fetching AtoZ: {e}", flush=True)

    print(f"Total verified seed targets collected from AtoZ: {len(all_targets)}", flush=True)
    return all_targets

def process_target(item):
    """Fetch webpage, extract clean text, contacts & tables, and find any embedded PDFs."""
    url, title = item
    parsed = urllib.parse.urlparse(url)
    
    # Direct PDF file
    if parsed.path.lower().endswith('.pdf'):
        clean_name = f"{sanitize_filename(title)}_{os.path.basename(parsed.path)}"
        clean_name = sanitize_filename(clean_name) + ".pdf"
        target_pdf = os.path.join(PDF_DIR, clean_name)
        if not os.path.exists(target_pdf):
            try:
                req = urllib.request.Request(url, headers=HEADERS)
                with urllib.request.urlopen(req, timeout=12) as r:
                    data = r.read()
                    if len(data) > 3000 and data.startswith(b'%PDF'):
                        with open(target_pdf, 'wb') as f:
                            f.write(data)
                        return ("PDF", clean_name, len(data))
            except Exception:
                pass
        return ("PDF_EXISTS", clean_name, 0)

    # HTML Webpage Processing
    clean_slug = sanitize_filename(f"{title}_{parsed.path.replace('/', '_')}")
    out_json = os.path.join(WEB_DIR, f"{clean_slug}.json")
    if os.path.exists(out_json):
        return ("WEB_EXISTS", title, 0)

    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=10) as resp:
            content_type = resp.headers.get('Content-Type', '').lower()
            if 'html' not in content_type:
                return ("SKIP", title, 0)
            html = resp.read().decode('utf-8', errors='ignore')

        soup = BeautifulSoup(html, 'html.parser')
        
        # Discover and download any embedded PDFs on this page
        for a in soup.find_all('a', href=True):
            h = a['href'].strip()
            if h.lower().endswith('.pdf'):
                pdf_full = urllib.parse.urljoin(url, h)
                p_parsed = urllib.parse.urlparse(pdf_full)
                if 'iitj.ac.in' in p_parsed.netloc:
                    pdf_name = f"{sanitize_filename(a.get_text(strip=True) or 'doc')}_{os.path.basename(p_parsed.path)}"
                    pdf_path = os.path.join(PDF_DIR, sanitize_filename(pdf_name) + ".pdf")
                    if not os.path.exists(pdf_path):
                        try:
                            p_req = urllib.request.Request(pdf_full, headers=HEADERS)
                            with urllib.request.urlopen(p_req, timeout=10) as pr:
                                p_data = pr.read()
                                if len(p_data) > 3000 and p_data.startswith(b'%PDF'):
                                    with open(pdf_path, 'wb') as pf:
                                        pf.write(p_data)
                        except Exception:
                            pass

        # Extract structured content
        main = soup.find('div', id='maincontent') or soup.find('div', class_='entry-content') or soup.find('body')
        if not main:
            return ("EMPTY", title, 0)

        for tag in main.find_all(['script', 'style', 'header', 'nav', 'footer', 'noscript', 'svg']):
            tag.decompose()

        content_lines = []
        for elem in main.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'li', 'tr', 'td', 'th']):
            txt = ' '.join(elem.get_text().split())
            if len(txt) > 15:
                content_lines.append(txt)

        page_text = "\n".join(content_lines)
        if len(page_text) < 40:
            return ("TOO_SHORT", title, 0)

        data = {
            "title": title,
            "url": url,
            "text": page_text
        }

        with open(out_json, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        return ("WEB_SAVED", title, len(page_text))

    except Exception:
        return ("ERROR", title, 0)

def run_deep_crawl(max_pages=1500, max_workers=16):
    targets = collect_all_urls()
    target_items = list(targets.items())[:max_pages]
    print(f"Executing Complete Comprehensive Crawl across ALL {len(target_items)} verified pages using {max_workers} threads...", flush=True)

    saved_web = 0
    saved_pdf = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_target, item): item for item in target_items}
        for i, fut in enumerate(as_completed(futures)):
            status, name, size = fut.result()
            if status == "WEB_SAVED":
                saved_web += 1
            elif status == "PDF":
                saved_pdf += 1
            if (i + 1) % 50 == 0 or (i + 1) == len(target_items):
                print(f"  Progress: {i+1}/{len(target_items)} processed ({saved_web} new web pages saved, {saved_pdf} new PDFs downloaded)", flush=True)

    print(f"\nComprehensive Deep Crawl Finished! Total newly saved web pages: {saved_web}", flush=True)
    return saved_web

if __name__ == "__main__":
    run_deep_crawl(max_pages=1500, max_workers=16)
