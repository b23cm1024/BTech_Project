import os
import urllib.request
import urllib.parse
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "pdfs")
HTML_DIR = os.path.join(os.path.dirname(__file__), "data", "html")
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(HTML_DIR, exist_ok=True)

OFFICIAL_PDFS = [
    (
        "Academic_Regulations_2019_onwards.pdf",
        "https://www.iitj.ac.in/PageImages/Gallery/03-2025/1_Academic_Regulations_Final_03_09_2019.pdf"
    ),
    (
        "Academic_Regulations_2020_onwards.pdf",
        "https://www.iitj.ac.in/PageImages/Gallery/03-2025/2_Academic_Regulations_Final_03_09_2019.pdf"
    ),
    (
        "Academic_Regulations_2021_onwards.pdf",
        "https://www.iitj.ac.in/PageImages/Gallery/03-2025/3_Academic_Regulations_Final_03_09_2019.pdf"
    ),
    (
        "PG_Regulations_2022_onwards.pdf",
        "https://www.iitj.ac.in/PageImages/Gallery/03-2025/4_Regulation_PG_2022-onwards_20022023.pdf"
    ),
    (
        "MS_by_Research_Regulations_2023.pdf",
        "https://www.iitj.ac.in/PageImages/Gallery/03-2025/6_2024-04-17-661f605b54457-1713332315.pdf"
    ),
    (
        "Amendment_Attendance_Policy_2024.pdf",
        "https://www.iitj.ac.in/PageImages/Gallery/03-2025/7.1_Circular_51Circular-No.-IITJOAACircular2024-2551682-dated-08-July-2024.pdf"
    ),
    (
        "Amendment_Examination_Policy_2024.pdf",
        "https://www.iitj.ac.in/PageImages/Gallery/03-2025/7.2_Circular_52Circular-No.-IITJOAACircular2024-2552683-dated-08-July-2024.pdf"
    ),
    (
        "Amendment_MTech_PhD_Continuation_Norms.pdf",
        "https://www.iitj.ac.in/PageImages/Gallery/03-2025/2.1_notification_26102020.pdf"
    ),
]

REGULATION_HTML_URL = "https://www.iitj.ac.in/Office-of-Academics/en/Academic-Regulations"

HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def download_all():
    print(f"=== 1. Downloading Official IITJ Regulation PDFs into {DATA_DIR} ===")
    for filename, url in OFFICIAL_PDFS:
        target_path = os.path.join(DATA_DIR, filename)
        if os.path.exists(target_path) and os.path.getsize(target_path) > 10000:
            print(f"  [Skipped] {filename} (already exists, {os.path.getsize(target_path)} bytes)")
            continue
        try:
            print(f"  Downloading {filename}...")
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=30) as resp:
                content = resp.read()
                if len(content) > 5000 and content.startswith(b'%PDF'):
                    with open(target_path, 'wb') as f:
                        f.write(content)
                    print(f"  [OK] Saved {filename} ({len(content)} bytes)")
                else:
                    print(f"  [Warning] Response for {filename} is not a valid PDF (length: {len(content)})")
        except Exception as e:
            print(f"  [Error] {filename}: {e}")

    print(f"\n=== 2. Downloading Full Academic Regulations Webpage into {HTML_DIR} ===")
    html_target = os.path.join(HTML_DIR, "Academic_Regulations_Page.html")
    try:
        req = urllib.request.Request(REGULATION_HTML_URL, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode('utf-8', errors='ignore')
            with open(html_target, 'w', encoding='utf-8') as f:
                f.write(html)
            print(f"  [OK] Saved Academic_Regulations_Page.html ({len(html)} characters)")
    except Exception as e:
        print(f"  [Error] Webpage fetch failed: {e}")

if __name__ == "__main__":
    download_all()
