import os
import glob
import json
import pickle
import re
import fitz  # PyMuPDF
import pandas as pd
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
HTML_DIR = os.path.join(DATA_DIR, "html")
WEB_DIR = os.path.join(DATA_DIR, "web_pages")
TABLES_DIR = os.path.join(DATA_DIR, "tables")
INDEX_PATH = os.path.join(DATA_DIR, "index_chunks.json")
TFIDF_PATH = os.path.join(DATA_DIR, "tfidf.pkl")

def clean_text(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 200) -> list[str]:
    """
    Split text strictly at sentence boundaries (period+space, numbered list items, section headings).
    Never cuts inside a sentence or word. Maintains trailing sentence overlap for continuity.
    """
    text = clean_text(text)
    if not text or len(text) < 40:
        return [text] if len(text) >= 20 else []

    sentence_pattern = re.compile(
        r'(?<=[.?!])\s+(?=[A-Z0-9\"\'\(\[])'  # End of sentence -> new capital
        r'|(?<=\n)(?=\d+\.\s)'                 # Numbered list item
        r'|(?<=\n)(?=[A-Z][A-Za-z ]{2,30}:)'  # Section heading like "Revised Policy:"
        r'|(?<=\n\n)'                          # Paragraph break
    )
    raw_sentences = sentence_pattern.split(text)
    sentences = [s.strip() for s in raw_sentences if s.strip()]

    if not sentences:
        return [text] if len(text) >= 40 else []

    result_chunks = []
    current_sentences = []
    current_len = 0

    for sent in sentences:
        sent_len = len(sent)
        if current_len + sent_len > chunk_size and current_sentences:
            chunk_body = ' '.join(current_sentences).strip()
            if len(chunk_body) >= 40:
                result_chunks.append(chunk_body)

            # Build trailing sentence overlap
            tail = []
            tail_len = 0
            for s in reversed(current_sentences):
                if tail_len + len(s) <= overlap:
                    tail.insert(0, s)
                    tail_len += len(s)
                else:
                    break
            current_sentences = tail + [sent]
            current_len = tail_len + sent_len
        else:
            current_sentences.append(sent)
            current_len += sent_len

    if current_sentences:
        chunk_body = ' '.join(current_sentences).strip()
        if len(chunk_body) >= 40:
            result_chunks.append(chunk_body)

    return result_chunks

PDF_TABLE_KEYWORDS = [
    "table", "s.no", "credits", "ltp", "l-t-p", "semester", "category",
    "fee", "fees", "amount", "elective", "compulsory", "graded", "audit",
    "course type", "programme core", "programme elective", "open elective",
    "designation", "department", "constitution", "head", "hod"
]

def extract_pdf_chunks(pdf_path: str, max_pages: int = 150):
    doc_name = os.path.basename(pdf_path)
    chunks = []
    
    # Skip non-text media files >85MB or known corrupted vector graphics streams
    if os.path.getsize(pdf_path) > 85 * 1024 * 1024:
        return []
    if any(corrupt in doc_name for corrupt in ["PG_Diploma_and_Executive_MTech_in_AI", "0546781a-2b22-48d4-a656-258b8e8e6e8b"]):
        return []

    # Strict Privacy Filter: Skip any historical payroll, individual salary or remuneration disclosure files
    payroll_terms = [
        'gross%20salary', 'gross_salary', 'gross salary',
        'salary%20jan', 'salary%20july', 'salary-jan', 'salary-july',
        'salary jan', 'salary july', 'staff members salary', 'faculty members salary',
        'monthly remuneration', 'monthly_remuneration',
        '_fm.pdf', '_sm.pdf', '-fm_', '-sm_'
    ]
    if any(k in doc_name.lower() for k in payroll_terms):
        return []

    is_circular = "circular" in doc_name.lower() or "amendment" in doc_name.lower()
    text_modality = "official_circular" if is_circular else "unstructured_prose"
    doc_type = "Official Circular / Statutory Order" if is_circular else "PDF Document"

    try:
        doc = fitz.open(pdf_path)
        total_p = min(len(doc), max_pages)
        for page_idx in range(total_p):
            page_num = page_idx + 1
            page = doc[page_idx]
            page_text = page.get_text()
            
            # 1. Precise Structured Table Extraction (PyMuPDF C++ Table Engine)
            if any(k in page_text.lower() for k in PDF_TABLE_KEYWORDS):
                try:
                    tabs = page.find_tables()
                    if tabs and tabs.tables:
                        for t_idx, tab in enumerate(tabs.tables):
                            md = tab.to_markdown()
                            if md and len(md.strip().split("\n")) >= 3:
                                chunks.append({
                                    "id": f"StructuredTable_{doc_name}_p{page_num}_{t_idx+1}",
                                    "source": doc_name,
                                    "page": f"Page {page_num} (Structured Table {t_idx+1})",
                                    "type": "Structured Table",
                                    "modality": "structured_table",
                                    "text": f"[STRUCTURED DATA TABLE | Source: {doc_name} | Page {page_num} | Table {t_idx+1}]\n{md}"
                                })
                except Exception:
                    pass

            # 2. Unstructured Narrative Prose Extraction
            if not page_text or len(page_text.strip()) < 20:
                continue
                
            raw_chunks = chunk_text(page_text)
            for c_idx, chunk in enumerate(raw_chunks):
                chunks.append({
                    "id": f"{doc_name}_p{page_num}_{c_idx}",
                    "source": doc_name,
                    "page": f"Page {page_num}",
                    "type": doc_type,
                    "modality": text_modality,
                    "text": chunk
                })
        doc.close()
    except Exception as e:
        pass
        
    return chunks

def extract_spreadsheet_chunks(file_path: str):
    """
    Extract structured tabular records from Excel (.xlsx, .xls) and CSV (.csv) files.
    Outputs markdown table chunks preserving column names, row associations, and cell values.
    """
    chunks = []
    fname = os.path.basename(file_path)
    # Strict Privacy Filter
    payroll_terms = ['salary', 'payroll', 'remuneration', 'payslip', 'property_returns']
    if any(k in fname.lower() for k in payroll_terms):
        return []

    try:
        if file_path.lower().endswith('.csv'):
            dfs = {"Data": pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')}
        else:
            dfs = pd.read_excel(file_path, sheet_name=None)

        for sheet_name, df in dfs.items():
            if df is None or df.empty:
                continue
            df = df.dropna(how='all')
            if df.empty:
                continue

            # Chunk into manageable tabular windows of 25 rows
            chunk_size = 25
            for start_idx in range(0, len(df), chunk_size):
                sub_df = df.iloc[start_idx:start_idx + chunk_size]
                try:
                    md_table = sub_df.to_markdown(index=False)
                except Exception:
                    md_table = sub_df.to_string(index=False)

                row_info = f"Rows {start_idx+1}-{min(start_idx+chunk_size, len(df))}"
                chunks.append({
                    "id": f"SpreadsheetTable_{fname}_{sheet_name}_{start_idx}",
                    "source": f"{fname} ({sheet_name})",
                    "page": f"Spreadsheet [{sheet_name} : {row_info}]",
                    "type": "Structured Table / Spreadsheet",
                    "modality": "structured_table",
                    "text": f"[STRUCTURED DATA TABLE | Source: {fname} | Sheet: {sheet_name} | {row_info}]\n{md_table}"
                })
    except Exception:
        pass
    return chunks

def extract_web_page_chunks(json_path: str):
    chunks = []
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        title = data.get("title", "IIT Jodhpur Portal")
        url = data.get("url", "https://iitj.ac.in/")
        text = data.get("text", "")
        
        raw_chunks = chunk_text(text, chunk_size=750, overlap=120)
        for c_idx, chunk in enumerate(raw_chunks):
            chunks.append({
                "id": f"Web_{os.path.basename(json_path)}_{c_idx}",
                "source": title,
                "page": url,
                "type": "Web Page / Portal",
                "modality": "semi_structured_web",
                "text": f"[{title}] {chunk}"
            })
    except Exception:
        pass
    return chunks

def extract_html_chunks(html_path: str):
    chunks = []
    if not os.path.exists(html_path):
        return chunks
        
    try:
        with open(html_path, 'r', encoding='utf-8') as f:
            html = f.read()
            
        soup = BeautifulSoup(html, 'html.parser')
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4'])
        for h in headings:
            heading_text = clean_text(h.get_text())
            if not heading_text or len(heading_text) < 3:
                continue
                
            section_texts = []
            curr = h.find_next_sibling()
            while curr and curr.name not in ['h1', 'h2', 'h3', 'h4']:
                t = clean_text(curr.get_text())
                if t:
                    section_texts.append(t)
                curr = curr.find_next_sibling()
                
            full_section_text = " ".join(section_texts)
            if full_section_text:
                for c_idx, chunk in enumerate(chunk_text(full_section_text, chunk_size=800, overlap=100)):
                    chunks.append({
                        "id": f"WebPage_{heading_text[:20]}_{c_idx}",
                        "source": f"Academic Regulations: {heading_text}",
                        "page": "https://www.iitj.ac.in/Office-of-Academics/en/Academic-Regulations",
                        "type": "Web Page / Portal",
                        "modality": "semi_structured_web",
                        "text": f"[{heading_text}] {chunk}"
                    })
    except Exception:
        pass
        
    return chunks

def build_index():
    print("=== Starting Comprehensive Multi-Modal Knowledge Base Ingestion ===", flush=True)
    all_chunks = []
    
    # 1. Process all Web Pages (Directories, Faculty, Departments, Contacts, Statutory Bodies)
    web_files = sorted(glob.glob(os.path.join(WEB_DIR, "*.json")))
    print(f"Found {len(web_files)} scraped Web Page archives in {WEB_DIR}", flush=True)
    for i, w_path in enumerate(web_files):
        c = extract_web_page_chunks(w_path)
        all_chunks.extend(c)
    print(f"  Processed {len(web_files)} web pages -> {len(all_chunks)} chunks generated so far", flush=True)

    # 2. Process all Structured Spreadsheets / CSVs
    spreadsheet_files = sorted(
        glob.glob(os.path.join(DATA_DIR, "*.csv")) +
        glob.glob(os.path.join(DATA_DIR, "*.xlsx")) +
        glob.glob(os.path.join(TABLES_DIR, "*.csv")) +
        glob.glob(os.path.join(TABLES_DIR, "*.xlsx"))
    )
    if spreadsheet_files:
        print(f"Found {len(spreadsheet_files)} Spreadsheet files", flush=True)
        for s_path in spreadsheet_files:
            c = extract_spreadsheet_chunks(s_path)
            all_chunks.extend(c)
        print(f"  Ingested {len(spreadsheet_files)} spreadsheets -> {len(all_chunks)} chunks total", flush=True)

    # 3. Process all PDFs (Curricula, Regulations, Handbooks, Fee Structures, Circulars)
    pdf_files = sorted(glob.glob(os.path.join(PDF_DIR, "*.pdf")))
    print(f"Found {len(pdf_files)} PDF files in {PDF_DIR}", flush=True)
    pdf_chunks_count = 0
    for i, pdf_path in enumerate(pdf_files):
        c = extract_pdf_chunks(pdf_path)
        all_chunks.extend(c)
        pdf_chunks_count += len(c)
        if (i + 1) % 150 == 0 or (i + 1) == len(pdf_files):
            print(f"  [PDF Progress] {i+1}/{len(pdf_files)} processed ({pdf_chunks_count} chunks)", flush=True)
    print(f"  Processed {len(pdf_files)} PDF documents -> {pdf_chunks_count} chunks generated", flush=True)
        
    # 4. Process Main Academic Regulations HTML Portal
    html_files = glob.glob(os.path.join(HTML_DIR, "*.html"))
    for html_path in html_files:
        c = extract_html_chunks(html_path)
        all_chunks.extend(c)

    print(f"\nTotal combined multi-modal knowledge base segments: {len(all_chunks)}", flush=True)
    
    if not all_chunks:
        print("No chunks to index.", flush=True)
        return len(all_chunks)
        
    # 5. Fit Unified Hybrid Vectorizer
    print("Building unified TF-IDF vocabulary across all website pages and documents...", flush=True)
    corpus = [chunk["text"] for chunk in all_chunks]
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        sublinear_tf=True,
        stop_words='english',
        max_features=65000
    )
    tfidf_matrix = vectorizer.fit_transform(corpus)
    
    # 6. Save Chunks JSON
    with open(INDEX_PATH, 'w', encoding='utf-8') as f:
        json.dump(all_chunks, f, ensure_ascii=False)
    print(f"Saved chunks to {INDEX_PATH}", flush=True)
    
    # 7. Save TF-IDF Model & Matrix
    with open(TFIDF_PATH, 'wb') as f:
        pickle.dump({"vectorizer": vectorizer, "matrix": tfidf_matrix}, f)
    print(f"Saved TF-IDF model to {TFIDF_PATH}", flush=True)
    
    unique_sources = set(c["source"] for c in all_chunks)
    print(f"=== Complete! Successfully indexed {len(all_chunks)} segments from {len(unique_sources)} distinct sources! ===", flush=True)
    return len(all_chunks)

if __name__ == "__main__":
    build_index()
