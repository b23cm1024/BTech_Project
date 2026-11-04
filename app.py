import os
import re
import base64
import streamlit as st
from dotenv import load_dotenv
from rag import rag_engine
from ingest import build_index, extract_pdf_chunks, PDF_DIR
from deep_crawl import run_deep_crawl

# Text sanitizer to eliminate any raw <br> or HTML tags from displaying in the UI
def clean_ui_text(text: str) -> str:
    if not text:
        return ""
    lines = str(text).split("\n")
    cleaned_lines = []
    for line in lines:
        if line.strip().startswith("|") and line.strip().endswith("|"):
            line = re.sub(r'<\s*br\s*/?\s*>', ' / ', line, flags=re.IGNORECASE)
        else:
            line = re.sub(r'<\s*br\s*/?\s*>', '\n', line, flags=re.IGNORECASE)
        cleaned_lines.append(line)
    cleaned = "\n".join(cleaned_lines)
    cleaned = re.sub(r'<\s*br\s*/?\s*>', '\n', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'</?(?:p|span|div)[^>]*>', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'【[^】]*】', '', cleaned)
    return cleaned

# Load local environment variables
load_dotenv()

st.set_page_config(
    page_title="IIT Jodhpur Intelligence Portal",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Helper function to convert local image to base64 for crisp rendering
def get_image_base64(path):
    if os.path.exists(path):
        with open(path, "rb") as f:
            data = f.read()
        ext = os.path.splitext(path)[1].lstrip('.').lower()
        if ext == 'jpg':
            ext = 'jpeg'
        return f"data:image/{ext};base64,{base64.b64encode(data).decode()}"
    return ""

LOGO_B64 = get_image_base64("assets/iitj_logo.png")

# Load / Cache RAG Engine
@st.cache_resource
def get_rag_engine():
    rag_engine.load_index()
    return rag_engine

engine = get_rag_engine()

# Custom Professional UI Styling (Institutional & Clean)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Keyframe Animations */
    @keyframes slideFadeUp {
        from {
            opacity: 0;
            transform: translateY(10px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }

    /* Clean Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #F8FAFC;
        border-right: 1px solid #E2E8F0;
        padding-top: 1.2rem;
    }
    
    .sidebar-brand-container {
        text-align: center;
        padding-bottom: 1.1rem;
        margin-bottom: 1.2rem;
        border-bottom: 1px solid #E2E8F0;
    }
    
    /* Crisp Logo with smooth interactive hover lift */
    .sidebar-logo {
        width: 105px;
        height: auto;
        object-fit: contain;
        image-rendering: -webkit-optimize-contrast;
        image-rendering: crisp-edges;
        transition: transform 0.3s cubic-bezier(0.34, 1.56, 0.64, 1), filter 0.3s ease;
        filter: drop-shadow(0 2px 4px rgba(0, 0, 0, 0.06));
        margin-bottom: 8px;
    }

    .sidebar-logo:hover {
        transform: scale(1.06);
        filter: drop-shadow(0 6px 12px rgba(30, 58, 138, 0.18));
    }

    .sidebar-brand-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: 0.5px;
        margin: 0;
    }

    .sidebar-brand-subtitle {
        font-size: 0.75rem;
        font-weight: 500;
        color: #64748B;
        margin-top: 2px;
        margin-bottom: 0;
    }

    /* Page Navigation Radio in Sidebar */
    div[data-testid="stRadio"] > div {
        gap: 0.4rem;
    }
    
    div[data-testid="stRadio"] label {
        padding: 0.6rem 0.9rem;
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.92rem;
        cursor: pointer;
        transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1);
        color: #334155;
        border: 1px solid transparent;
    }

    div[data-testid="stRadio"] label:hover {
        background-color: #FFFFFF;
        color: #1E40AF;
        border-color: #CBD5E1;
        transform: translateX(4px);
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
    }

    /* Buttons Styling */
    div[data-testid="stButton"] > button {
        border-radius: 8px;
        font-weight: 500;
        transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
        border: 1px solid #E2E8F0;
    }

    div[data-testid="stButton"] > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.15);
        border-color: #2563EB;
        color: #2563EB;
    }

    /* Headers */
    .page-main-header {
        font-size: 2.15rem;
        font-weight: 700;
        color: #0F172A;
        letter-spacing: -0.5px;
        margin-bottom: 0.25rem;
        animation: slideFadeUp 0.35s ease-out;
    }

    .page-sub-header {
        font-size: 1.02rem;
        color: #475569;
        margin-bottom: 1.4rem;
        line-height: 1.5;
        animation: slideFadeUp 0.4s ease-out;
    }

    /* Citation Card */
    .citation-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-left: 4px solid #1D4ED8;
        padding: 12px 16px;
        margin-top: 8px;
        margin-bottom: 8px;
        border-radius: 6px;
        font-size: 0.88rem;
        color: #1E293B;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }

    .citation-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 3px 6px rgba(0, 0, 0, 0.05);
    }

    .citation-badge {
        display: inline-block;
        background-color: #EFF6FF;
        color: #1D4ED8;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.76rem;
        font-weight: 600;
        margin-right: 8px;
    }

    .stChatMessage {
        animation: slideFadeUp 0.3s ease-out;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# Session State
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_question" not in st.session_state:
    st.session_state.selected_question = None

# Active Model Config (Stored in session state or .env)
if "ai_provider" not in st.session_state:
    st.session_state.ai_provider = "Groq" if os.getenv("GROQ_API_KEY") else "Deterministic"
if "model_name" not in st.session_state:
    st.session_state.model_name = "openai/gpt-oss-120b"

# Retrieve all loaded documents and web pages
unique_sources = sorted(list(set(c["source"] for c in engine.chunks)))

# ==========================================
# 1. CLEAN & PROFESSIONAL SIDEBAR
# ==========================================
with st.sidebar:
    # Institute Header & Crisp Logo
    st.markdown(f"""
    <div class="sidebar-brand-container">
        <img class="sidebar-logo" src="{LOGO_B64}" alt="IIT Jodhpur Official Seal">
        <h2 class="sidebar-brand-title">IIT JODHPUR</h2>
        <p class="sidebar-brand-subtitle">Academic & Campus Intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    # Clean Navigation Menu
    nav_page = st.radio(
        "Navigation",
        ["Academic Advisor", "Document Repository"],
        label_visibility="collapsed"
    )

    st.write("")
    if st.button("Clear Conversation", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

    with st.expander("Model Configuration", expanded=False):
        default_p_idx = 0 if os.getenv("GEMINI_API_KEY") else 1
        selected_provider = st.selectbox(
            "AI Provider",
            ["Google Gemini", "Groq"],
            index=default_p_idx
        )
        st.session_state.ai_provider = selected_provider

        if selected_provider == "Google Gemini":
            st.session_state.model_name = st.selectbox(
                "Gemini Model (Free Tier)",
                ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-3.8-flash"],
                index=0
            )
            custom_gem_key = st.text_input("Gemini API Key (optional)", type="password", help="Overrides GEMINI_API_KEY from .env")
            if custom_gem_key:
                os.environ["GEMINI_API_KEY"] = custom_gem_key
        else:
            st.session_state.model_name = st.selectbox(
                "Groq Model (Free Tier)",
                ["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"],
                index=0
            )

# ==========================================
# 2. PAGE 1: ACADEMIC ADVISOR (CHAT)
# ==========================================
if nav_page == "Academic Advisor":
    st.markdown('<div class="page-main-header">IIT Jodhpur Academic Advisor</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub-header">Search across institute documents, regulations, curricula, and administrative records.</div>', unsafe_allow_html=True)

    # Quick Suggestion Chips (Expanded across institute scope)
    st.markdown("**Suggested Queries:**")
    q_col1, q_col2, q_col3 = st.columns(3)
    with q_col1:
        if st.button("Minimum Attendance Requirements", use_container_width=True):
            st.session_state.selected_question = "What is the minimum attendance requirement to appear in examinations at IIT Jodhpur?"
        if st.button("Leadership & Key Functionaries", use_container_width=True):
            st.session_state.selected_question = "Who is the Director, Registrar, and Chairman of Board of Governors at IIT Jodhpur?"
    with q_col2:
        if st.button("B.Tech Electronics Curriculum", use_container_width=True):
            st.session_state.selected_question = "What is the detailed curriculum and course structure for B.Tech in Electronics Engineering?"
        if st.button("Administrative Contacts & Offices", use_container_width=True):
            st.session_state.selected_question = "What are the administrative contact details and offices at IIT Jodhpur?"
    with q_col3:
        if st.button("Branch Change Criteria", use_container_width=True):
            st.session_state.selected_question = "What are the eligibility criteria and rules for branch change for B.Tech students?"
        if st.button("Health Center & Campus Facilities", use_container_width=True):
            st.session_state.selected_question = "Where is the Health Center located and what facilities are available on campus?"

    st.divider()

    # Chat Transcript
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(clean_ui_text(message["content"]))
            if "sources" in message and message["sources"]:
                with st.expander("Statutory Evidence Cards & Document Excerpts", expanded=False):
                    for idx, src in enumerate(message["sources"]):
                        clean_src_text = clean_ui_text(src.get('text', ''))
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <span class="citation-badge">Source {idx+1}</span> 
                                <div style="font-weight: 600; margin-top: 4px; color: #1e293b;">{src.get('source')} <span style="font-size: 0.82em; font-family: monospace; color: #64748b;">(Page {src.get('page')})</span></div>
                                <div style="color: #334155; font-style: italic; margin-top: 6px;">"{clean_src_text}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

    # Chat Input
    user_prompt = st.chat_input("Ask any question regarding faculty, departments, contacts, courses, grading, attendance, or guidelines...")

    if st.session_state.selected_question:
        user_prompt = st.session_state.selected_question
        st.session_state.selected_question = None

    if user_prompt:
        st.session_state.chat_history.append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)

        # Active credentials
        active_api_key = os.getenv("GROQ_API_KEY" if st.session_state.ai_provider == "Groq" else "GEMINI_API_KEY", "")

        with st.chat_message("assistant"):
            with st.spinner("Searching institute web portals & verified documents..."):
                response = engine.generate_answer(
                    query=user_prompt,
                    api_provider=st.session_state.ai_provider,
                    api_key=active_api_key,
                    model_name=st.session_state.model_name,
                    chat_history=st.session_state.chat_history
                )

            cleaned_answer = clean_ui_text(response["answer"])
            st.markdown(cleaned_answer)

            if response["sources"]:
                with st.expander("Statutory Evidence Cards & Document Excerpts", expanded=True):
                    for idx, src in enumerate(response["sources"]):
                        clean_src_text = clean_ui_text(src.get('text', ''))
                        st.markdown(
                            f"""
                            <div class="citation-card">
                                <span class="citation-badge">Source {idx+1}</span> 
                                <div style="font-weight: 600; margin-top: 4px; color: #1e293b;">{src.get('source')} <span style="font-size: 0.82em; font-family: monospace; color: #64748b;">(Page {src.get('page')})</span></div>
                                <div style="color: #334155; font-style: italic; margin-top: 6px;">"{clean_src_text}"</div>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

            st.session_state.chat_history.append({
                "role": "assistant",
                "content": cleaned_answer,
                "sources": response["sources"]
            })

# ==========================================
# 3. PAGE 2: DOCUMENT & WEB REPOSITORY
# ==========================================
elif nav_page == "Document Repository":
    st.markdown('<div class="page-main-header">Institute Document & Web Repository</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-sub-header">Search, inspect, and manage official IIT Jodhpur documents, web portals, and directories indexed in the intelligence system.</div>', unsafe_allow_html=True)

    # Overview Metrics Row
    m_col1, m_col2, m_col3 = st.columns(3)
    with m_col1:
        st.metric("Total Indexed Sources", f"{len(unique_sources)} sources")
    with m_col2:
        st.metric("Total Searchable Segments", f"{len(engine.chunks):,} clauses")
    with m_col3:
        st.metric("Repository Scope", "Institute-Wide (A to Z)")

    st.markdown("---")

    # Document Search & Filter Bar
    f_col1, f_col2 = st.columns([2, 1])
    with f_col1:
        search_query = st.text_input("Search sources by keyword:", placeholder="e.g. director, deans, btech, nirf, chemistry, health center...")
    with f_col2:
        category_filter = st.selectbox(
            "Category Filter:",
            [
                "All Sources",
                "Web Portals, People & Administration",
                "Curricula & Syllabi",
                "Academic Regulations & Ordinances",
                "Academic Calendars",
                "NIRF & Institutional Data",
                "Circulars, Committees & Tenders"
            ]
        )

    # Enhanced Categorization
    def classify_source(name: str):
        nl = name.lower()
        if "(web portal)" in nl or any(w in nl for w in ["office of", "contact", "director", "deans", "warden", "statutory", "infrastructure", "health", "dia", "crf", "department", "functionaries"]):
            return "Web Portals, People & Administration"
        elif any(w in nl for w in ["regulation", "ordinance", "norm", "rule"]):
            return "Academic Regulations & Ordinances"
        elif any(w in nl for w in ["curriculum", "syllabus", "mtech", "btech", "msc", "phd", "concept"]):
            return "Curricula & Syllabi"
        elif any(w in nl for w in ["calendar", "ay-", "summer-term"]):
            return "Academic Calendars"
        elif any(w in nl for w in ["nirf", "ranking", "annual", "consultancy"]):
            return "NIRF & Institutional Data"
        else:
            return "Circulars, Committees & Tenders"

    filtered_docs = []
    for s in unique_sources:
        cat = classify_source(s)
        if category_filter != "All Sources" and cat != category_filter:
            continue
        if search_query and search_query.lower() not in s.lower():
            continue
        filtered_docs.append((s, cat))

    st.write(f"Showing **{len(filtered_docs)}** of **{len(unique_sources)}** indexed sources:")

    # Document List
    for s, cat in filtered_docs[:80]:
        doc_chunks = [c for c in engine.chunks if c.get("source") == s]
        source_type = (doc_chunks[0].get("type") or doc_chunks[0].get("modality") or doc_chunks[0].get("content_type") or "Document") if doc_chunks else "Record"
        with st.expander(f"{s} ({len(doc_chunks)} segments indexed)", expanded=False):
            st.caption(f"Category: {cat} | Format: {source_type}")
            if doc_chunks:
                sample_preview = clean_ui_text(doc_chunks[0].get("text", "")[:380])
                page_info = doc_chunks[0].get("page", "N/A")
                st.markdown(f"**Sample Content Preview ({page_info}):**")
                st.info(f'"{sample_preview}..."')

    if len(filtered_docs) > 80:
        st.caption(f"... and {len(filtered_docs) - 80} additional sources matching your criteria.")

    st.markdown("---")

    # Document Management & Ingestion Controls
    st.subheader("Repository Synchronization & Updates")
    act_col1, act_col2 = st.columns(2)
    with act_col1:
        st.markdown("**Upload Additional Document:**")
        up_file = st.file_uploader("Select PDF file to index:", type=["pdf"], key="doc_lib_uploader")
        if up_file is not None:
            save_path = os.path.join(PDF_DIR, up_file.name)
            if not os.path.exists(save_path):
                with open(save_path, "wb") as f:
                    f.write(up_file.getbuffer())
                with st.spinner("Indexing new document into AI knowledge base..."):
                    build_index()
                    engine.load_index()
                st.success(f"Successfully indexed {up_file.name}")
                st.rerun()

    with act_col2:
        st.markdown("**Synchronize Entire Website:**")
        st.caption("Deep-crawl all 1,300+ A-to-Z directories, portals, and newly published circulars from iitj.ac.in.")
        if st.button("Deep-Crawl & Synchronize Website", use_container_width=True):
            with st.spinner("Executing full deep crawl and re-indexing..."):
                run_deep_crawl(max_pages=350, max_workers=8)
                build_index()
                engine.load_index()
            st.success("Website synchronization completed.")
            st.rerun()
