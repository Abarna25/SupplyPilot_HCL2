import os
import sys
import site
import time
from pathlib import Path

# Ensure user site-packages are accessible
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import streamlit as st
from dotenv import load_dotenv

from config import (
    DATA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LLM_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    DEFAULT_TOP_K,
)
from ingest import (
    ingest_pdf_bytes,
    ingest_all_data_pdfs,
    get_db_stats,
    clear_knowledge_base,
)
from rag import RAGEngine

load_dotenv()

# Page Setup
st.set_page_config(
    page_title="SupplyPilot — AI Supply Chain Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Enterprise Styling (CSS)
st.markdown("""
<style>
    /* Global Container Adjustments */
    .main .block-container {
        padding-top: 1.8rem;
        padding-bottom: 2rem;
        max-width: 1250px;
    }
    
    /* Header Card Styling */
    .sp-header {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 24px 30px;
        color: #FFFFFF;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    .sp-header h1 {
        color: #F8FAFC !important;
        font-size: 2.2rem !important;
        font-weight: 700 !important;
        margin: 0 0 6px 0 !important;
        letter-spacing: -0.5px;
    }
    .sp-header .tagline {
        color: #38BDF8;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 8px;
    }
    .sp-header .desc {
        color: #94A3B8;
        font-size: 0.95rem;
        margin: 0;
    }
    
    /* Metric Cards */
    .metric-card {
        background: #1E293B;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 16px 20px;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    .metric-value {
        font-size: 1.7rem;
        font-weight: 700;
        color: #38BDF8;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #94A3B8;
    }
    
    /* Answer Card */
    .answer-card {
        background-color: #0F172A;
        border-left: 4px solid #38BDF8;
        border-top: 1px solid #334155;
        border-right: 1px solid #334155;
        border-bottom: 1px solid #334155;
        border-radius: 8px;
        padding: 24px;
        margin-top: 15px;
        margin-bottom: 20px;
        color: #F8FAFC;
    }
    
    /* Source Badges */
    .source-badge {
        display: inline-block;
        background: #1E293B;
        border: 1px solid #0EA5E9;
        color: #38BDF8;
        border-radius: 6px;
        padding: 6px 12px;
        font-size: 0.85rem;
        font-weight: 600;
        margin-right: 8px;
        margin-bottom: 8px;
    }
    
    /* Suggested Questions Buttons */
    .stButton > button {
        border-radius: 8px !important;
        font-weight: 500 !important;
    }
    
    /* Refusal Banner */
    .refusal-card {
        background-color: #450A0A;
        border-left: 4px solid #EF4444;
        border-radius: 8px;
        padding: 18px 22px;
        color: #FCA5A5;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    /* Sidebar Section Divider */
    .sidebar-section {
        background: #1E293B;
        padding: 14px;
        border-radius: 8px;
        border: 1px solid #334155;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Session States
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "last_query" not in st.session_state:
    st.session_state.last_query = ""

if "last_result" not in st.session_state:
    st.session_state.last_result = None


def load_stats():
    return get_db_stats()


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION & KNOWLEDGE BASE
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📦 SupplyPilot KB")
    
    # API Key check banner
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        st.error("⚠️ `GEMINI_API_KEY` missing from `.env` file!")
    else:
        st.success("🔑 Gemini API Key Active", icon="✅")

    st.markdown("---")
    st.markdown("### 📑 Knowledge Base")
    
    # PDF Upload Widget
    uploaded_files = st.file_uploader(
        "Upload PDF Document(s)",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload Meridian Procurement Policy or Performance Review PDFs."
    )

    col_idx1, col_idx2 = st.columns(2)
    
    with col_idx1:
        if st.button("⚡ Index KB", use_container_width=True, type="primary"):
            with st.spinner("Indexing documents into ChromaDB..."):
                indexed_count = 0
                messages = []
                
                # First ingest any uploaded files
                if uploaded_files:
                    for uf in uploaded_files:
                        res = ingest_pdf_bytes(uf.getvalue(), uf.name)
                        messages.append(res["message"])
                        if res["status"] == "success":
                            indexed_count += res["chunks_added"]
                
                # Next ingest default files in data/ directory if not indexed
                dir_results = ingest_all_data_pdfs()
                for r in dir_results:
                    if r["status"] == "success":
                        indexed_count += r["chunks_added"]
                        messages.append(r["message"])
                    elif r["status"] == "already_indexed":
                        messages.append(r["message"])
                
                st.success("✓ Ingestion Completed")
                for m in messages:
                    st.caption(m)
                time.sleep(1)
                st.rerun()

    with col_idx2:
        confirm_clear = st.checkbox("Confirm Clear", key="chk_clear")
        if st.button("🗑 Clear KB", use_container_width=True, disabled=not confirm_clear):
            clear_knowledge_base()
            st.session_state.chat_history = []
            st.session_state.last_result = None
            st.success("Knowledge Base Cleared!")
            time.sleep(1)
            st.rerun()

    # Current Stats
    stats = load_stats()
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("##### 📄 Indexed Documents")
    if stats["indexed_documents"]:
        for doc in stats["indexed_documents"]:
            st.markdown(f"• `{doc}`")
    else:
        st.info("No documents currently indexed.")

    st.markdown("---")
    st.markdown("### ⚙️ Retrieval Settings")
    st.markdown(f"""
    - **Vector Store:** `ChromaDB`
    - **Embedding:** `gemini-embedding-001`
    - **LLM Model:** `gemini-flash-latest`
    - **Temperature:** `0.1`
    - **Chunk Size:** `1000 chars`
    - **Chunk Overlap:** `150 chars`
    - **Top-K Retrieval:** `5`
    """)

    st.markdown("---")
    debug_mode = st.toggle("🛠 Developer / Debug Mode", value=False)


# -----------------------------------------------------------------------------
# MAIN APP HEADER & DASHBOARD METRICS
# -----------------------------------------------------------------------------
st.markdown("""
<div class="sp-header">
    <div class="tagline">Navigate Supply Chain Decisions with AI</div>
    <h1>SupplyPilot</h1>
    <p class="desc">Ask complex single & cross-document questions across Meridian Components' Procurement Policies & Quarterly Performance Reviews.</p>
</div>
""", unsafe_allow_html=True)

# Dashboard Metrics Row
stats = load_stats()
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{stats['document_count']}</div>
        <div class="metric-label">Documents Indexed</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{stats['total_chunks']}</div>
        <div class="metric-label">Chunks Stored</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">ChromaDB</div>
        <div class="metric-label">Vector Store</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">Gemini AI</div>
        <div class="metric-label">AI Engine</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# SUGGESTED EVALUATION QUESTIONS
# -----------------------------------------------------------------------------
st.markdown("### 💡 Assignment Test Questions")

test_questions = [
    "Q1: Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?",
    "Q2: How many line stoppages happened in Q1, what was the total downtime, and what caused them?",
    "Q3: What is the approval authority for a purchase order worth ₹1.4 crore?",
    "Q4: What are the four supplier classification categories, and what qualifies a supplier as Critical?",
    "Q5: Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do?",
    "Q6: The microcontroller supplier is single-source. What does the sourcing policy require in this situation, and what is the company already doing about it?",
    "Q7: Microcontrollers are imported with a 46-day lead time. Using the safety-stock policy, how many days of stock should be held for this part?",
    "Q8: Trident Circuit Boards had a defect rate of 640 parts per million. What is the cost consequence under the policy?",
    "Q9: Which suppliers would fall below the B rating band on on-time delivery alone, and what is the escalation path for them?",
    "Q10: What is the annual salary of the Head of Procurement?"
]

# Quick selection grid
q_cols = st.columns(2)
selected_prompt = None

for idx, q_text in enumerate(test_questions):
    col = q_cols[idx % 2]
    if col.button(q_text, key=f"btn_q_{idx}", use_container_width=True):
        selected_prompt = q_text.split(": ", 1)[1] if ": " in q_text else q_text


# -----------------------------------------------------------------------------
# QUESTION INPUT AREA
# -----------------------------------------------------------------------------
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("### 🔍 Ask SupplyPilot")

query_input = st.text_input(
    "Enter your supply chain question:",
    value=selected_prompt if selected_prompt else "",
    placeholder="e.g., Kaveri Metals recorded 88.1% OTD and 1150 PPM. What policy clauses are triggered and what must the buyer do?",
    key="input_query_field"
)

col_ask1, col_ask2 = st.columns([1, 5])
with col_ask1:
    ask_clicked = st.button("Ask SupplyPilot →", type="primary", use_container_width=True)

if ask_clicked or selected_prompt:
    query_to_process = query_input if query_input else selected_prompt
    if query_to_process:
        if stats["total_chunks"] == 0:
            st.warning("⚠️ Knowledge base is empty. Please click '⚡ Index KB' in the sidebar first.")
        else:
            with st.spinner("Retrieving evidence & generating grounded answer via Gemini AI..."):
                rag_engine = RAGEngine(top_k=DEFAULT_TOP_K)
                res = rag_engine.answer_question(query_to_process)
                st.session_state.last_result = res
                st.session_state.chat_history.append((query_to_process, res))


# -----------------------------------------------------------------------------
# ANSWER & EVIDENCE DISPLAY
# -----------------------------------------------------------------------------
if st.session_state.last_result:
    res = st.session_state.last_result
    query = res["query"]
    answer = res["answer"]
    sources = res["sources"]
    evidence = res["evidence"]
    refused = res.get("refused", False)

    st.markdown("---")
    st.markdown(f"#### ❓ Question: *{query}*")

    if refused:
        st.markdown(f"""
        <div class="refusal-card">
            <h4>⛔ Information Not Found / Refusal</h4>
            <p>{answer}</p>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("#### 🤖 AI Answer")
        st.markdown(f"""
        <div class="answer-card">
            {answer}
        </div>
        """, unsafe_allow_html=True)

        # Source Citations Section
        if sources:
            st.markdown("#### 📄 Sources")
            src_html = ""
            for s in sources:
                src_html += f'<span class="source-badge">📄 {s["source"]} — Page {s["page"]}</span>'
            st.markdown(src_html, unsafe_allow_html=True)

        # Expandable Evidence
        if evidence:
            with st.expander("▸ View Retrieved Evidence (Top-K Chunks)", expanded=False):
                for i, chunk in enumerate(evidence, 1):
                    st.markdown(f"**Chunk {i}** | Document: `{chunk['source']}` | **Page {chunk['page']}** | Distance: `{chunk['distance']}`")
                    st.code(chunk["text"], language="text")
                    st.markdown("---")

    # Developer Debug Drawer
    if debug_mode:
        with st.expander("🛠 Developer / Debug Info", expanded=True):
            st.json({
                "query": query,
                "refused": refused,
                "top_k": DEFAULT_TOP_K,
                "retrieved_chunk_count": len(evidence),
                "sources_retrieved": sources,
                "raw_chunks": [
                    {
                        "source": c["source"],
                        "page": c["page"],
                        "distance": c["distance"],
                        "snippet": c["text"][:150] + "..."
                    } for c in evidence
                ]
            })


# -----------------------------------------------------------------------------
# SESSION CHAT HISTORY
# -----------------------------------------------------------------------------
if st.session_state.chat_history:
    st.markdown("<br><br>", unsafe_allow_html=True)
    h_col1, h_col2 = st.columns([4, 1])
    with h_col1:
        st.markdown("### 💬 Session Q&A History")
    with h_col2:
        if st.button("Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.last_result = None
            st.rerun()

    for idx, (h_q, h_res) in enumerate(reversed(st.session_state.chat_history), 1):
        with st.expander(f"Q: {h_q}", expanded=False):
            st.markdown(h_res["answer"])
            if h_res["sources"]:
                st.caption("Sources: " + ", ".join([f"{s['source']} (Page {s['page']})" for s in h_res["sources"]]))
