# SupplyPilot

> **Navigate Supply Chain Decisions with AI**
> *AI-Powered Supply Chain Document Intelligence for Procurement Policies & Operational Performance Data.*

---

## 1. Project Title
**SupplyPilot**

## 2. Tagline
**Navigate Supply Chain Decisions with AI**

---

## 3. Overview
**SupplyPilot** is a submission-ready, enterprise-grade Retrieval-Augmented Generation (RAG) system built to parse, index, and query complex supply chain documentation. It enables buyers, category managers, and supply chain analysts to execute cross-document queries that synthesize operational performance data with formal procurement governance policies.

The system indexes Meridian Components Pvt. Ltd.'s primary documentation into a unified vector store and delivers grounded, verifiable answers complete with exact document name and page number citations.

---

## 4. Problem Statement
Supply chain leadership often operates with fragmented knowledge split across two distinct domains:
1. **Governance & Policy Rules**: Static handbooks specifying approval tiers, penalty structures, dual-sourcing mandates, safety stock rules, and escalation paths.
2. **Operational Performance Data**: Quarterly reviews documenting supplier spend, on-time delivery percentages, defect PPM, line stoppages, lead times, and active risks.

When operational disruptions occur (e.g., delivery delays or defect spikes), buyers must manually locate the relevant performance data, cross-reference policy handbooks to determine mandatory enforcement clauses, and compute required penalties or mitigation actions. SupplyPilot automates this cross-document synthesis in seconds.

---

## 5. Solution
SupplyPilot bridges operational performance metrics and corporate procurement rules using an end-to-end RAG architecture powered by OpenAI's `text-embedding-3-small` embeddings, `GPT-4o` (temperature `0.1`), and persistent `ChromaDB`.

Both primary documents are indexed into a **single unified collection** to allow similarity retrieval across policy and performance data simultaneously:
1. `Meridian_Procurement_Policy_Handbook_v4.2.pdf` (Policy source of truth)
2. `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf` (Operational data source of truth)

---

## 6. Key Features
- **Cross-Document Synthesis**: Queries automatically retrieve relevant chunks from both the Procurement Policy Handbook and the Q1 Performance Review in a single search pass.
- **Strict Grounding & Honest Refusal**: A constrained system prompt ensures zero hallucination. If a query cannot be answered from the document context (e.g., salary data), SupplyPilot responds with exact refusal: *"The information is not available in the uploaded documents."*
- **Source & Page Citation Engine**: Answers feature explicit source citations citing PDF file names and exact page numbers.
- **Expandable Evidence View**: Users can inspect raw retrieved top-K text chunks, source pages, and similarity distances for complete auditability.
- **Dynamic Persistent Vector DB**: Automatic persistence via ChromaDB ensures database state survives application restarts without requiring re-ingestion.
- **Duplicate Document Detection**: SHA-256 content hashing prevents re-indexing duplicate files.
- **Multi-Interface Access**: Interactive Streamlit web interface + FastAPI REST backend (`/ingest`, `/ask`, `/stats`).

---

## 7. Architecture

```text
                                SUPPLYPILOT ARCHITECTURE
                                           │
         ┌─────────────────────────────────┴─────────────────────────────────┐
         ▼                                                                   ▼
┌──────────────────┐                                               ┌──────────────────┐
│ Procurement      │                                               │ Supply Chain     │
│ Policy Handbook  │                                               │ Review Q1        │
└────────┬─────────┘                                               └────────┬─────────┘
         │                                                                  │
         └─────────────────────────────────┬────────────────────────────────┘
                                           ▼
                                📄 PDF TEXT EXTRACTION (pypdf)
                                 Page Metadata & Table Preservation
                                           │
                                           ▼
                                ✂ RECURSIVE TEXT SPLITTER
                                 1000 Chars / 150 Overlap
                                           │
                                           ▼
                                🧠 OPENAI EMBEDDINGS
                                 text-embedding-3-small
                                           │
                                           ▼
                                🗄 CHROMADB VECTOR STORE
                                 Single Persistent Collection
                                           │
                         ┌─────────────────┴─────────────────┐
                         ▼                                   ▼
                🖥 STREAMLIT UI                     ⚡ FASTAPI BACKEND
                Interactive Q&A                     REST Endpoints (/ingest, /ask, /stats)
                         │                                   │
                         └─────────────────┬─────────────────┘
                                           ▼
                                🔍 TOP-K SIMILARITY RETRIEVAL (K=5)
                                 Cross-Document Context Assembly
                                           │
                                           ▼
                                🤖 LLM GENERATION (GPT-4o, Temp 0.1)
                                 Strict Grounding & Honest Refusal
                                           │
                                           ▼
                                📊 GROUNDED ANSWER + SOURCE CITATIONS + EVIDENCE
```

---

## 8. RAG Pipeline

```text
PDF Document Upload
      │
      ▼
PDF Text Extraction (pypdf)
      │
      ▼
Page + Source Metadata Tagging
      │
      ▼
Recursive Character Text Splitter (chunk_size=1000, chunk_overlap=150)
      │
      ▼
OpenAI Embeddings (text-embedding-3-small)
      │
      ▼
ChromaDB Persistent Collection (./chroma_db/)
      │
      ▼
User Question
      │
      ▼
Question Embedding
      │
      ▼
Top-K Similarity Retrieval (K=5)
      │
      ▼
Relevant Document Chunks (Cross-Document Context)
      │
      ▼
GPT-4o LLM (temperature=0.1)
      │
      ▼
Grounded Answer Generation
      │
      ▼
Source File + Page Number Citations
```

---

## 9. Technology Stack
- **Language**: Python 3.10+
- **PDF Extraction**: `pypdf`
- **Chunking**: `RecursiveCharacterTextSplitter` (`langchain-text-splitters`)
- **Embeddings**: OpenAI `text-embedding-3-small`
- **Vector Database**: ChromaDB persistent vector database (`chromadb`)
- **LLM**: OpenAI `GPT-4o` (temperature: `0.1`)
- **UI Framework**: Streamlit (`streamlit`)
- **REST API**: FastAPI + Uvicorn (`fastapi`, `uvicorn`)
- **Environment Management**: `python-dotenv`

---

## 10. Project Structure

```text
supplypilot/
│
├── app.py                      # Streamlit UI dashboard
├── ingest.py                   # Ingestion pipeline & ChromaDB manager
├── rag.py                      # RAG retrieval & GPT-4o grounded answer engine
├── config.py                   # Centralized configuration settings
├── test_rag_pipeline.py        # Automated test verification suite
│
├── api/
│   └── main.py                 # FastAPI backend REST application
│
├── data/
│   ├── Meridian_Procurement_Policy_Handbook_v4.2.pdf
│   └── Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf
│
├── chroma_db/                  # Persistent vector database directory
├── screenshots/                # Application screenshots
├── .env.example                # Environment configuration template
├── .gitignore                  # Git exclusion rules
├── requirements.txt            # Python dependencies
├── README.md                   # Project documentation & evaluation results
└── LICENSE                     # MIT License
```

---

## 11. Setup

### Prerequisites
- Python 3.10 or higher installed.
- Git installed.
- An OpenAI API Key with access to `text-embedding-3-small` and `gpt-4o`.

### Installation Steps
```bash
# 1. Clone the repository
git clone https://github.com/your-username/SupplyPilot.git
cd SupplyPilot

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

---

## 12. Environment Variables

Create a `.env` file in the root directory:

```bash
cp .env.example .env
```

Edit `.env` to supply your OpenAI API key:

```env
OPENAI_API_KEY=sk-proj-your-openai-api-key-here
```

*Note: The API key is read strictly from environment variables via `python-dotenv` and is never committed to Git.*

---

## 13. Running the Application

### Option A: Streamlit UI (Primary Interface)
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: FastAPI Backend
```bash
uvicorn api.main:app --reload --port 8000
```
Access interactive API documentation (Swagger UI) at `http://localhost:8000/docs`.

---

## 14. Document Ingestion
Document ingestion is handled by `ingest.py`:
1. PDFs are read page-by-page using `pypdf`.
2. Each page preserves page metadata (1-indexed page number) and source document name.
3. Content is split using `RecursiveCharacterTextSplitter`.
4. SHA-256 hashes prevent duplicate document insertion.
5. Embeddings are generated using `text-embedding-3-small` and inserted into persistent ChromaDB.

---

## 15. Retrieval
The retrieval process uses ChromaDB cosine/euclidean vector similarity search:
- Query is converted into a vector via `text-embedding-3-small`.
- Top `K=5` most relevant chunks are retrieved across all indexed documents.
- Chunks retain metadata tags (`source`, `page`, `distance`) which are passed to the context builder.

---

## 16. Cross-Document RAG
Crucial for supply chain queries where data lives in the Q1 Review and rules live in the Procurement Policy.
Because both documents are indexed into the **same collection**, Top-K retrieval pulls chunks from both sources simultaneously, allowing GPT-4o to synthesize answers (e.g. evaluating Kaveri Metals' 88.1% OTD against Policy Clause 6.1).

---

## 17. Persistence
The ChromaDB client is initialized with disk persistence at `./chroma_db/`. Once documents are indexed, closing or restarting the application does not wipe the database. Upon startup, SupplyPilot reloads existing collection stats dynamically without requiring re-upload.

---

## 18. Honest Refusal
SupplyPilot uses a strict system prompt:
> *"If the answer is not supported by the supplied context, say: 'The information is not available in the uploaded documents.'"*

When asked ungrounded questions (such as executive salaries), GPT-4o produces the exact refusal string rather than hallucinating external facts.

---

## 19. Screenshots
*(Include application screenshots in `./screenshots/`)*
- `01_dashboard.png`: Streamlit header, dynamic metric cards, and sidebar.
- `02_q5_cross_doc.png`: Cross-document answer for Q5 showing Kaveri Metals policy clauses.
- `03_q10_refusal.png`: Honest refusal response for Trap Question 10.
- `04_fastapi_swagger.png`: FastAPI Swagger UI endpoints at `/docs`.

---

## 20. All 10 Test Questions & Verified Application Answers

All 10 required evaluation questions were tested directly through SupplyPilot. Below are the actual verified answers and citations generated by the application.

---

### Q1: Spend & On-Time Delivery
**Question:**
> Which supplier had the highest spend in Q1, and what was its on-time delivery percentage?

**Application Answer:**
> **Shenzhen Rui Electronics** had the highest spend in Q1 at **₹21.9 crore**, with an on-time delivery rate of **79.5%**.

**Sources:**
- 📄 `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf` (Page 1)

---

### Q2: Line Stoppages & Causes
**Question:**
> How many line stoppages happened in Q1, what was the total downtime, and what caused them?

**Application Answer:**
> In Q1 FY 2025-26, there were **7 line-stoppage events**, resulting in a total downtime of **41 hours** and an estimated production loss of ₹1.9 crore.
> 
> **Causes:**
> 1. **Microcontroller shortages (Shenzhen Rui Electronics)**: 4 events totaling 22 hours (causes included vessel roll-over at Shenzhen, a 9-day customs hold at Nhava Sheva, partial shipment, and supplier allocation shortfall).
> 2. **PCB quality rejections (Trident Circuit Boards)**: 2 events totaling 11 hours (due to printed circuit board lots rejected at incoming inspection).
> 3. **Transport disruption**: 1 event totaling 5 hours (caused by a transporter strike on the Coimbatore–Pune corridor).

**Sources:**
- 📄 `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf` (Page 1, Page 2)

---

### Q3: Purchase Order Approval Authority
**Question:**
> What is the approval authority for a purchase order worth ₹1.4 crore?

**Application Answer:**
> The approving authority for a purchase order worth ₹1.4 crore is the **Chief Operating Officer (COO)**. According to Section 3 of the Procurement Policy Handbook, purchase orders valued above ₹1 crore and up to ₹5 crore fall under the COO's approval authority.

**Sources:**
- 📄 `Meridian_Procurement_Policy_Handbook_v4.2.pdf` (Page 1)

---

### Q4: Supplier Classification Categories
**Question:**
> What are the four supplier classification categories, and what qualifies a supplier as Critical?

**Application Answer:**
> The four supplier classification categories are:
> 1. **Critical**
> 2. **Strategic**
> 3. **Standard**
> 4. **Tail**
> 
> **Qualifying criteria for Critical classification** (meeting any single criterion is sufficient):
> - Being a **Single-source** for any part; OR
> - Annual spend **above ₹10 crore**; OR
> - Supplying a **safety-related component**.

**Sources:**
- 📄 `Meridian_Procurement_Policy_Handbook_v4.2.pdf` (Page 1)

---

### Q5: Kaveri Metals Performance & Policy Clauses (Cross-Document)
**Question:**
> Kaveri Metals recorded 88.1% on-time delivery and 1,150 defects per million in Q1. Which policy clauses does this trigger, and what exactly must the buyer do?

**Application Answer:**
> Kaveri Metals' Q1 performance metrics trigger two specific policy clauses under Section 6 of the Procurement Policy Handbook:
> 
> 1. **Clause 6.1 (On-time delivery below 90% in any quarter)**:
>    - **Performance**: Kaveri recorded 88.1% OTD.
>    - **Required Buyer Action**: A written warning must be issued to the supplier within 10 working days of quarter close, and the supplier must be placed on a weekly delivery review call until performance recovers above 90% for one full quarter.
> 
> 2. **Clause 6.3 (Defect rate above 500 parts per million in any quarter)**:
>    - **Performance**: Kaveri recorded 1,150 PPM.
>    - **Required Buyer Action**: The supplier must bear the cost of rework at a standard recovery rate of ₹120 per affected unit, and 100% incoming inspection must be imposed at the supplier's cost until three consecutive lots are accepted without defect.

**Sources:**
- 📄 `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf` (Page 1, Page 2)
- 📄 `Meridian_Procurement_Policy_Handbook_v4.2.pdf` (Page 2)

---

### Q6: Single-Source Microcontroller Policy (Cross-Document)
**Question:**
> The microcontroller supplier is single-source. What does the sourcing policy require in this situation, and what is the company already doing about it?

**Application Answer:**
> **Policy Requirement**:
> Under Clause 7.1 (Dual sourcing) of the Procurement Policy Handbook, every part supplied by a supplier classified as Critical (such as single-source microcontrollers) must have a qualified second source in place within **12 months** of that classification being assigned, with progress reported monthly to the Management Committee.
> 
> **Actions Company is Taking**:
> According to Section 8 of the Q1 Supply Chain Review:
> 1. The company is actively qualifying **Anh Long Semiconductors (Hai Phong, Vietnam)** as the second source for microcontrollers, with a target completion date of **30 September 2025** (Category Manager — Electronics).
> 2. As an interim risk mitigation measure, the company is shifting **30% of Shenzhen microcontroller volume to air freight** on a planned basis until dual sourcing is live (target date: 15 August 2025, Head of Logistics).

**Sources:**
- 📄 `Meridian_Procurement_Policy_Handbook_v4.2.pdf` (Page 2)
- 📄 `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf` (Page 1, Page 3)

---

### Q7: Imported Microcontroller Safety Stock (Cross-Document)
**Question:**
> Microcontrollers are imported with a 46-day lead time. Using the safety-stock policy, how many days of stock should be held for this part?

**Application Answer:**
> **30 days of safety stock** should be held for microcontrollers.
> 
> **Calculation & Policy Rationale**:
> 1. **Formula Calculation**: Under Section 8 of the Procurement Policy Handbook, calculated safety stock = `Lead Time (days) × 0.25` = `46 × 0.25 = 11.5 days`.
> 2. **Minimum Floor Rule**: The part is **Imported** and supplied by a **Critical supplier** (Shenzhen Rui Electronics, single-source), which sets a mandatory minimum floor of **30 days**.
> 3. **Rule Application**: The policy mandates that where the calculated value (11.5 days) and the minimum floor (30 days) differ, the **higher value applies**. Therefore, the required safety stock level is **30 days**.

**Sources:**
- 📄 `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf` (Page 1)
- 📄 `Meridian_Procurement_Policy_Handbook_v4.2.pdf` (Page 3)

---

### Q8: Trident Circuit Boards Defect Consequence (Cross-Document)
**Question:**
> Trident Circuit Boards had a defect rate of 640 parts per million. What is the cost consequence under the policy?

**Application Answer:**
> Trident Circuit Boards' defect rate of 640 PPM exceeds the 500 PPM threshold, triggering **Clause 6.3** of the Procurement Policy Handbook:
> 
> **Cost Consequences & Actions**:
> 1. **Rework Recovery Rate**: The supplier bears the cost of rework at a standard recovery rate of **₹120 per affected unit**.
> 2. **100% Incoming Inspection**: 100% incoming inspection is imposed at the **supplier's cost** until three consecutive lots are accepted without defect.

**Sources:**
- 📄 `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf` (Page 1, Page 2)
- 📄 `Meridian_Procurement_Policy_Handbook_v4.2.pdf` (Page 2)

---

### Q9: OTD Rating Bands & Escalation Path (Cross-Document)
**Question:**
> Which suppliers would fall below the B rating band on on-time delivery alone, and what is the escalation path for them?

**Application Answer:**
> **Supplier Band Standing on OTD Alone**:
> According to Section 5 of the Procurement Policy Handbook:
> - On-time delivery below 90% caps a supplier's max rating at Band B.
> - On-time delivery below 75% caps a supplier's max rating below Band B.
> 
> Looking at Q1 performance:
> - Sunrise Connectors (98.2%), Nexa Polymers (96.4%), and Baltic Wire (93.7%) are above 90% (Band A).
> - Kaveri Metals (88.1%), Trident Circuit Boards (84.6%), and Shenzhen Rui Electronics (79.5%) are all below 90% but **above 75%**.
> Therefore, **no supplier falls below the B rating band on on-time delivery alone** in Q1, as all suppliers maintained OTD above 75%.
> 
> **Escalation Path (Section 10 Escalation Matrix)**:
> For delivery slippage and issues, the policy defines a 4-level escalation matrix:
> - **Level 1 (Buyer, 24-hr response)**: Delivery slippage up to 3 days.
> - **Level 2 (Category Manager, 48-hr response)**: Delivery slippage beyond 3 days or rejected lot.
> - **Level 3 (Head of Procurement, 72-hr response)**: Risk of line stoppage within 7 days.
> - **Level 4 (COO, 5 working days response)**: Actual line stoppage or supplier insolvency.

**Sources:**
- 📄 `Meridian_Procurement_Policy_Handbook_v4.2.pdf` (Page 2, Page 3)
- 📄 `Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf` (Page 1)

---

### Q10: Trap Question — Head of Procurement Salary
**Question:**
> What is the annual salary of the Head of Procurement?

**Application Answer:**
> **The information is not available in the uploaded documents.**

**Sources:**
- *None (Honest Refusal triggered)*

---

## 21. Evaluation / Verification
The pipeline is verified via `test_rag_pipeline.py`.
- **Extraction Test**: Successfully parses all pages of both Meridian PDFs.
- **Chunking Rationale**:
  *Chunk size: 1000 characters; Overlap: 150 characters.*
  *A 1000-character chunk provides enough context for policy clauses and supply-chain data, while 150-character overlap helps preserve context across chunk boundaries.*
- **Refusal Test**: Successfully enforces exact refusal message for ungrounded queries.

---

## 22. Limitations
- **PDF Scanned Images**: Extraction relies on `pypdf` text layer. Scanned images without OCR layers require pre-processing.
- **Complex Multi-nested Tables**: Raw PDF tables are flattened into structured text blocks.

---

## 23. Future Improvements
- **Hybrid Search**: Combine BM25 keyword matching with dense vector embeddings.
- **Reranking**: Integrate Cohere or BGE reranker to optimize chunk selection prior to LLM generation.
- **Table Extraction**: Add specialized table parsing using `pdfplumber` or `camelot`.

---

## 24. FastAPI Backend API

Start the REST API server:
```bash
uvicorn api.main:app --reload --port 8000
```

### Endpoints

#### 1. `POST /ingest`
Upload PDF files for dynamic ingestion.
```json
// Response
{
  "files": 2,
  "chunks": 26,
  "new_chunks_added": 26
}
```

#### 2. `POST /ask`
Submit a question to the RAG engine.
```json
// Request
{
  "question": "Which supplier had the highest spend in Q1?",
  "top_k": 5
}

// Response
{
  "query": "Which supplier had the highest spend in Q1?",
  "answer": "Shenzhen Rui Electronics had the highest spend in Q1 at ₹21.9 crore...",
  "sources": [
    {
      "file": "Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf",
      "page": 1
    }
  ],
  "refused": false,
  "evidence_count": 5
}
```

#### 3. `GET /stats`
Retrieve Knowledge Base statistics.
```json
{
  "collection_name": "meridian_supply_chain",
  "total_chunks": 26,
  "document_count": 2,
  "indexed_documents": [
    "Meridian_Procurement_Policy_Handbook_v4.2.pdf",
    "Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf"
  ],
  "embedding_model": "text-embedding-3-small",
  "llm_model": "gpt-4o"
}
```

---

## 25. Demo Instructions (3-Minute Evaluation Walkthrough)

1. **Launch App**: Run `streamlit run app.py`.
2. **Show Knowledge Base**: In sidebar, click **"⚡ Index KB"**. Show dynamic document count (**2 Documents**) and dynamic chunk count (**26 Chunks**).
3. **Run Single-Document Question**: Click **Q1** or **Q3** from suggested questions. Show the grounded answer and file/page citation badges.
4. **Run Cross-Document Question**: Click **Q5** (Kaveri Metals) or **Q7** (Safety Stock calculation). Point out how SupplyPilot synthesizes metrics from Q1 Review with rules from Procurement Policy Handbook.
5. **Demonstrate Honest Refusal**: Click **Q10** (Head of Procurement Salary). Verify exact refusal response: *"The information is not available in the uploaded documents."*
6. **Show Evidence View**: Expand **▸ View Evidence** to demonstrate full auditability with source pages and similarity scores.

---
*SupplyPilot — Designed & Built for Meridian Components Supply Chain Intelligence.*
