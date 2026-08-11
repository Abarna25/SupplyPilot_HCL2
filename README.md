# SupplyPilot — AI-Powered Supply Chain Document Intelligence

> **Navigate Supply Chain Decisions with AI**  
> *Ask. Retrieve. Verify. Decide.*

---

## 1. Project Overview
**SupplyPilot** is an enterprise-grade supply-chain document intelligence platform designed to parse, index, and query complex operational and procurement documentation for Meridian Components Pvt. Ltd. By combining localized semantic search with generative capabilities powered by the native **Gemini API** (`models/gemini-embedding-001` and `gemini-flash-latest`) and **ChromaDB**, SupplyPilot retrieves facts directly from policy handbooks and quarterly performance reviews, synthesizing grounded answers complete with direct citations and verification evidence.

---

## 2. Problem Statement
Supply chain leadership operates with fragmented knowledge split across two distinct domain sources:
1. **Governance & Policy Rules**: Static handbooks specifying approval tiers, penalty structures, dual-sourcing mandates, safety stock rules, and escalation paths.
2. **Operational Performance Data**: Quarterly reviews documenting supplier spend, on-time delivery percentages, defect PPM, line stoppages, lead times, and active risks.

When operational disruptions occur (e.g., delivery delays or defect spikes), buyers must manually locate the relevant performance data, cross-reference policy handbooks to determine mandatory enforcement clauses, and compute required penalties or mitigation actions. Standard keyword search tools fail to link figures to their semantic context, while vanilla LLMs suffer from hallucinations.

---

## 3. Solution
SupplyPilot implements a localized, low-latency **Retrieval-Augmented Generation (RAG)** pipeline. Key policy handbooks and operational performance reviews are parsed, chunked recursively, and embedded into a persistent local vector store. 

When a query is made, SupplyPilot retrieves the most semantically relevant text fragments across single or multiple documents, feeds them to Gemini Flash as grounding context, and presents a structured answer with exact page numbers, source files, and similarity scores.

---

## 4. Key Features
- **Direct Streamlit RAG (Default)**: Executes RAG logic locally to minimize latency, completely bypassing external microservices unless explicitly enabled.
- **FastAPI Backend (Optional Bonus)**: Standalone FastAPI server (`/ingest`, `/ask`, `/stats`) for remote indexing and query APIs.
- **Dynamic KPI Dashboard**: Real-time metric cards for Documents Indexed, Chunks Stored, Vector Database (`ChromaDB`), and AI Engine (`Gemini AI`).
- **Strict Grounding & Refusal Engine**: Visual indicators and prompt rules enforcing strict context-only answers. Unanswerable queries (e.g. executive salary trap questions) trigger exact refusal: *"The information is not available in the uploaded documents."*
- **Source Evidence Panel**: Structured card grids detailing source filenames, exact page numbers, similarity distances, and collapsable excerpt blockquotes.
- **Cross-Document Synthesis**: Single-pass vector retrieval across both Procurement Policy Handbooks and Q1 Performance Reviews.
- **Duplicate Document Skipping**: SHA-256 file content hashing prevents re-indexing identical files.
- **Interactive Session Chat History**: Persists query-response pairs during current session with a 1-click "Clear Chat" option.

---

## 5. Why RAG?
RAG ensures **explainability**, **factuality**, and **auditability**. In supply chain management, an ungrounded policy interpretation or hallucinated defect penalty can trigger invalid debit notes or supplier legal disputes. By forcing the LLM to restrict its output strictly to retrieved document segments and refusing queries unsupported by context, SupplyPilot eliminates hallucinations. Every claim made by the assistant is immediately verifiable.

---

## 6. Architecture
The system architecture flows as follows:

```text
PDF Document Upload (Meridian Procurement Policy + Q1 Review)
    ↓
pypdf (page-by-page text extraction & metadata preservation)
    ↓
Recursive Character Splitting (1000 chunk size / 150 overlap)
    ↓
Gemini Embeddings (models/gemini-embedding-001)
    ↓
ChromaDB (Persistent cosine-similarity store at ./chroma_db/)
    ↓
User Question
    ↓
Gemini Embedding (models/gemini-embedding-001)
    ↓
Top-K Retrieval (K=5 chunks across unified collection)
    ↓
Gemini Flash (gemini-flash-latest)
    ↓
Grounded Answer + Source File & Page Number Evidence
```

> [!NOTE]
> The original assignment prompt mentions OpenAI models. This implementation uses Gemini API (`models/gemini-embedding-001` and `gemini-flash-latest`) as the model substitution requested for this project.

---

## 7. Technology Stack
- **Frontend**: Streamlit (`streamlit`)
- **Backend / API**: FastAPI & Uvicorn (`fastapi`, `uvicorn`)
- **Orchestration**: Google Generative AI Python SDK (`google-generativeai`)
- **Vector Database**: ChromaDB persistent store (`chromadb`)
- **Document Parsing**: PyPDF (`pypdf`)
- **Text Chunking**: LangChain Text Splitters (`langchain-text-splitters`)
- **Environment Management**: Python Dotenv (`python-dotenv`)

---

## 8. Dataset
The project primary knowledge base consists of the two provided Meridian Components PDFs located in `data/`:
1. **`Meridian_Procurement_Policy_Handbook_v4.2.pdf`**: Procurement governance rules, approval authorities, supplier classification, penalty clauses, dual sourcing, safety stock formulas, and escalation matrix.
2. **`Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf`**: Q1 FY2025-26 operational data, supplier scorecards, spend, OTD %, defect PPM, freight performance, line stoppages, and Q2 planned actions.

---

## 9. Environment Setup

### Prerequisites
- Python 3.10 or higher
- A valid **Gemini API Key**

### Installation
```bash
# 1. Clone repository
git clone https://github.com/Abarna25/SupplyPilot_HCL2.git
cd SupplyPilot_HCL2

# 2. Install dependencies
pip install -r requirements.txt
```

### Environment Variables
Create a `.env` file in the project root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here
```

*(Note: The API key is loaded via `python-dotenv` and ignored by `.gitignore` to keep credentials secure.)*

---

## 10. Running the Application

### Option A: Streamlit UI (Primary Interface)
```bash
streamlit run app.py
```
Open browser at **`http://localhost:8501`**.

### Option B: FastAPI Backend Server
```bash
uvicorn api.main:app --reload --port 8000
```
Access interactive API documentation (Swagger UI) at **`http://localhost:8000/docs`**.

---

## 11. All 10 Test Questions & Verified Application Answers

All 10 evaluation questions were tested directly through SupplyPilot powered by Gemini API.

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

## 12. FastAPI API Endpoints

### `POST /ingest`
Upload PDFs for vector database indexing.
```json
{
  "files": 2,
  "chunks": 26,
  "new_chunks_added": 26
}
```

### `POST /ask`
Query the RAG engine with a question.
```json
{
  "question": "Which supplier had the highest spend in Q1?",
  "top_k": 5
}
```

### `GET /stats`
Get dynamic Knowledge Base statistics.
```json
{
  "collection_name": "meridian_supply_chain",
  "total_chunks": 26,
  "document_count": 2,
  "indexed_documents": [
    "Meridian_Procurement_Policy_Handbook_v4.2.pdf",
    "Meridian_Supply_Chain_Review_Q1_FY2025-26.pdf"
  ],
  "embedding_model": "models/gemini-embedding-001",
  "llm_model": "gemini-flash-latest"
}
```

---

## 13. Evaluation & Verification

- **Extraction Test**: Parses 100% of text and tables across both PDFs.
- **Chunking Rationale**:
  *Chunk size: 1000 characters; Overlap: 150 characters.*
  *A 1000-character chunk provides enough context for policy clauses and supply-chain data, while 150-character overlap helps preserve context across chunk boundaries.*
- **Refusal Test**: Successfully enforces exact refusal string for ungrounded queries.

---
*SupplyPilot — Designed & Built for Meridian Components Supply Chain Intelligence.*
