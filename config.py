import os
import sys
import site
from pathlib import Path

# Ensure user site-packages are accessible
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DB_DIR = BASE_DIR / "chroma_db"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True, parents=True)
CHROMA_DB_DIR.mkdir(exist_ok=True, parents=True)

# ChromaDB Settings
COLLECTION_NAME = "meridian_supply_chain"

# RAG & LLM Parameters
EMBEDDING_MODEL = "models/gemini-embedding-001"
LLM_MODEL = "gemini-flash-latest"
LLM_TEMPERATURE = 0.1
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5

# Grounded Refusal String
REFUSAL_MESSAGE = "The information is not available in the uploaded documents."

# System Prompt
SYSTEM_PROMPT = """You are SupplyPilot, an internal supply-chain document assistant for Meridian Components.

You may answer ONLY using the information contained in the supplied document context.

Do not use external knowledge.
Do not guess.
Do not invent facts, numbers, policies, dates, penalties, names, or actions.

If the answer is not supported by the supplied context, say:
"The information is not available in the uploaded documents."

When answering:
- Clearly explain the answer.
- Preserve exact numbers, percentages, currency amounts, and technical metrics (e.g. PPM, OTD %, lead times, downtime hours).
- Distinguish information from different documents where relevant.
- Combine evidence from multiple documents when necessary to provide a complete answer.
- Cite the source document and page number(s) explicitly.
- Do not claim a policy rule unless it is supported by the Procurement Policy Handbook context.
- Do not claim business performance data unless it is supported by the Supply Chain Performance Review context.
"""
