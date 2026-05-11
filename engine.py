import json
import os
from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- 1. GLOBAL RESOURCE CONTAINERS ---
# We keep these empty at first to save RAM during startup
_embeddings = None
_db = None
_client = None

def get_resources():
    """Lazy loads memory-heavy models only when needed."""
    global _embeddings, _db, _client
    
    # Initialize the Google Client once
    if _client is None:
        _client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
    
    # Initialize the heavy RAG components
    if _embeddings is None:
        print("--- LAZY LOADING EMBEDDINGS & VECTOR DB ---")
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder="./model_cache"
        )
        _db = FAISS.load_local(
            "faiss_index", 
            _embeddings, 
            allow_dangerous_deserialization=True
        )
    
    return _client, _db

# --- 2. RERANKER & FORMATTER ---
def advanced_reranker_and_formatter(docs):
    pure_docs = []
    for doc in docs:
        name = doc.metadata.get("name", "").lower()
        impurities = ["report", "profile", "narrative", "feedback", "development"]
        if any(x in name for x in impurities):
            continue
        link = doc.metadata.get("link", "Contact SHL for Link")
        formatted_entry = f"ASSESSMENT: {doc.metadata.get('name')}\nCONTENT: {doc.page_content}\nURL: <{link}>\n---"
        pure_docs.append(formatted_entry)
    return "\n".join(pure_docs) if pure_docs else "No matches found."

# --- 3. THE MAIN FUNCTION ---
def get_consultant_response(question, history):
    # Step 0: Get resources (Loads them now if it's the first run)
    client, db = get_resources()
    
    # Step 1: Retrieve context from FAISS
    # Using k=2 instead of 3 to save extra RAM during processing
    retriever = db.as_retriever(search_kwargs={"k": 2})
    docs = retriever.invoke(question)
    context = advanced_reranker_and_formatter(docs)

    # Step 2: Prepare the Prompt
    prompt = f"""
You are a Senior SHL Solutions Consultant. You don't just find tests; you design assessment strategies.

### CONSULTANT GUIDELINES:
1. **Proactive Bundling:** Recommend a "Standard Industry Stack" immediately if possible.
2. **Instrument vs. Report:** Distinguish between OPQ32r (test) and reports (outputs).
3. **The "Always-On" Recommendation:** Never return an empty list.
4. **Consistency:** Keep the same stack unless changed.

### CONTEXT:
{context}

### CONVERSATION HISTORY:
{history}

### USER QUESTION:
{question}

### RESPONSE FORMAT (STRICT JSON):
{{
  "reply": "<Explain strategy, distinguish test vs report. Be authoritative.>",
  "recommendations": [
    {{
      "name": "<Exact product name>",
      "url": "<Exact URL>",
      "test_type": "<Personality, Cognitive, etc.>"
    }}
  ],
  "end_of_conversation": false
}}

### CRITICAL RULES:
- Mentioned products MUST be in recommendations array.
- Max one clarifying question.
- 'end_of_conversation' only true on user confirmation.
"""

    # Step 3: Call the model
    try:
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return json.dumps({
            "reply": f"Model Error: {str(e)}",
            "recommendations": [],
            "end_of_conversation": False
        })