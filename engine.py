import json
import os
from dotenv import load_dotenv
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- 1. GLOBAL INITIALIZATION ---
# These load once on startup. This solves your 30-second delay.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 3})

# Initialize the new Google GenAI Client
# Ensure your .env has GOOGLE_API_KEY
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

# --- 2. RERANKER & FORMATTER ---
def advanced_reranker_and_formatter(docs):
    pure_docs = []
    for doc in docs:
        name = doc.metadata.get("name", "").lower()
        # Filter out "report" specific items to keep context clean
        impurities = ["report", "profile", "narrative", "feedback", "development"]
        if any(x in name for x in impurities):
            continue
        link = doc.metadata.get("link", "Contact SHL for Link")
        formatted_entry = f"ASSESSMENT: {doc.metadata.get('name')}\nCONTENT: {doc.page_content}\nURL: <{link}>\n---"
        pure_docs.append(formatted_entry)
    return "\n".join(pure_docs) if pure_docs else "No matches found."

# --- 3. THE MAIN FUNCTION ---
def get_consultant_response(question, history):
    # Step 1: Retrieve context from FAISS
    docs = retriever.invoke(question)
    context = advanced_reranker_and_formatter(docs)

    # Step 2: Prepare the Prompt
    # We build the string manually because we aren't using the LangChain 'pipe' anymore
    prompt = f"""
 
You are a Senior SHL Solutions Consultant. You don't just find tests; you design assessment strategies.

### CONSULTANT GUIDELINES:
1. **Proactive Bundling:** If a user mentions a domain (Sales, Leadership, Tech), immediately recommend a "Standard Industry Stack" from the CONTEXT. Do not wait for seniority/goal if you can make an educated guess.
2. **Instrument vs. Report:** Understand that OPQ32r is an "Instrument" (the test) and reports (like MQ Sales or Leadership) are "Outputs." Explain this to the user to show expertise.
3. **The "Always-On" Recommendation:** Never return an empty "recommendations" list if there are relevant products in the CONTEXT. Provide the "Best Fit" now and refine later.
4. **Consistency:** If you suggest a 5-product stack, keep those same 5 products in the list throughout the conversation unless the user asks to change them.

### CONTEXT:
{context}

### CONVERSATION HISTORY:
{history}

### USER QUESTION:
{question}

### RESPONSE FORMAT (STRICT JSON):
{{
  "reply": "<A consultant-style response. Explain the 'Why' behind the stack. Distinguish between taking the test and getting the report. Be authoritative.>",
  "recommendations": [
    {{
      "name": "<Exact product name from context>",
      "url": "<Exact URL from context>",
      "test_type": "<e.g., Personality, Cognitive, Behavior>"
    }}
  ],
  "end_of_conversation": false
}}

### CRITICAL RULES:
- If you mention a product in the 'reply', it MUST be in the 'recommendations' array.
- Do NOT ask more than one clarifying question per turn. Provide value first, then ask.
- 'end_of_conversation' is ONLY true when the user confirms satisfaction (e.g., "Perfect", "Thanks").
 
"""

    # Step 3: Call the model using the new SDK syntax
    # If gemini-3.1-pro-preview gives a 404, switch to "gemini-2.0-flash"
    try:
        response = client.models.generate_content(
            model="gemini-3.1-flash-lite", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return json.dumps({
            "reply": f"Model Error: {str(e)}",
            "recommendations": [],
            "end_of_conversation": False
        })