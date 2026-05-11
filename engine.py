import os
import json
from dotenv import load_dotenv
from google import genai
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# --- 1. GLOBAL INITIALIZATION (RAM Efficient) ---
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001", 
    google_api_key=os.getenv("GOOGLE_API_KEY")
)

# Ensure the folder name matches what you pushed to GitHub
# If you renamed it to 'faiss_index', change 'faiss_index_google' below to 'faiss_index'
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
retriever = db.as_retriever(search_kwargs={"k": 2})

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

def advanced_reranker_and_formatter(docs):
    pure_docs = []
    for doc in docs:
        name = doc.metadata.get("name", "").lower()
        impurities = ["report", "profile", "narrative", "feedback", "development"]
        if any(x in name for x in impurities): continue
        link = doc.metadata.get("link", "Contact SHL")
        # Included page_content so the LLM actually knows what the test is!
        pure_docs.append(f"ASSESSMENT: {doc.metadata.get('name')}\nDETAILS: {doc.page_content}\nURL: <{link}>\n---")
    return "\n".join(pure_docs)

def get_consultant_response(question, history):
    docs = retriever.invoke(question)
    context_text = advanced_reranker_and_formatter(docs)
    
    # Use f""" at the start so {question}, {history}, and {context_text} actually work!
    prompt = f"""
You are a Senior SHL Solutions Consultant. You design assessment strategies.

### CONSULTANT GUIDELINES:
1. **Proactive Bundling:** Recommend a "Standard Industry Stack" immediately.
2. **Instrument vs. Report:** Distinguish between OPQ32r (test) and reports (outputs).
3. **The "Always-On" Recommendation:** Never return an empty list.
4. **Consistency:** Keep the same stack unless changed.

### CONTEXT:
{context_text}

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

    try:
        # Use gemini-1.5-flash for maximum speed to stay under 30s
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt
        )
        return response.text
    except Exception as e:
        return json.dumps({
            "reply": f"Consultant Service is temporarily busy. Error: {str(e)}",
            "recommendations": [],
            "end_of_conversation": False
        })