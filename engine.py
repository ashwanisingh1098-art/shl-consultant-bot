import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

# ----------------------------
# OPENROUTER CLIENT
# ----------------------------
client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# ----------------------------
# LAZY LOADED RESOURCES
# ----------------------------
_embeddings = None
_db = None
_retriever = None


def load_resources():
    global _embeddings, _db, _retriever

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            cache_folder="./model_cache"
        )

    if _db is None:
        _db = FAISS.load_local(
            "faiss_index",
            _embeddings,
            allow_dangerous_deserialization=True
        )

    if _retriever is None:
        _retriever = _db.as_retriever(search_kwargs={"k": 3})


# ----------------------------
# FORMATTER
# ----------------------------
def advanced_reranker_and_formatter(docs):
    pure_docs = []

    for doc in docs:
        name = doc.metadata.get("name", "").lower()

        impurities = ["report", "profile", "narrative", "feedback", "development"]
        if any(x in name for x in impurities):
            continue

        link = doc.metadata.get("link", "Contact SHL for Link")

        pure_docs.append(
            f"ASSESSMENT: {doc.metadata.get('name')}\n"
            f"CONTENT: {doc.page_content}\n"
            f"URL: <{link}>\n---"
        )

    return "\n".join(pure_docs) if pure_docs else "No matches found."


# ----------------------------
# MAIN FUNCTION
# ----------------------------
def get_consultant_response(question, history):

    load_resources()

    # Step 1: FAISS retrieval
    docs = _retriever.invoke(question)
    context = advanced_reranker_and_formatter(docs)

    # Step 2: Prompt
    prompt = f"""
You are a Senior SHL Solutions Consultant. You design assessment strategies.

### CONTEXT:
{context}

### CONVERSATION HISTORY:
{history}

### USER QUESTION:
{question}

### RESPONSE FORMAT (STRICT JSON):
{{
  "reply": "<consultant explanation>",
  "recommendations": [
    {{
      "name": "<exact product name>",
      "url": "<exact URL>",
      "test_type": "<Personality/Cognitive/Behavior>"
    }}
  ],
  "end_of_conversation": false
}}

### RULES:
- Must include products from context if mentioned
- Be precise and consultative
- Max 1 clarifying question
"""

    # Step 3: OpenRouter call
    try:
        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-thinking:free",
            messages=[
                {
                    "role": "system",
                    "content": "You are a strict JSON-only SHL assessment consultant."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response.choices[0].message.content

    except Exception as e:
        return json.dumps({
            "reply": f"Model Error: {str(e)}",
            "recommendations": [],
            "end_of_conversation": False
        })
    

