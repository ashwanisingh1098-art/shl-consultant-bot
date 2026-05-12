import json
import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

# --- Global Resource Management ---
_embeddings = None
_db = None
_retriever = None

def load_resources():
    global _embeddings, _db, _retriever
    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}
        )
    if _db is None:
        _db = FAISS.load_local(
            "faiss_index",
            _embeddings,
            allow_dangerous_deserialization=True
        )
    if _retriever is None:
        _retriever = _db.as_retriever(search_type="mmr", search_kwargs={"k": 3})

load_resources()

def advanced_reranker_and_formatter(docs):
    pure_docs = []
    for doc in docs:
        name = doc.metadata.get("name", "").lower()
        impurities = ["report", "profile", "narrative", "feedback", "development"]
        # Standard filter to ensure we suggest base assessments first
        link = doc.metadata.get("link", "Contact SHL for Link")
        pure_docs.append(
            f"ASSESSMENT: {doc.metadata.get('name')}\n"
            f"CONTENT: {doc.page_content}\n"
            f"URL: <{link}>\n---"
        )
    return "\n".join(pure_docs) if pure_docs else "No matches found."

def get_consultant_response(question, history):
    docs = _retriever.invoke(question)
    context = advanced_reranker_and_formatter(docs)

    # COMPANY SPECIFIC PROMPT
    # We define the prompt text first
    prompt_text = f"""
You are an expert SHL Assessment Consultant. 
Follow the consultation protocol strictly based on the history provided.

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION:
{question}

CONSULTATION RULES:
1. STEP 1: If role details (Seniority, Purpose, Tech Stack) are missing in HISTORY, ask clarifying questions. 'recommendations' must be [].
2. STEP 2: If details are present, recommend 1-5 SHL assessments from CONTEXT.
3. STEP 3: Handle refinement/changes to the stack.
4. STEP 4: Explain differences if asked.
5. STEP 5: Set end_of_conversation=true only when user is satisfied.

OUTPUT FORMAT (STRICT JSON):
{{
  "reply": "<conversational text>",
  "recommendations": [{{ "name": "...", "url": "...", "test_type": "..." }}],
  "end_of_conversation": false
}}
"""

    try:
        # NOTICE: We removed the 'f' from the messages list to avoid the Dict error
        response = client.chat.completions.create(
            model="arcee-ai/trinity-large-thinking:free", 
            messages=[
                {"role": "system", "content": "You are a strict JSON-only SHL consultant."},
                {"role": "user", "content": prompt_text}
            ],
            stream=True,
            max_tokens=4000
        )

        for chunk in response:
            if hasattr(chunk.choices[0], 'delta') and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content

    except Exception as e:
        # Final safety for the error message
        error_json = json.dumps({"reply": f"Error: {str(e)}", "recommendations": []})
        yield error_json

     