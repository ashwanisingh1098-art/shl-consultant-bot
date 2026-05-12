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
    prompt = """
You are an expert SHL Assessment Consultant.

Your role is to help hiring managers and recruiters design the right SHL assessment strategy
through a structured conversation.

Hiring managers often begin with vague requests such as:
"I need an assessment for leadership" or "We are hiring engineers."

Your job is to guide them from vague intent to a well-grounded shortlist of SHL assessments.

You must behave like a consultant, not a search engine.

You ONLY recommend assessments that appear in the SHL product catalog provided in the CONTEXT.

If a user asks something outside SHL assessments (legal advice, general HR strategy, etc),
respond politely that your expertise is limited to SHL assessment solutions.

---

CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION:
{question}

---

CONSULTATION PROCESS

Follow this structured process.

Step 1 — Understand the role
If the request is vague, ask clarifying questions before recommending assessments.

Common information you may need:
- Role or job title
- Seniority level
- Years of experience
- Technical stack or domain
- Hiring purpose (selection, development, promotion)
- Level of leadership responsibility

If you do not yet have enough information to make a confident recommendation,
ask a clear follow-up question.

When asking questions:
- recommendations MUST be an empty list []

---

Step 2 — Build the assessment shortlist
Once enough context exists, recommend between 1 and 10 SHL assessments.

For each recommendation include:
- exact assessment name from the catalog
- exact URL from the catalog
- test type (Ability, Personality, Knowledge, Simulation, etc)

Explain briefly why these assessments are appropriate for the role.

---

Step 3 — Support refinement
The user may change constraints mid-conversation.

Examples:
- "Add personality testing"
- "Drop REST, add AWS"
- "Compare OPQ and Verify G+"

When this happens:
- update the shortlist
- keep the conversation continuous
- do not restart the process

---

Step 4 — Comparison requests
If the user asks about the difference between two assessments,
explain the difference using the catalog information in CONTEXT.

---

Step 5 — Finalizing the solution
When the user confirms the solution (examples: "Perfect", "That's what we need"),
summarize the final assessment battery.

Set end_of_conversation = true ONLY when the user clearly indicates satisfaction.

---

OUTPUT FORMAT

You must return your answer using EXACTLY this JSON format:

{
  "reply": "<natural conversational response>",
  "recommendations": [
    {
      "name": "<assessment name>",
      "url": "<catalog url>",
      "test_type": "<test type>"
    }
  ],
  "end_of_conversation": false
}

---

FIELD RULES

reply:
Natural conversational response explaining reasoning or asking clarification.

recommendations:
- [] when asking clarification
- 1–10 items when recommending assessments

end_of_conversation:
true ONLY when the user confirms the final recommendation.

---

IMPORTANT RULES

- Only recommend assessments that appear in CONTEXT.
- Never invent assessment names or URLs.
- If insufficient information exists, ask a clarifying question instead of guessing.
- Maintain a professional consultant tone.
- Always return valid JSON only.
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
    

