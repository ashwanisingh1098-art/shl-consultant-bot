from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
from engine import get_consultant_response

app = FastAPI()

# ----------------------------
# SCHEMAS
# ----------------------------
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]


# ----------------------------
# HEALTH CHECK (Render uses this)
# ----------------------------
@app.get("/")
def root():
    return {"status": "RUNNING"}

@app.get("/health")
def health():
    return {"status": "ok"}


# ----------------------------
# CHAT ENDPOINT (STREAMLIT CALLS THIS)
# ----------------------------
@app.post("/chat")
def chat(request: ChatRequest):

    # Build history string
    history_str = "\n".join(
        [f"{msg.role}: {msg.content}" for msg in request.messages[:-1]]
    )

    # Current user query
    current_q = request.messages[-1].content

    # Call AI engine
    raw_ai_output = get_consultant_response(current_q, history_str)

    # Safe JSON parsing
    try:
        response_data = json.loads(raw_ai_output)
    except Exception:
        response_data = {
            "reply": raw_ai_output,
            "recommendations": [],
            "end_of_conversation": False
        }

    return response_data