from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import json
from engine import get_consultant_response

app = FastAPI()

# SCHEMA DEFINITIONS (Strictly following PDF)
class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/chat")
async def chat(request: ChatRequest):
    # 1. Transform history list into a single string to keep it stateless
    history_str = ""
    for msg in request.messages[:-1]: # All turns except the current one
        history_str += f"{msg.role}: {msg.content}\n"
    
    # 2. Get current question
    current_q = request.messages[-1].content
    
    # 3. Get AI Response
    raw_ai_output = get_consultant_response(current_q, history_str)
    
    # 4. Parse string output into JSON object for the response
    try:
        response_data = json.loads(raw_ai_output)
    except:
        # Fallback if LLM fails to output valid JSON
        response_data = {"reply": raw_ai_output, "recommendations": [], "end_of_conversation": False}
        
    return response_data