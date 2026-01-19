import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# ✅ Burayı sabitle
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

SYSTEM_PROMPT = "Sen gezgin ruhsar uygulamasının gezi asistanısın..."

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing (env var)")

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT}\n\nKullanıcı: {req.message}"}]}
        ]
    }

    r = requests.post(f"{GEMINI_URL}?key={GEMINI_API_KEY}", json=payload, timeout=20)

    if r.status_code != 200:
        raise HTTPException(status_code=500, detail=f"Gemini HTTP {r.status_code}: {r.text}")

    data = r.json()
    try:
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Parse error. Raw JSON: {data}")

    return ChatResponse(reply=text)
