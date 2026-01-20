import os
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# CORS (Android/başka clientlar için sorun çıkarmaz)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent"

SYSTEM_PROMPT = "Sen gezgin ruhsar uygulamasının gezi asistanısın..."

# ✅ Render uyandırma / health-check
@app.get("/")
def root():
    return {"ok": True, "service": "gezginruhsar-backend", "model": MODEL_NAME}

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str

def call_gemini(payload: dict) -> requests.Response:
    # connect: 10sn, read: 90sn (Render free + Gemini gecikmelerine uygun)
    return requests.post(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        json=payload,
        timeout=(10, 90),
    )

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY missing (env var)")

    msg = (req.message or "").strip()
    if not msg:
        raise HTTPException(status_code=400, detail="message is empty")

    payload = {
        "contents": [
            {"role": "user", "parts": [{"text": f"{SYSTEM_PROMPT}\n\nKullanıcı: {msg}"}]}
        ]
    }

    # ✅ 1 kez retry: özellikle timeout/connection reset durumlarında işe yarıyor
    try:
        r = call_gemini(payload)
    except requests.RequestException:
        try:
            r = call_gemini(payload)
        except requests.RequestException as e:
            raise HTTPException(status_code=504, detail=f"Gemini request failed: {str(e)}")

    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Gemini HTTP {r.status_code}: {r.text}")

    try:
        data = r.json()
        text = data["candidates"][0]["content"]["parts"][0]["text"]
    except Exception:
        raise HTTPException(status_code=500, detail=f"Parse error. Raw JSON: {r.text}")

    return ChatResponse(reply=text)
