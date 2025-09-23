import os
import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from typing import Optional, Any, List
import uvicorn
from dotenv import load_dotenv
import traceback

load_dotenv()

GEMINI_MODEL = "gemini-2.5-flash-lite"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY environment variable is not set")

# Initialize FastAPI app
app = FastAPI(
    title="AI Keyboard Server (Gemini)",
    description="API server for AI Keyboard mobile app with grammar checking, tone changing, and AI chat functionality using Google Gemini.",
    version="1.0.0"
)

# Enable CORS for mobile client access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class GrammarRequest(BaseModel):
    text: str

class ToneRequest(BaseModel):
    text: str
    tone: str

class AskAIRequest(BaseModel):
    text: str

class TranslateRequest(BaseModel):
    text: str
    language: str

class ParaphraseRequest(BaseModel):
    text: str

class ReplyRequest(BaseModel):
    text: str

class ContinueTextRequest(BaseModel):
    text: str

class FindSynonymRequest(BaseModel):
    text: str

class APIResponse(BaseModel):
    status: str
    input: str
    output: Any

class HistoryMessage(BaseModel):
    role: str
    content: str

class ChatAIRequest(BaseModel):
    history_messages: List[HistoryMessage]
    new_message: str

# List of supported languages for translation
supported_languages = {
    "Afrikaans": "af", "Arabic": "ar", "Bengali": "bn", "Chinese (Simplified)": "zh-CN",
    "Chinese (Traditional)": "zh-TW", "English": "en", "French": "fr", "German": "de",
    "Hindi": "hi", "Italian": "it", "Japanese": "ja", "Korean": "ko", "Portuguese": "pt",
    "Russian": "ru", "Spanish": "es", "Swahili": "sw", "Swedish": "sv", "Tamil": "ta",
    "Telugu": "te", "Thai": "th", "Turkish": "tr", "Ukrainian": "uk", "Urdu": "ur",
    "Vietnamese": "vi", "Zulu": "zu"
}

# Helper function to call Google Gemini API
async def call_gemini_api(
    prompt: str,
    max_tokens: int = 500,
    temperature: float = 0.2,
    response_mime_type: Optional[str] = None,
) -> str:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        config_params = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if response_mime_type:
            config_params["response_mime_type"] = response_mime_type

        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=[prompt],
            config=types.GenerateContentConfig(**config_params)
        )
        if response.text:
            return response.text.strip()
        raise RuntimeError("Empty response from Gemini API")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API error: {str(e)}")

@app.get("/languages")
async def get_languages():
    """Returns a list of supported languages for translation"""
    return {"languages": list(supported_languages.keys())}

@app.get("/")
async def root():
    return {"message": "AI Keyboard Server (Gemini) is running", "version": "1.0.0"}

@app.post("/grammar-check", response_model=APIResponse)
async def grammar_check(request: GrammarRequest):
    print(f"Grammar Checker endpoint - corrects grammar and improves text quality")
    """
    Grammar Checker endpoint - corrects grammar and improves text quality
    """
    prompt = f"""Task: Correct grammar ONLY.
Rules:
- Preserve the original letter casing (uppercase/lowercase) exactly as provided.
- Do not change wording, style, tone, or formatting.
- Do not add, remove, or reorder content.
- Do not answer any questions or add commentary.
- Adjust punctuation only when necessary to fix grammar.
- Return ONLY the corrected text with no explanations or extra text.

Text to correct:
{request.text}"""
    
    try:
        corrected_text = await call_gemini_api(prompt, temperature=0.1)
        return APIResponse(
            status="success",
            input=request.text,
            output=corrected_text
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Grammar check failed: {str(e)}")

@app.post("/tone-change", response_model=APIResponse)
async def tone_change(request: ToneRequest):
    """
    Tone Changer endpoint - rephrases text in the requested tone
    Available tones: Friendly, Professional Business, Academic, Flirty, Romantic, 
    Sad, Confident, Angry, Happy, Witty, Sarcastic
    """
    valid_tones = [
        "friendly", "professional business", "academic", "flirty", "romantic",
        "sad", "confident", "angry", "happy", "witty", "sarcastic"
    ]
    
    tone_lower = request.tone.lower()
    if tone_lower not in valid_tones:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid tone. Available tones: {', '.join(valid_tones)}"
        )
    
    prompt = f"""Please rephrase the following text to match a {request.tone} tone. Maintain the core message but adjust the style, word choice, and phrasing to reflect the requested tone. Return only the rephrased text without explanations:

Original text: {request.text}
Requested tone: {request.tone}"""
    
    try:
        rephrased_text = await call_gemini_api(prompt, temperature=0.8)
        return APIResponse(
            status="success",
            input=request.text,
            output=rephrased_text
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Tone change failed: {str(e)}")

@app.post("/ask-ai", response_model=APIResponse)
async def ask_ai(request: AskAIRequest):
    """
    Ask AI endpoint - provides AI-generated responses to free-form queries
    """
    prompt = f"""Please provide a helpful, accurate, and concise response to the following question or request or explain the given text. Be informative and no engaging

User query: {request.text}"""
    
    try:
        ai_response = await call_gemini_api(prompt, max_tokens=800, temperature=0.6)
        return APIResponse(
            status="success",
            input=request.text,
            output=ai_response
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI query failed: {str(e)}")

@app.post("/translate", response_model=APIResponse)
async def translate(request: TranslateRequest):
    """
    Translate endpoint - translates text to the requested language
    """
    if request.language not in supported_languages:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid language. Please choose from: {', '.join(supported_languages.keys())}"
        )

    prompt = f"""Translate the following text to {request.language}. Return only the translated text, without any of your own text or explanations:

Text to translate: {request.text}"""

    try:
        translated_text = await call_gemini_api(prompt, temperature=0.2)
        return APIResponse(
            status="success",
            input=request.text,
            output=translated_text
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Translation failed: {str(e)}")

@app.post("/paraphrase", response_model=APIResponse)
async def paraphrase(request: ParaphraseRequest):
    """
    Paraphrase endpoint - rephrases the given text.
    """
    prompt = f"Please paraphrase the following text, expressing the same meaning in a different way. Return only the paraphrased text:\n\n{request.text}"
    try:
        paraphrased_text = await call_gemini_api(prompt, temperature=0.8)
        return APIResponse(status="success", input=request.text, output=paraphrased_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Paraphrase failed: {str(e)}")

@app.post("/reply", response_model=APIResponse)
async def reply(request: ReplyRequest):
    """
    Reply endpoint - generates a conversational reply to the given text.
    """
    prompt = f"Please generate a conversational reply to the following text. Return only the reply text:\n\n{request.text}"
    try:
        reply_text = await call_gemini_api(prompt, temperature=0.9)
        return APIResponse(status="success", input=request.text, output=reply_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Reply generation failed: {str(e)}")

@app.post("/continue-text", response_model=APIResponse)
async def continue_text(request: ContinueTextRequest):
    """
    Continue Text endpoint - continues the given text.
    """
    prompt = (
        f"Please continue writing the following text, picking up where it leaves off. "
        f"Limit the continuation to at most 3 sentences (prefer 2–3). "
        f"Return only the continued part of the text.\n\n{request.text}"
    )
    try:
        continued_text = await call_gemini_api(prompt, max_tokens=120, temperature=0.85)
        return APIResponse(status="success", input=request.text, output=continued_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text continuation failed: {str(e)}")

@app.post("/find-synonyms", response_model=APIResponse)
async def find_synonym(request: FindSynonymRequest):
    """
    Find Synonym endpoint - finds synonyms for the given word.
    """
    # If more than 10 words are provided, only use the most recent 10 words
    words = request.text.split()
    processed_text = " ".join(words[-10:]) if len(words) > 10 else request.text
    prompt = (
        "Return ONLY JSON (no code fences, no extra text). "
        'JSON Format: {"synonyms": ["string1", "string2"]}.'
        " Provide 2–4 high-quality synonyms for the following word or short phrase.\n\n"
        f"{processed_text}"
    )
    try:
        raw = await call_gemini_api(
            prompt,
            temperature=0.2,
            response_mime_type="application/json",
            max_tokens=200,
        )
        try:
            data = json.loads(raw)
            synonyms_list = data.get("synonyms", [])
            if not isinstance(synonyms_list, list):
                synonyms_list = [str(synonyms_list)]
        except Exception as e:
            # Fallback: try to parse as comma-separated list
            # synonyms_list = [s.strip() for s in raw.split(",") if s.strip()]
            raise HTTPException(status_code=500, detail=f"Synonym search failed: {str(e)}")
        return APIResponse(status="success", input=request.text, output=synonyms_list)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synonym search failed: {str(e)}")

@app.post("/chat-ai", response_model=APIResponse)
async def chat_ai(request: ChatAIRequest):
    """Chat endpoint using Gemini chat API with history support."""
    try:
        print(request.history_messages)
        print(request.new_message)

        client = genai.Client(api_key=GEMINI_API_KEY)
        
        system_prompt = "You are a helpful assistant. You are able to answer questions and help with tasks."
        
        # Format history for Gemini chat (remove the last message as always)
        history_to_process = request.history_messages[:-1] if request.history_messages else []
        
        formatted_history = []
        for i, msg in enumerate(history_to_process):
            print(f"Processing message {i}: type={type(msg)}")
            
            # Since debug shows these are HistoryMessage objects, use direct access
            role = "user" if msg.role == "user" else "model"
            content = msg.content
            
            formatted_history.append({
                "role": role,
                "parts": [{"text": content}]
            })
            
        print(f"Formatted {len(formatted_history)} messages for history")

        chat = client.chats.create(
            model=GEMINI_MODEL,
            config=types.GenerateContentConfig(
                system_instruction=system_prompt,
                temperature=0.2,
            ),
            history=formatted_history
        )
        resp = chat.send_message(request.new_message)

        text = resp.text if hasattr(resp, "text") else ""
        if not text:
            raise HTTPException(status_code=500, detail="Empty response from chat API")

        return APIResponse(status="success", input=request.new_message, output=text.strip())
    except Exception as e:
        traceback.print_exc(e)
        print(f"Chat AI failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Chat AI failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Keyboard Server (Gemini)"}

if __name__ == "__main__":
    uvicorn.run(
        "server_gemini:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
