import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from typing import Optional
import uvicorn
from dotenv import load_dotenv

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
    output: str

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
async def call_gemini_api(prompt: str, max_tokens: int = 500) -> str:
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=[prompt],
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    max_output_tokens=max_tokens
                )
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
    prompt = f"""Please correct the grammar of the following text. It is very important that you preserve the original letter casing (uppercase and lowercase). Only fix grammar mistakes, without altering the casing. Return only the corrected text without any explanations or additional formatting:

Text to correct: {request.text}"""
    
    try:
        corrected_text = await call_gemini_api(prompt)
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
        rephrased_text = await call_gemini_api(prompt)
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
        ai_response = await call_gemini_api(prompt, max_tokens=800)
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
        translated_text = await call_gemini_api(prompt)
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
        paraphrased_text = await call_gemini_api(prompt)
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
        reply_text = await call_gemini_api(prompt)
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
    prompt = f"Please continue writing the following text, picking up where it leaves off. Return only the continued part of the text:\n\n{request.text}"
    try:
        continued_text = await call_gemini_api(prompt)
        return APIResponse(status="success", input=request.text, output=continued_text)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text continuation failed: {str(e)}")

@app.post("/find-synonym", response_model=APIResponse)
async def find_synonym(request: FindSynonymRequest):
    """
    Find Synonym endpoint - finds synonyms for the given word.
    """
    prompt = f"Please provide a single synonym for the following word or text. Return only the synonym:\n\n{request.text}"
    try:
        synonyms = await call_gemini_api(prompt)
        return APIResponse(status="success", input=request.text, output=synonyms)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synonym search failed: {str(e)}")

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
