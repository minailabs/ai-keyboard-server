import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
from typing import Optional
import uvicorn
from dotenv import load_dotenv

load_dotenv()

# Initialize OpenAI client using environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY environment variable is not set")
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize FastAPI app
app = FastAPI(
    title="AI Keyboard Server",
    description="API server for AI Keyboard mobile app with grammar checking, tone changing, and AI chat functionality",
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

class APIResponse(BaseModel):
    status: str
    input: str
    output: str

# Helper function to call OpenAI Responses API
async def call_openai_api(prompt: str, max_tokens: int = 10500) -> str:
    try:
        # Using Responses API with the requested model name
        response = client.responses.create(
            model="gpt-5-nano",
            input=prompt,
            max_output_tokens=max_tokens
        )
        # Extract text from the response
        if response and response.output and len(response.output) > 0:
            # Concatenate all text outputs
            texts = []
            for item in response.output:
                if hasattr(item, "content") and item.content:
                    for block in item.content:
                        if getattr(block, "type", None) == "output_text" and hasattr(block, "text"):
                            texts.append(block.text)
            if texts:
                return "".join(texts).strip()
        # Fallback for SDKs that expose top-level text
        if hasattr(response, "output_text") and response.output_text:
            return response.output_text.strip()
        raise RuntimeError("Empty response from OpenAI Responses API")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {str(e)}")

@app.get("/")
async def root():
    return {"message": "AI Keyboard Server is running", "version": "1.0.0"}

@app.post("/grammar-check", response_model=APIResponse)
async def grammar_check(request: GrammarRequest):
    """
    Grammar Checker endpoint - corrects grammar and improves text quality
    """
    prompt = f"""Please correct the grammar and improve the following text while maintaining its original meaning and style. Return only the corrected text without any explanations or additional formatting:

Text to correct: {request.text}"""
    
    try:
        corrected_text = await call_openai_api(prompt)
        return APIResponse(
            status="success",
            input=request.text,
            output=corrected_text
        )
    except Exception as e:
        print(f"Grammar check failed: {str(e)}")
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
        rephrased_text = await call_openai_api(prompt)
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
    prompt = f"""Please provide a helpful, accurate, and concise response to the following question or request. Be informative and engaging:

User query: {request.text}"""
    
    try:
        ai_response = await call_openai_api(prompt, max_tokens=800)
        return APIResponse(
            status="success",
            input=request.text,
            output=ai_response
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"AI query failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "AI Keyboard Server"}

if __name__ == "__main__":
    uvicorn.run(
        "server_gpt:app",
        host="0.0.0.0",
        port=8001,
        reload=True
    )