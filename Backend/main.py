from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
load_dotenv()
from ai import summarize, ask, summarizePlaylist, cancel_task



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class Video(BaseModel):
    video_id: str
    userId: str
class Playlist(BaseModel):
    playlist_id: str
    userId: str

class Question(BaseModel):
    question: str
    video_id: Optional[str] = None
    userId: str
    playlist_id: Optional[str] = None

class CancelRequest(BaseModel):
    userId: str
    videoId: Optional[str] = None
    playlistId: Optional[str] = None

def get_api_key_from_request(request: Request) -> str:
    """Extract Gemini API key from request headers."""
    api_key = request.headers.get("X-Gemini-API-Key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Gemini API key is required in X-Gemini-API-Key header")
    return api_key

@app.get("/")
def home():
    return {"message": "hello world"}

@app.post("/validate-api-key")
async def validate_api_key(request: Request):
    """Validate the Gemini API key by making a simple test request"""
    try:
        api_key = get_api_key_from_request(request)
        
        # Test the API key with a simple model creation and actual API call
        from ai import create_model
        from langchain_core.output_parsers import StrOutputParser
        
        model = create_model(api_key)
        parser = StrOutputParser()
        
        # Create a simple chain and test it
        chain = model | parser
        
        # Make a minimal test request to verify the API key works
        test_response = chain.invoke("Hello")
        
        # If we get here, the API key is valid
        if test_response and isinstance(test_response, str):
            return {"valid": True, "message": "API key is valid"}
        else:
            return {"valid": False, "error": "API key validation failed"}
            
    except HTTPException:
        raise  # Re-raise HTTPException (missing API key)
    except Exception as e:
        error_msg = str(e).lower()
        
        # Check for specific authentication errors
        if "api_key" in error_msg or "authentication" in error_msg or "unauthorized" in error_msg:
            return {"valid": False, "error": "Invalid API key"}
        elif "quota" in error_msg or "limit" in error_msg:
            return {"valid": False, "error": "API key quota exceeded"}
        elif "permission" in error_msg or "forbidden" in error_msg:
            return {"valid": False, "error": "API key lacks required permissions"}
        else:
            return {"valid": False, "error": f"API key validation failed: {str(e)}"}

@app.post("/summarize/video")
async def summarize_video(data: Video, request: Request):
    api_key = get_api_key_from_request(request)
    res = summarize(data.video_id, data.userId, api_key)
    return {"summary": res}

@app.post("/ask")
async def askQuestion(data: Question, request: Request):
    api_key = get_api_key_from_request(request)
    res = ask(data.video_id, data.playlist_id, data.question, data.userId, api_key)
    return {"answer": res}

@app.post("/summarize/playlist")
async def summarize_playlist(data: Playlist, request: Request):
    api_key = get_api_key_from_request(request)
    res = await summarizePlaylist(data.playlist_id, data.userId, api_key)
    return {"summary": res}

@app.post("/cancel")
async def cancel_processing(data: CancelRequest, request: Request):
    """Cancel ongoing summarization tasks for a specific user/content"""
    # Cancel requests don't need API key since they don't make LLM calls
    if data.videoId:
        success = cancel_task(data.userId, data.videoId)
    elif data.playlistId:
        success = cancel_task(data.userId, data.playlistId, is_playlist=True)
    else:
        success = False
    
    return {"success": success, "message": "Cancellation request received"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

