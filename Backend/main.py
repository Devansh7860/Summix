from fastapi import FastAPI, Request, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from typing import Optional
import asyncio
import json
load_dotenv()
from ai import summarize, ask, summarizePlaylist
from websocket_task_manager import task_manager, TaskType, TaskStatus, periodic_cleanup

# Background task functions
async def summarize_background(video_id: str, user_id: str, api_key: str, browserless_api_key: str, task_id: str):
    """Background task for video summarization"""
    try:
        print(f"🚀 Starting background summarization for video {video_id}, task {task_id}")
        
        # Run the summarization
        result = await summarize(video_id, user_id, api_key, browserless_api_key, task_id)
        
        # Check if task was cancelled during processing
        task_status = await task_manager.get_task_status(task_id)
        if task_status == TaskStatus.CANCELLED:
            print(f"❌ Task {task_id} was cancelled - not sending result")
            return
        
        # Send final result via WebSocket
        if result and result != "Task was cancelled":
            await task_manager.update_task(
                task_id, 
                status=TaskStatus.COMPLETED, 
                progress=100, 
                message="Summary completed", 
                result=result
            )
            print(f"✅ Background summarization completed for task {task_id}")
        else:
            await task_manager.update_task(
                task_id, 
                status=TaskStatus.FAILED, 
                progress=100, 
                message="Summarization failed",
                error="Failed to generate summary"
            )
            print(f"❌ Background summarization failed for task {task_id}")
            
    except Exception as e:
        print(f"💥 Background summarization error for task {task_id}: {str(e)}")
        # Only update if task wasn't cancelled
        try:
            task_status = await task_manager.get_task_status(task_id)
            if task_status != TaskStatus.CANCELLED:
                await task_manager.update_task(
                    task_id, 
                    status=TaskStatus.FAILED, 
                    progress=100, 
                    message="Summarization error",
                    error=str(e)
                )
        except:
            pass  # Task might not exist anymore

async def summarize_playlist_background(playlist_id: str, user_id: str, api_key: str, browserless_api_key: str, task_id: str):
    """Background task for playlist summarization"""
    try:
        print(f"🚀 Starting background playlist summarization for {playlist_id}, task {task_id}")
        
        # Run the playlist summarization
        result = await summarizePlaylist(playlist_id, user_id, api_key, browserless_api_key, task_id)
        
        # Check if task was cancelled during processing
        task_status = await task_manager.get_task_status(task_id)
        if task_status == TaskStatus.CANCELLED:
            print(f"❌ Task {task_id} was cancelled - not sending result")
            return
        
        # Send final result via WebSocket
        if result and result != "Task was cancelled":
            await task_manager.update_task(
                task_id, 
                status=TaskStatus.COMPLETED, 
                progress=100, 
                message="Playlist summary completed", 
                result=result
            )
            print(f"✅ Background playlist summarization completed for task {task_id}")
        else:
            await task_manager.update_task(
                task_id, 
                status=TaskStatus.FAILED, 
                progress=100, 
                message="Playlist summarization failed",
                error="Failed to generate playlist summary"
            )
            print(f"❌ Background playlist summarization failed for task {task_id}")
            
    except Exception as e:
        print(f"💥 Background playlist summarization error for task {task_id}: {str(e)}")
        # Only update if task wasn't cancelled
        try:
            task_status = await task_manager.get_task_status(task_id)
            if task_status != TaskStatus.CANCELLED:
                await task_manager.update_task(
                    task_id, 
                    status=TaskStatus.FAILED, 
                    progress=100, 
                    message="Playlist summarization error",
                    error=str(e)
                )
        except:
            pass  # Task might not exist anymore



app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["*"] for all
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Start periodic cleanup task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(periodic_cleanup())

# WebSocket endpoint for real-time task management
@app.websocket("/ws/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    await task_manager.connect_user(websocket, user_id)
    try:
        while True:
            # Keep connection alive and handle any incoming messages
            data = await websocket.receive_text()
            try:
                message = json.loads(data)
                # Handle client messages if needed (ping/pong, etc.)
                if message.get("type") == "ping":
                    await websocket.send_text(json.dumps({"type": "pong"}))
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        await task_manager.disconnect_user(user_id)



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

def get_browserless_api_key_from_request(request: Request) -> str:
    """Extract Browserless API key from request headers."""
    api_key = request.headers.get("X-Browserless-API-Key")
    if not api_key:
        raise HTTPException(status_code=400, detail="Browserless API key is required in X-Browserless-API-Key header")
    return api_key

async def validate_both_api_keys_quick(gemini_api_key: str, browserless_api_key: str) -> None:
    """Quick validation - just check if keys exist and have basic format"""
    # Quick Gemini API key check
    if not gemini_api_key or not gemini_api_key.startswith('AIza'):
        raise HTTPException(status_code=400, detail="Invalid Gemini API key format")
    
    # Quick Browserless API key check  
    if not browserless_api_key or len(browserless_api_key) < 20:
        raise HTTPException(status_code=400, detail="Invalid Browserless API key format")
    
    # Skip actual API calls since frontend already validated
    print(f"✅ Quick validation passed for both API keys")

def validate_token(token: str, gemini_api_key: str, browserless_api_key: str) -> bool:
    """Validate the token against the provided API keys"""
    try:
        import time
        import hashlib
        
        if not token or '_' not in token:
            return False
            
        # Parse token format: timestamp_hash
        parts = token.split('_')
        if len(parts) != 2:
            return False
            
        timestamp_str, token_hash = parts
        timestamp = int(timestamp_str)
        
        # Recreate hash and verify
        token_data = f"{gemini_api_key[:10]}{browserless_api_key[:10]}{timestamp}"
        expected_hash = hashlib.sha256(token_data.encode()).hexdigest()[:16]
        
        return token_hash == expected_hash
        
    except Exception as e:
        print(f"Token validation error: {e}")
        return False

async def validate_with_token_or_quick(gemini_api_key: str, browserless_api_key: str, token: str = None) -> None:
    """Validate using token if available, otherwise use quick validation"""
    if token and validate_token(token, gemini_api_key, browserless_api_key):
        print(f"✅ Token validation passed - skipping API re-validation")
        return
    
    if token:
        print(f"⚠️ Token validation failed - falling back to quick validation")
    
    # Fallback to quick validation
    await validate_both_api_keys_quick(gemini_api_key, browserless_api_key)

async def validate_both_api_keys(gemini_api_key: str, browserless_api_key: str) -> None:
    """Validate both Gemini and Browserless API keys, raise HTTPException if invalid"""
    # Validate Gemini API key
    try:
        from ai import create_model
        from langchain_core.output_parsers import StrOutputParser
        
        model = create_model(gemini_api_key)
        parser = StrOutputParser()
        chain = model | parser
        test_response = chain.invoke("Hello")
        
        if not test_response or not isinstance(test_response, str):
            raise HTTPException(status_code=400, detail="Invalid Gemini API key")
            
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Gemini API key: {str(e)}")
    
    # Validate Browserless API key
    try:
        import requests
        
        test_url = f"https://production-sfo.browserless.io/content?token={browserless_api_key}"
        test_payload = {
            "url": "https://httpbin.org/status/200",
            "gotoOptions": {"timeout": 3000, "waitUntil": "load"}
        }
        response = requests.post(test_url, json=test_payload, timeout=8)
        
        if response.status_code not in [200]:
            raise HTTPException(status_code=400, detail="Invalid Browserless API key")
            
    except requests.exceptions.RequestException:
        raise HTTPException(status_code=400, detail="Invalid Browserless API key")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid Browserless API key: {str(e)}")

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

@app.post("/validate-browserless-key")
async def validate_browserless_key(request: Request):
    """Validate the Browserless API key by making a simple test request"""
    try:
        # Extract Browserless API key from request headers
        api_key = request.headers.get("X-Browserless-API-Key")
        if not api_key:
            raise HTTPException(status_code=400, detail="Browserless API key is required in X-Browserless-API-Key header")
        
        # Basic format validation (Browserless keys are typically long alphanumeric strings)
        if len(api_key) < 20:
            return {"valid": False, "error": "Browserless API key appears to be too short"}
        
        # Test the API key with the same endpoint used in production
        import requests
        
        # Use the content endpoint with a very minimal test
        test_url = f"https://production-sfo.browserless.io/content?token={api_key}"
        
        test_payload = {
            "url": "https://httpbin.org/status/200",  # Simple, fast test URL
            "gotoOptions": {
                "timeout": 3000,  # 3 seconds
                "waitUntil": "load"  # Faster than networkidle2
            }
        }
        
        # Add some randomization to prevent caching issues
        import time
        cache_buster = int(time.time())
        test_payload["url"] = f"https://httpbin.org/status/200?t={cache_buster}"
        
        response = requests.post(
            test_url,
            json=test_payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "YoutubeSummary-Validator/1.0"
            },
            timeout=10  # Increased timeout for reliability
        )
        
        print(f"Browserless validation response: {response.status_code}")  # Debug logging
        
        # Check if the request was successful
        if response.status_code == 200:
            return {"valid": True, "message": "Browserless API key is valid"}
        elif response.status_code == 401:
            return {"valid": False, "error": "Invalid Browserless API key"}
        elif response.status_code == 402:
            return {"valid": False, "error": "Browserless API key quota exceeded"}
        elif response.status_code == 403:
            return {"valid": False, "error": "Browserless API key lacks required permissions"}
        elif response.status_code == 429:
            return {"valid": False, "error": "Too many requests. Please wait a moment and try again."}
        else:
            return {"valid": False, "error": f"Browserless API validation failed with status {response.status_code}"}
            
    except HTTPException:
        raise  # Re-raise HTTPException (missing API key)
    except requests.exceptions.Timeout:
        print("Browserless validation timed out")  # Debug logging
        return {"valid": False, "error": "Browserless API validation timed out. Please try again."}
    except requests.exceptions.RequestException as e:
        print(f"Browserless validation network error: {str(e)}")  # Debug logging
        return {"valid": False, "error": f"Network error during validation: {str(e)}"}
    except Exception as e:
        print(f"Browserless validation error: {str(e)}")  # Debug logging
        return {"valid": False, "error": f"Browserless API key validation failed: {str(e)}"}

@app.post("/validate-both-keys")
async def validate_both_keys(request: Request):
    """Validate both Gemini and Browserless API keys and return a validation token"""
    import uuid
    import time
    import hashlib
    
    try:
        # Extract both API keys
        gemini_api_key = get_api_key_from_request(request)
        browserless_api_key = get_browserless_api_key_from_request(request)
        
        # Validate both keys using existing validation functions
        await validate_both_api_keys(gemini_api_key, browserless_api_key)
        
        # Both keys are valid - generate validation token
        timestamp = int(time.time())
        token_data = f"{gemini_api_key[:10]}{browserless_api_key[:10]}{timestamp}"
        token_hash = hashlib.sha256(token_data.encode()).hexdigest()[:16]
        validation_token = f"{timestamp}_{token_hash}"
        
        return {
            "valid": True,
            "validation_token": validation_token,
            "message": "Both API keys are valid"
        }
        
    except HTTPException as e:
        return {"valid": False, "error": e.detail}
    except Exception as e:
        return {"valid": False, "error": f"Validation failed: {str(e)}"}

@app.post("/summarize/video")
async def summarize_video(data: Video, request: Request):
    # Extract both API keys and validation token
    api_key = get_api_key_from_request(request)
    browserless_api_key = get_browserless_api_key_from_request(request)
    validation_token = request.headers.get("X-Validation-Token")
    
    # Use token validation if available, otherwise quick validation
    await validate_with_token_or_quick(api_key, browserless_api_key, validation_token)
    
    # Create task in WebSocket manager
    task_id = await task_manager.create_task(data.userId, TaskType.VIDEO_SUMMARY, data.video_id)
    
    # Start summarization in background (non-blocking)
    asyncio.create_task(summarize_background(data.video_id, data.userId, api_key, browserless_api_key, task_id))
    
    # Return immediately with task info
    return {"task_id": task_id, "status": "started", "message": "Summarization started"}

@app.post("/ask")
async def askQuestion(data: Question, request: Request):
    # Extract both API keys and validation token
    api_key = get_api_key_from_request(request)
    browserless_api_key = get_browserless_api_key_from_request(request)
    validation_token = request.headers.get("X-Validation-Token")
    
    # Use token validation if available, otherwise quick validation
    await validate_with_token_or_quick(api_key, browserless_api_key, validation_token)
    
    # Keys validated, proceed with question answering
    import asyncio
    res = await asyncio.to_thread(ask, data.video_id, data.playlist_id, data.question, data.userId, api_key)
    return {"answer": res}

@app.post("/summarize/playlist")
async def summarize_playlist(data: Playlist, request: Request):
    # Extract both API keys and validation token
    api_key = get_api_key_from_request(request)
    browserless_api_key = get_browserless_api_key_from_request(request)
    validation_token = request.headers.get("X-Validation-Token")
    
    # Use token validation if available, otherwise quick validation
    await validate_with_token_or_quick(api_key, browserless_api_key, validation_token)
    
    # Create task in WebSocket manager
    task_id = await task_manager.create_task(data.userId, TaskType.PLAYLIST_SUMMARY, data.playlist_id)
    
    # Start playlist summarization in background (non-blocking)
    asyncio.create_task(summarize_playlist_background(data.playlist_id, data.userId, api_key, browserless_api_key, task_id))
    
    # Return immediately with task info
    return {"task_id": task_id, "status": "started", "message": "Playlist summarization started"}

@app.post("/cancel")
async def cancel_processing(data: CancelRequest, request: Request):
    """Cancel ongoing summarization tasks for a specific user/content"""
    success_count = 0
    
    if data.videoId:
        success_count = await task_manager.cancel_user_tasks(data.userId, data.videoId)
    elif data.playlistId:
        success_count = await task_manager.cancel_user_tasks(data.userId, data.playlistId)
    else:
        success_count = await task_manager.cancel_user_tasks(data.userId)
    
    return {"success": success_count > 0, "cancelled_tasks": success_count, "message": "Cancellation request processed"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

