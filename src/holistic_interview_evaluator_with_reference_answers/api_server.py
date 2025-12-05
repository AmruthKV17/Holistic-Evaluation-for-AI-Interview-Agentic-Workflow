"""
FastAPI server for running CrewAI workflow locally.
Exposes /kickoff and /status endpoints matching the CrewAI platform API.
"""
import os
import uuid
import threading
from datetime import datetime
from typing import Any, Dict, Optional
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from holistic_interview_evaluator_with_reference_answers.crew import (
    HolisticInterviewEvaluatorWithReferenceAnswersCrew,
)

# Load environment variables
load_dotenv()

# FORCE GOOGLE AI STUDIO: Unset Vertex AI credentials if present
# This prevents litellm from defaulting to Vertex AI (which requires billing)
if "GOOGLE_APPLICATION_CREDENTIALS" in os.environ:
    print(f"Removing GOOGLE_APPLICATION_CREDENTIALS to force Google AI Studio usage")
    del os.environ["GOOGLE_APPLICATION_CREDENTIALS"]

# Ensure we're using the API key
if not os.getenv("GROQ_API_KEY"):
    print("WARNING: GROQ_API_KEY not found in environment variables!")

app = FastAPI(
    title="Holistic Interview Evaluator API",
    description="Local API server for interview evaluation using CrewAI",
    version="1.0.0",
)

# Add CORS middleware for cross-origin requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for execution states
executions: Dict[str, Dict[str, Any]] = {}


class KickoffRequest(BaseModel):
    """Request body for /kickoff endpoint"""
    inputs: Dict[str, Any]


class KickoffResponse(BaseModel):
    """Response body for /kickoff endpoint"""
    kickoff_id: str
    status: str = "PENDING"
    message: str = "Crew execution started"


class StatusResponse(BaseModel):
    """Response body for /status endpoint"""
    state: str
    last_executed_task: Optional[Dict[str, Any]] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None


def run_crew_async(kickoff_id: str, inputs: Dict[str, Any]):
    """Run the crew in a background thread"""
    try:
        executions[kickoff_id]["state"] = "RUNNING"
        
        # Create and run the crew
        crew_instance = HolisticInterviewEvaluatorWithReferenceAnswersCrew()
        result = crew_instance.crew().kickoff(inputs=inputs)
        
        # Store the result
        executions[kickoff_id]["state"] = "SUCCESS"
        executions[kickoff_id]["completed_at"] = datetime.now().isoformat()
        executions[kickoff_id]["last_executed_task"] = {
            "output": str(result.raw) if hasattr(result, 'raw') else str(result),
            "task_name": "final_output_assembly",
        }
        
    except Exception as e:
        executions[kickoff_id]["state"] = "FAILED"
        executions[kickoff_id]["error"] = str(e)
        executions[kickoff_id]["completed_at"] = datetime.now().isoformat()
        print(f"Crew execution failed: {e}")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}


@app.post("/kickoff", response_model=KickoffResponse)
async def kickoff(request: KickoffRequest):
    """
    Start a new crew execution.
    
    Accepts the same input format as the CrewAI platform:
    {
        "inputs": {
            "interview_data": {
                "topic": "...",
                "difficulty": "...",
                "expected_keywords": [...],
                "questions_and_answers": [...]
            }
        }
    }
    """
    # Generate unique ID for this execution
    kickoff_id = str(uuid.uuid4())
    
    # Initialize execution state
    executions[kickoff_id] = {
        "state": "PENDING",
        "started_at": datetime.now().isoformat(),
        "completed_at": None,
        "last_executed_task": None,
        "error": None,
        "inputs": request.inputs,
    }
    
    # Start crew execution in background thread
    thread = threading.Thread(
        target=run_crew_async,
        args=(kickoff_id, request.inputs),
        daemon=True,
    )
    thread.start()
    
    return KickoffResponse(
        kickoff_id=kickoff_id,
        status="PENDING",
        message="Crew execution started",
    )


@app.get("/status/{kickoff_id}", response_model=StatusResponse)
async def get_status(kickoff_id: str):
    """
    Get the status of a crew execution.
    
    Returns the same format as the CrewAI platform:
    {
        "state": "SUCCESS" | "RUNNING" | "PENDING" | "FAILED",
        "last_executed_task": {
            "output": "..."
        }
    }
    """
    if kickoff_id not in executions:
        raise HTTPException(status_code=404, detail="Execution not found")
    
    execution = executions[kickoff_id]
    
    return StatusResponse(
        state=execution["state"],
        last_executed_task=execution["last_executed_task"],
        started_at=execution["started_at"],
        completed_at=execution["completed_at"],
        error=execution["error"],
    )


def start():
    """Start the server using uvicorn"""
    import uvicorn
    uvicorn.run(
        "holistic_interview_evaluator_with_reference_answers.api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )


if __name__ == "__main__":
    start()
