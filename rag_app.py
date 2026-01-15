import os
import asyncio
import time
from typing import Optional, List, Dict, Any
from contextlib import asynccontextmanager

import requests 
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Depends, status, Query, Path
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field, validator
from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage 

from rag_backend import ingest_pdf, chatbot, logger, _THREAD_METADATA, _THREAD_RETRIEVERS

# Config
MAX_FILE_SIZE_MB = 10
ALLOWED_FILE_TYPES = {".pdf"}
RATE_LIMIT_PER_MINUTE = 60

# Models 
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    thread_id: str = Field(..., min_length=1, max_length=100, description="Thread identifier")

class ChatResponse(BaseModel):
    response: str
    thread_id: str
    processing_time_ms: int
    metadata: Optional[Dict[str, Any]] = None

class UploadResponse(BaseModel):
    filename: str
    thread_id: str
    documents: int
    chunks: int
    processing_time_ms: int
    message: str

class HealthResponse(BaseModel):
    status: str
    version: str
    uptime_seconds: float
    active_threads: int

class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    thread_id: Optional[str] = None

class ThreadInfoResponse(BaseModel):
    thread_id: str
    has_document: bool
    metadata: Optional[Dict[str, Any]] = None

# Lifespan Management 
startup_time = time.time() 

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown events"""
    # Startup
    logger.info("Agentic RAG API starting up...")
    logger.info(f"Max file size: {MAX_FILE_SIZE_MB}MB")
    yield
    
    # Shutdown
    logger.info("Agentic RAG API shutting down...")
    # Cleanup resources if needed
    _THREAD_RETRIEVERS.clear()
    _THREAD_METADATA.clear()

# FastAPI App 
app = FastAPI(
    title="Agentic RAG API",
    description="RAG system with hybrid search and agentic workflows",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
) 

# Middlewares
# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# DEPENDENCIES Injection
async def verify_thread_exists(thread_id: str) -> bool:
    """Check if thread has an indexed document"""
    return thread_id in _THREAD_RETRIEVERS

async def get_file_size(file: UploadFile) -> int:
    """
    Get file size safely for FastAPI UploadFile
    """
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    return size

# End-points
@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Agentic RAG API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint for monitoring"""
    return HealthResponse(
        status="healthy",
        version="2.0.0",
        uptime_seconds=round(time.time() - startup_time, 2),
        active_threads=len(_THREAD_RETRIEVERS)
    )

@app.post(
    "/upload",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Processing error"}
    }
)
async def upload_pdf(thread_id: str = Query(..., description="Unique thread identifier"), file: UploadFile = File(..., description="PDF file to upload"), background_tasks: BackgroundTasks = None):
    """
    Upload and index a PDF document for a specific thread
    
    - thread_id: Unique identifier for the chat thread
    - file: PDF file (max 10MB)
    
    Returns indexing summary with processing time
    """
    start_time = time.time() 

    # Validate file type
    if not file.filename.endswith(".pdf"):
        logger.warning(f"Invalid file type uploaded: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Only PDF files are allowed. Received: {file.filename}"
        )

    # Validate file size
    file_size = await get_file_size(file)
    max_size_bytes = MAX_FILE_SIZE_MB * 1024 * 1024

    if file_size > max_size_bytes:
        logger.warning(f"File too large: {file_size / 1024 / 1024:.2f}MB")
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File size {file_size / 1024 / 1024:.2f}MB exceeds maximum {MAX_FILE_SIZE_MB}MB"
        )
    
    logger.info(f"Uploading PDF | thread_id={thread_id} | filename={file.filename} | size={file_size / 1024:.2f}KB")

    try:
        # Read file data
        data = await file.read()
        
        # Process PDF (in background if desired)
        result = ingest_pdf(data, thread_id, file.filename)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        logger.info(f"PDF indexed successfully | thread_id={thread_id} | chunks={result['chunks']} | time={processing_time}ms")
        
        return UploadResponse(
            filename=result["filename"],
            thread_id=thread_id,
            documents=result["documents"],
            chunks=result["chunks"],
            processing_time_ms=processing_time,
            message=f"Successfully indexed {result['chunks']} chunks from {result['documents']} pages"
        )
    
    except ValueError as e:
        logger.error(f"Validation error during PDF processing: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    
    except Exception as e:
        logger.exception(f"Error processing PDF | thread_id={thread_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {str(e)}"
        )

# @app.post("/upload")
# async def upload_pdf(thread_id: str, file: UploadFile = File(...)):
#     if not file.filename.endswith(".pdf"):
#         raise HTTPException(status_code=400, detail="Only PDF's Allowed")

#     data = await file.read() 
#     return ingest_pdf(data, thread_id, file.filename)

@app.post("/chat", response_model=ChatResponse, responses={400: {"model": ErrorResponse, "description":"Invalid request"},500: {"model": ErrorResponse, "description": "Processing error"}})
async def chat(req: ChatRequest):
    """
    Send a message and get AI response
    
    - message: User's question or instruction
    - thread_id: Thread identifier (must have uploaded PDF if asking about documents)
    
    Returns AI-generated response with processing time
    """
    start_time = time.time()
    
    logger.info(f"Chat request | thread_id={req.thread_id} | message_length={len(req.message)}")
    
    try:
        config = {"configurable": {"thread_id": req.thread_id}}
        
        # Invoke chatbot
        result = chatbot.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config=config,
        )
        
        processing_time = int((time.time() - start_time) * 1000)
        
        # Extract response
        response_content = result["messages"][-1].content
        
        # Get metadata if available
        metadata = None
        if req.thread_id in _THREAD_METADATA:
            metadata = _THREAD_METADATA[req.thread_id]
        
        logger.info(f"Chat response generated | thread_id={req.thread_id} | time={processing_time}ms | response_length={len(response_content)}")
        
        return ChatResponse(
            response=response_content,
            thread_id=req.thread_id,
            processing_time_ms=processing_time,
            metadata=metadata
        )
    
    except Exception as e:
        logger.exception(f"Error during chat | thread_id={req.thread_id}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate response: {str(e)}"
        )

@app.get(
    "/thread/{thread_id}",
    response_model=ThreadInfoResponse,
    responses={404: {"model": ErrorResponse, "description": "Thread not found"}}
)
async def get_thread_info(
    thread_id: str = Path(
        ...,
        min_length=1,
        max_length=100,
        description="Thread ID"
    )
):
    has_document = thread_id in _THREAD_RETRIEVERS
    metadata = _THREAD_METADATA.get(thread_id) if has_document else None

    logger.info(f"Thread info request | thread_id={thread_id} | has_document={has_document}")

    return ThreadInfoResponse(
        thread_id=thread_id,
        has_document=has_document,
        metadata=metadata
    )

@app.delete(
    "/thread/{thread_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": ErrorResponse}}
)
async def delete_thread(
    thread_id: str = Path(..., min_length=1, max_length=100)
):    
    """
    Delete a thread and its associated data
    
    Removes indexed documents and metadata for the thread
    """
    if thread_id not in _THREAD_RETRIEVERS:
        logger.warning(f"Attempted to delete non-existent thread: {thread_id}")
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Thread {thread_id} not found"
        )
    
    # Clean up
    del _THREAD_RETRIEVERS[thread_id]
    if thread_id in _THREAD_METADATA:
        del _THREAD_METADATA[thread_id]
    
    logger.info(f"Thread deleted | thread_id={thread_id}")
    return None

@app.get("/threads", response_model=List[ThreadInfoResponse])
async def list_threads():
    """
    List all active threads
    
    Returns list of all threads with their metadata
    """
    threads = [
        ThreadInfoResponse(
            thread_id=thread_id,
            has_document=True,
            metadata=_THREAD_METADATA.get(thread_id)
        )
        for thread_id in _THREAD_RETRIEVERS.keys()
    ]
    
    logger.info(f"Thread list request | count={len(threads)}")
    return threads

if __name__ == "__main__":
    import uvicorn
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", 8000))
    logger.info(f"Starting server | host={host} | port={port}")

    uvicorn.run(
        "rag_app:app",   
        host=host,
        port=port,
        reload=True
    )