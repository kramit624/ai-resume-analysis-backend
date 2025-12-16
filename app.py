from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
from ingest import ingest_resume
from query import answer_question, analyze_resume
from jobs import search_jobs

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = FastAPI(title="AI Resume Analyzer API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class AnalysisResult(BaseModel):
    ats_score: int
    suggestions: list
    summary: str

latest_analysis = None

def process_resume(file_path: str):
    """Process resume: ingest + analyze"""
    global latest_analysis
    
    ingest_resume(file_path)
    

    analysis = analyze_resume()
    latest_analysis = analysis
    print(f"Analysis complete: ATS Score = {analysis['ats_score']}/100")


@app.post("/upload")
async def upload_resume(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    global latest_analysis

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    try:
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")

        if os.path.exists(UPLOAD_DIR):
            for old_file in os.listdir(UPLOAD_DIR):
                os.remove(os.path.join(UPLOAD_DIR, old_file))

        latest_analysis = None

        path = os.path.join(UPLOAD_DIR, file.filename)
        content = await file.read()
        with open(path, "wb") as f:
            f.write(content)

        background_tasks.add_task(process_resume, path)

        return {
            "status": "uploaded",
            "filename": file.filename,
            "message": "Previous resume cleared. New resume is being analyzed."
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/analysis")
def get_analysis():
    """Get the latest resume analysis"""
    if latest_analysis is None:
        return {"status": "processing", "message": "Analysis in progress..."}
    
    return {
        "status": "complete",
        "analysis": latest_analysis
    }


@app.post("/ask")
def ask(payload: QuestionRequest):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        job_keywords = [
            "find job",
            "find jobs",
            "job opening",
            "job openings",
            "job opportunities",
            "search jobs",
            "available jobs",
            "recommend jobs",
            "job listings",
        ]

        is_job_query = any(
            keyword in payload.question.lower() for keyword in job_keywords
        )

        if is_job_query:
            if not latest_analysis:
                return {
                    "question": payload.question,
                    "answer": "Please upload a resume first so I can find relevant jobs for you."
                }

            answer = search_jobs(
                user_query=payload.question,
                resume_context=latest_analysis
            )

        else:
            answer = answer_question(payload.question)

        return {
            "question": payload.question,
            "answer": answer
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.get("/status")
def status():
    vectorstore_exists = os.path.exists("vectorstore/index.faiss")
    uploaded_files = os.listdir(UPLOAD_DIR) if os.path.exists(UPLOAD_DIR) else []
    
    return {
        "vectorstore_exists": vectorstore_exists,
        "uploaded_files": uploaded_files,
        "analysis_ready": latest_analysis is not None
    }

@app.delete("/clear")
def clear_all():
    """Clear all data"""
    global latest_analysis
    
    try:
        if os.path.exists("vectorstore"):
            shutil.rmtree("vectorstore")
        
        if os.path.exists(UPLOAD_DIR):
            for file in os.listdir(UPLOAD_DIR):
                os.remove(os.path.join(UPLOAD_DIR, file))
        
        latest_analysis = None
        
        return {"status": "success", "message": "All data cleared"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Clear failed: {str(e)}")