# AI Resume Analyzer with Job Suggestions

A RAG-based system that analyzes resumes, generates ATS scores, provides improvement suggestions, and recommends relevant jobs.

## Features

✅ **Resume Analysis**
- Upload PDF resume
- Automatic ATS score generation (0-100)
- Identify key strengths
- Get actionable improvement suggestions

✅ **Q&A System**
- Ask questions about the uploaded resume
- RAG-powered answers from resume content only
- No hallucinations or made-up information

✅ **Job Search**
- Ask for job recommendations
- Fetches real jobs from external API
- AI-formatted job listings with apply links

## Architecture

```
1. Upload Resume → Chunking → Embeddings → Vector Store
2. Auto-analyze → Generate ATS Score + Suggestions
3. User Q&A → RAG retrieves from resume
4. Job queries → Fetch from API → LLM formats response
```

## Tech Stack

**Backend:**
- FastAPI
- LangChain
- FAISS (Vector Store)
- FastEmbed (Embeddings)
- Groq (LLM)
- JSearch API (Job Search)

**Frontend:**
- React + Vite
- Tailwind CSS
- Lucide React Icons
- React Toastify

## Setup

### 1. Clone Repository

```bash
git clone <your-repo>
cd resume-analyzer
```

### 2. Install Backend Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
cp .env.example .env
# Edit .env and add your API keys
```

**Required:**
- `GROQ_API_KEY` - Get from [Groq Console](https://console.groq.com)

**Optional:**
- `RAPIDAPI_KEY` - Get from [RapidAPI JSearch](https://rapidapi.com/letscrape-6bRBa3QguO5/api/jsearch)
  - If not provided, mock job data will be used

### 4. Run Backend

```bash
uvicorn app:app --reload
```

Backend runs at: `http://localhost:8000`

### 5. Setup Frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend runs at: `http://localhost:5173`

## API Endpoints

### `POST /upload`
Upload resume PDF
```json
{
  "status": "uploaded",
  "filename": "resume.pdf",
  "message": "Resume is being analyzed..."
}
```

### `GET /analysis`
Get resume analysis result
```json
{
  "status": "complete",
  "analysis": {
    "ats_score": 75,
    "strengths": ["Strong technical skills", "..."],
    "suggestions": ["Add more action verbs", "..."],
    "summary": "Experienced software engineer..."
  }
}
```

### `POST /ask`
Ask questions about resume or search jobs
```json
{
  "question": "What are my key skills?"
}
```

Response:
```json
{
  "question": "What are my key skills?",
  "answer": "Based on your resume, your key skills include..."
}
```

### `GET /status`
Check system status
```json
{
  "vectorstore_exists": true,
  "uploaded_files": ["resume.pdf"],
  "analysis_ready": true
}
```

### `DELETE /clear`
Clear all data
```json
{
  "status": "success",
  "message": "All data cleared"
}
```

## Usage Flow

### 1. Upload Resume
- Click upload button
- Select PDF resume
- Wait for processing (~30-60 seconds)

### 2. View Analysis
- ATS Score displayed automatically
- See strengths and suggestions
- Review detailed analysis

### 3. Ask Questions
Examples:
- "What are my key skills?"
- "Summarize my work experience"
- "What certifications do I have?"

### 4. Find Jobs
Examples:
- "Find me software engineer jobs"
- "Search for remote developer positions"
- "Show me job opportunities"

## Project Structure

```
resume-analyzer/
├── app.py              # FastAPI main 
├── ingest.py           # Resume 
├── query.py            # ATS analysis & Q&A
├── jobs.py             # Job search integration
├── requirements.txt    # Python dependencies
├── .env.example        # Environment variables template
├── uploads/            # Uploaded resumes (auto-created)
├── vectorstore/        # FAISS index (auto-created)
```

## Deployment

### Railway (Backend)

1. Push code to GitHub
2. Connect Railway to your repo
3. Add environment variables:
   - `GROQ_API_KEY`
   - `RAPIDAPI_KEY` (optional)
4. Deploy!

### Vercel (Frontend)

1. Build frontend: `npm run build`
2. Deploy to Vercel
3. Update API URL in frontend code

## Customization

### Change Embedding Model

In `ingest.py` and `query.py`:
```python
embeddings = FastEmbedEmbeddings(
    model_name="your-model-name",
    max_length=256
)
```

### Change LLM Model

In `query.py` and `jobs.py`:
```python
model="llama-3.1-70b-versatile"  # or other Groq models
```

### Adjust ATS Scoring

Modify `ATS_ANALYSIS_PROMPT` in `query.py` to change scoring criteria.

### Add More Job APIs

Add more job sources in `jobs.py`:
- Indeed API
- LinkedIn API
- GitHub Jobs
- RemoteOK API

## Troubleshooting

**Issue: Processing takes too long**
- Reduce `MAX_CHUNKS` in `ingest.py`
- Use smaller embedding model

**Issue: Low ATS score**
- Check resume format (PDF quality)
- Ensure proper sections (Skills, Experience, etc.)

**Issue: No jobs found**
- Verify `RAPIDAPI_KEY` is set
- Check API quota/limits
- Falls back to mock data if API fails

## Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## License

MIT License
---
Built with ❤️ using FastAPI, LangChain, and Groq
