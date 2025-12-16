import os
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
RAPIDAPI_KEY = os.getenv("RAPIDAPI_KEY")

RAPIDAPI_HOST = "jsearch.p.rapidapi.com"
RAPIDAPI_URL = "https://jsearch.p.rapidapi.com/search"

TECH_ROLES = [
    "frontend developer",
    "backend developer",
    "full stack developer",
    "react developer",
    "web developer",
    "python developer",
    "nodejs developer",
    "software engineer",
    "data analyst",
    "data scientist",
    "machine learning engineer",
    "ai engineer",
]

TECH_KEYWORDS = [
    "engineer", "developer", "software",
    "frontend", "backend", "full stack",
    "data", "ml", "ai"
]


def fetch_jobs_from_api(query: str):
    """
    Fetch top 5 TECH jobs from RapidAPI JSearch
    """
    if not RAPIDAPI_KEY:
        return []

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST
    }

    params = {
        "query": query,
        "page": "1",
        "num_pages": "1",
        "date_posted": "week"
    }

    try:
        response = requests.get(
            RAPIDAPI_URL,
            headers=headers,
            params=params,
            timeout=10
        )

        if response.status_code != 200:
            print(f"RapidAPI error: {response.status_code}")
            return []

        data = response.json().get("data", [])
        jobs = []

        role_lower = query.replace(" jobs", "").lower()

        for job in data:
            title = (job.get("job_title") or "").lower()

            if not any(keyword in title for keyword in TECH_KEYWORDS):
                continue

            if role_lower not in title:
                continue

            jobs.append({
                "title": job.get("job_title"),
                "company": job.get("employer_name"),
                "location": job.get("job_location"),
                "employment_type": job.get("job_employment_type_text"),
                "remote": job.get("job_is_remote"),
                "posted": job.get("job_posted_human_readable"),
                "description": (job.get("job_description") or "")[:300] + "...",
                "apply_link": job.get("job_apply_link")
            })

            if len(jobs) == 5:
                break

        return jobs

    except Exception as e:
        print(f"Job fetch error: {str(e)}")
        return []


def search_jobs(user_query: str, resume_context: dict | None = None):
    """
    Resume-driven job search
    Role comes ONLY from resume analysis
    user_query is intentionally ignored
    """
    print("🔍 Job search triggered")

    role = "software developer"

    if resume_context and resume_context.get("primary_role"):
        detected_role = resume_context["primary_role"].lower()

        for tech_role in TECH_ROLES:
            if tech_role in detected_role:
                role = tech_role
                break

    search_query = f"{role} jobs"
    print(f"Final job search query: {search_query}")

    jobs = fetch_jobs_from_api(search_query)

    if not jobs:
        return (
            "No matching technical job openings were found for your resume role at the moment."
        )

    jobs_text = ""
    for idx, job in enumerate(jobs, 1):
        jobs_text += f"""
Job {idx}:
Title: {job['title']}
Company: {job['company']}
Location: {job['location']}
Employment Type: {job['employment_type']}
Remote: {"Yes" if job['remote'] else "No"}
Posted: {job['posted']}
Apply Link: {job['apply_link']}

Description:
{job['description']}
---

"""

    prompt = f"""
You are a job listing formatter.

STRICT RULES:
- DO NOT explain anything
- DO NOT analyze the resume
- DO NOT recommend jobs
- DO NOT add introductions or conclusions
- ONLY format the given jobs
- DO NOT invent information

Format EXACTLY as provided.

{jobs_text}
"""

    try:
        response = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=1500
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        print(f"LLM formatting error: {str(e)}")

        fallback = "Top matching technical jobs:\n\n"
        for i, job in enumerate(jobs, 1):
            fallback += (
                f"{i}. {job['title']} at {job['company']} "
                f"({job['location']})\n"
                f"Apply: {job['apply_link']}\n\n"
            )
        return fallback
