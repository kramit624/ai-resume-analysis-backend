import os
from dotenv import load_dotenv
from groq import Groq
import json
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_core.prompts import PromptTemplate

load_dotenv()

VECTORSTORE_DIR = "vectorstore"
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


ATS_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
You are an ATS resume analyzer.

Context (resume content):
{context}

Tasks:
1. List missing or weak technical skills
2. Suggest clear, actionable improvements
3. Be honest and critical (no sugar-coating)
4. Do NOT add information not present in the resume
5. DO NOT calculate ATS score

Respond strictly in this JSON format:
{{
  "missing_skills": [string],
  "suggestions": [string],
  "summary": string
}}
"""
)


ROLE_PROMPT = PromptTemplate(
    input_variables=["context"],
    template="""
Based ONLY on the resume content below, identify the SINGLE most suitable technical job role.

Rules:
- One role only
- Must be a technical role
- No explanation

Resume:
{context}

Respond in JSON:
{{ "primary_role": "string" }}
"""
)

QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are answering questions based ONLY on the resume content below.

Context:
{context}

Question: {question}

Rules:
- Answer ONLY from the context
- If information is missing, say so clearly
- Do NOT assume or hallucinate
- Be concise

Answer:
"""
)


def load_vectorstore():
    if not os.path.exists(VECTORSTORE_DIR):
        return None

    embeddings = FastEmbedEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        max_length=256
    )

    return FAISS.load_local(
        VECTORSTORE_DIR,
        embeddings,
        allow_dangerous_deserialization=True
    )



def calculate_ats_score(context: str) -> int:
    context = context.lower()

    score = 0


    skills = [
        "python", "java", "sql", "api", "docker", "aws",
        "react", "node", "machine learning", "data"
    ]
    score += min(sum(1 for s in skills if s in context) * 5, 40)


    if "experience" in context or "years" in context:
        score += 25

    if "project" in context or "developed" in context:
        score += 20

    if "bachelor" in context or "master" in context or "certification" in context:
        score += 15

    return min(score, 100)


def analyze_resume():
    vectorstore = load_vectorstore()
    if not vectorstore:
        return None

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 12}
    )

    docs = retriever.invoke("skills experience projects technologies")
    context = "\n\n".join(d.page_content for d in docs)

    prompt = ATS_PROMPT.format(context=context)

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=700
    )

    raw_output = response.choices[0].message.content.strip()

    try:
        analysis = json.loads(raw_output)
    except Exception:
        print("JSON parse failed, raw output:")
        print(raw_output)
        analysis = {
            "missing_skills": [],
            "suggestions": ["Unable to generate suggestions due to formatting issue."],
            "summary": "Resume analysis could not be completed properly."
        }


    analysis["ats_score"] = calculate_ats_score(context)


    role_response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{
            "role": "user",
            "content": ROLE_PROMPT.format(context=context)
        }],
        temperature=0.0,
        max_tokens=50
    )

    try:
        role_data = json.loads(role_response.choices[0].message.content.strip())
        analysis["primary_role"] = role_data.get("primary_role", "Software Developer")
    except Exception:
        analysis["primary_role"] = "Software Developer"

    return analysis



def answer_question(question: str):
    vectorstore = load_vectorstore()
    if not vectorstore:
        return "No resume uploaded yet."

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 6}
    )

    docs = retriever.invoke(question)

    if not docs:
        return "This information is not present in the resume."

    context = "\n\n".join(d.page_content for d in docs)
    prompt = QA_PROMPT.format(context=context, question=question)

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=500
    )

    return response.choices[0].message.content.strip()
