from fastapi import FastAPI
from pydantic import BaseModel, Field
from rag_engine import ask_llm

app = FastAPI(
    title="PICT InC RAG API",
    description="FAISS-based RAG API with strict answers and redirect support",
    version="1.0.0",
)

# -------------------- MODELS --------------------

class Query(BaseModel):
    question: str = Field(..., min_length=2, max_length=1000)

class Answer(BaseModel):
    answer: str
    redirect: bool = False
    redirect_url: str | None = None

# -------------------- ROUTES --------------------

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/ask", response_model=Answer)
def ask(query: Query):
    result = ask_llm(query.question)

    if result.get("redirect"):
        return {
            "answer": result["answer"],
            "redirect": True,
            "redirect_url": "https://pictinc.org"
        }

    return {
        "answer": result["answer"],
        "redirect": False
    }
