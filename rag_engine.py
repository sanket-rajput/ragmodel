import faiss
import pickle
import os
import re
import difflib
import logging
from functools import lru_cache
from typing import List
import numpy as np
import requests
from fastembed import TextEmbedding
from dotenv import load_dotenv

# --------------------------------------------------
# ENV + CONFIG
# --------------------------------------------------

load_dotenv()
logger = logging.getLogger("rag_engine")

OPENROUTER_KEYS = [
    k.strip() for k in os.getenv("OPENROUTER_KEYS", "").split(",") if k.strip()
]
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL", "deepseek/deepseek-r1:free")
OPENROUTER_TEMPERATURE = float(os.getenv("OPENROUTER_TEMPERATURE", "0.2"))
OPENROUTER_TIMEOUT = int(os.getenv("OPENROUTER_TIMEOUT", "30"))
REDIRECT_URL = os.getenv("REDIRECT_URL", "https://pictinc.org")

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
INDEX_PATH = "faiss_store/index.faiss"
META_PATH = "faiss_store/meta.pkl"
TOP_K = int(os.getenv("RAG_TOP_K", "6"))

# --------------------------------------------------
# LOADERS (CACHED)
# --------------------------------------------------

@lru_cache(maxsize=1)
def _embedder():
    return TextEmbedding(model_name=EMBEDDING_MODEL)

@lru_cache(maxsize=1)
def _index():
    return faiss.read_index(INDEX_PATH)

@lru_cache(maxsize=1)
def _documents():
    with open(META_PATH, "rb") as f:
        return pickle.load(f)

# --------------------------------------------------
# UTILS
# --------------------------------------------------

STOPWORDS = {
    "the","a","an","and","or","to","of","in","on","for","is","are","was","were",
    "be","by","with","as","at","from","that","this","it","its","their","your",
    "you","we","our","they","them","will","should","can","may","must","not"
}

def _tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"\b\w+\b", text.lower()) if t not in STOPWORDS]

def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())

def _is_greeting(text: str) -> bool:
    return _normalize(text) in {"hi","hello","hey","hi there","hello there"}

def _clean(text: str) -> str:
    text = re.sub(r"[*_`]+", "", text)
    return re.sub(r"\s+", " ", text).strip()

def _to_one_liner(text: str) -> str:
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return sentences[0] if sentences else text

# --------------------------------------------------
# VECTOR SEARCH
# --------------------------------------------------

def _embed(texts: List[str]) -> np.ndarray:
    vecs = list(_embedder().embed(texts))
    return np.vstack(vecs).astype("float32")

def retrieve(query: str) -> List[str]:
    q_vec = _embed([query])
    docs = _documents()
    index = _index()

    scores, indices = index.search(q_vec, TOP_K * 3)
    ranked = []

    for i, idx in enumerate(indices[0]):
        doc = docs[idx]
        vec_score = float(scores[0][i])
        token_score = len(set(_tokenize(query)) & set(_tokenize(doc)))
        fuzzy = difflib.SequenceMatcher(None, _normalize(query), _normalize(doc)).ratio()
        score = (vec_score * 0.6) + (token_score * 0.25) + (fuzzy * 0.15)
        ranked.append((score, doc))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:TOP_K]]

# --------------------------------------------------
# FALLBACK (NO LLM)
# --------------------------------------------------

def _extract_answer(question: str, blocks: List[str]) -> str:
    q_tokens = set(_tokenize(question))
    best, best_score = None, 0

    for block in blocks:
        for sent in re.split(r"(?<=[.!?])\s+", block):
            s_tokens = set(_tokenize(sent))
            score = len(q_tokens & s_tokens)
            if score > best_score:
                best, best_score = sent.strip(), score

    return best if best_score > 0 else "Information not available"

# --------------------------------------------------
# PROMPT (YOUR EXACT PROMPT)
# --------------------------------------------------

def _build_prompt(question: str, context: str) -> str:
    return f"""
You are a helpful hackathon information assistant.

RULES (must follow exactly):
- Use ONLY the provided CONTEXT to answer.
- If the answer cannot be found in CONTEXT, reply exactly: "Information not available".
- For short/simple factual questions, reply concisely (default ≤ 3 sentences).
- For broad or explicitly requested detailed answers (question contains words like "detail", "explain", "steps", "how to", "comprehensive", "full", "long"), return a structured, longer response with headings and bullet points.
- If the user's input is a simple greeting (e.g., "hi", "hello"), reply exactly: "Hello, this is the PICT InC Assistant. How can I help you?"
- If the user's question has typos or poor grammar, interpret intent and answer as if corrected.
- If the question is unclear and the CONTEXT does not resolve ambiguity, ask one short clarification question.
- Do NOT repeat or mirror the CONTEXT or the QUESTION in the answer.
- Use clear, human-friendly language. Prefer bullets or numbered steps when helpful.
- Always keep the reply focused and relevant to the user's QUESTION.

CONTEXT:
{context}

QUESTION:
{question}
""".strip()

# --------------------------------------------------
# OPENROUTER (KEY ROTATION)
# --------------------------------------------------

def _call_openrouter(payload: dict) -> str | None:
    for key in OPENROUTER_KEYS:
        try:
            r = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json"
                },
                json=payload,
                timeout=OPENROUTER_TIMEOUT
            )
            if r.status_code != 200:
                continue

            data = r.json()
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if content:
                return content.strip()
        except requests.RequestException:
            continue
    return None

# --------------------------------------------------
# MAIN ENTRY
# --------------------------------------------------

def ask_llm(question: str) -> dict:
    if not question or not question.strip():
        return {"answer": "Information not available", "redirect": True}

    if _is_greeting(question):
        return {
            "answer": "Hello, this is the PICT InC Assistant. How can I help you?",
            "redirect": False
        }

    context_blocks = retrieve(question)
    context = "\n\n".join(context_blocks).strip()

    if not context:
        return {"answer": "Information not available", "redirect": True}

    prompt = _build_prompt(question, context)

    payload = {
        "model": OPENROUTER_MODEL,
        "temperature": OPENROUTER_TEMPERATURE,
        "messages": [
            {"role": "system", "content": "Answer strictly from context."},
            {"role": "user", "content": prompt}
        ]
    }

    llm_answer = _call_openrouter(payload)

    # ❌ LLM unavailable → fallback
    if not llm_answer:
        fallback = _extract_answer(question, context_blocks)
        return {
            "answer": _clean(fallback),
            "redirect": fallback == "Information not available"
        }

    # ❌ Explicit not-found
    if llm_answer.lower().strip() == "information not available":
        return {"answer": llm_answer, "redirect": True}

    # ✅ FORCE one-liner unless explicitly asked for detail
    if not any(
        kw in question.lower()
        for kw in ["detail", "explain", "steps", "how to", "comprehensive", "full", "long"]
    ):
        llm_answer = _to_one_liner(llm_answer)

    return {"answer": _clean(llm_answer), "redirect": False}
