import os
import io
import json
import re
import tempfile
import shutil
import asyncio
from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

try:
    import faiss
    _HAS_FAISS = True
except Exception:
    faiss = None
    _HAS_FAISS = False

try:
    from sentence_transformers import CrossEncoder
    _HAS_RERANKER = True
except Exception:
    CrossEncoder = None
    _HAS_RERANKER = False

try:
    from rank_bm25 import BM25Okapi
    _HAS_BM25 = True
except Exception:
    BM25Okapi = None
    _HAS_BM25 = False

try:
    import fitz
    _HAS_FITZ = True
except Exception:
    fitz = None
    _HAS_FITZ = False

try:
    import nltk
    nltk.data.find("tokenizers/punkt")
except Exception:
    import nltk
    nltk.download("punkt", quiet=True)

try:
    from PIL import Image
    import pytesseract
    _HAS_OCR = True
except Exception:
    Image = None
    pytesseract = None
    _HAS_OCR = False

PDF_PATH = "data.pdf"
CHUNK_MIN_WORDS = 140
CHUNK_MAX_WORDS = 320
MODEL_NAME = "all-MiniLM-L6-v2"
TOP_K = 3
CACHE_DIR = Path("embedding_cache")
CACHE_DIR.mkdir(exist_ok=True)

def _stem(name: str) -> str:
    return Path(name).stem.replace(" ", "_")

app = FastAPI(title="Smart Knowledge Assistant (Enhanced RAG)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading sentence-transformers model:", MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

reranker = None
if _HAS_RERANKER:
    try:
        print("Loading cross-encoder reranker (may take time)...")
        reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    except Exception as e:
        print("Could not load reranker:", e)
        reranker = None

current_pdf_name: Optional[str] = None
raw_text: Optional[str] = None
chunks: List[str] = []
metadata: List[Dict[str, Any]] = []
chunk_embeddings: Optional[np.ndarray] = None
faiss_index = None
bm25_index = None

def save_cache(pdf_name: str, chunks_list: List[str], embeddings: np.ndarray, metadata_list: List[Dict[str, Any]]):
    stem = _stem(pdf_name)
    np.save(CACHE_DIR / f"{stem}_chunks.npy", np.array(chunks_list, dtype=object))
    np.save(CACHE_DIR / f"{stem}_emb.npy", embeddings)
    with open(CACHE_DIR / f"{stem}_meta.json", "w", encoding="utf-8") as f:
        json.dump(metadata_list, f, ensure_ascii=False)

def load_cache(pdf_name: str) -> Tuple[List[str], np.ndarray, List[Dict[str, Any]]]:
    stem = _stem(pdf_name)
    chunks_list = list(np.load(CACHE_DIR / f"{stem}_chunks.npy", allow_pickle=True))
    embeddings = np.load(CACHE_DIR / f"{stem}_emb.npy")
    with open(CACHE_DIR / f"{stem}_meta.json", "r", encoding="utf-8") as f:
        meta = json.load(f)
    return chunks_list, embeddings, meta

def faiss_save(index, pdf_name: str):
    if not _HAS_FAISS:
        return
    faiss.write_index(index, str(CACHE_DIR / f"{_stem(pdf_name)}_faiss.index"))

def faiss_load(pdf_name: str):
    if not _HAS_FAISS:
        return None
    p = CACHE_DIR / f"{_stem(pdf_name)}_faiss.index"
    if p.exists():
        return faiss.read_index(str(p))
    return None

def build_faiss_index(emb: np.ndarray):
    if not _HAS_FAISS:
        return None
    d = emb.shape[1]
    emb_norm = emb.copy()
    faiss.normalize_L2(emb_norm)
    index = faiss.IndexFlatIP(d)
    index.add(emb_norm)
    return index

def build_bm25(chunks_list: List[str]):
    global bm25_index
    if not _HAS_BM25:
        bm25_index = None
        return
    tokenized = [re.findall(r"\w+", c.lower()) for c in chunks_list]
    bm25_index = BM25Okapi(tokenized)

def extract_text_from_pdf(path: str) -> Tuple[List[str], List[int]]:
    if not _HAS_FITZ:
        raise RuntimeError("PyMuPDF (fitz) not installed. Install PyMuPDF to extract PDF text.")
    doc = fitz.open(path)
    page_texts = []
    page_nums = []
    for i, page in enumerate(doc):
        text = page.get_text() or ""
        if not text.strip() and _HAS_OCR:
            try:
                pix = page.get_pixmap(matrix=fitz.Matrix(2,2))
                img_bytes = pix.tobytes(output="png")
                img = Image.open(io.BytesIO(img_bytes))
                text = pytesseract.image_to_string(img)
            except Exception as e:
                text = ""
        page_texts.append(text)
        page_nums.append(i+1)
    return page_texts, page_nums

import nltk
from nltk.tokenize import sent_tokenize

def split_pages_to_chunks(page_texts: List[str], page_nums: List[int], min_words=CHUNK_MIN_WORDS, max_words=CHUNK_MAX_WORDS):
    chunks_out = []
    meta_out = []
    for text, page_num in zip(page_texts, page_nums):
        if not text or not text.strip():
            continue
        sentences = sent_tokenize(text)
        current = []
        current_words = 0
        for s in sentences:
            w = len(s.split())
            if w == 0:
                continue
            if current_words + w > max_words:
                # flush if current big enough, else attach and flush
                if current_words >= min_words:
                    chunks_out.append(" ".join(current).strip())
                    meta_out.append({"page": page_num})
                    current = [s]
                    current_words = w
                else:
                    current.append(s)
                    chunks_out.append(" ".join(current).strip())
                    meta_out.append({"page": page_num})
                    current = []
                    current_words = 0
            else:
                current.append(s)
                current_words += w
        if current:
            chunks_out.append(" ".join(current).strip())
            meta_out.append({"page": page_num})
    cleaned_chunks = []
    cleaned_meta = []
    for c, m in zip(chunks_out, meta_out):
        if len(c.split()) >= 20:
            cleaned_chunks.append(c)
            cleaned_meta.append(m)
    if not cleaned_chunks:
        return chunks_out[:5], meta_out[:5]
    return cleaned_chunks, cleaned_meta

def process_pdf(file_path: str, filename: str, force_reindex: bool=False) -> dict:
    global current_pdf_name, raw_text, chunks, metadata, chunk_embeddings, faiss_index

    stem = _stem(filename)
    cache_chunks_file = CACHE_DIR / f"{stem}_chunks.npy"
    cache_emb_file = CACHE_DIR / f"{stem}_emb.npy"
    cache_meta_file = CACHE_DIR / f"{stem}_meta.json"
    if not force_reindex and cache_chunks_file.exists() and cache_emb_file.exists() and cache_meta_file.exists():
        try:
            print("Loading from cache:", stem)
            chunks_list, embeddings, meta = load_cache(filename)
            chunks = chunks_list
            metadata = meta
            chunk_embeddings = embeddings
            faiss_index = faiss_load(filename) if _HAS_FAISS else None
            build_bm25(chunks)
            current_pdf_name = filename
            raw_text = " ".join(chunks)
            return {"filename": filename, "status": "loaded_from_cache", "chunks_count": len(chunks)}
        except Exception as e:
            print("Cache load failed, will re-index:", e)

    page_texts, page_nums = extract_text_from_pdf(file_path)
    if not any(p.strip() for p in page_texts):
        raise ValueError("PDF appears empty or contains no extractable text")

    new_chunks, new_meta = split_pages_to_chunks(page_texts, page_nums)
    if not new_chunks:
        raise ValueError("No chunks produced from PDF")

    print("Generating embeddings for", len(new_chunks), "chunks...")
    emb = model.encode(new_chunks, convert_to_numpy=True, show_progress_bar=True)
    chunk_embeddings = emb
    chunks = new_chunks
    metadata = new_meta
    raw_text = " ".join(page_texts)
    current_pdf_name = filename

    if _HAS_FAISS:
        try:
            faiss_index = build_faiss_index(chunk_embeddings)
            faiss_save(faiss_index, filename)
        except Exception as e:
            print("FAISS build failed:", e)
            faiss_index = None
    else:
        faiss_index = None

    if _HAS_BM25:
        build_bm25(chunks)
    else:
        pass

    try:
        save_cache(filename, chunks, chunk_embeddings, metadata)
    except Exception as e:
        print("Failed to save cache:", e)

    return {"filename": filename, "status": "processed", "chunks_count": len(chunks)}

def search_faiss(query: str, k:int=TOP_K*5) -> Tuple[List[int], List[float]]:
    if not _HAS_FAISS or faiss_index is None or chunk_embeddings is None:
        return [], []
    q_emb = model.encode([query], convert_to_numpy=True)
    q_norm = q_emb.copy()
    faiss.normalize_L2(q_norm)
    D, I = faiss_index.search(q_norm, k)
    indices = [int(i) for i in I[0] if i != -1]
    scores = [float(s) for s in D[0][:len(indices)]]
    return indices, scores

def search_bm25(query: str, k:int=TOP_K*5) -> List[int]:
    if not _HAS_BM25 or bm25_index is None:
        return []
    tokenized = re.findall(r"\w+", query.lower())
    scores = bm25_index.get_scores(tokenized)
    idxs = list(np.argsort(-scores)[:k])
    return idxs

def hybrid_search(query: str, top_k:int=TOP_K) -> List[Tuple[int, float]]:
    faiss_idx, faiss_scores = search_faiss(query, k=top_k*5)
    bm25_idx = search_bm25(query, k=top_k*5)
    merged = []
    seen = set()
    for i, s in zip(faiss_idx, faiss_scores):
        if i not in seen:
            merged.append((i, s))
            seen.add(i)
    for i in bm25_idx:
        if i not in seen:
            merged.append((i, 0.0))
            seen.add(i)

    candidates = merged[:top_k*5]
    if reranker is not None and candidates:
        pairs = [(query, chunks[i]) for i, _ in candidates]
        try:
            scores = reranker.predict(pairs)
            reranked = sorted(zip([i for i,_ in candidates], scores), key=lambda x: x[1], reverse=True)
            return [(int(i), float(s)) for i, s in reranked[:top_k]]
        except Exception as e:
            print("Reranker failed, fallback to cosine:", e)
    if chunk_embeddings is not None and candidates:
        cand_idxs = [i for i,_ in candidates]
        q_emb = model.encode([query], convert_to_numpy=True)[0]
        sims = cosine_similarity(q_emb.reshape(1,-1), chunk_embeddings[cand_idxs]).flatten()
        scored = sorted(zip(cand_idxs, sims.tolist()), key=lambda x: x[1], reverse=True)
        return [(int(i), float(s)) for i, s in scored[:top_k]]
    return []

def simulate_answer(query: str, retrieved_indices: List[int]) -> Tuple[str, List[Dict[str,Any]]]:
    q_tokens = [w.lower() for w in re.findall(r"\w+", query) if len(w) > 2]
    sentences = []
    sources = []
    for idx in retrieved_indices:
        chunk = chunks[idx]
        sents = re.split(r'(?<=[.!?])\s+', chunk)
        matched = []
        for s in sents:
            lw = s.lower()
            if any(qt in lw for qt in q_tokens):
                matched.append(s.strip())
        if matched:
            sentences.extend(matched[:3])
        else:
            sentences.extend(sents[:2])
        meta = metadata[idx] if idx < len(metadata) else {}
        sources.append({"index": idx, "page": meta.get("page"), "file": current_pdf_name})
        if len(sentences) >= 6:
            break
    if sentences:
        answer = " ".join(sentences)[:2000]
    else:
        if retrieved_indices:
            first = chunks[retrieved_indices[0]]
            answer = first.split(".")[:2]
            answer = ". ".join(answer).strip()
        else:
            answer = "Sorry, I couldn't find relevant information in the indexed documents."
    return answer, sources

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file.")
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        try:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {e}")
    try:
        result = process_pdf(tmp_path, file.filename)
        return JSONResponse({"message":"uploaded", "result": result})
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing error: {e}")
    finally:
        try:
            os.unlink(tmp_path)
        except:
            pass

@app.post("/ask")
async def ask(request: Request):
    payload = await request.json()
    query = payload.get("query")
    if not query or not isinstance(query, str) or query.strip() == "":
        raise HTTPException(status_code=400, detail="Please provide a non-empty 'query' string in JSON body.")
    if not chunks or chunk_embeddings is None:
        raise HTTPException(status_code=500, detail="No PDF indexed. Upload a PDF via /upload or place data.pdf and restart.")
    candidates = hybrid_search(query, top_k=TOP_K)
    indices = [i for i, _ in candidates]
    answer, sources = simulate_answer(query, indices)
    retrieved_previews = []
    for idx, score in candidates:
        retrieved_previews.append({
            "index": int(idx),
            "score": float(score),
            "preview": chunks[int(idx)][:400] + ("..." if len(chunks[int(idx)]) > 400 else ""),
            "page": metadata[int(idx)].get("page") if int(idx) < len(metadata) else None
        })
    return JSONResponse({"query": query, "answer": answer, "sources": sources, "retrieved": retrieved_previews})

@app.post("/ask_stream")
async def ask_stream(request: Request):
    payload = await request.json()
    query = payload.get("query")
    if not query:
        raise HTTPException(status_code=400, detail="Please provide 'query' in JSON body.")
    if not chunks or chunk_embeddings is None:
        raise HTTPException(status_code=500, detail="No PDF indexed. Upload a PDF via /upload or place data.pdf and restart.")
    candidates = hybrid_search(query, top_k=TOP_K)
    indices = [i for i, _ in candidates]
    answer, sources = simulate_answer(query, indices)

    async def streamer():
        meta = {"type":"meta", "query": query, "sources": sources}
        yield f"data: {json.dumps(meta)}\n\n"
        words = answer.split()
        for i, w in enumerate(words):
            chunk = {"type":"chunk", "content": w + (" " if i < len(words)-1 else ""), "is_final": i == len(words)-1}
            yield f"data: {json.dumps(chunk)}\n\n"
            await asyncio.sleep(0.03)
        yield f"data: {json.dumps({'type':'done','full_answer':answer})}\n\n"
    return StreamingResponse(streamer(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status":"ok","current_pdf": current_pdf_name, "chunks_indexed": len(chunks)}

@app.get("/list_cached")
async def list_cached():
    files = []
    for p in CACHE_DIR.glob("*_chunks.npy"):
        name = p.stem.replace("_chunks", "")
        files.append(name)
    return {"cached": files, "has_faiss": _HAS_FAISS, "has_bm25": _HAS_BM25, "has_reranker": _HAS_RERANKER, "has_ocr": _HAS_OCR}

def initialize_default():
    if os.path.exists(PDF_PATH):
        try:
            print("Initializing default PDF:", PDF_PATH)
            process_pdf(PDF_PATH, Path(PDF_PATH).name)
        except Exception as e:
            print("Default init failed:", e)

initialize_default()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)