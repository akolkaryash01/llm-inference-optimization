"""
LLM Inference Optimization - Full Backend
Run: python app.py
Then open: http://localhost:8000
"""

from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_cpp import Llama
import time, hashlib, json, os, threading
import uvicorn

app = FastAPI(title="LLM Inference Optimization")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load model ──────────────────────────────────────────────
print("Loading Phi-2 model...")
llm = Llama(
    model_path="./models/phi-2.Q4_K_M.gguf",
    n_ctx=2048,
    n_threads=4,
    verbose=False
)
print("✅ Model loaded!")

# ── Cache ────────────────────────────────────────────────────
cache = {}
cache_hits = 0
total_requests = 0
total_tokens = 0
total_time = 0.0

# ── Request schema ───────────────────────────────────────────
class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.7
    use_cache: bool = True
    stream: bool = False

# ── Routes ───────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()

@app.post("/generate")
async def generate(req: GenerateRequest):
    global cache_hits, total_requests, total_tokens, total_time
    total_requests += 1

    # Cache check
    if req.use_cache:
        key = hashlib.md5(req.prompt.strip().lower().encode()).hexdigest()
        if key in cache:
            cache_hits += 1
            return {**cache[key], "cached": True}

    start = time.time()
    output = llm(
        req.prompt,
        max_tokens=req.max_tokens,
        temperature=req.temperature,
        echo=False
    )
    elapsed = time.time() - start
    text = output["choices"][0]["text"].strip()
    n_tokens = len(text.split())
    tps = round(n_tokens / elapsed, 1)

    total_tokens += n_tokens
    total_time += elapsed

    result = {
        "response": text,
        "latency_ms": round(elapsed * 1000),
        "tokens_per_sec": tps,
        "tokens_generated": n_tokens,
        "cached": False
    }

    if req.use_cache:
        cache[key] = result

    return result

@app.post("/stream")
async def stream_generate(req: GenerateRequest):
    """Stream tokens one by one like ChatGPT"""
    def token_gen():
        output = llm(
            req.prompt,
            max_tokens=req.max_tokens,
            temperature=req.temperature,
            echo=False,
            stream=True
        )
        for chunk in output:
            token = chunk["choices"][0]["text"]
            yield f"data: {json.dumps({'token': token})}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(token_gen(), media_type="text/event-stream")

@app.get("/benchmark")
async def benchmark():
    """Run live benchmark — threads 1,2,4,8"""
    results = {}
    prompt = "What is machine learning? Explain briefly."

    for threads in [1, 2, 4, 8]:
        m = Llama(
            model_path="./models/phi-2.Q4_K_M.gguf",
            n_ctx=256, n_threads=threads, verbose=False
        )
        start = time.time()
        out = m(prompt, max_tokens=40, echo=False)
        elapsed = time.time() - start
        tps = round(len(out["choices"][0]["text"].split()) / elapsed, 1)
        results[f"{threads}_threads"] = tps

    best = max(results, key=results.get)
    return {"thread_results": results, "best": best, "best_tps": results[best]}

@app.get("/stats")
async def stats():
    return {
        "total_requests": total_requests,
        "cache_hits": cache_hits,
        "cache_miss": total_requests - cache_hits,
        "cache_hit_rate": f"{round(cache_hits/max(total_requests,1)*100)}%",
        "cached_prompts": len(cache),
        "avg_tokens_per_sec": round(total_tokens / max(total_time, 0.001), 1),
        "total_tokens_generated": total_tokens,
    }

@app.delete("/cache")
async def clear_cache():
    cache.clear()
    return {"message": "Cache cleared"}

@app.get("/health")
async def health():
    return {"status": "ok", "model": "phi-2.Q4_K_M.gguf", "device": "cpu"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
