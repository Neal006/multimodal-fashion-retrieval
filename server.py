"""FastAPI dashboard server. Model + index load once at startup; each request only
encodes the query and rescores the FAISS top-50 candidates — that's what keeps
per-query latency low (no reload, no GPU round-trip at query time).

Run: python server.py  (single command, opens the dashboard in your browser)
"""
import sys, pathlib, time, threading, webbrowser
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from config import DATA_DIR, TOP_K
from retriever.search import FashionSearch

ROOT = pathlib.Path(__file__).resolve().parent
HOST, PORT = "127.0.0.1", 8000

app = FastAPI(title="Fashion Retrieval")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_search: FashionSearch | None = None


@app.on_event("startup")
def _load():
    global _search
    t0 = time.time()
    _search = FashionSearch()
    print(f"[server] model + index loaded in {time.time() - t0:.1f}s, ready at http://{HOST}:{PORT}")


class Hit(BaseModel):
    id: str
    url: str
    score: float
    dense: float
    min_clause: float
    attr: float
    caption: str


class SearchResponse(BaseModel):
    query: str
    latency_ms: float
    results: list[Hit]


@app.get("/api/search", response_model=SearchResponse)
def search(q: str, k: int = TOP_K):
    t0 = time.time()
    results = _search.search(q, k)
    for r in results:
        r["url"] = f"/images/{r['id']}"
    latency_ms = round((time.time() - t0) * 1000, 1)
    return {"query": q, "latency_ms": latency_ms, "results": results}


app.mount("/images", StaticFiles(directory=str(DATA_DIR)), name="images")
app.mount("/static", StaticFiles(directory=str(ROOT / "static")), name="static")


@app.get("/")
def index():
    return FileResponse(str(ROOT / "static" / "index.html"))


def _open_browser():
    webbrowser.open(f"http://{HOST}:{PORT}")


if __name__ == "__main__":
    import uvicorn
    threading.Timer(1.2, _open_browser).start()
    uvicorn.run(app, host=HOST, port=PORT, log_level="warning")
