# Multimodal Fashion Retrieval — Decompose-and-Verify Hybrid

Dense recall (Marqo-FashionSigLIP) + compositional verification: min-pooled clause
scoring + exact attribute matching against a SmolVLM-built structured index + BM25.
Binding (color↔garment) is verified explicitly instead of inferred from a global
bag-of-words embedding, which is where vanilla CLIP fails.

```
QUERY "a red tie and a white shirt in a formal setting"
  parser  -> clauses [(red,tie),(white,shirt)] + style=formal      (lexicon, zero-LLM)
  SigLIP  -> full-query emb -> FAISS top-50 (recall)
          -> per-clause emb  -> min-pool over clause sims (binding penalty)
  attr    -> query pairs vs SmolVLM caption attributes (same-item co-occurrence)
  bm25    -> over captions
  final = 0.35*dense + 0.25*min_clause + 0.25*attr + 0.15*bm25 -> top-k
```

## Quickstart — dashboard (1 command)

```bash
pip install -r requirements.txt
python server.py
```

Opens `http://127.0.0.1:8000` in your browser automatically. Type a query, get the top-5
images with score + binding badge. Model and index load once at startup (a few seconds);
each search only encodes the query and rescores the FAISS top-50 candidates, so per-query
latency stays in the **~150–250ms** range on CPU, measured end-to-end in the response
(`latency_ms` field / shown live in the UI).

## Rebuilding the index from scratch

```bash
python indexer/01_dataset.py             # Fashionpedia (local) + optional Pexels top-up
python indexer/03_encode.py --baseline   # SigLIP + CLIP image embeddings (GPU)
python indexer/04_build_index.py         # FAISS flat (cosine)
python indexer/02_caption.py             # SmolVLM captions -> structured attrs (incremental)
python -m retriever.cli "a red gown on a runway"
python -m eval.make_qrels                # auto-labels from captions
python -m eval.run_ablation              # CLIP vs SigLIP vs +minpool vs hybrid
python -m eval.hard_negative             # color-swap binding accuracy (the real proof)
```

Pexels top-up requires `PEXELS_API_KEY` in a `.env` file (free tier, no server-side cost or
query-time latency — it only runs once at index time). Without a key, `01_dataset.py` is a
no-op and the corpus stays Fashionpedia-only.

## Corpus

**1150 images**: 1000 Fashionpedia (`config.N_IMAGES`, the spec's design scale) + 150
Pexels top-up targeting the environments Fashionpedia lacks (office, park, home, raincoat).
`config.image_paths()` is the single source of truth for this set — stateless and
deterministic, so re-running the indexer never silently drifts the corpus size.

| Environment | Before top-up | After top-up |
|---|---|---|
| Office | 0 | 12 |
| Home | ~2 | 30 |
| Park | 5 | 58 |
| Street | 15 | 155 |

## Deltas from the original Colab-T4 spec (adapted to this machine)
- **transformers 5.x**: `AutoModelForVision2Seq` → `AutoModelForImageTextToText`.
- **SmolVLM-500M won't emit reliable JSON.** It captions well, so `02_caption.py` takes
  a prose caption and derives the structured `{garments,environment,style}` index by
  reusing the query parser's lexicon — deterministic, model-agnostic, no JSON fragility.
- **4 GB laptop GPU (RTX 3050)**: image batch 32, `num_workers=0` (Windows).
- **ONNX int8 export deferred**: CUDA present and `encoder.py` falls back to torch-CPU
  when `text_encoder.int8.onnx` is absent; export buys only query-time RAM, not needed here.

## Results (see `Fashion_Retrieval_Report.pdf` for full writeup)

Hard-negative compositional binding test (the PRD's "red shirt/blue pants" case — 80
image pairs, correct query must outrank its color-swapped negative):

| Scorer | Binding accuracy |
|---|---|
| vanilla CLIP | 0.450 (≈ chance) |
| SigLIP dense | 0.613 |
| **full hybrid** | **1.000** |

PRD eval-query ablation (qrels auto-labeled from captions; q2/q4/q5 still sparse — see
report §1 for why, and the hard-negative test above for the model-independent proof):

| Config | P@5 | P@10 | nDCG@10 |
|---|---|---|---|
| vanilla_clip | 0.160 | 0.180 | 0.228 |
| siglip_dense | 0.200 | 0.200 | 0.250 |
| siglip_minpool | 0.200 | 0.220 | 0.264 |
| **full_hybrid** | **0.280** | **0.240** | **0.400** |

## Repo layout

```
server.py             FastAPI dashboard — the "1 command" entry point
static/index.html     minimal search UI (vanilla JS, no build step)
indexer/   01_dataset (Fashionpedia + Pexels top-up) · 02_caption (SmolVLM->structured,
           incremental) · 03_encode (image embeddings) · 04_build_index (FAISS)
retriever/ parser (zero-LLM decomposition) · encoder · search (hybrid scorer) · cli
eval/      queries · make_qrels · metrics · run_ablation · hard_negative
config.py  all paths/weights/image-selection — logic fully separated from data
```
