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

## Run
```bash
python indexer/03_encode.py --baseline   # SigLIP + CLIP image embeddings (GPU)
python indexer/04_build_index.py         # FAISS flat (cosine)
python indexer/02_caption.py             # SmolVLM captions -> structured attrs
python -m retriever.cli "a red gown on a runway"
python -m eval.make_qrels                # auto-labels from captions
python -m eval.run_ablation              # CLIP vs SigLIP vs +minpool vs hybrid
python -m eval.hard_negative             # color-swap binding accuracy (the real proof)
```

## Deltas from the original Colab-T4 spec (adapted to this machine)
- **Data**: 3200 Fashionpedia jpgs already local in `test/`; skipped `01_dataset.py`
  download. Indexed a **1000-image subset** (`config.N_IMAGES`) — the spec's own design
  scale — keeping captions/embeddings/index aligned on one subset.
- **transformers 5.x**: `AutoModelForVision2Seq` → `AutoModelForImageTextToText`.
- **SmolVLM-500M won't emit reliable JSON.** It captions well, so `02_caption.py` takes
  a prose caption and derives the structured `{garments,environment,style}` index by
  reusing the query parser's lexicon — deterministic, model-agnostic, no JSON fragility.
- **4 GB laptop GPU (RTX 3050)**: image batch 32, `num_workers=0` (Windows).
- **ONNX int8 export deferred**: CUDA present and `encoder.py` falls back to torch-CPU
  when `text_encoder.int8.onnx` is absent; export buys only query-time RAM, not needed here.

## Corpus caveat
This is a **runway fashion** dataset. The stock eval queries (office, park bench,
raincoat, city walk) were designed for the skipped Pexels top-up, so their qrels are
sparse — `run_ablation` prints per-query counts. The **hard-negative binding test** is
the corpus-appropriate, model-independent proof of the compositionality claim.
```
