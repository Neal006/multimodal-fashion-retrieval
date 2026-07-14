from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "test"            # images already present locally
ART = ROOT / "artifacts"
ART.mkdir(parents=True, exist_ok=True)

SIGLIP_ID = "hf-hub:Marqo/marqo-fashionSigLIP"   # 203M, Apache-2.0
CLIP_BASELINE = ("ViT-B-32", "openai")           # ablation baseline
VLM_ID = "HuggingFaceTB/SmolVLM-500M-Instruct"   # ~1GB, Apache-2.0

EMB_FILE = ART / "embeddings_siglip.npy"
EMB_BASELINE_FILE = ART / "embeddings_clip.npy"
IDS_FILE = ART / "image_ids.json"
CAPTIONS_FILE = ART / "captions.jsonl"
FAISS_FILE = str(ART / "index_siglip.faiss")
ONNX_FILE = ART / "text_encoder.int8.onnx"       # deferred: torch-cpu fallback used

N_IMAGES = 1000   # Fashionpedia base (spec design scale); see image_paths() for the full set

TOP_CAND = 50
TOP_K = 5

# scoring weights: dense, min_clause, attr, bm25
W_FULL = (0.35, 0.25, 0.25, 0.15)
W_NO_CLAUSE = (0.60, 0.00, 0.20, 0.20)


def image_paths():
    """Fixed base: first N_IMAGES Fashionpedia images (alphabetical over hex filenames),
    plus every Pexels top-up image ('px_*.jpg'). Stateless and deterministic — the ~2200
    unindexed Fashionpedia images stay excluded regardless of what's already on disk."""
    fashionpedia = sorted(p for p in DATA_DIR.glob("*.jpg") if not p.name.startswith("px_"))
    topup = sorted(DATA_DIR.glob("px_*.jpg"))
    return fashionpedia[:N_IMAGES] + topup
