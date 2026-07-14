"""Ablation: vanilla CLIP vs SigLIP dense vs +min-pool vs full hybrid. Numbers, not claims."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, open_clip
from config import EMB_FILE, EMB_BASELINE_FILE, IDS_FILE, CAPTIONS_FILE, W_FULL, ROOT
from retriever.parser import parse
from retriever.encoder import QueryEncoder
from retriever.search import FashionSearch, _norm
from eval.metrics import precision_at_k, ndcg_at_k

EVAL = ROOT / "eval"
ids = json.load(open(IDS_FILE))
caps = {json.loads(l)["id"]: json.loads(l) for l in open(CAPTIONS_FILE, encoding="utf-8")}
qrels = json.load(open(EVAL / "qrels.json"))
queries = json.load(open(EVAL / "queries.json"))
fs = FashionSearch.__new__(FashionSearch)

print("qrel counts:", {q["qid"]: len(qrels[q["qid"]]) for q in queries})


def clip_text_encoder():
    m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32"); m.eval()
    def f(texts):
        with torch.inference_mode():
            e = m.encode_text(tok(texts))
            return torch.nn.functional.normalize(e, dim=-1).numpy().astype("float32")
    return f


siglip_enc = QueryEncoder().encode
CONFIGS = {
    "vanilla_clip":    dict(emb=np.load(EMB_BASELINE_FILE), enc=clip_text_encoder(), w=(1, 0, 0, 0)),
    "siglip_dense":    dict(emb=np.load(EMB_FILE), enc=siglip_enc, w=(1, 0, 0, 0)),
    "siglip_minpool":  dict(emb=np.load(EMB_FILE), enc=siglip_enc, w=(0.6, 0.4, 0, 0)),
    "full_hybrid":     dict(emb=np.load(EMB_FILE), enc=siglip_enc, w=W_FULL),
}


def run(cfg, qtext):
    parsed = parse(qtext)
    embs = cfg["enc"]([qtext] + parsed["clauses"])
    dense = cfg["emb"] @ embs[0]
    mc = (cfg["emb"] @ embs[1:].T).min(axis=1) if len(embs) > 1 else dense
    attr = np.array([fs._attr(parsed, caps[i]) for i in ids])
    w = cfg["w"]
    score = w[0] * _norm(dense) + w[1] * _norm(mc) + w[2] * attr   # bm25 omitted in ablation core
    return [ids[i] for i in np.argsort(-score)]


print(f"{'config':<16}{'P@5':>7}{'P@10':>7}{'nDCG@10':>9}")
for name, cfg in CONFIGS.items():
    p5, p10, nd = [], [], []
    for q in queries:
        ranked, rel = run(cfg, q["text"]), qrels[q["qid"]]
        p5.append(precision_at_k(ranked, rel, 5))
        p10.append(precision_at_k(ranked, rel, 10))
        nd.append(ndcg_at_k(ranked, rel, 10))
    print(f"{name:<16}{np.mean(p5):>7.3f}{np.mean(p10):>7.3f}{np.mean(nd):>9.3f}")
