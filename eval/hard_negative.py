"""Hard-negative compositional binding test — the model-independent proof.

The PRD hint's exact failure case: "red shirt with blue pants" vs "blue shirt with red pants".
For each image with two differently-colored garments, we build the CORRECT query and its
COLOR-SWAPPED negative and ask the scorer to rank the image higher under the correct query.
A bag-of-words global embedding (vanilla CLIP) sees the same token set either way => ~chance.
Explicit attribute verification binds color->garment => approaches 1.0.

Also reports the easier single-garment color test for context.
"""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import numpy as np, torch, open_clip
from config import EMB_FILE, EMB_BASELINE_FILE, IDS_FILE, CAPTIONS_FILE, W_FULL
from retriever.parser import parse
from retriever.encoder import QueryEncoder
from retriever.search import FashionSearch, _norm

ids = json.load(open(IDS_FILE))
pos = {v: i for i, v in enumerate(ids)}
caps = {json.loads(l)["id"]: json.loads(l) for l in open(CAPTIONS_FILE, encoding="utf-8")}
fs = FashionSearch.__new__(FashionSearch)

# images with two differently-colored, differently-typed garments
two = []
for iid, c in caps.items():
    g = [(x["color"], x["type"]) for x in c.get("garments", []) if x.get("color")]
    seen, uniq = set(), []
    for col, typ in g:
        if typ not in seen and col:
            seen.add(typ); uniq.append((col, typ))
    if len(uniq) >= 2 and uniq[0][0] != uniq[1][0]:
        two.append((iid, uniq[0], uniq[1]))
np.random.default_rng(0).shuffle(two)
two = two[:80]
print(f"compositional pairs (2-garment color swap): {len(two)} images")


def clip_enc():
    m, _, _ = open_clip.create_model_and_transforms("ViT-B-32", pretrained="openai")
    tok = open_clip.get_tokenizer("ViT-B-32"); m.eval()
    def f(texts):
        with torch.inference_mode():
            e = m.encode_text(tok(texts))
            return torch.nn.functional.normalize(e, dim=-1).numpy().astype("float32")
    return f


siglip = QueryEncoder().encode
SCORERS = {
    "vanilla_clip": dict(emb=np.load(EMB_BASELINE_FILE), enc=clip_enc(), w=(1, 0, 0, 0)),
    "siglip_dense": dict(emb=np.load(EMB_FILE), enc=siglip, w=(1, 0, 0, 0)),
    "full_hybrid":  dict(emb=np.load(EMB_FILE), enc=siglip, w=W_FULL),
}


def value(cfg, qtext, iid):
    parsed = parse(qtext)
    embs = cfg["enc"]([qtext] + parsed["clauses"])
    dense = cfg["emb"] @ embs[0]
    mc = (cfg["emb"] @ embs[1:].T).min(axis=1) if len(embs) > 1 else dense
    attr = np.array([fs._attr(parsed, caps[i]) for i in ids])
    w = cfg["w"]
    s = w[0] * _norm(dense) + w[1] * _norm(mc) + w[2] * attr
    return s[pos[iid]]


def binding_acc(cfg):
    ok = 0
    for iid, (c1, t1), (c2, t2) in two:
        correct = f"a {c1} {t1} and a {c2} {t2}"
        swapped = f"a {c2} {t1} and a {c1} {t2}"          # same tokens, colors swapped
        ok += value(cfg, correct, iid) > value(cfg, swapped, iid)
    return ok / len(two)


print(f"{'scorer':<16}{'binding_acc':>12}")
for name, cfg in SCORERS.items():
    print(f"{name:<16}{binding_acc(cfg):>12.3f}")
