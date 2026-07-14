"""Auto-label relevance from structured captions: attr==1.0 => every parsed component satisfied.
Judge is model-independent, so dense ablations are unbiased. full_hybrid shares attr, so
hand-verify before treating the ablation table as final (hard-negative test is the clean proof)."""
import sys, pathlib, json
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import CAPTIONS_FILE, ROOT
from retriever.parser import parse
from retriever.search import FashionSearch

EVAL = ROOT / "eval"
caps = [json.loads(l) for l in open(CAPTIONS_FILE, encoding="utf-8")]
queries = json.load(open(EVAL / "queries.json"))
fs = FashionSearch.__new__(FashionSearch)   # reuse _attr without loading models

qrels = {}
for q in queries:
    parsed = parse(q["text"])
    rel = [c["id"] for c in caps if fs._attr(parsed, c) == 1.0]
    qrels[q["qid"]] = rel
    print(q["qid"], len(rel), "relevant  |", q["text"])
json.dump(qrels, open(EVAL / "qrels.json", "w"), indent=1)
