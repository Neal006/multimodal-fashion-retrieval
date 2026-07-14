import json
import faiss, numpy as np
from rank_bm25 import BM25Okapi
from config import (FAISS_FILE, IDS_FILE, CAPTIONS_FILE, TOP_CAND, TOP_K,
                    W_FULL, W_NO_CLAUSE)
from retriever.parser import parse, family
from retriever.encoder import QueryEncoder


def _norm(x: np.ndarray) -> np.ndarray:
    lo, hi = x.min(), x.max()
    return (x - lo) / (hi - lo) if hi > lo else np.zeros_like(x)


class FashionSearch:
    def __init__(self):
        self.index = faiss.read_index(FAISS_FILE)
        self.ids = json.load(open(IDS_FILE))
        self.caps = {json.loads(l)["id"]: json.loads(l)
                     for l in open(CAPTIONS_FILE, encoding="utf-8")}

        def doc_text(r):
            return " ".join([r.get("caption", ""),
                             " ".join(f"{g.get('color','')} {g.get('type','')}"
                                      for g in r.get("garments", [])),
                             r.get("environment", ""), r.get("style", "")]).lower().split()

        # bm_all[idxs] in search() indexes BM25 docs positionally against FAISS rows,
        # so doc order must match self.ids order exactly — build from self.ids, not file order.
        docs = [doc_text(self.caps[i]) for i in self.ids]
        self.bm25 = BM25Okapi(docs)
        self.enc = QueryEncoder()
        print("backend:", self.enc.backend, "| images:", self.index.ntotal)

    def _attr(self, parsed, cap) -> float:
        parts = []
        img_garments = [(str(g.get("color", "")).lower(), str(g.get("type", "")).lower())
                        for g in cap.get("garments", [])]
        for qc, qg in parsed["pairs"]:            # (color, garment) must co-occur on SAME item
            hit = any((qg in ic_type or ic_type in qg) and
                      (family(qc) == family(ic_col) or qc in ic_col)
                      for ic_col, ic_type in img_garments)
            parts.append(1.0 if hit else 0.0)
        for qg in parsed["garments"]:
            parts.append(1.0 if any(qg in t or t in qg for _, t in img_garments) else 0.0)
        if parsed["environment"]:
            parts.append(1.0 if cap.get("environment") == parsed["environment"] else 0.0)
        if parsed["style"]:
            s = cap.get("style", "")
            ok = s == parsed["style"] or (parsed["style"] == "formal" and s == "business_casual")
            parts.append(1.0 if ok else 0.0)
        return float(np.mean(parts)) if parts else 0.5

    def search(self, query: str, k: int = TOP_K):
        parsed = parse(query)
        texts = [query] + parsed["clauses"]
        embs = self.enc.encode(texts)
        q_full, q_clauses = embs[0:1], embs[1:]

        sims, idxs = self.index.search(q_full, TOP_CAND)
        idxs, dense = idxs[0], sims[0]
        cand_vecs = np.vstack([self.index.reconstruct(int(i)) for i in idxs])

        if len(q_clauses):
            clause_sims = cand_vecs @ q_clauses.T
            min_clause = clause_sims.min(axis=1)
        else:
            min_clause = dense.copy()
        attr = np.array([self._attr(parsed, self.caps[self.ids[i]]) for i in idxs])
        bm_all = self.bm25.get_scores(query.lower().split())
        bm = bm_all[idxs]

        w = W_FULL if len(q_clauses) else W_NO_CLAUSE
        final = (w[0] * _norm(dense) + w[1] * _norm(min_clause)
                 + w[2] * attr + w[3] * _norm(bm))

        order = np.argsort(-final)[:k]
        return [{"id": self.ids[idxs[o]], "score": round(float(final[o]), 4),
                 "dense": round(float(dense[o]), 4),
                 "min_clause": round(float(min_clause[o]), 4),
                 "attr": round(float(attr[o]), 2),
                 "caption": self.caps[self.ids[idxs[o]]].get("caption", "")}
                for o in order]
