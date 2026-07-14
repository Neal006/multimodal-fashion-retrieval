"""FAISS inner-product index (vectors pre-normalized => cosine). Flat is exact at this scale."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import faiss, numpy as np
from config import EMB_FILE, FAISS_FILE

emb = np.load(EMB_FILE)
index = faiss.IndexFlatIP(emb.shape[1])
index.add(emb)
faiss.write_index(index, FAISS_FILE)
print("indexed", index.ntotal, "vectors, dim", emb.shape[1])
