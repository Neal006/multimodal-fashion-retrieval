import numpy as np


def precision_at_k(ranked, rel, k):
    return sum(r in rel for r in ranked[:k]) / k


def ndcg_at_k(ranked, rel, k):
    dcg = sum(1 / np.log2(i + 2) for i, r in enumerate(ranked[:k]) if r in rel)
    ideal = sum(1 / np.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / ideal if ideal else 0.0
