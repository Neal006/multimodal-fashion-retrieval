import sys, json, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse
from retriever.search import FashionSearch

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("query"); ap.add_argument("-k", type=int, default=5)
    args = ap.parse_args()
    s = FashionSearch()
    for r in s.search(args.query, args.k):
        print(json.dumps(r))
