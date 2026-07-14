"""Dataset assembly.

Fashionpedia images are already present locally in `test/` (the working corpus), so the
Fashionpedia download is a no-op here. This module keeps the optional Pexels top-up, which
is the intended fix for the environment-coverage gap (office / home / raincoat are sparse in
Fashionpedia — see report §1). It is gated on PEXELS_API_KEY and writes into the same DATA_DIR
so `03_encode` / `02_caption` pick the images up transparently.
"""
import io, os, requests
from PIL import Image, ImageOps
from tqdm import tqdm
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from config import DATA_DIR
from dotenv import load_dotenv

load_dotenv()  # load PEXELS_API_KEY from .env

DATA_DIR.mkdir(parents=True, exist_ok=True)

# graded axes under-represented in Fashionpedia
PEXELS_QUERIES = [
    "business attire modern office", "person sitting park bench",
    "casual outfit city walk", "person yellow raincoat",
    "casual clothes at home living room", "formal shirt tie office",
]


def _save(img: Image.Image, name: str):
    img = ImageOps.exif_transpose(img).convert("RGB")
    img.thumbnail((640, 640))
    img.save(DATA_DIR / name, "JPEG", quality=90)


def pexels_topup(per_query: int = 25):
    key = os.environ.get("PEXELS_API_KEY")
    if not key:
        print("PEXELS_API_KEY unset — skipping top-up (Fashionpedia-only corpus).")
        return
    n = 0
    for q in PEXELS_QUERIES:
        r = requests.get("https://api.pexels.com/v1/search",
                         headers={"Authorization": key},
                         params={"query": q, "per_page": per_query}, timeout=30).json()
        for p in tqdm(r.get("photos", []), desc=q):
            try:
                img = Image.open(io.BytesIO(requests.get(p["src"]["large"], timeout=20).content))
                _save(img, f"px_{n:05d}.jpg"); n += 1
            except (requests.RequestException, OSError) as e:
                print("skip:", e)
    print("added", n, "environment images")


if __name__ == "__main__":
    print("Fashionpedia images expected in", DATA_DIR, "->",
          len(list(DATA_DIR.glob("*.jpg"))), "present")
    pexels_topup()
