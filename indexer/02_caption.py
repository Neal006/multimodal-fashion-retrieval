"""SmolVLM-500M -> one structured record per image.

The 500M model won't reliably emit JSON, but it captions well. So we take a rich prose
caption (its strength) and derive the structured attribute index by reusing the query
parser's lexicon (deterministic, model-agnostic). caption doubles as the BM25 document.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from config import DATA_DIR, CAPTIONS_FILE, VLM_ID, N_IMAGES
from retriever.parser import parse

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

PROMPT = ("Describe the person's outfit and surroundings in one detailed sentence. "
          "Name every visible garment together with its color, and the setting.")


def structured(caption: str) -> dict:
    p = parse(caption)
    garments = [{"type": g, "color": c} for c, g in p["pairs"]] + \
               [{"type": g, "color": ""} for g in p["garments"]]
    return {"garments": garments,
            "environment": p["environment"] or "other",
            "style": p["style"] or "other",
            "caption": caption}


def main():
    proc = AutoProcessor.from_pretrained(VLM_ID)
    model = AutoModelForImageTextToText.from_pretrained(VLM_ID, dtype=DTYPE).to(DEVICE).eval()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    chat = proc.apply_chat_template(msgs, add_generation_prompt=True)

    paths = sorted(DATA_DIR.glob("*.jpg"))[:N_IMAGES]
    with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
        for p in tqdm(paths, desc="captioning"):
            img = Image.open(p).convert("RGB")
            inputs = proc(text=chat, images=[img], return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=72, do_sample=False)
            cap = proc.batch_decode(out, skip_special_tokens=True)[0].split("Assistant:")[-1].strip()
            rec = structured(cap)
            rec["id"] = p.name
            f.write(json.dumps(rec) + "\n")
    print("wrote", CAPTIONS_FILE)


if __name__ == "__main__":
    main()
