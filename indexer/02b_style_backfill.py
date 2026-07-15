"""One-off: recompute the `style` field on every existing captions.jsonl record
via constrained decoding (see 02_caption.py docstring). Everything else
(garments/environment/caption) is untouched -- this only fixes the >99% "other"
style-tagging gap on records captioned before the style pass existed.

Resumable: writes incrementally to a side file keyed by id, so a kill/interrupt
loses at most the in-flight record, not the whole run.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, torch
from enum import Enum
import outlines
from outlines.inputs import Chat, Image as OImage
from PIL import Image
from transformers import AutoProcessor, AutoModelForImageTextToText
from config import CAPTIONS_FILE, DATA_DIR, VLM_ID, ART
from retriever.parser import STYLES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
STYLE_PROMPT = ("Judge the overall style of this outfit: formal (business/dressy), "
                 "casual (everyday/relaxed), sporty (athletic/gym), or other. "
                 "Answer with just the style.")
Style = Enum("Style", {s: s for s in list(STYLES) + ["other"]})
OUT_FILE = ART / "captions_style_new.jsonl"


def _with_format(p):
    img = Image.open(p)
    fmt = img.format
    img = img.convert("RGB")
    img.format = fmt
    return img


def main():
    records = [json.loads(l) for l in open(CAPTIONS_FILE, encoding="utf-8")]
    done = set()
    if OUT_FILE.exists():
        done = {json.loads(l)["id"] for l in open(OUT_FILE, encoding="utf-8")}
    todo = [r for r in records if r["id"] not in done]
    print(f"{len(done)} done, {len(todo)} remaining of {len(records)}", flush=True)
    if not todo:
        print("nothing left to backfill", flush=True)
        return

    proc = AutoProcessor.from_pretrained(VLM_ID)
    model = AutoModelForImageTextToText.from_pretrained(VLM_ID, dtype=DTYPE).to(DEVICE).eval()
    style_gen = outlines.Generator(outlines.from_transformers(model, proc), output_type=Style)

    with open(OUT_FILE, "a", encoding="utf-8") as f:
        for i, rec in enumerate(todo):
            img = _with_format(DATA_DIR / rec["id"])
            prompt = Chat([{"role": "user", "content": [
                {"type": "image", "image": OImage(img)}, {"type": "text", "text": STYLE_PROMPT}]}])
            rec["style"] = style_gen(prompt, max_new_tokens=10)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if (i + 1) % 25 == 0 or (i + 1) == len(todo):
                print(f"{i + 1}/{len(todo)}", flush=True)

    new_done = {json.loads(l)["id"]: json.loads(l) for l in open(OUT_FILE, encoding="utf-8")}
    if len(new_done) == len(records):
        with open(CAPTIONS_FILE, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(new_done[r["id"]]) + "\n")
        OUT_FILE.unlink()
        print(f"merged {len(records)} records into {CAPTIONS_FILE}", flush=True)
    else:
        print(f"{len(new_done)}/{len(records)} done so far, run again to continue", flush=True)


if __name__ == "__main__":
    main()
