"""SmolVLM-500M -> one structured record per image.

The 500M model won't reliably emit JSON for a full multi-field schema (tried it:
forces hallucinated garments to fill the schema, degrades the caption too). It
captions well in free-form prose, so garments/environment/caption still come from
that + the query parser's lexicon. Style is the one field the lexicon leaves >99%
"other" on (queries rarely say "formal"/"casual" in caption prose), so that single
field is asked directly via outlines grammar-constrained decoding -- one enum
token, no schema-filling pressure, cheap and reliable at this model size.
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import json, torch
from enum import Enum
import outlines
from outlines.inputs import Chat, Image as OImage
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForImageTextToText
from config import CAPTIONS_FILE, VLM_ID, image_paths
from retriever.parser import parse, STYLES

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

PROMPT = ("Describe the person's outfit and surroundings in one detailed sentence. "
          "Name every visible garment together with its color, and the setting.")
STYLE_PROMPT = ("Judge the overall style of this outfit: formal (business/dressy), "
                 "casual (everyday/relaxed), sporty (athletic/gym), or other. "
                 "Answer with just the style.")
Style = Enum("Style", {s: s for s in list(STYLES) + ["other"]})


def _with_format(p) -> Image.Image:
    img = Image.open(p)
    fmt = img.format
    img = img.convert("RGB")
    img.format = fmt
    return img


def structured(caption: str) -> dict:
    p = parse(caption)
    garments = [{"type": g, "color": c} for c, g in p["pairs"]] + \
               [{"type": g, "color": ""} for g in p["garments"]]
    return {"garments": garments,
            "environment": p["environment"] or "other",
            "style": p["style"] or "other",
            "caption": caption}


def main():
    paths = image_paths()
    already = set()
    if CAPTIONS_FILE.exists():
        already = {json.loads(l)["id"] for l in open(CAPTIONS_FILE, encoding="utf-8")}
    todo = [p for p in paths if p.name not in already]
    if not todo:
        print("nothing new to caption"); return
    print(f"{len(already)} cached, captioning {len(todo)} new image(s)")

    proc = AutoProcessor.from_pretrained(VLM_ID)
    model = AutoModelForImageTextToText.from_pretrained(VLM_ID, dtype=DTYPE).to(DEVICE).eval()
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    chat = proc.apply_chat_template(msgs, add_generation_prompt=True)
    style_gen = outlines.Generator(outlines.from_transformers(model, proc), output_type=Style)

    with open(CAPTIONS_FILE, "a", encoding="utf-8") as f:
        for p in tqdm(todo, desc="captioning"):
            img = _with_format(p)
            inputs = proc(text=chat, images=[img], return_tensors="pt").to(DEVICE)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=72, do_sample=False)
            cap = proc.batch_decode(out, skip_special_tokens=True)[0].split("Assistant:")[-1].strip()
            rec = structured(cap)
            style_prompt = Chat([{"role": "user", "content": [
                {"type": "image", "image": OImage(img)}, {"type": "text", "text": STYLE_PROMPT}]}])
            rec["style"] = style_gen(style_prompt, max_new_tokens=10)
            rec["id"] = p.name
            f.write(json.dumps(rec) + "\n")
    print("wrote", CAPTIONS_FILE)


if __name__ == "__main__":
    main()
