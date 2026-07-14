"""FashionSigLIP image embeddings (batched, GPU). --baseline adds vanilla CLIP."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import argparse, json
import numpy as np, torch, open_clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from config import (DATA_DIR, EMB_FILE, EMB_BASELINE_FILE, IDS_FILE,
                    SIGLIP_ID, CLIP_BASELINE, N_IMAGES)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class ImgDS(Dataset):
    def __init__(self, paths, preprocess):
        self.paths, self.pp = paths, preprocess
    def __len__(self): return len(self.paths)
    def __getitem__(self, i):
        return self.pp(Image.open(self.paths[i]).convert("RGB"))


@torch.inference_mode()
def encode(model_arg, out_file):
    if isinstance(model_arg, tuple):
        model, _, pp = open_clip.create_model_and_transforms(model_arg[0], pretrained=model_arg[1])
    else:
        model, _, pp = open_clip.create_model_and_transforms(model_arg)
    model = model.to(DEVICE).eval()

    paths = sorted(DATA_DIR.glob("*.jpg"))[:N_IMAGES]
    dl = DataLoader(ImgDS(paths, pp), batch_size=32, num_workers=0)  # 4GB VRAM, Windows
    chunks = []
    for batch in tqdm(dl, desc=str(out_file.name)):
        batch = batch.to(DEVICE)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=(DEVICE == "cuda")):
            f = model.encode_image(batch)
        f = torch.nn.functional.normalize(f.float(), dim=-1)
        chunks.append(f.cpu().numpy())
    emb = np.concatenate(chunks).astype("float32")
    np.save(out_file, emb)
    json.dump([p.name for p in paths], open(IDS_FILE, "w"))
    print(out_file, emb.shape)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    args = ap.parse_args()
    encode(SIGLIP_ID, EMB_FILE)
    if args.baseline:
        encode(CLIP_BASELINE, EMB_BASELINE_FILE)
