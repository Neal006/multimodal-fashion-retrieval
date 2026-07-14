"""Text encoder. ONNX int8 if exported (deferred here), else torch CPU fp32."""
import numpy as np, open_clip
from config import SIGLIP_ID, ONNX_FILE


class QueryEncoder:
    def __init__(self):
        self.tok = open_clip.get_tokenizer(SIGLIP_ID)
        if ONNX_FILE.exists():
            import onnxruntime as ort
            self.sess = ort.InferenceSession(str(ONNX_FILE),
                                             providers=["CPUExecutionProvider"])
            self.backend = "onnx-int8"
        else:
            import torch
            self.model, _, _ = open_clip.create_model_and_transforms(SIGLIP_ID)
            self.model.eval(); self.torch = torch
            self.backend = "torch-cpu"

    def encode(self, texts) -> np.ndarray:
        toks = self.tok(texts)
        if self.backend == "onnx-int8":
            out = self.sess.run(None, {"tokens": toks.numpy().astype(np.int64)})[0]
            return out.astype("float32")
        with self.torch.inference_mode():
            f = self.model.encode_text(toks)
            f = self.torch.nn.functional.normalize(f, dim=-1)
        return f.numpy().astype("float32")
