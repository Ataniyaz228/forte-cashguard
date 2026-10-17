"""Frame-level COCO object detection (RT-DETR, Apache-2.0)."""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

DET_MODEL = "PekingU/rtdetr_r50vd_coco_o365"
KEEP = {"person", "laptop", "cell phone", "cup", "bottle", "handbag", "backpack", "book",
        "chair", "tvmonitor", "refrigerator", "microwave", "suitcase", "sandwich", "donut",
        "cake", "banana", "apple", "wine glass", "bowl", "remote", "keyboard", "mouse"}


@dataclass
class Det:
    label: str
    score: float
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in pixels
    look: dict | None = None  # appearance stats for people (see appearance())

    def to_list(self):
        out = [self.label, round(self.score, 3), [round(v, 1) for v in self.box]]
        if self.look:
            out.append(self.look)
        return out


def appearance(frame: np.ndarray, box) -> dict:
    """Colour saturation and sharpness inside a box. People seen through the tinted glass
    wall are washed out and soft; used relative to the video's own people (timeline.py)."""
    h, w = frame.shape[:2]
    x1, y1 = int(max(0, box[0])), int(max(0, box[1]))
    x2, y2 = int(min(w, box[2])), int(min(h, box[3]))
    c = frame[y1:y2, x1:x2].astype(np.float32)
    if c.shape[0] < 4 or c.shape[1] < 4:
        return {"sat": 0.0, "sharp": 0.0}
    g = c.mean(axis=2)
    sat = float((c.max(axis=2) - c.min(axis=2)).mean())
    sharp = float(np.abs(np.diff(g, axis=0)).mean() + np.abs(np.diff(g, axis=1)).mean())
    return {"sat": round(sat, 2), "sharp": round(sharp, 2)}


def pick_device(pref: str = "auto") -> str:
    if pref != "auto":
        return pref
    return "cuda" if torch.cuda.is_available() else "cpu"


class Detector:
    def __init__(self, device: str = "auto", threshold: float = 0.3):
        from transformers import AutoImageProcessor, RTDetrForObjectDetection

        self.device = pick_device(device)
        self.threshold = threshold
        self.proc = AutoImageProcessor.from_pretrained(DET_MODEL)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = RTDetrForObjectDetection.from_pretrained(DET_MODEL, dtype=dtype)
        self.model.to(self.device).eval()
        self.dtype = dtype
        self.names = self.model.config.id2label

    @torch.inference_mode()
    def __call__(self, frames: list[np.ndarray], batch: int = 8) -> list[list[Det]]:
        out: list[list[Det]] = []
        for i in range(0, len(frames), batch):
            imgs = [Image.fromarray(f) for f in frames[i:i + batch]]
            inp = self.proc(images=imgs, return_tensors="pt")
            pv = inp["pixel_values"].to(self.device, self.dtype)
            res = self.model(pixel_values=pv)
            res.logits = res.logits.float()
            res.pred_boxes = res.pred_boxes.float()
            sizes = [(im.height, im.width) for im in imgs]
            post = self.proc.post_process_object_detection(res, target_sizes=sizes,
                                                           threshold=self.threshold)
            for p in post:
                dets = []
                for s, lab, b in zip(p["scores"].tolist(), p["labels"].tolist(),
                                     p["boxes"].tolist()):
                    name = self.names[lab]
                    if name in KEEP:
                        look = appearance(frames[len(out)], b) if name == "person" else None
                        dets.append(Det(name, s, tuple(b), look))
                out.append(dets)
        return out
