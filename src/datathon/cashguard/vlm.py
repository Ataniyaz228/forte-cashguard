"""Yes/no scene-state questions answered by a local VLM (Qwen3-VL, Apache-2.0).

Instead of free text we read P(yes) from the next-token logits: one forward pass per
(image, question) pair, batched, and a calibrated score we can threshold on dev data.
"""

from dataclasses import dataclass

import numpy as np
import torch
from PIL import Image

VLM_MODEL = "Qwen/Qwen3-VL-2B-Instruct"

QUESTIONS: dict[str, str] = {
    "safe_open": "Look at the grey metal safe (a cabinet with a lockable door). "
                 "Is the safe door open?",
    "cash_on_desk": "Are banknotes, cash or bundles of money lying on the desk?",
    "bag_on_desk": "Is there a green fabric cash bag lying on the desk?",
    "docs_on_desk": "Are papers, documents, a folder or a tablet lying on the desk?",
    "screen_on": "Is the laptop screen switched on and showing an application or desktop?",
    "food": "Is there food, a drink or a cup on the desk, or is someone eating or drinking?",
    "personal_items": "Is there a handbag, a backpack or a jacket lying on the desk?",
    "phone": "Is someone holding or looking at a mobile phone?",
    "counting": "Is someone counting banknotes by hand or with a money counting machine?",
    "id_shown": "Is someone holding up an ID card or document so the other person can check it?",
    "room_empty": "Is the room empty, with no person inside it (ignore people behind the glass)?",
}
SUFFIX = " Answer only yes or no."
# asked on every sampled frame; the rest of QUESTIONS are kept for experiments
PERIODIC_KEYS = ["safe_open", "cash_on_desk", "bag_on_desk", "food", "personal_items",
                 "counting", "id_shown"]


@dataclass
class VLMConfig:
    model: str = VLM_MODEL
    max_pixels: int = 320 * 600
    batch: int = 12
    load_4bit: bool = False  # needed for Qwen3-VL-4B on a 6 GB GPU


class SceneVLM:
    def __init__(self, device: str = "cuda", cfg: VLMConfig | None = None):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.cfg = cfg or VLMConfig()
        self.device = device
        self.proc = AutoProcessor.from_pretrained(self.cfg.model)
        self.proc.tokenizer.padding_side = "left"
        ip = self.proc.image_processor
        if hasattr(ip, "max_pixels"):
            ip.max_pixels = self.cfg.max_pixels
        if hasattr(ip, "size") and isinstance(ip.size, dict) and "longest_edge" in ip.size:
            ip.size["longest_edge"] = self.cfg.max_pixels
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        if self.cfg.load_4bit:
            from transformers import BitsAndBytesConfig

            q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                   bnb_4bit_compute_dtype=torch.bfloat16)
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.cfg.model, quantization_config=q, dtype=dtype, device_map={"": 0})
        else:
            self.model = AutoModelForImageTextToText.from_pretrained(self.cfg.model, dtype=dtype)
            self.model.to(device)
        self.model.eval()
        tok = self.proc.tokenizer
        self.yes_ids = sorted({tok.encode(w, add_special_tokens=False)[0]
                               for w in ("yes", "Yes", " yes", " Yes")})
        self.no_ids = sorted({tok.encode(w, add_special_tokens=False)[0]
                              for w in ("no", "No", " no", " No")})

    def _prompt(self, question: str) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"},
                                             {"type": "text", "text": question + SUFFIX}]}]
        return self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    @torch.inference_mode()
    def yes_prob(self, images: list[Image.Image], keys: list[str]) -> np.ndarray:
        """P(yes) for every image x question -> array [len(images), len(keys)]."""
        pairs = [(i, j) for i in range(len(images)) for j in range(len(keys))]
        # a key is either a name from QUESTIONS or a literal question
        prompts = {k: self._prompt(QUESTIONS.get(k, k)) for k in keys}
        out = np.zeros((len(images), len(keys)), dtype=np.float32)
        b = self.cfg.batch
        for s in range(0, len(pairs), b):
            chunk = pairs[s:s + b]
            inp = self.proc(text=[prompts[keys[j]] for _, j in chunk],
                            images=[images[i] for i, _ in chunk],
                            return_tensors="pt", padding=True).to(self.device)
            logits = self.model(**inp, logits_to_keep=1).logits[:, -1, :].float()
            ly = torch.logsumexp(logits[:, self.yes_ids], dim=-1)
            ln = torch.logsumexp(logits[:, self.no_ids], dim=-1)
            p = torch.sigmoid(ly - ln).cpu().numpy()
            for (i, j), v in zip(chunk, p):
                out[i, j] = v
        return out

    @torch.inference_mode()
    def describe(self, image: Image.Image, question: str, max_new_tokens: int = 60) -> str:
        msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text",
                                                                 "text": question}]}]
        text = self.proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inp = self.proc(text=[text], images=[image], return_tensors="pt").to(self.device)
        out = self.model.generate(**inp, max_new_tokens=max_new_tokens, do_sample=False)
        return self.proc.batch_decode(out[:, inp["input_ids"].shape[1]:],
                                      skip_special_tokens=True)[0].strip()
