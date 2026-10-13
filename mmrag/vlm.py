"""Qwen2.5-VL reader: answers a question from one page image plus retrieved text context."""

from __future__ import annotations

from pathlib import Path

DEFAULT_VLM = "Qwen/Qwen2.5-VL-3B-Instruct"

PROMPT = (
    "You are reading a document page. Use the page image and the retrieved text below to answer.\n"
    "Retrieved text (may contain OCR errors; trust the image if they disagree):\n{context}\n\n"
    "Question: {question}\n"
    "Reply with only the exact value from the document (a number, name or short phrase), no explanation."
)


def trim_margins(image, threshold: int = 200, pad: int = 16):
    """Crop blank page margins so the visual-token budget is spent on text, not white paper."""
    from PIL import ImageOps

    ink = ImageOps.invert(image.convert("L")).point(lambda v: 255 if v > 255 - threshold else 0)
    box = ink.getbbox()
    if box is None:
        return image
    left, top, right, bottom = box
    return image.crop((max(left - pad, 0), max(top - pad, 0),
                       min(right + pad, image.width), min(bottom + pad, image.height)))


def build_prompt(question: str, context: list[str]) -> str:
    ctx = "\n".join(f"[{i}] {c}" for i, c in enumerate(context, start=1)) or "(none)"
    return PROMPT.format(context=ctx, question=question)


class QwenVLReader:
    """Loads Qwen2.5-VL once and answers questions about page images.

    load_in_4bit=True (default) uses bitsandbytes NF4 so the 3B model fits in a 4 GB GPU.
    max_pixels caps the number of visual tokens (one token per 28x28 patch).
    """

    def __init__(self, model_id: str = DEFAULT_VLM, load_in_4bit: bool = True,
                 max_pixels: int = 768 * 28 * 28, max_new_tokens: int = 48):
        import torch
        from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

        kwargs: dict = {"device_map": "auto"}
        if load_in_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True)
            kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **kwargs).eval()
        self.processor = AutoProcessor.from_pretrained(model_id, min_pixels=256 * 28 * 28, max_pixels=max_pixels)
        self.max_pixels = max_pixels
        self.max_new_tokens = max_new_tokens

    def answer(self, question: str, image_path: str | Path, context: list[str]) -> str:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        image = trim_margins(Image.open(image_path).convert("RGB"))
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image, "max_pixels": self.max_pixels},
            {"type": "text", "text": build_prompt(question, context)},
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs,
                                padding=True, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)
        new_tokens = out[:, inputs.input_ids.shape[1]:]
        return self.processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
