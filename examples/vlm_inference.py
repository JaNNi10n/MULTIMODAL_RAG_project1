"""Minimal Qwen2.5-VL document question answering on one page image.

Fixed version of the original notebook export: it builds the chat message, runs
process_vision_info + the processor to create `inputs`, and loads the model in 4-bit so it fits
a 4 GB GPU. Needs the VLM extras: `pip install -r requirements-vlm.txt`.
Usage: python examples/vlm_inference.py path/to/page.jpg "What is the total amount due?"
"""

import sys

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2_5_VLForConditionalGeneration

MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
image_path = sys.argv[1] if len(sys.argv) > 1 else "docs/sample_invoice.jpg"
question = sys.argv[2] if len(sys.argv) > 2 else "What is the total amount due?"

model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float16,
    device_map="auto",
    quantization_config=BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.float16),
)
processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=256 * 28 * 28, max_pixels=1280 * 28 * 28)

img = Image.open(image_path).convert("RGB")
img.thumbnail((1024, 1024))  # keep the visual token count (and VRAM) bounded

messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
image_inputs, video_inputs = process_vision_info(messages)
inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
inputs = inputs.to(model.device)

with torch.inference_mode():
    output_ids = model.generate(**inputs, max_new_tokens=200, do_sample=False)
output_text = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(output_text[0])
