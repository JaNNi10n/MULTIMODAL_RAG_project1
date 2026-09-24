import torch 
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor 
from qwen_vl_utils import process_vision_info 

model = Qwen2_5_VLForConditionalGeneration.from_pretrained( 
    "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
     ) 
processor = AutoProcessor.from_pretrained( "Qwen/Qwen2.5-VL-3B-Instruct", 
                                          min_pixels=256*28*28, 
                                           max_pixels=1280*28*28 
                                           )

 from PIL import Image 
 img = Image.open("your_document.jpg") 
 img.thumbnail((1024, 1024)) 
 img.save("your_document_resized.jpg")
 
  from PIL import Image 
 img = Image.open("your_document.jpg") 
 img.thumbnail((1024, 1024)) 
 img.save("your_document_resized.jpg")
 
 output_ids = model.generate(**inputs, max_new_tokens=200)
output_text = processor.batch_decode(output_ids[:, inputs.input_ids.shape[1]:], skip_special_tokens=True)
print(output_text[0])

