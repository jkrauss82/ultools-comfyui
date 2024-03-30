import torch
import sys
import os
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import numpy as np
import folder_paths

model_path = folder_paths.models_dir

def do_blip(image, blip_model, mode, min_length, max_length, device):
    device = torch.device("cuda" if torch.cuda.is_available() and device == "cuda" else "cpu")
    if blip_model == "base":
        blipprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=f"{model_path}/blip")
        blipmodel = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base", cache_dir=f"{model_path}/blip").to(device)
    else:
        blipprocessor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=f"{model_path}/blip")
        blipmodel = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large", cache_dir=f"{model_path}/blip").to(device)
    img = Image.fromarray(np.clip((255. * image[0].cpu().numpy()), 0, 255).astype(np.uint8))
    with torch.no_grad():
        text = blipprocessor.decode(blipmodel.generate(**blipprocessor(img, return_tensors="pt").to(device), min_length=min_length, max_length=max_length)[0], skip_special_tokens=True)
        print(f'BLIP: {text}')
    return text

class BLIPAnalyzer:
    @classmethod
    def INPUT_TYPES(s):
        ui_widgets = {
            "required": {
                "image": ("IMAGE",),
                "blip_model": (["base", "large"], {"default": "base"}),
                "mode": (["caption","interrogate"], {"default": "caption"}),
            },
            "optional": {
                "min_length": ("INT", {"default": 20, "min": 0, "max": 200, "step": 1}),
                "max_length": ("INT", {"default": 50, "min": 0, "max": 200, "step": 1}),
                "device": (["cpu", "cuda"], {"default": "cpu"}),
            }
        }
        return ui_widgets

    RETURN_TYPES = ("STRING",)
    FUNCTION = "analyze"
    CATEGORY = "utils"

    def analyze(self, image, blip_model, mode, min_length, max_length, device):
        caption = do_blip(image, blip_model, mode, min_length, max_length, device)
        return (caption,)

NODE_CLASS_MAPPINGS = {
    "BLIPAnalyzer": BLIPAnalyzer,
}
