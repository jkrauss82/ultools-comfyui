import torch
import numpy
from nodes import MAX_RESOLUTION
from PIL import Image, ImageDraw

class SolidMaskAdv:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "width": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "height": ("INT", {"default": 512, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
                "type": (["rectangle", "ellipsis"], {"default": "rectangle"}),
            }
        }

    CATEGORY = "mask"

    RETURN_TYPES = ("MASK",)

    FUNCTION = "solid"

    def solid(self, value, width, height, type):
        out = None
        if type == "rectangle":
            out = torch.full((1, height, width), value, dtype=torch.float32, device="cpu")
        else:
            ellipsis = Image.new("F", (width, height), color=0)
            mask = ImageDraw.Draw(ellipsis)
            mask.ellipse([0, 0, width, height], fill=value, outline=value, width=1)
            np = numpy.array(ellipsis)
            # print(np)
            out = torch.from_numpy(np)

        return (out,)
