import io
import torch
import comfy.model_management
from nodes import MAX_RESOLUTION
from kornia.enhance import equalize_clahe
from kornia.color import rgb_to_lab, lab_to_rgb
from kornia.utils import tensor_to_image
from kornia.filters import unsharp_mask
from matplotlib import pyplot as plt
from PIL import Image
import numpy as np
import matplotlib
matplotlib.use('agg')


class Clahe:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "clip_limit": ("FLOAT", {"default": 1.0, "min": 0, "max": 9999.0, "step": 0.1}),
                "grid_size_wh": ("INT", {"default": 8, "min": 1, "max": MAX_RESOLUTION, "step": 1}),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "doit"

    def doit(self, image, clip_limit, grid_size_wh):
        if grid_size_wh == 1:
            return (image,)

        device = torch.device("cpu") # comfy.model_management.get_torch_device() #
        image_k = image.to(device).movedim(-1, 1)

        # clamp image values to the valid range [0, 1]
        image_k = torch.clamp(image_k, 0.0, 1.0)

        img_lab = rgb_to_lab(image_k)

        lightness = img_lab[..., :1, :, :] / 100.0  # L in lab is in range [0, 100]

        try:
            lightness_eq = equalize_clahe(
                lightness,
                clip_limit=clip_limit,
                grid_size=(grid_size_wh, grid_size_wh),
                slow_and_differentiable=False)

            img_rgb_new = lab_to_rgb(torch.cat([lightness_eq * 100.0, img_lab[:, 1:]], dim=1))
            img_out = img_rgb_new.to(comfy.model_management.intermediate_device()).movedim(1, -1)
            return (img_out,)
        except Exception as e:
            print('UL TOOLS ERROR: error during applying clahe, returning unprocessed image!', e)
        return (image,)


class UnsharpMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", ),
                "kernel_size": ("INT", {"default": 9, "min": 1, "max": MAX_RESOLUTION, "step": 2}),
                "sigma": ("FLOAT", {"default": 2.5, "min": 0, "max": 9999.0, "step": 0.1}),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "doit"

    def doit(self, image, kernel_size, sigma):
        if kernel_size == 1 and sigma == 0:
            return (image,)

        device = comfy.model_management.get_torch_device()
        image_k = image.to(device).movedim(-1, 1)

        # clamp image values to the valid range [0, 1]
        image_k = torch.clamp(image_k, 0.0, 1.0)

        sharpened_tensor = unsharp_mask(image_k, (kernel_size, kernel_size), (sigma, sigma))

        img_out = sharpened_tensor.to(comfy.model_management.intermediate_device()).movedim(1, -1)

        return (img_out,)


class Histogram:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", ),
                "mode": ([ "lightness", "rgb" ], { "default": "lightness" }),
            }
        }

    CATEGORY = "image"

    RETURN_TYPES = ("IMAGE",)

    FUNCTION = "doit"

    def doit(self, images, mode):
        my_dpi = 96

        device = comfy.model_management.get_torch_device()
        image_k = images.to(device).movedim(-1, 1)

        lightness = None
        retval = []

        for idx in range(len(images)):
            if mode == "lightness":
                if lightness is None:
                    img_lab = rgb_to_lab(image_k)
                    lightness = img_lab[..., :1, :, :] / 100.0  # L in lab is in range [0, 100]
                CH, H, W = lightness[idx].shape
                img = tensor_to_image(lightness[idx].mul(255.0).byte())
            else:
                CH, H, W = image_k[idx].shape
                img = tensor_to_image(image_k[idx].mul(255.0).byte())

            fig = plt.figure(figsize=(1600/my_dpi, 400/my_dpi), dpi=my_dpi)
            ax2 = plt.subplot(1, 1, 1)
            fig.patch.set_facecolor('#353535')
            ax2.patch.set_facecolor('#353535')
            # set various colors
            ax2.spines['bottom'].set_color('white')
            ax2.spines['top'].set_color('#353535')
            ax2.spines['right'].set_color('#353535')
            ax2.spines['right'].set_linewidth(0)
            ax2.spines['left'].set_color('white')
            ax2.spines['left'].set_lw(1)
            ax2.xaxis.label.set_color('white')
            ax2.yaxis.label.set_color('white')
            ax2.tick_params(colors='white', which='both')

            colors = ("b", "g", "r")
            kwargs = dict(histtype="stepfilled", alpha=0.3, density=True, ec="k")

            for i in range(CH):
                img_vec = img[..., i].flatten()
                ax2.hist(img_vec, range=(0, 255), bins=256, color=colors[i] if mode == "rgb" else "white", **kwargs)

            # Save the plot to a BytesIO object
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)

            # Create a PIL image from the buffer and move channels to the end.
            pil_image = Image.open(buf)
            pil_image = pil_image.resize((1600, 400)) # matplotlib produces differently sized images, will cause errors in torch batch
            image_tensor = torch.from_numpy(np.array(pil_image).astype(np.float32) / 255.0).unsqueeze(0)

            retval.append(image_tensor)

            matplotlib.pyplot.close()

        return ( torch.cat(tuple(retval), dim=0), )
