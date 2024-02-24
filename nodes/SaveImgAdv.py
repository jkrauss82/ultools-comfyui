import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo
import piexif
import piexif.helper
from . import helper
import folder_paths
import os
import json


'''
by Kaharos94 and jkrauss82
forked from Kaharos94 / https://github.com/Kaharos94/ComfyUI-Saveaswebp
comfyUI node to save an image in webp, jpeg and png formats
'''
class SaveImgAdv:
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE", ),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "mode": (["lossy", "lossless"],),
                "format": ([ "webp", "jpg", "png" ], { "default": "webp" }),
                "compression": ("INT", {"default": 90, "min": 1, "max": 100, "step": 1},),
                "calc_model_hashes": ("BOOLEAN", {"default": False}),
                "add_automatic1111_meta": ("BOOLEAN", {"default": False}),
                "keywords": ("STRING", { "placeholder": "List of keywords to be added as exif tag, separated by commas" }),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            }
        }

    INPUT_IS_LIST = True

    RETURN_TYPES = ()
    FUNCTION = "Save_as_format"

    OUTPUT_NODE = True

    CATEGORY = "image"


    def Save_as_format(self, mode, format, compression, images, calc_model_hashes, add_automatic1111_meta, filename_prefix="ComfyUI",
                       keywords=None, prompt=None, extra_pnginfo=None, ):

        # we have set INPUT_IS_LIST = True, need to map regular parameters from their lists
        images = images[0]
        mode = mode[0]
        format = format[0]
        compression = compression[0]
        calc_model_hashes = calc_model_hashes[0]
        add_automatic1111_meta = add_automatic1111_meta[0]
        filename_prefix = filename_prefix[0]
        prompt = prompt[0]
        extra_pnginfo = extra_pnginfo[0] if extra_pnginfo is not None else None

        def map_filename(filename):
            prefix_len = len(os.path.basename(filename_prefix))
            prefix = filename[:prefix_len + 1]
            try:
                digits = int(filename[prefix_len + 1:].split('_')[0])
            except:
                digits = 0
            return (digits, prefix)

        def compute_vars(input):
            input = input.replace("%width%", str(images[0].shape[1]))
            input = input.replace("%height%", str(images[0].shape[0]))
            return input

        results = list()

        filename_prefix = compute_vars(filename_prefix)

        subfolder = os.path.dirname(os.path.normpath(filename_prefix))
        filename = os.path.basename(os.path.normpath(filename_prefix))

        full_output_folder = os.path.join(self.output_dir, subfolder)

        if os.path.commonpath((self.output_dir, os.path.abspath(full_output_folder))) != self.output_dir:
            print("Saving image outside the output folder is not allowed.")
            return {}

        # sanitize mode option
        if format == 'jpg': mode = 'lossy'
        elif format == 'png': mode = 'lossless'

        # TODO: get rid of the trailing underscore
        try:
            counter = max(filter(lambda a: a[1][:-1] == filename and a[1][-1] == "_", map(map_filename, os.listdir(full_output_folder))))[0] + 1
        except ValueError:
            counter = 1
        except FileNotFoundError:
            os.makedirs(full_output_folder, exist_ok=True)
            counter = 1

        for idx in range(len(images)):
            i = 255. * images[idx].cpu().numpy()
            img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

            file = f"{filename}_{counter:05}_.{format}"

            # format webp or jpg
            if format != 'png':
                workflowmetadata = str()
                promptstr = str()

                if prompt is not None:
                    promptstr="".join(json.dumps(prompt)) #prepare prompt String
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        workflowmetadata += "".join(json.dumps(extra_pnginfo[x]))

                exifdict = {
                        "0th": {
                            piexif.ImageIFD.Make: promptstr,
                            piexif.ImageIFD.ImageDescription: workflowmetadata
                        }
                    }
                if add_automatic1111_meta:
                    exifdict['Exif'] = {
                            piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(helper.automatic1111Format(prompt, img, calc_model_hashes) or "", encoding="unicode")
                        }

                kidx = idx if keywords != None and len(keywords) > 1 else 0
                if keywords[kidx] != None and isinstance(keywords[kidx], str) and keywords[kidx] != '':
                    # keywords maxlength in iptc standard 64 characters
                    klist = keywords[kidx].split(",")
                    final_list = []
                    for word in klist:
                        if len(word) < 65: final_list.append(word.strip())
                    exifdict["0th"][piexif.ImageIFD.XPKeywords] = ", ".join(final_list).encode("utf-16le")

                exif_bytes = piexif.dump(exifdict)

                img.save(os.path.join(full_output_folder, file), method=6, exif=exif_bytes, lossless=(mode =="lossless"), quality=compression)

            # format png (method from ComfyUI SaveImage class)
            else:
                metadata = None
                metadata = PngInfo()
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

                img.save(os.path.join(full_output_folder, file), pnginfo=metadata, compress_level=4)

            results.append({
                "filename": file,
                "subfolder": subfolder,
                "type": self.type
            });
            counter += 1

        return { "ui": { "images": results } }
