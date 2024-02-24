from .nodes import CLIPTextEncodeWithStats, SaveImgAdv
import shutil
import folder_paths
import os

class colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    GREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

comfy_path = os.path.dirname(folder_paths.__file__)

def setup_js():
    ultools_path = os.path.dirname(__file__)
    js_dest_path = os.path.join(comfy_path, "web", "extensions", "ultools")
    legacy_js_dest_path = os.path.join(comfy_path, "web", "extensions", "imginfo")
    js_files = ["ultools.js", "exif-reader.js"]

    # check presence of legacy folder, print hint it can be removed
    if os.path.isdir(legacy_js_dest_path):
        print(f"{colors.BLUE}ULTools: {colors.WARNING}Found legacy SaveImgAdv path at {legacy_js_dest_path}, this folder and its content can be removed.{colors.ENDC}")

    ## Creating folder if it's not present, then Copy.
    if not os.path.isdir(js_dest_path):
        os.mkdir(js_dest_path)
    logged = False
    for js in js_files:
        if not os.path.isfile(f"{js_dest_path}/{js}"):
            if logged == False:
                print(f"{colors.BLUE}ULTools:{colors.ENDC} Copying JS files")
                logged = True
        shutil.copy(os.path.join(ultools_path, "js", js), js_dest_path)

    print(f"{colors.BLUE}ULTools: {colors.GREEN}Loaded{colors.ENDC}")

setup_js()

NODE_CLASS_MAPPINGS = {
    "SaveImgAdv": SaveImgAdv.SaveImgAdv,
    "CLIPTextEncodeWithStats": CLIPTextEncodeWithStats.CLIPTextEncodeWithStats
}
