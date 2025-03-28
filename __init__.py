from .nodes import CLIPTextEncodeWithStats, SaveImgAdv, OpenPoseEditorAdv, SolidMaskAdv
import folder_paths
import os

WEB_DIRECTORY = "./js"

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

def checkForLegacyFolders():
    js_dest_path = os.path.join(comfy_path, "web", "extensions", "ultools")
    legacy_js_dest_path = os.path.join(comfy_path, "web", "extensions", "imginfo")

    # check presence of legacy folder, print hint it can be removed
    if os.path.isdir(legacy_js_dest_path):
        print(f"{colors.BLUE}ULTools: {colors.WARNING}Found legacy SaveImgAdv path at {legacy_js_dest_path}, this folder and its content can be removed.{colors.ENDC}")
    if os.path.isdir(js_dest_path):
        print(f"{colors.BLUE}ULTools: {colors.WARNING}Found legacy SaveImgAdv path at {js_dest_path}, this folder and its content can be removed.{colors.ENDC}")

checkForLegacyFolders()

NODE_CLASS_MAPPINGS = {
    "SaveImgAdv": SaveImgAdv.SaveImgAdv,
    "CLIPTextEncodeWithStats": CLIPTextEncodeWithStats.CLIPTextEncodeWithStats,
    "OpenPoseEditorAdv": OpenPoseEditorAdv.OpenPoseEditorAdv,
    "SolidMaskAdv": SolidMaskAdv.SolidMaskAdv
}

__all__ = ["NODE_CLASS_MAPPINGS", "WEB_DIRECTORY"]
