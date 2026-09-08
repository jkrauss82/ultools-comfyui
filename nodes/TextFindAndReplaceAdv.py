import re

'''
by WASasquatch and jkrauss82
forked from class WAS_Search_and_Replace by WASasquatch / https://github.com/WASasquatch/was-node-suite-comfyui/
ComfyUI node to find text in a string and replace it, optionally setting a maximum number of times to replace it
'''
class TextFindAndReplaceAdv:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
                "find": ("STRING", {"default": '', "multiline": False}),
                "replace": ("STRING", {"default": '', "multiline": False}),
                "max_count": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            }
        }

    RETURN_TYPES = ("STRING", "NUMBER", "FLOAT", "INT")
    RETURN_NAMES = ("result_text", "replacement_count_number", "replacement_count_float", "replacement_count_int")
    FUNCTION = "text_search_and_replace"

    CATEGORY = "utils"

    DESCRIPTION = "Search and replace with regex support and option to set number of replacements (0 for replace all)."

    def text_search_and_replace(self, text, find, replace, max_count):
        finds = find.split("\n")
        replaces = replace.split("\n")
        # print(f"\nfind: {finds}\nreplace: {replaces}\n\n")
        if len(finds) != len(replaces):
            print(f"Warning: length of supplied find != length of supplied replace ({len(finds)} / {len(replaces)})")
        modified_text = text
        counts = 0
        for idx in range(len(finds)):
            find = finds[idx] if len(finds) > idx else finds[-1]
            replace = replaces[idx] if len(replaces) > idx else replaces[-1]
            modified_text, count = self.replace_substring(modified_text, find, replace, max_count)
            counts += count
        # print(f'\n=== TextFindAndReplaceAdv ===\nfind [{find}]\nreplace [{replace}]\nmax count [{max_count}]\nmodified text [{modified_text}]\norg text [{text}]\n\n')
        return (modified_text, counts, float(counts), int(counts))

    def replace_substring(self, text, find, replace, max_count):
        # print(f"\nreplace_substring(\n\t[{text}],\n\t[{find}],\n\t[{replace}],\n\t[{max_count}]\n)\n\n")
        modified_text, count = re.subn(re.escape(find), replace, text, max_count)
        # print(f'\n=== find [{find}] replace [{replace}] ===\nin [{text}]\nout [{modified_text}]\nreplaced [{count}] times\n\n')
        return (modified_text, count)
