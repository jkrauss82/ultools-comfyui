import hashlib
import json
import re
from pathlib import Path
from folder_paths import folder_names_and_paths, models_dir

hashed = {}
def sha256sum(filename):
    global hashed
    hashstore = f'{models_dir}/hashes.json'
    if len(dict.keys(hashed)) == 0 and Path(hashstore).is_file():
        with open(hashstore) as f:
            hashed = json.load(f)
            print(hashed)
    if filename in hashed: return hashed[filename]
    h  = hashlib.sha256()
    b  = bytearray(128*1024)
    mv = memoryview(b)
    with open(filename, 'rb', buffering=0) as f:
        for n in iter(lambda : f.readinto(mv), 0):
            h.update(mv[:n])
    hashed[filename] = h.hexdigest()
    with open(hashstore, 'w', encoding='utf-8') as f:
        json.dump(hashed, f, ensure_ascii=False, indent=4)
    return hashed[filename]

def stripFileExtension(filename):
    if filename == None: return ''
    return str(filename).rsplit('.', 1)[0]

sampler_props_map = {
    'steps': 'Steps',
    'cfg': 'CFG scale',
    'seed': 'Seed',
    'denoise': 'Denoising strength',
    # handled by convSamplerA1111()
    'sampler_name': None,
    'scheduler': None }

def isSamplerNode(node):
    sampler_classes = [ "KSampler", "KSamplerAdvanced", "SamplerCustom", "DetailerForEach" ]
    prop = "class_type" if "class_type" in node else "type"
    return node[prop] in sampler_classes

def getSamplerProps(node_id, prompt):
    node = prompt[node_id]
    all = True
    props = {}
    for prop in sampler_props_map.keys():
        if prop in node["inputs"] and isinstance(node["inputs"][prop], (str, int, float)):
            props[prop] = node["inputs"][prop]
        else: all = False
    if not all:
        # calculate denoise and correct steps if KSamplerAdvanced
        if node["class_type"] == "KSamplerAdvanced":
            props["steps"] = node["inputs"]["end_at_step"] - node["inputs"]["start_at_step"]
            props["denoise"] = props["steps"] / node["inputs"]["steps"]
        # pull values from plugged input nodes for SamplerCustom
        if node["class_type"] == "SamplerCustom":
            props["seed"] = node["inputs"]["noise_seed"]
            source = prompt[node["inputs"]["sigmas"][0]]
            if "steps" in source["inputs"]: props["steps"] = source["inputs"]["steps"]
            if "denoise" in source["inputs"]:
                props["denoise"] = source["inputs"]["denoise"]
            else:
                props["denoise"] = 1.0 if node["inputs"].get("add_noise", True) else 0.0
            if source["class_type"] == "BasicScheduler":
                props["scheduler"] = source["inputs"]["scheduler"]
            else:
                props["scheduler"] = source["class_type"].replace("Scheduler", "").lower()
            props["sampler_name"] = None
            tries = 0
            try:
                while props["sampler_name"] is None and tries < 3:
                    source = prompt[node["inputs"]["sampler"][0]]
                    if "sampler_name" in source["inputs"]: props["sampler_name"] = source["inputs"]["sampler_name"]
                    else: source = prompt[source["inputs"]["sampler"][0]]
                    tries +=1
            except:
                pass
    return props

def convSamplerA1111(sampler, scheduler):
    if sampler is None: sampler = "N/A"
    sampler = sampler.replace('dpm', 'DPM').replace('pp', '++').replace('_ancestral', ' a').replace('DPM_2', 'DPM2').replace('_sde', ' SDE').replace('2m', '2M').replace('3m', '3M').replace('2s', '2S').replace('euler', 'Euler').replace('ddim', 'DDIM').replace('heun', 'Heun').replace('uni_pc', 'UniPC').replace('_', ' ')
    if scheduler == 'normal': return sampler
    scheduler = scheduler.title()
    return sampler+' '+scheduler

def traverseToPropValue(input, prompt, prop):
    if input[0] in prompt:
        node = prompt[input[0]]
        if 'inputs' in node:
            if prop in node['inputs']:
                if isinstance(node['inputs'][prop], (str, int, float)):
                    return node['inputs'][prop]
                else:
                    return traverseToPropValue(node['inputs'][prop], prompt, prop)

def resolveStringInput(value, prompt):
    """Resolve a single string-producing input of a text node to a string.

    A value is either a plain string (return as-is) or a node reference
    [node_id, slot] (recurse into that node to pull its text). Used by the
    per-node handlers below, where an input may be a literal or the output of
    another node (e.g. a PrimitiveStringMultiline).
    """
    if isinstance(value, str):
        return value
    if isinstance(value, list) and value[0] in prompt:
        return traverseOrGetText(value, prompt)
    return ''

# --------------------------------------------------------------------------- #
# Text-node handlers
#
# Each handler takes (node, prompt) and returns the string that node produces.
# `node` is the full node dict (with 'inputs'); `prompt` is the whole workflow
# so handlers can recurse into upstream nodes via resolveStringInput /
# traverseOrGetText. Add a new ComfyUI string node by writing a handler and
# registering its class_type in _TEXT_NODE_HANDLERS -- no other changes needed.
# --------------------------------------------------------------------------- #

def _handle_default(node, prompt):
    """Generic handler: a node that IS the text source.

    Looks for a text-ish input prop (text / clip_l / clip_g / t5xxl / value)
    that is either a literal string or a reference to an upstream node. This
    covers CLIPTextEncode, PrimitiveString(Multiline), and any other node that
    simply passes a string through. If no such prop exists, falls back to
    traversing any inputs whose name starts with 'conditioning' (Conditioning
    nodes), collecting and de-duplicating their text.
    """
    inputs = node['inputs']
    for text_prop in ("text", "clip_l", "clip_g", "t5xxl", "value"):
        if text_prop in inputs:
            return resolveStringInput(inputs[text_prop], prompt)
    # conditioning-style node: gather text from each conditioning input
    parts = []
    for prop, val in inputs.items():
        if prop.startswith('conditioning') and isinstance(val, list):
            parts.append(traverseOrGetText(val, prompt))
    retVal = []
    for txts in parts:
        for txt in txts.split(','):
            if txt.strip(' ') and txt.strip(' ') not in retVal:
                retVal.append(txt.strip(' '))
    return ', '.join(retVal)

def _handle_string_concatenate(node, prompt):
    """StringConcatenate ("Concatenate Text"): string_a + delimiter + string_b.

    Each string_* input is either a literal or a node reference. Keys are
    sorted so the order is deterministic (string_a, string_b, ...).
    """
    inputs = node['inputs']
    delimiter = inputs.get('delimiter', '')
    parts = [resolveStringInput(inputs[k], prompt)
             for k in sorted(k for k in inputs if k.startswith('string_'))]
    return delimiter.join(parts)

def _handle_string_replace(node, prompt):
    """StringReplace ("Replace Text"): string.replace(find, replace)."""
    inputs = node['inputs']
    s = resolveStringInput(inputs.get('string', ''), prompt)
    find = resolveStringInput(inputs.get('find', ''), prompt)
    replace = resolveStringInput(inputs.get('replace', ''), prompt)
    return s.replace(find, replace)

def _handle_switch(node, prompt):
    """ComfySwitchNode / ComfySoftSwitchNode ("If/Else Switch" / "Soft Switch").

    Picks on_true or on_false based on the boolean `switch` input, then
    resolves the chosen branch (a node reference or literal).
    """
    inputs = node['inputs']
    sw = inputs.get('switch', False)
    branch = inputs.get('on_true' if sw else 'on_false', '')
    return resolveStringInput(branch, prompt)

def _handle_string_format(node, prompt):
    """StringFormat ("Format Text"): f_string.format(**values).

    `values` is an autogrow dict of name -> (string | node ref); `f_string`
    is the Python format template. Only keys that resolve to a string are
    supplied; a template referencing an unresolvable key returns the raw
    template rather than failing prompt extraction.
    """
    inputs = node['inputs']
    f_string = inputs.get('f_string', '')
    values = inputs.get('values', {})
    kwargs = {}
    if isinstance(values, dict):
        for k, v in values.items():
            kwargs[k] = resolveStringInput(v, prompt)
    try:
        return f_string.format(**kwargs)
    except (KeyError, IndexError, ValueError):
        return f_string

def _handle_case_converter(node, prompt):
    """CaseConverter ("Convert Text Case"): applies the selected case mode."""
    inputs = node['inputs']
    s = resolveStringInput(inputs.get('string', ''), prompt)
    mode = inputs.get('mode', '')
    if mode == 'UPPERCASE': return s.upper()
    if mode == 'lowercase': return s.lower()
    if mode == 'Capitalize': return s.capitalize()
    if mode == 'Title Case': return s.title()
    return s

def _handle_string_trim(node, prompt):
    """StringTrim ("Trim Text"): strips whitespace per the selected mode."""
    inputs = node['inputs']
    s = resolveStringInput(inputs.get('string', ''), prompt)
    mode = inputs.get('mode', 'Both')
    if mode == 'Left': return s.lstrip()
    if mode == 'Right': return s.rstrip()
    return s.strip()

def _handle_string_substring(node, prompt):
    """StringSubstring ("Substring"): string[start:end]."""
    inputs = node['inputs']
    s = resolveStringInput(inputs.get('string', ''), prompt)
    start = inputs.get('start', 0)
    end = inputs.get('end', len(s))
    try:
        return s[int(start):int(end)]
    except (TypeError, ValueError):
        return s

def _handle_json_extract_string(node, prompt):
    """JsonExtractString ("Extract Text from JSON"): json.loads(json_string)[key]."""
    inputs = node['inputs']
    json_string = resolveStringInput(inputs.get('json_string', ''), prompt)
    key = inputs.get('key', '')
    try:
        data = json.loads(json_string)
        if isinstance(data, dict) and key in data:
            value = data[key]
            return '' if value is None else str(value)
    except (json.JSONDecodeError, TypeError):
        pass
    return ''

# class_type -> handler. The default handler (generic text/value/clip lookup)
# is used for any class_type not listed here.
_TEXT_NODE_HANDLERS = {
    'StringConcatenate': _handle_string_concatenate,
    'StringReplace': _handle_string_replace,
    'ComfySwitchNode': _handle_switch,
    'ComfySoftSwitchNode': _handle_switch,
    'StringFormat': _handle_string_format,
    'CaseConverter': _handle_case_converter,
    'StringTrim': _handle_string_trim,
    'StringSubstring': _handle_string_substring,
    'JsonExtractString': _handle_json_extract_string,
}

def traverseOrGetText(order, prompt):
    # print(f'check node {order[0]} input {order[1]}')
    if order[0] in prompt:
        node = prompt[order[0]]
        if 'inputs' in node:
            handler = _TEXT_NODE_HANDLERS.get(node.get('class_type'), _handle_default)
            return handler(node, prompt)
    return ''

def automatic1111Format(prompt, image, add_hashes):
    positive_input = ''
    negative_input = ''
    gensampler = ''
    genmodel = ''
    hires = ''
    controlnet = ''
    ultimate_sd_upscale = ''
    hashes = {}
    loras = []

    # print(json.dumps(prompt, indent=4))

    for order in prompt:
        params = None
        if 'inputs' in prompt[order]:
            params = prompt[order]['inputs']
        if params != None and 'class_type' in prompt[order]:
            if 'positive' in params:
                if positive_input == '':
                    # print('positive...')
                    positive_input = traverseOrGetText(params['positive'], prompt)
                    # print(f'found pos: {positive_input}')
            if 'negative' in params:
                if negative_input == '':
                    # print('negative...')
                    negative_input = '\nNegative prompt: ' + traverseOrGetText(params['negative'], prompt)
                    # print(f'found neg: {negative_input}')
            if 'LoraLoader' in prompt[order]['class_type']:
                if 'lora_name' in params and params['lora_name'] != None and params['strength_clip'] + params['strength_model'] > 0:
                    loras.append({ "name": stripFileExtension(params['lora_name']), "weight_clip": params['strength_clip'], "weight_model": params['strength_model'] })
                    # calculate the sha256sum for this lora. TODO: store hashes in .txt file next to loras
                    if add_hashes:
                        hash = sha256sum(folder_names_and_paths['loras'][0][0]+'/'+params['lora_name'])
                        hashes[f'lora:{params["lora_name"]}'] = hash[0:10]
            if isSamplerNode(prompt[order]) and gensampler == '':
                sampler_props = getSamplerProps(order, prompt)
                sampler = convSamplerA1111(sampler_props['sampler_name'], sampler_props['scheduler'])
                width, height = image.size
                if "seed" in sampler_props:
                    gensampler = f'\nSteps: {sampler_props["steps"]}, Sampler: {sampler}, CFG scale: {sampler_props["cfg"]}, Seed: {sampler_props["seed"]}, Denoising strength: {sampler_props["denoise"]}, Size: {width}x{height}'
                else:
                    gensampler = f'\nSteps: {sampler_props["steps"]}, Sampler: {sampler}, CFG scale: {sampler_props["cfg"]}, Denoising strength: {sampler_props["denoise"]}, Size: {width}x{height}'
            if prompt[order]['class_type'] == 'UltimateSDUpscale' and ultimate_sd_upscale == '':
                if 'upscale_model' in params and params['upscale_model'] != None and params['upscale_model'][0] in prompt and prompt[params['upscale_model'][0]]['class_type'] == 'UpscaleModelLoader':
                    model = stripFileExtension(prompt[params['upscale_model'][0]]['inputs']['model_name'])
                ultimate_sd_upscale = f', Ultimate SD upscale upscaler: {model}'
                if 'tile_width' in params: ultimate_sd_upscale += f', Ultimate SD upscale tile_width: {params["tile_width"]}'
                if 'tile_height' in params: ultimate_sd_upscale += f', Ultimate SD upscale tile_height: {params["tile_height"]}'
                if 'mask_blur' in params: ultimate_sd_upscale += f', Ultimate SD upscale mask_blur: {params["mask_blur"]}'
                if 'tile_padding' in params: ultimate_sd_upscale += f', Ultimate SD upscale tile_padding: {params["tile_padding"]}'
            if prompt[order]['class_type'] in ['CheckpointLoaderSimple', 'UNETLoader']:
                prop = 'ckpt_name' if prompt[order]['class_type'] == 'CheckpointLoaderSimple' else 'unet_name'
                if prop in params:
                    model = stripFileExtension(params[prop])
                    # first found model gets selected as creator
                    if genmodel == '': genmodel = f', Model: {model}'
                    # calculate the sha256sum for this model. TODO: store hashes in .txt file next to models
                    if add_hashes:
                        model_dir = 'checkpoints' if prompt[order]['class_type'] == 'CheckpointLoaderSimple' else 'diffusion_models'
                        if model_dir in folder_names_and_paths:
                            for loc in folder_names_and_paths[model_dir][0]:
                                if model_dir in loc and Path(f'{loc}/{params[prop]}').is_file():
                                    hash = sha256sum(f'{loc}/{params[prop]}')
                                    hashes[f'model:{model}'] = hash[0:10]
                        else:
                            print(f'[ULTools SaveImgAdv] Warning: "{model_dir}" no in folder_names_and_paths! Cannot calculate model hash.')
            if prompt[order]['class_type'] == 'UpscaleModelLoader' and hires == '':
                model = stripFileExtension(params['model_name'])
                hires = f', Hires upscaler: {model}'
            if prompt[order]['class_type'] == 'ControlNetApply' and controlnet == '':
                if 'control_net' in params and params['control_net'] != None and params['control_net'][0] in prompt and prompt[params['control_net'][0]]['class_type'] == 'ControlNetLoader':
                    model = stripFileExtension(prompt[params['control_net'][0]]['inputs']['control_net_name'])
                    controlnet = f', ControlNet: "model: {model}, weight: {params["strength"]}"'
    # find embeddings
    prompt_parts = (positive_input + negative_input).split(',')
    for part in prompt_parts:
        if part.strip().startswith('embedding:'):
            result = re.search(r'embedding:([^\s]+)\b', part)
            if result:
                embedding = result.group(1)
                if add_hashes:
                    hash = sha256sum(folder_names_and_paths['embeddings'][0][0]+'/'+embedding)
                    hashes[f'embed:{embedding}'] = hash[0:10]
    # add lora prompt part just like in a1111
    lora_prompt_add = ''
    if len(loras) > 0:
        lora_prompt_add = ', <lora:'+'>, <lora:'.join(f'{l["name"]}:{l["weight_model"]}' for l in loras)+'>'
    uc = positive_input + lora_prompt_add + negative_input + gensampler + genmodel + controlnet + hires + ultimate_sd_upscale + (', Hashes: ' + json.dumps(hashes) if add_hashes else '')
    return uc
