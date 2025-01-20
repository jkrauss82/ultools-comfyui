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
            if "denoise" in source["inputs"]: props["denoise"] = source["inputs"]["denoise"]
            if source["class_type"] == "BasicScheduler":
                props["scheduler"] = source["inputs"]["scheduler"]
            else:
                props["scheduler"] = source["class_type"].replace("Scheduler", "").lower()
            source = prompt[node["inputs"]["sampler"][0]]
            if "sampler_name" in source["inputs"]: props["sampler_name"] = source["inputs"]["sampler_name"]
    return props

def convSamplerA1111(sampler, scheduler):
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

def traverseOrGetText(order, prompt):
    # print(f'check node {order[0]} input {order[1]}')
    text = []
    text_props = ["text", "clip_l", "clip_g", "t5xxl"]
    if order[0] in prompt:
        node = prompt[order[0]]
        # print(f': {type(list(node.values())[int(order[1])])}')
        if 'inputs' in node:
            # node which has some sort of text input - either as a text field or coming from yet another node
            for text_prop in text_props:
                if text_prop in node['inputs']:
                    if isinstance(node['inputs'][text_prop], str):
                        return node['inputs'][text_prop]
                    else:
                        # check if is find/replace node
                        if 'find' in node['inputs'] and 'replace' in node['inputs']:
                            find_node = prompt[node['inputs']['find'][0]]
                            # print(f'find node {json.dumps(find_node)}')
                            find = ''
                            if 'inputs' in find_node:
                                find = list(find_node['inputs'].values())[node['inputs']['find'][1]]
                                # print(f'find {json.dumps(find)}')
                                if not isinstance(find, str):
                                    # print('traversing find dict')
                                    find = traverseOrGetText(node['inputs']['find'], prompt)
                            replace_node = prompt[node['inputs']['replace'][0]]
                            # print(f'repl node {json.dumps(replace_node)}')
                            replace = ''
                            if 'inputs' in replace_node:
                                replace = list(replace_node['inputs'].values())[node['inputs']['replace'][1]]
                                # print(f'replace {json.dumps(replace)}')
                                if not isinstance(replace, str):
                                    # print('traversing replace dict')
                                    replace = traverseOrGetText(node['inputs']['replace'], prompt)
                            # print('traversing text replace dict')
                            return traverseOrGetText(node['inputs'][text_prop], prompt).replace(find, replace)
                        else:
                            # print('traversing text dict')
                            return traverseOrGetText(node['inputs'][text_prop], prompt)
                # we most likely have a node which handles some properties of condition, let's see how many there are and traverse back each of them
            for prop in node['inputs']:
                if prop.find('conditioning') == 0:
                    text.append(traverseOrGetText(node['inputs'][prop], prompt))

    retVal = []
    for txts in text:
        parts = txts.split(',')
        for txt in parts:
            if not txt.strip(' ') in retVal: retVal.append(txt.strip(' '))
    return ', '.join(retVal)

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
                if 'tile_padding' in params: ultimate_sd_upscale += f', Ultimate SD upscale padding: {params["tile_padding"]}'
            if prompt[order]['class_type'] in ['CheckpointLoaderSimple', 'UNETLoader']:
                prop = 'ckpt_name' if prompt[order]['class_type'] == 'CheckpointLoaderSimple' else 'unet_name'
                if prop in params:
                    model = stripFileExtension(params[prop])
                    # first found model gets selected as creator
                    if genmodel == '': genmodel = f', Model: {model}'
                    # calculate the sha256sum for this model. TODO: store hashes in .txt file next to models
                    if add_hashes:
                        hash = sha256sum(folder_names_and_paths['checkpoints' if prop == 'ckpt_name' else 'unet'][0][0]+'/'+params[prop])
                        hashes[f'model:{model}'] = hash[0:10]
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
