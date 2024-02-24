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

def isSamplerNode(params):
    for param in sampler_props_map:
        if param not in params:
            return False
    return True

def convSamplerA1111(sampler, scheduler):
    sampler = sampler.replace('dpm', 'DPM').replace('pp', '++').replace('_ancestral', ' a').replace('DPM_2', 'DPM2').replace('_sde', ' SDE').replace('2m', '2M').replace('3m', '3M').replace('2s', '2S').replace('euler', 'Euler').replace('ddim', 'DDIM').replace('heun', 'Heun').replace('uni_pc', 'UniPC').replace('_', ' ')
    if scheduler == 'normal': return sampler
    scheduler = scheduler.title()
    return sampler+' '+scheduler

def traverseOrGetText(order, prompt):
    # print(f'check node {order[0]} input {order[1]}')
    text = []
    if order[0] in prompt:
        node = prompt[order[0]]
        # print(f': {type(list(node.values())[int(order[1])])}')
        if 'inputs' in node:
            # node which has some sort of text input - either as a text field or coming from yet another node
            if 'text' in node['inputs']:
                if isinstance(node['inputs']['text'], str):
                    return node['inputs']['text']
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
                        return traverseOrGetText(node['inputs']['text'], prompt).replace(find, replace)
                    else:
                        # print('traversing text dict')
                        return traverseOrGetText(node['inputs']['text'], prompt)
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
            if prompt[order]['class_type'] == 'LoraLoader':
                if 'lora_name' in params and params['lora_name'] != None:
                    loras.append({ "name": stripFileExtension(params['lora_name']), "weight_clip": params['strength_clip'], "weight_model": params['strength_model'] })
                    # calculate the sha256sum for this lora. TODO: store hashes in .txt file next to loras
                    if add_hashes:
                        hash = sha256sum(folder_names_and_paths['loras'][0][0]+'/'+params['lora_name'])
                        hashes[f'lora:{params["lora_name"]}'] = hash[0:10]
            if isSamplerNode(params) and gensampler == '':
                sampler = convSamplerA1111(params['sampler_name'], params['scheduler'])
                width, height = image.size
                gensampler = f'\nSteps: {params["steps"]}, Sampler: {sampler}, CFG scale: {params["cfg"]}, Seed: {params["seed"]}, Denoising strength: {params["denoise"]}, Size: {width}x{height}'
            if prompt[order]['class_type'] == 'UltimateSDUpscale' and ultimate_sd_upscale == '':
                if 'upscale_model' in params and params['upscale_model'] != None and params['upscale_model'][0] in prompt and prompt[params['upscale_model'][0]]['class_type'] == 'UpscaleModelLoader':
                    model = stripFileExtension(prompt[params['upscale_model'][0]]['inputs']['model_name'])
                ultimate_sd_upscale = f', Ultimate SD upscale upscaler: {model}'
                if 'tile_width' in params: ultimate_sd_upscale += f', Ultimate SD upscale tile_width: {params["tile_width"]}'
                if 'tile_height' in params: ultimate_sd_upscale += f', Ultimate SD upscale tile_height: {params["tile_height"]}'
                if 'mask_blur' in params: ultimate_sd_upscale += f', Ultimate SD upscale mask_blur: {params["mask_blur"]}'
                if 'tile_padding' in params: ultimate_sd_upscale += f', Ultimate SD upscale padding: {params["tile_padding"]}'
            if prompt[order]['class_type'] == 'CheckpointLoaderSimple':
                model = stripFileExtension(params['ckpt_name'])
                # first found model gets selected as creator
                if genmodel == '': genmodel = f', Model: {model}'
                # calculate the sha256sum for this model. TODO: store hashes in .txt file next to models
                if add_hashes:
                    hash = sha256sum(folder_names_and_paths['checkpoints'][0][0]+'/'+params['ckpt_name'])
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
