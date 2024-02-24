import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import os
import json
from comfy.sd1_clip import escape_important, unescape_important, token_weights
import torch


class CLIPTextEncodeWithStats:
    def __init__(self):
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "text": ("STRING", {"multiline": True}),
                "clip": ("CLIP", )
            }
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "IMAGE",)
    FUNCTION = "encode"

    CATEGORY = "conditioning"

    OUTPUT_NODE = True

    embedding_identifier = "embedding:"

    def encode(self, clip, text):
        tokens = clip.tokenize(text, return_word_ids=True)
        words = self.getWords(text)
        stats = {}
        for clipn in tokens:
            clip_name = f'clip_{clipn}'
            if not clip_name in stats: stats[clip_name] = { 'num_batches': len(tokens[clipn]), 'num_tokens': 0, 'batches': [] }
            for batch in tokens[clipn]:
                bat = { 'num_tokens': 0, 'words': {}, 'text': [] }
                for token in batch:
                    # exclude start and stop tokens or avoid index out of bounds (should never happen)
                    if token[2] == 0 or len(words) < token[2]: continue
                    bat['num_tokens'] += 1
                    if not f'word{token[2]}' in bat['words']:
                        bat['words'][f'word{token[2]}'] = { 'word': words[token[2] -1], 'num_tokens': 0, 'weight': 0 }
                    bat['words'][f'word{token[2]}']['num_tokens'] += 1
                    bat['words'][f'word{token[2]}']['weight'] += token[1]
                    stats[clip_name]['num_tokens'] += 1
                stats[clip_name]['batches'].append(bat)

        # build the stats dict for the dataframe
        dfdict = { 'batch': [], 'num_tokens': [], 'num_words': [], 'text': [] }
        for clipn in stats:
            bidx = 0
            for batch in stats[clipn]['batches']:
                bidx += 1
                for word_idx in batch['words']:
                    batch['words'][word_idx]['weight'] = batch['words'][word_idx]['weight'] / batch['words'][word_idx]['num_tokens']
                    batch['text'].append(batch['words'][word_idx]['word'])
                batch['text'] = " ".join(batch['text'])
                dfdict['batch'].append(str(bidx))
                dfdict['num_tokens'].append(batch['num_tokens'])
                dfdict['num_words'].append(len(dict.keys(batch['words'])))
                txt = batch['text'] if len(batch['text']) < 90 else str(batch['text'][:42]+" (..) "+batch['text'][-42:])
                dfdict['text'].append(txt)
            # create sum row
            dfdict['batch'].append('total')
            dfdict['num_tokens'].append(sum(dfdict['num_tokens']))
            dfdict['num_words'].append(sum(dfdict['num_words']))
            dfdict['text'].append(text if len(text) < 90 else str(text[:42]+" (..) "+text[-42:]))

        dfstr = pd.DataFrame.from_dict(dfdict).set_index('batch').to_string()
        # print(dfstr)
        lines = len(dfstr.splitlines())

        fnt = ImageFont.truetype(f"{os.path.dirname(__file__)}/../font/RobotoMono.ttf", 14)

        out = Image.new("RGB", (1024, 20 * lines), (60, 60, 60))

        ImageDraw.Draw(
            out  # Image
        ).text(
            (6, 0),  # Coordinates
            dfstr,  # Text
            (255, 255, 255),  # Color
            fnt
        )

        # Convert the PIL image to Torch tensor
        img_tensor = np.array(out).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_tensor)[None,]

        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled}]], json.dumps(stats, indent=3), img_tensor, )


    # replicates some functionality of SDTokenizer.tokenize_with_weights to extract the prompt words
    def getWords(self, text):
        words = []
        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                # TODO: what about embeddings?
                words.append(word)

        return words
