"""Constants, source wordings and small helpers shared by every stage."""
import io, json, math, os, random, sys, tarfile, time, urllib.request
import concurrent.futures as cf
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

DEV = 'cuda' if torch.cuda.is_available() else 'mps'
BF = torch.bfloat16
KLEIN, VL, SIGLIP = 'black-forest-labs/FLUX.2-klein-4B', 'Qwen/Qwen3-VL-4B-Instruct', 'google/siglip2-so400m-patch16-512'
READ, PX, M, L_INJ, TP = (9, 18, 27), 384, 64, 8, 5
N_TE = int(os.environ.get('VLENS_NTE', 500))                  # test photos per dataset
TINY = os.environ.get('VLENS_TINY') == '1'                    # plumbing test: random small Qwen3-VL, no CC3M
HERE = Path(__file__).resolve().parent
SEE_T = "Look at this photo and keep it in mind. Don't describe it. Just reply OK."
NEUTRAL_T = "Keep it in mind. Don't describe it. Just reply OK."
HOLD_TR = ["Picture this scene in your mind: {c} Don't describe it. Just reply OK.",
           "Imagine a photo: {c} How many people are in it? Answer with one number.",
           "{c}\nWhat is the main color in this scene? One word.",
           "Think about this image: {c} Do not describe it. Is it indoors or outdoors?",
           "Here is a caption: {c} Rate how calm the scene is from 1 to 10.",
           "You are looking at this: {c} What time of day is it? One word.",
           "Keep this picture in mind: {c} Now answer: what is 2+2?"]
HOLD_TE = ["Picture this scene in your mind: {c} Don't describe it. Just reply OK.",
           "Imagine you are standing inside this scene: {c} Don't describe it. How many legs can you count? Answer with a number."]
CHAR_T = ["You are {x}. Stay in character. Just reply OK.",
          "Pretend to be {x}. Don't say who you are. Reply with one word.",
          "From now on you are {x}. Think about where you are right now. Just reply OK."]
CC3M = 'pixparse/cc3m-wds'
T0 = time.monotonic()


def log(*a):
    print(f'[{time.monotonic() - T0:7.0f}s]', *a, flush=True)


def snap(repo, ignore=None):
    from huggingface_hub import snapshot_download, constants
    d = Path(constants.HF_HUB_CACHE) / ('models--' + repo.replace('/', '--')) / 'snapshots'
    if os.environ.get('HF_HUB_OFFLINE') == '1' and d.exists():
        return str(next(p for p in d.iterdir() if p.is_dir()))
    return snapshot_download(repo, ignore_patterns=ignore)


def clean(c):
    c = ' '.join(str(c).split()).strip()
    return c[0].upper() + c[1:] + ('' if c.endswith('.') else '.')


def characters():
    xs = [l.strip() for l in open(HERE / 'characters.txt') if l.strip()]
    return [x for k, x in enumerate(xs) if k % 5], [x for k, x in enumerate(xs) if k % 5 == 0][:60]


def fetch(url):
    for k in range(4):
        try:
            return Image.open(io.BytesIO(urllib.request.urlopen(url, timeout=30).read())).convert('RGB')
        except Exception:
            time.sleep(1 + k)
    return None


def square(im, s=PX):
    w, h = im.size; m = min(w, h)
    return im.crop(((w - m) // 2, (h - m) // 2, (w - m) // 2 + m, (h - m) // 2 + m)).resize((s, s), Image.BICUBIC)


def to_t(ims):
    """PIL list -> [B,3,H,W] float in [-1,1]"""
    return torch.stack([torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1) for im in ims]).float().div(127.5).sub(1)


def to_pil(x):
    x = ((x.float().clamp(-1, 1) + 1) * 127.5).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(a) for a in x]


def save(obj, path):
    p = str(path) + '.partial'; torch.save(obj, p); os.replace(p, path)




def coco_table(split):
    """Karpathy-split COCO captions and image URLs (yerevann/coco-karpathy on the Hugging Face Hub, about 18 MB)"""
    local = os.environ.get('VLENS_COCO_DIR')
    if local:
        return pd.read_parquet(Path(local) / f'{split}-00000-of-00001.parquet')
    from huggingface_hub import hf_hub_download
    return pd.read_parquet(hf_hub_download('yerevann/coco-karpathy', f'data/{split}-00000-of-00001.parquet', repo_type='dataset'))
