"""Pictures of activations made with no picture and no description anywhere in the context.

In a neutral text-only chat (NEUTRAL_T), Qwen3-VL's residual stream is steered at the reply-start positions:
  concept  after layer L_INJ: + alpha * (mean reply-start state of photos whose captions name the concept
           - mean over all photos)
  neuron   one MLP neuron (the input of down_proj at a layer): + alpha * that neuron's spread over photos. The neuron is
           the one whose activation separates the concept's photos from all photos best (a z-score). Random neurons
           at the same layers and strengths are the control.
The steered activation (layers 9/18/27, last position) is drawn by a trained adapter. In the same steered chat
Qwen3-VL is asked what it is thinking about, so its own words sit beside the picture.
Usage: python -m vlens explore ADAPTER OUT [N_PHOTOS]   (ADAPTER: con_B_fp16.pt or con_C_fp16.pt)"""
import re
from .common import *
from .models import Reader, Klein, Connector

CONCEPTS = ['dog', 'cat', 'horse', 'elephant', 'giraffe', 'pizza', 'banana', 'train', 'airplane', 'boat', 'snow',
            'beach', 'kitchen', 'bus', 'clock', 'umbrella']
NEURON_LAYERS = (12, 16, 20)
ASK_T = "What are you thinking about right now? Answer in at most eight words."
A_CONCEPT = (1.0, 2.0, 4.0)
A_NEURON = (10.0, 20.0, 40.0)


def load_adapter(path):
    st = torch.load(path, weights_only=False)['state']
    st = {k: v.float() for k, v in st.items()}
    con = Connector(st['mu'].shape[0], st['q0'], st['mu'], st['sd'])
    con.load_state_dict(st); return con.to(DEV).eval()


def photos_for(df, word, n):
    pat = re.compile(rf'\b{word}s?\b', re.I)
    hit = df[df['sentences'].map(lambda caps: any(pat.search(str(c)) for c in caps))]
    return hit.head(n)


def fetch_square(urls):
    with cf.ThreadPoolExecutor(32) as ex:
        ims = list(ex.map(fetch, urls))
    return [square(im) for im in ims if im is not None]


class Steer:
    """forward hooks that add to the reply-start positions (the last TP positions of every forward call)"""
    def __init__(self, R):
        self.R, self.h = R, []

    def state(self, v):                                       # v [TP, d] after layer L_INJ
        def f(mod, a, o):
            t = (o[0] if isinstance(o, tuple) else o).clone(); k = min(TP, t.shape[1])
            t[:, -k:] += v[-k:].to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        self.h.append(self.R.layers[L_INJ - 1].register_forward_hook(f)); return self

    def neuron(self, layer, n, alpha):                       # layer 1-indexed decoder layer, n MLP neuron index
        def f(mod, a):
            x = a[0].clone(); k = min(TP, x.shape[1]); x[:, -k:, n] += alpha
            return (x,) + tuple(a[1:])
        self.h.append(self.R.layers[layer - 1].mlp.down_proj.register_forward_pre_hook(f)); return self

    def __enter__(self):
        return self

    def __exit__(self, *e):
        for h in self.h:
            h.remove()


@torch.no_grad()
def photo_stats(R, ims):
    """reply-start states after L_INJ [B,TP,d] and MLP neuron activations at NEURON_LAYERS (last position) of see()"""
    cap = {}
    hooks = [R.layers[l - 1].mlp.down_proj.register_forward_pre_hook(lambda m, a, l=l: cap.__setitem__(l, a[0][:, -1].float().cpu()))
             for l in NEURON_LAYERS]
    G, N = [], {l: [] for l in NEURON_LAYERS}
    try:
        x = to_t(ims)
        for b in range(0, len(x), 16):
            G.append(R.see(x[b:b + 16], grab=True)[1].float().cpu())
            for l in NEURON_LAYERS:
                N[l].append(cap[l])
    finally:
        for h in hooks:
            h.remove()
    return torch.cat(G), {l: torch.cat(v) for l, v in N.items()}


@torch.no_grad()
def neutral_activation(R):
    ids = R.neutral_ids
    return R._run(torch.full((1,), ids.shape[1] - 1, device=DEV), input_ids=ids, attention_mask=torch.ones_like(ids))


@torch.no_grad()
def words(full, proc, steer_fn):
    """Qwen3-VL's own answer to ASK_T in the steered model (greedy, 16 tokens)"""
    text = proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': ASK_T}]}], tokenize=False, add_generation_prompt=True)
    enc = proc.tokenizer([text], return_tensors='pt').to(DEV)
    with steer_fn():
        out = full.generate(**enc, max_new_tokens=16, do_sample=False)
    return proc.tokenizer.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def explore(adapter, out, n_photos=32):
    out = Path(out); out.mkdir(parents=True, exist_ok=True)
    from transformers import Qwen3VLForConditionalGeneration
    full = None if TINY else Qwen3VLForConditionalGeneration.from_pretrained(snap(VL), dtype=BF).to(DEV).eval()
    R = Reader(model=full, cut=full is None)                  # shares the full model's weights (all layers, so it can also answer)
    df = coco_table('train').sample(frac=1, random_state=0).iloc[40000:].reset_index(drop=True)
    base_ims = fetch_square(df['url'].head(256 if not TINY else 8).tolist())
    G0, N0 = photo_stats(R, base_ims); g_mean = G0.mean(0); n_mean = {l: v.mean(0) for l, v in N0.items()}; n_sd = {l: v.std(0) + 1e-3 for l, v in N0.items()}
    log('baseline photos', len(base_ims))
    rows, rec = [], []
    neutral = neutral_activation(R)
    for c in (CONCEPTS if not TINY else CONCEPTS[:2]):
        sel = photos_for(df, c, n_photos if not TINY else 4)
        ims = fetch_square(sel['url'].tolist())
        if len(ims) < 2:
            continue
        G, N = photo_stats(R, ims)
        v = (G.mean(0) - g_mean).to(DEV)
        z = {l: (N[l].mean(0) - n_mean[l]) / n_sd[l] for l in NEURON_LAYERS}
        l_best = max(NEURON_LAYERS, key=lambda l: float(z[l].max())); n_best = int(z[l_best].argmax())
        acts, labels = [neutral], ['no steering']
        for a in A_CONCEPT:
            with Steer(R).state(a * v):
                acts.append(neutral_activation(R)); labels.append(f'concept x{a:g}')
        for a in A_NEURON:
            with Steer(R).neuron(l_best, n_best, a * float(n_sd[l_best][n_best])):
                acts.append(neutral_activation(R)); labels.append(f'neuron L{l_best} #{n_best} +{a:g} sd')
        said = {}
        if full is not None:
            said['concept'] = words(full, R.proc, lambda: Steer(R).state(A_CONCEPT[-1] * v))
            said['neuron'] = words(full, R.proc, lambda: Steer(R).neuron(l_best, n_best, A_NEURON[-1] * float(n_sd[l_best][n_best])))
        rows.append((c, ims[:3], torch.cat(acts), labels))
        rec.append(dict(concept=c, photos=len(ims), neuron=[l_best, n_best, round(float(z[l_best][n_best]), 2)], labels=labels, said=said))
        log(c, 'photos', len(ims), 'neuron', l_best, n_best, 'z', round(float(z[l_best][n_best]), 2), said)
    g = torch.Generator().manual_seed(3); ctrl = []
    for l in NEURON_LAYERS:
        for n in torch.randint(len(n_sd[l]), (2,), generator=g).tolist():
            with Steer(R).neuron(l, n, A_NEURON[-1] * float(n_sd[l][n])):
                ctrl.append((f'random neuron L{l} #{n}', neutral_activation(R)))
    del R; torch.cuda.empty_cache()
    if full is not None:
        del full
    K = Klein(); con = load_adapter(adapter)
    S = 160; ncol = 3 + 1 + len(A_CONCEPT) + len(A_NEURON)
    Gd = Image.new('RGB', (ncol * S, (len(rows) + 1) * S), 'white')
    with torch.no_grad():
        for r, (c, ims, A, labels) in enumerate(rows):
            pics = to_pil(K.draw(con(A.to(DEV)), [r] * len(A)))
            for k, im in enumerate(ims):
                Gd.paste(im.resize((S, S)), (k * S, r * S))
            for k, im in enumerate(pics):
                Gd.paste(im.resize((S, S)), ((3 + k) * S, r * S)); im.save(out / f'{c}_{k}.jpg', quality=90)
        cp = to_pil(K.draw(con(torch.cat([a for _, a in ctrl]).to(DEV)), [99] * len(ctrl)))
        for k, im in enumerate(cp[:ncol]):
            Gd.paste(im.resize((S, S)), (k * S, len(rows) * S)); im.save(out / f'random_{k}.jpg', quality=90)
    Gd.save(out / 'explore_grid.jpg', quality=88)
    json.dump(dict(concepts=rec, controls=[n for n, _ in ctrl], columns=['photo'] * 3 + rows[0][3] if rows else []), open(out / 'explore.json', 'w'), indent=1)
    log('saved', out / 'explore_grid.jpg')
