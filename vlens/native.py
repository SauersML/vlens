"""Pictures of a real LLM's own internal states, with no training: Qwen3-VL-8B-Instruct -> Qwen-Image-2.1-Turbo.

Qwen-Image-2.1's picture generator reads the last-layer states (before the final norm) of Qwen3-VL-8B, and its encoder
is the public Qwen3-VL-8B-Instruct checkpoint (checked tensor by tensor). When the pipeline edits a photo, the photo's
own positions in that sequence are replaced by the photo's VAE latents, so Qwen3-VL's states at the image tokens never
reach the generator; only its states at the text positions do. Those include the positions after the photo, which have
attended to it: the reply start.

Two tests:
  photo rows   Qwen3-VL-8B looks at a photo inside the pipeline's own edit template. The generator gets only Qwen's
               last-layer states at the text positions (no pixels, no latents, no image tokens):
                 with pixels    the normal edit path with the photo's latents, for reference
                 states, blank  the user text is " "
                 states, see    the user text is the vlens see prompt ("Look at this photo ... Just reply OK.")
                 reply start    only the last 5 positions of "states, see"
  concept rows Qwen3-VL-8B reads a blank prompt while its states are pushed along a direction, at layer STEER_L, at
               every position after the system prompt. Strength s is in units of the typical state size there:
                 concept  the mean reply-start state of photos whose captions name the concept minus that of all photos
                 neuron   the one MLP neuron at STEER_L that best separates those photos from all photos, raised until
                          its write has the same size
               In the same steered model Qwen3-VL-8B is asked what it is thinking about, so its words sit beside them.
Every picture uses the same noise.
Usage: python -m vlens native OUT [N_PHOTOS]"""
import re
from .common import *

GEN = 'Qwen/Qwen-Image-2.1-Turbo'
STEER_L = 18
CONCEPTS = ['dog', 'cat', 'horse', 'elephant', 'giraffe', 'pizza', 'banana', 'train', 'airplane', 'boat', 'snow',
            'beach', 'kitchen', 'bus', 'clock', 'umbrella']
S_CONCEPT, S_NEURON = (0.5, 1.0, 2.0, 4.0), (1.0, 2.0, 4.0)
ASK_T = "What are you thinking about right now? Answer in at most eight words."
RES = int(os.environ.get('VLENS_NATIVE_RES', 768))
N_TEST = int(os.environ.get('VLENS_NATIVE_TEST', 16))


class Native:
    def __init__(self):
        from diffusers import QwenImage21Pipeline
        self.pipe = QwenImage21Pipeline.from_pretrained(GEN, dtype=BF).to(DEV)
        self.pipe.set_progress_bar_config(disable=True)
        self.enc = self.pipe.text_encoder.eval(); self.proc = self.pipe.processor
        self.layers = self.enc.model.language_model.layers
        self.drop = self.pipe._drop_idx

    def hook_add(self, layer, v, start):
        """+ v to the output of decoder layer `layer` (1-indexed) at positions >= start"""
        def f(mod, a, o):
            t = (o[0] if isinstance(o, tuple) else o).clone(); t[:, start:] += v.to(t.device, t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return self.layers[layer - 1].register_forward_hook(f)

    def hook_neuron(self, layer, n, add, start):
        """+ add to MLP neuron n of decoder layer `layer` (the input of down_proj) at positions >= start"""
        def f(mod, a):
            x = a[0].clone(); x[:, start:, n] += add
            return (x,) + tuple(a[1:])
        return self.layers[layer - 1].mlp.down_proj.register_forward_pre_hook(f)

    @torch.no_grad()
    def encode(self, text=' ', im=None, hooks=(), capture=None):
        """the pipeline's own encoding of `text` (and photo `im`) with `hooks` active: (states at the text positions
        after the system prompt, all positions' image-pad mask, full states, captured). With `capture` = a layer, also
        returns that layer's output and the MLP neuron activations there, at the last 5 positions."""
        cap = {}
        hs = list(hooks)
        if capture:
            hs.append(self.layers[capture - 1].register_forward_hook(
                lambda m, a, o: cap.__setitem__('h', (o[0] if isinstance(o, tuple) else o)[0, -5:].float().cpu())))
            hs.append(self.layers[capture - 1].mlp.down_proj.register_forward_pre_hook(
                lambda m, a: cap.__setitem__('n', a[0][0, -5:].float().cpu())))
        try:
            e, m, pad = self.pipe._get_qwen_prompt_embeds([text], None if im is None else [im], DEV)
        finally:
            for h in hs:
                h.remove()
        return e[0][~pad[0].bool()], pad, (e, m), cap

    @torch.no_grad()
    def picture(self, e, seed, image=None):
        """render states e (positions x 4096, text positions only), or the normal edit of `image` when e is None"""
        kw = dict(height=RES, width=RES, generator=torch.Generator('cpu').manual_seed(seed))
        if e is None:
            out = self.pipe(prompt=' ', image=[image], **kw)
        else:
            out = self.pipe(prompt_embeds=e[None], prompt_embeds_mask=torch.ones(1, len(e), dtype=torch.long, device=e.device), **kw)
        return out.images[0].convert('RGB')   # the generator outputs RGBA

    @torch.no_grad()
    def words(self, hook_fn):
        text = self.proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': ASK_T}]}], tokenize=False, add_generation_prompt=True)
        enc = self.proc.tokenizer([text], return_tensors='pt').to(DEV)
        hs = hook_fn()
        try:
            out = self.enc.generate(**enc, max_new_tokens=16, do_sample=False)
        finally:
            for h in hs:
                h.remove()
        return self.proc.tokenizer.decode(out[0, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip()


def photos(df, word, n):
    pat = re.compile(rf'\b{word}s?\b', re.I)
    hit = df[df['sentences'].map(lambda caps: any(pat.search(str(c)) for c in caps))].head(int(n * 1.3))
    with cf.ThreadPoolExecutor(32) as ex:
        ims = [square(im) for im in ex.map(fetch, hit['url'].tolist()) if im is not None]
    return ims[:n]


def native(out, n_photos=32):
    out = Path(out); (out / 'pics').mkdir(parents=True, exist_ok=True)
    Q = Native(); log('loaded', GEN, 'drop', Q.drop)
    SEED, S = 11, 256
    save = lambda p, name: p.save(out / 'pics' / f'{name}.jpg', quality=90) or p

    # photo rows: the generator gets only Qwen's states, never the pixels
    te = coco_table('test').head(int(N_TEST * 1.5))
    with cf.ThreadPoolExecutor(32) as ex:
        tims = [square(im) for im in ex.map(fetch, te['url'].tolist()) if im is not None][:N_TEST]
    prow, pcols = [], ['photo', 'with pixels', 'states, blank', 'states, see', 'reply start']
    for i, im in enumerate(tims):
        eb, pad, _, _ = Q.encode(' ', im)
        es, _, _, _ = Q.encode(SEE_T, im)
        assert int(pad.sum()) > 0 and len(eb) < pad.shape[1], (pad.shape, len(eb))
        row = [im, save(Q.picture(None, SEED, im), f'photo{i}_pixels'), save(Q.picture(eb, SEED), f'photo{i}_blank'),
               save(Q.picture(es, SEED), f'photo{i}_see'), save(Q.picture(es[-5:], SEED), f'photo{i}_reply')]
        prow.append(row)
        if i == 0:
            log('text positions, blank', len(eb), 'see', len(es), 'image pads', int(pad.sum()))
    log('photo rows', len(prow))

    # concept rows: steering at STEER_L, strength in units of the typical state size there
    blank, _, _, bc = Q.encode(' ', capture=STEER_L)
    hn = float(bc['h'].norm(dim=-1).median()); ref = save(Q.picture(blank, SEED), 'blank')
    wd = Q.layers[STEER_L - 1].mlp.down_proj.weight.float().norm(dim=0).cpu()        # write size per neuron
    df = coco_table('train').sample(frac=1, random_state=0).iloc[40000:].reset_index(drop=True)
    with cf.ThreadPoolExecutor(32) as ex:
        base = [square(im) for im in ex.map(fetch, df['url'].head(300).tolist()) if im is not None][:256]

    def stats(ims):
        H, N = [], []
        for im in ims:
            c = Q.encode(' ', im, capture=STEER_L)[3]; H.append(c['h'].mean(0)); N.append(c['n'].mean(0))
        return torch.stack(H), torch.stack(N)
    H0, N0 = stats(base); h_mean, n_mean, n_sd = H0.mean(0), N0.mean(0), N0.std(0) + 1e-3
    log('baseline photos', len(base), 'state size', round(hn, 1))
    crow, rec = [], []
    for c in CONCEPTS:
        ims = photos(df.iloc[300:], c, n_photos)
        if len(ims) < 4:
            continue
        H, N = stats(ims)
        d = H.mean(0) - h_mean; u = (d / d.norm()).to(DEV, BF)
        z = (N.mean(0) - n_mean) / n_sd; nb = int(z.argmax()); per = hn / float(wd[nb])
        pics = [ref]
        for s in S_CONCEPT:
            e = Q.encode(' ', hooks=[Q.hook_add(STEER_L, s * hn * u, Q.drop)])[0]; pics.append(save(Q.picture(e, SEED), f'{c}_concept{s:g}'))
        for s in S_NEURON:
            e = Q.encode(' ', hooks=[Q.hook_neuron(STEER_L, nb, s * per, Q.drop)])[0]; pics.append(save(Q.picture(e, SEED), f'{c}_neuron{s:g}'))
        said = {f'concept {s:g}': Q.words(lambda s=s: [Q.hook_add(STEER_L, s * hn * u, 1)]) for s in (1.0, 2.0)}
        said['neuron 2'] = Q.words(lambda: [Q.hook_neuron(STEER_L, nb, 2 * per, 1)])
        crow.append((ims[:2], pics))
        rec.append(dict(concept=c, photos=len(ims), mean_gap_over_state_size=round(float(d.norm()) / hn, 3),
                        neuron=[STEER_L, nb, round(float(z[nb]), 2)], said=said))
        log(c, 'photos', len(ims), 'gap/size', rec[-1]['mean_gap_over_state_size'], 'neuron', nb, 'z', round(float(z[nb]), 2), said)
    ccols = ['photo', 'photo', 'blank prompt'] + [f'concept s{s:g}' for s in S_CONCEPT] + [f'neuron s{s:g}' for s in S_NEURON]

    W = max(len(pcols), len(ccols)); Gd = Image.new('RGB', (W * S, (len(prow) + len(crow)) * S), 'white')
    for r, row in enumerate(prow + [ims + pics for ims, pics in crow]):
        for k, im in enumerate(row):
            Gd.paste(im.resize((S, S)), (k * S, r * S))
    Gd.save(out / 'native_grid.jpg', quality=88)
    json.dump(dict(generator=GEN, steer_layer=STEER_L, seed=SEED, res=RES, state_size=hn, photo_columns=pcols,
                   concept_columns=ccols, concepts=rec), open(out / 'native.json', 'w'), indent=1)
    log('saved', out / 'native_grid.jpg')
