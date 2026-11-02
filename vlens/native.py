"""Pictures of a real LLM's own internal states, with no training: Qwen3-VL-8B-Instruct -> Qwen-Image-2.1-Turbo.

Qwen-Image-2.1's picture generator was trained to read the last-layer states (before the final norm) of Qwen3-VL-8B,
and its encoder is the public Qwen3-VL-8B-Instruct checkpoint (checked tensor by tensor). Here Qwen3-VL-8B reads a
blank prompt while its states are changed from the inside, and the generator renders whatever its states now hold:
  concept   + a x (mean state of photos whose captions name the concept - mean over all photos), layer STEER_L,
            at every position of the user turn
  neuron    + k sd on the one MLP neuron at STEER_L that best separates those photos from all photos
  random    + 40 sd on random MLP neurons at the same layer (control)
  transplant  the states of the last 5 positions after layer L while Qwen3-VL-8B looks at a real photo ("Look at this
            photo and keep it in mind. Don't describe it. Just reply OK."), copied into the blank prompt at layer L
Every picture in a row uses the same noise. In the same steered model Qwen3-VL-8B is also asked what it is thinking
about, so its words sit beside the pictures.
Usage: python -m vlens native OUT [N_PHOTOS]"""
import re
from .common import *

GEN = 'Qwen/Qwen-Image-2.1-Turbo'
STEER_L, TRANS_L = 18, (8, 18)
CONCEPTS = ['dog', 'cat', 'horse', 'elephant', 'giraffe', 'pizza', 'banana', 'train', 'airplane', 'boat', 'snow',
            'beach', 'kitchen', 'bus', 'clock', 'umbrella']
A_CONCEPT, A_NEURON = (1.0, 2.0, 4.0), (10.0, 20.0, 40.0)
ASK_T = "What are you thinking about right now? Answer in at most eight words."
RES = int(os.environ.get('VLENS_NATIVE_RES', 768))


class Native:
    def __init__(self):
        from diffusers import QwenImage21Pipeline
        self.pipe = QwenImage21Pipeline.from_pretrained(GEN, dtype=BF).to(DEV)
        self.pipe.set_progress_bar_config(disable=True)
        self.enc = self.pipe.text_encoder.eval(); self.proc = self.pipe.processor
        self.layers = self.enc.model.language_model.layers
        self.drop = self.pipe._drop_idx

    def hook_state(self, layer, fn):
        """fn(t) -> t on the output of decoder layer `layer` (1-indexed)"""
        def f(mod, a, o):
            t = fn((o[0] if isinstance(o, tuple) else o).clone())
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        return self.layers[layer - 1].register_forward_hook(f)

    def hook_add(self, layer, v, start):
        """+ v to the output of decoder layer `layer` at positions >= start"""
        def fn(t):
            t[:, start:] += v.to(t.device, t.dtype); return t
        return self.hook_state(layer, fn)

    def hook_neuron(self, layer, n, add, start):
        """+ add to MLP neuron n of decoder layer `layer` (the input of down_proj) at positions >= start"""
        def f(mod, a):
            x = a[0].clone(); x[:, start:, n] += add
            return (x,) + tuple(a[1:])
        return self.layers[layer - 1].mlp.down_proj.register_forward_pre_hook(f)

    @torch.no_grad()
    def embeds(self, hooks=()):
        """the generator's input for a blank prompt, computed by the pipeline's own code with `hooks` active"""
        try:
            return self.pipe._get_qwen_prompt_embeds([' '], None, DEV)[:2]
        finally:
            for h in hooks:
                h.remove()

    @torch.no_grad()
    def picture(self, e, seed):
        pe, pm = e
        return self.pipe(prompt_embeds=pe, prompt_embeds_mask=pm, height=RES, width=RES,
                         generator=torch.Generator('cpu').manual_seed(seed)).images[0]

    @torch.no_grad()
    def see(self, ims, layers):
        """reply-start stats of the vlens see chat: last-5-position states after each layer in `layers` and the last
        position's MLP neuron activations at STEER_L"""
        cap, out = {}, {l: [] for l in layers}; neu = []
        hs = [self.layers[l - 1].register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[:, -5:].float().cpu()))
              for l in layers]
        hs.append(self.layers[STEER_L - 1].mlp.down_proj.register_forward_pre_hook(lambda m, a: cap.__setitem__('n', a[0][:, -1].float().cpu())))
        text = self.proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': SEE_T}]}],
                                             tokenize=False, add_generation_prompt=True)
        try:
            for i in range(0, len(ims), 8):
                b = self.proc(text=[text] * len(ims[i:i + 8]), images=ims[i:i + 8], return_tensors='pt').to(DEV)
                self.enc.model(**{k: v for k, v in b.items()}, use_cache=False)
                for l in layers:
                    out[l].append(cap[l])
                neu.append(cap['n'])
        finally:
            for h in hs:
                h.remove()
        return {l: torch.cat(v) for l, v in out.items()}, torch.cat(neu)

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
    see_t = Q.proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': SEE_T}]}], tokenize=False, add_generation_prompt=True)
    tail = lambda text: Q.proc.tokenizer.convert_ids_to_tokens(Q.proc.tokenizer(text)['input_ids'][-5:])
    assert tail(see_t) == tail(Q.pipe.prompt_template_t2i.format(' ')), (tail(see_t), tail(Q.pipe.prompt_template_t2i.format(' ')))
    df = coco_table('train').sample(frac=1, random_state=0).iloc[40000:].reset_index(drop=True)
    with cf.ThreadPoolExecutor(32) as ex:
        base = [square(im) for im in ex.map(fetch, df['url'].head(300).tolist()) if im is not None][:256]
    G0, N0 = Q.see(base, (STEER_L,)); g_mean = G0[STEER_L][:, -1].mean(0); n_mean, n_sd = N0.mean(0), N0.std(0) + 1e-3
    log('baseline photos', len(base))
    blank = Q.embeds(); SEED = 11
    S = 256; rows, rec = [], []
    ref = Q.picture(blank, SEED); ref.save(out / 'pics' / 'blank.jpg', quality=90)
    for c in CONCEPTS:
        ims = photos(df.iloc[300:], c, n_photos)
        if len(ims) < 4:
            continue
        G, N = Q.see(ims, (STEER_L,))
        v = (G[STEER_L][:, -1].mean(0) - g_mean).to(DEV, BF)
        z = (N.mean(0) - n_mean) / n_sd; nb = int(z.argmax())
        pics, labels = [ref], ['blank prompt']
        for a in A_CONCEPT:
            pics.append(Q.picture(Q.embeds([Q.hook_add(STEER_L, a * v, Q.drop)]), SEED)); labels.append(f'concept x{a:g}')
        for a in A_NEURON:
            pics.append(Q.picture(Q.embeds([Q.hook_neuron(STEER_L, nb, a * float(n_sd[nb]), Q.drop)]), SEED)); labels.append(f'neuron {nb} +{a:g} sd')
        said = dict(concept=Q.words(lambda: [Q.hook_add(STEER_L, A_CONCEPT[-1] * v, 0)]),
                    neuron=Q.words(lambda: [Q.hook_neuron(STEER_L, nb, A_NEURON[-1] * float(n_sd[nb]), 0)]))
        for k, p in enumerate(pics):
            p.save(out / 'pics' / f'{c}_{k}.jpg', quality=90)
        rows.append((ims[:2], pics)); rec.append(dict(concept=c, photos=len(ims), neuron=[STEER_L, nb, round(float(z[nb]), 2)], labels=labels, said=said))
        log(c, 'photos', len(ims), 'neuron', nb, 'z', round(float(z[nb]), 2), said)
    g = torch.Generator().manual_seed(3); ctrl = []
    for n in torch.randint(len(n_sd), (6,), generator=g).tolist():
        p = Q.picture(Q.embeds([Q.hook_neuron(STEER_L, n, A_NEURON[-1] * float(n_sd[n]), Q.drop)]), SEED); p.save(out / 'pics' / f'random_{n}.jpg', quality=90)
        ctrl.append((n, p))
    te = coco_table('test').head(40)
    with cf.ThreadPoolExecutor(32) as ex:
        tims = [square(im) for im in ex.map(fetch, te['url'].tolist()) if im is not None][:8]
    T, _ = Q.see(tims, TRANS_L); trans = []
    for i, im in enumerate(tims):
        row = [im]
        for l in TRANS_L:
            def put(t, H=T[l][i]):
                t[:, -5:] = H.to(t.device, t.dtype); return t
            p = Q.picture(Q.embeds([Q.hook_state(l, put)]), SEED); p.save(out / 'pics' / f'transplant_L{l}_{i}.jpg', quality=90); row.append(p)
        trans.append(row)
    ncol = 2 + len(rows[0][1]) if rows else 3
    Gd = Image.new('RGB', (ncol * S, (len(rows) + 1 + len(trans)) * S), 'white')
    for r, (ims, pics) in enumerate(rows):
        for k, im in enumerate(ims + pics):
            Gd.paste(im.resize((S, S)), (k * S, r * S))
    for k, (_, p) in enumerate(ctrl):
        Gd.paste(p.resize((S, S)), ((2 + k) * S, len(rows) * S))
    for r, row in enumerate(trans):
        for k, im in enumerate(row):
            Gd.paste(im.resize((S, S)), (k * S, (len(rows) + 1 + r) * S))
    Gd.save(out / 'native_grid.jpg', quality=88)
    json.dump(dict(generator=GEN, steer_layer=STEER_L, transplant_layers=TRANS_L, seed=SEED, res=RES, concepts=rec,
                   random_neurons=[n for n, _ in ctrl],
                   columns=['photo', 'photo', 'blank prompt'] + [f'concept x{a:g}' for a in A_CONCEPT] + [f'neuron +{a:g} sd' for a in A_NEURON],
                   transplant_columns=['photo'] + [f'transplant after layer {l}' for l in TRANS_L]), open(out / 'native.json', 'w'), indent=1)
    log('saved', out / 'native_grid.jpg')
