"""Swap test: is a depth-lens picture faithful to what the model holds, judged where we set the truth ourselves?

For pairs (X, Y) of the PAIRS answers, the think prompt for X ("Think of X." + the think tail) runs with the states at
the tail positions (after "X.", the positions the depth lens draws) at layer LP mixed toward the same positions of the
think prompt for Y: h <- (1 - a) h_X + a h_Y. The words still say X, so later layers can pull the thought back to X;
how far each pair moves differs. The model's side is read in the same chat after it replied OK, with the same mix:
a forced choice "Which were you thinking of? A) X B) Y" (logit toward Y minus toward X; the order flips for half the
pairs) and the free answer to "What were you thinking of?". The pictures are drawn from the same states (tail
positions, mixed forward): raw layers (no map) and the last layer, only layers after LP.
A faithful picture says Y exactly where the model says Y. The mix weight a predicts both, so the scores compare
pairs at the same (LP, a): AUC of the picture's SigLIP 2 Y-minus-X score for the model's side, and the 8B judge's
A/B on the picture against the cell's majority side.
Env: VLENS_SWAP_LP (patch layers, default 20,24,28), VLENS_SWAP_ALPHA (default 0.3,0.5,0.7,1), VLENS_SWAP_COLS
(raw layers to draw; the last layer is always drawn; default 28,30,32), VLENS_SWAP_N (pairs, default 48).
Usage: python -m vlens swap OUT"""
from .common import *
from .native import Native
from .depth import PAIRS, TAIL, ASK_THING, tail_len, judge, same

LP = tuple(int(x) for x in os.environ.get('VLENS_SWAP_LP', '20,24,28').split(','))
ALPHA = tuple(float(x) for x in os.environ.get('VLENS_SWAP_ALPHA', '0.3,0.5,0.7,1').split(','))
COLS = tuple(int(x) for x in os.environ.get('VLENS_SWAP_COLS', '28,30,32').split(','))
N, SEED = int(os.environ.get('VLENS_SWAP_N', 48)), 11
ASK_FC = "OK<|im_end|>\n<|im_start|>user\nWhich were you thinking of? A) {a} B) {b} Answer with A or B only.<|im_end|>\n<|im_start|>assistant\n"


def mix_hook(Q, lp, h_y, a, end=None):
    """output of decoder layer lp at the len(h_y) positions ending at `end` (None: the sequence end) <- mix toward
    h_y; skipped on the cached one-token decode steps"""
    k = len(h_y)

    def f(mod, inp, o):
        t = (o[0] if isinstance(o, tuple) else o)
        e = t.shape[1] if end is None else end
        if t.shape[1] < e:
            return None
        t = t.clone(); t[0, e - k:e] = (1 - a) * t[0, e - k:e] + a * h_y.to(t.dtype)
        return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
    return Q.layers[lp - 1].register_forward_hook(f)


@torch.no_grad()
def tail_states(Q, text, k, layers, hooks=()):
    """{layer: [k, D]} at the last k positions of the generator's input, + Q.last from the pipeline's own output"""
    cap = {}
    hs = list(hooks) + [Q.layers[l - 1].register_forward_hook(lambda m, i, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0, -k:]))
                        for l in layers]
    try:
        e, _, pad = Q.pipe._get_qwen_prompt_embeds([text], None, DEV)
    finally:
        for h in hs:
            h.remove()
    cap[Q.last] = e[0][~pad[0].bool()][-k:]
    return cap


@torch.no_grad()
def side(Q, text, x, y, flip, hook=None):
    """the model's answers in the same chat: forced-choice logit toward y minus toward x, and the free answer"""
    pre = Q.pipe.prompt_template_t2i.format(text); tok = Q.proc.tokenizer
    a, b = (y, x) if flip else (x, y)
    ids = tok([pre + ASK_FC.format(a=a, b=b)], return_tensors='pt').to(DEV)
    A, B = tok('A', add_special_tokens=False)['input_ids'][0], tok('B', add_special_tokens=False)['input_ids'][0]
    n_pre = len(tok(pre)['input_ids']); hs = [hook(n_pre)] if hook else []
    try:
        lg = Q.enc(**ids).logits[0, -1].float()
        ids2 = tok([pre + ASK_THING], return_tensors='pt').to(DEV); n = ids2['input_ids'].shape[1]
        o = Q.enc.generate(**ids2, max_new_tokens=12, do_sample=False)
    finally:
        for h in hs:
            h.remove()
    d = float(lg[B] - lg[A]); d = d if not flip else -d          # toward b=y unflipped; flipped, a=y
    return d, tok.decode(o[0, n:], skip_special_tokens=True).strip().strip('.!"').strip()


def swap(out):
    out = Path(out); (out / 'pics').mkdir(parents=True, exist_ok=True)
    Q = Native(); Q.last = len(Q.layers); log('loaded; layers', Q.last, 'patch', LP, 'alpha', ALPHA, 'cols', COLS)
    names = [x for x, _ in PAIRS][:N]; T = lambda x: f'Think of {x}.' + TAIL
    rows, pics = [], {}
    for i, x in enumerate(names):
        y = names[(i + 17) % len(names)]; flip = i % 2 == 1
        k = min(tail_len(Q, T(x)), tail_len(Q, T(y)))
        hy = tail_states(Q, T(y), k, LP)
        cells = [(0, 0.0)] + [(lp, a) for lp in LP for a in ALPHA]
        for lp, a in cells:
            hook = (lambda n_pre, lp=lp, a=a: mix_hook(Q, lp, hy[lp], a, n_pre)) if a else None
            d, free = side(Q, T(x), x, y, flip, hook)
            st = tail_states(Q, T(x), k, [c for c in COLS if c > lp], [mix_hook(Q, lp, hy[lp], a)] if a else [])
            for c in [c for c in COLS if c > lp] + [Q.last]:
                p = Q.picture(st[c].to(DEV, BF), SEED); key = (i, lp, a, c); pics[key] = p
                p.save(out / 'pics' / f'{i}_{lp}_{a}_{c}.jpg', quality=85)
                j = judge(Q, p, *((y, x) if flip else (x, y)))
                rows.append(dict(pair=i, x=x, y=y, lp=lp, a=a, col=c, fc=round(d, 4), free=free,
                                 free_side='y' if same(free, y) else 'x' if same(free, x) else '-',
                                 judge_y=None if j < 0 else int((j == 0) if flip else (j == 1))))
        log('pair', i, x, '->', y, [(r['lp'], r['a'], r['fc'], r['free']) for r in rows if r['pair'] == i and r['col'] == Q.last])
    from transformers import AutoModel, AutoProcessor
    root = snap(SIGLIP); sm = AutoModel.from_pretrained(root, dtype=torch.float16).to(DEV).eval(); sp = AutoProcessor.from_pretrained(root)
    with torch.no_grad():
        o = sm.get_text_features(**sp(text=[f'a photo of {x}.' for x in names], padding='max_length', max_length=64, return_tensors='pt').to(DEV))
        tf = F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu()
        for r in rows:
            o = sm.get_image_features(pixel_values=sp(images=[pics[(r['pair'], r['lp'], r['a'], r['col'])]], return_tensors='pt')['pixel_values'].to(DEV).half())
            f = F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu()[0]
            r['sig'] = round(float(f @ tf[names.index(r['y'])] - f @ tf[names.index(r['x'])]), 4)
    res = dict(lp=LP, alpha=ALPHA, cols=list(COLS) + [Q.last], rows=rows, scores={})
    for c in res['cols']:
        aucs, jacc, base, n_cells = [], [], [], 0
        for lp in LP:
            for a in ALPHA:
                cell = [r for r in rows if r['lp'] == lp and r['a'] == a and r['col'] == c]
                if not cell:
                    continue
                pos = [r['sig'] for r in cell if r['fc'] > 0]; neg = [r['sig'] for r in cell if r['fc'] <= 0]
                if len(pos) >= 5 and len(neg) >= 5:
                    aucs.append(sum((p > q) + 0.5 * (p == q) for p in pos for q in neg) / (len(pos) * len(neg))); n_cells += 1
                jj = [r for r in cell if r['judge_y'] is not None]
                jacc += [r['judge_y'] == (r['fc'] > 0) for r in jj]
                base += [max(len(pos), len(neg)) / len(cell)] * len(jj)
        res['scores'][str(c)] = dict(siglip_auc_within_cell=round(sum(aucs) / max(len(aucs), 1), 4), n_cells=n_cells,
                                     judge_agrees_with_model=round(sum(jacc) / max(len(jacc), 1), 4),
                                     cell_majority=round(sum(base) / max(len(base), 1), 4))
    flips = {f'{lp}_{a}': round(sum(r['fc'] > 0 for r in rows if r['lp'] == lp and r['a'] == a and r['col'] == Q.last) / len(names), 3)
             for lp in LP for a in ALPHA}
    res['model_says_y'] = flips; res['model_says_y_unpatched'] = round(sum(r['fc'] > 0 for r in rows if r['a'] == 0 and r['col'] == Q.last) / len(names), 3)
    log('model says y', res['model_says_y_unpatched'], flips)
    log('swap', res['scores'])
    json.dump(res, open(out / 'swap.json', 'w'), indent=1)
    log('saved', out / 'swap.json')
