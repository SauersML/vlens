"""Swap test: is a depth-lens picture faithful to what the model holds, judged where we set the truth ourselves?

For pairs (X, Y) of the PAIRS answers whose think prompts ("Think of X." + the think tail) have the same token length,
the prompt for X runs with the states at layer LP mixed toward the prompt for Y at every position from the first token
where the two differ to the end: h <- (1 - a) h_X + a h_Y (the words and the tail after them). Mixing only the tail
positions, the ones the depth lens draws, changed nothing in the model's answer (the answer reads the word positions),
so the words are mixed too. At a = 1 the forward from LP on is Y's; in between, where each pair lands differs.
The model's side is read in the same chat after it replied OK, with the same mix: a forced choice "Which were you
thinking of? A) X B) Y" (logit toward Y minus toward X; the order flips for every other pair) and the free answer to
"What were you thinking of?". The pictures are drawn from the same mixed forward at the tail positions only (after
"X.", as in the depth lens): raw layers after LP (no map) and the last layer.
A faithful picture says Y exactly where the model says Y. The mix weight a predicts both, so the scores compare pairs
at the same (LP, a): AUC of the picture's SigLIP 2 Y-minus-X score for the model's side, and the 8B judge's A/B on
the picture against the cell's majority side.
Env: VLENS_SWAP_LP (patch layers, default 16,24), VLENS_SWAP_ALPHA (default 0.25,0.4,0.55,0.7), VLENS_SWAP_COLS
(raw layers to draw; the last layer is always drawn; default 28,30,32).
Usage: python -m vlens swap OUT"""
from .common import *
from .native import Native
from .depth import PAIRS, TAIL, ASK_THING, tail_len, judge, same

LP = tuple(int(x) for x in os.environ.get('VLENS_SWAP_LP', '16,24').split(','))
ALPHA = tuple(float(x) for x in os.environ.get('VLENS_SWAP_ALPHA', '0.25,0.4,0.55,0.7').split(','))
COLS = tuple(int(x) for x in os.environ.get('VLENS_SWAP_COLS', '28,30,32').split(','))
SEED = 11
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
    names = [x for x, _ in PAIRS]; T = lambda x: f'Think of {x}.' + TAIL
    full = lambda x: Q.proc.tokenizer(Q.pipe.prompt_template_t2i.format(T(x)))['input_ids']; groups = {}
    for x in names:
        groups.setdefault(len(full(x)), []).append(x)
    pairs = [(x, g[(j + 1) % len(g)]) for g in groups.values() if len(g) > 1 for j, x in enumerate(g)]
    log('pairs', len(pairs), 'of', len(names), 'answers (same prompt length)')
    rows, pics = [], {}
    for i, (x, y) in enumerate(pairs):
        flip = i % 2 == 1; a_, b_ = full(x), full(y); assert len(a_) == len(b_)
        kp = len(a_) - next(j for j, (u, v) in enumerate(zip(a_, b_)) if u != v)   # mixed: from the first differing token
        k = tail_len(Q, T(x)); assert k < kp, (x, y, k, kp)                        # drawn: the tail after the words
        hy = tail_states(Q, T(y), kp, LP)
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
    res = score(dict(lp=LP, alpha=ALPHA, cols=list(COLS) + [Q.last], pairs=pairs, rows=rows))
    log('model says y', res['model_says_y']); log('swap', res['scores'])
    json.dump(res, open(out / 'swap.json', 'w'), indent=1)
    log('saved', out / 'swap.json')


def score(res):
    """the model's side is its free answer (y, x, or neither: dropped). The forced choice is kept in the rows but not
    scored: it never moved (both names are in the question, and the model can match them against the prompt's words,
    which are mixed only from LP on). Per (LP, a, column) cell: AUC of SigLIP Y-minus-X for answers y against x, pooled
    over cells by their pair counts (a random y answer and a random x answer from the same cell, ordered right), and
    the 8B judge's and SigLIP's sign agreement with the answer against the cell's majority answer."""
    rows = [r for r in res['rows'] if r['free_side'] in ('x', 'y')]; out = {}
    for c in res['cols']:
        num = den = 0; agree_j = agree_s = maj = n_j = n = 0
        for lp in res['lp']:
            for a in res['alpha']:
                cell = [r for r in rows if r['lp'] == lp and r['a'] == a and r['col'] == c]
                pos = [r['sig'] for r in cell if r['free_side'] == 'y']; neg = [r['sig'] for r in cell if r['free_side'] == 'x']
                num += sum((p > q) + 0.5 * (p == q) for p in pos for q in neg); den += len(pos) * len(neg)
                m = max(len(pos), len(neg)); maj += m; n += len(cell)
                agree_s += sum((r['sig'] > 0) == (r['free_side'] == 'y') for r in cell)
                jj = [r for r in cell if r['judge_y'] is not None]; n_j += len(jj)
                agree_j += sum(bool(r['judge_y']) == (r['free_side'] == 'y') for r in jj)
        out[str(c)] = dict(siglip_auc_within_cell=round(num / max(den, 1), 4), n_pairs_compared=den,
                           judge_agrees=round(agree_j / max(n_j, 1), 4), siglip_sign_agrees=round(agree_s / max(n, 1), 4),
                           cell_majority=round(maj / max(n, 1), 4), n=n)
    last = res['cols'][-1]; cells = lambda lp, a: [r for r in res['rows'] if r['lp'] == lp and r['a'] == a and r['col'] == last]
    res['model_says_y'] = {f'{lp}_{a}': round(sum(r['free_side'] == 'y' for r in cells(lp, a)) / max(len(cells(lp, a)), 1), 3)
                           for lp, a in [(0, 0.0)] + [(lp, a) for lp in res['lp'] for a in res['alpha']]}
    res['model_says_neither'] = round(sum(r['free_side'] == '-' for r in res['rows'] if r['col'] == last) / max(sum(r['col'] == last for r in res['rows']), 1), 3)
    res['scores'] = out
    return res
