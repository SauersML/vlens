"""Adapter into Qwen-Image-2.1, fit in closed form: Qwen3-VL-4B's reply-start activation -> the states Qwen3-VL-8B
hands Qwen-Image-2.1's generator after it has looked at the same photo.

Target per photo: Qwen3-VL-8B's last-layer states (before the final norm) at the text positions after the photo in the
pipeline's own edit template, user text TGT_T (`vlens native` renders photos from exactly these, with no pixels). The
positions before the photo cannot see it and are the same for every photo. Map: ridge regression per source kind,
on standardized activations, strength picked on the last tenth of the training photos. Sources (all Qwen3-VL-4B,
layers 9/18/27 at the reply start): see (it looks at the photo), inj (its reply-start states after layer 8 moved into
a chat with no photo), hold (it reads one of the photo's captions). The generator is never trained and gets no words.
Test photos: pictures from Qwen3-VL-8B's own states (the ceiling), from each source, and from the mean target (the
collapse control). Scores: SigLIP 2 rank of the true photo among the test photos; readback (Qwen3-VL-4B looks at the
picture) rank among the test photos' see activations.
Explore (hold map, never trained on these): Qwen3-VL-4B is told to think of an object without naming it ("Think of
a lighthouse. Don't name it or describe it. Just reply OK."), gets a riddle whose answer it must not say, or plays a
character; SigLIP 2 ranks the picture against the labels of its set.
Usage: python -m vlens qimg OUT [N_TRAIN]"""
from .common import *
from .models import Reader
from .native import Native, RES

TGT_T = os.environ.get('VLENS_QIMG_TEXT', SEE_T)
N_SHOW, SEED = 12, 11
LAMS = (1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5)
THINK = [('a lighthouse', 'a tall tower on a rocky coast with a bright lamp that guides ships at night'),
         ('a giraffe', 'an animal with a very long neck that eats leaves from the tops of trees'),
         ('an elephant', 'the largest land animal, gray, with a trunk and big ears'),
         ('a hot air balloon', 'something that floats in the sky carrying people in a basket under a huge bag of warm air'),
         ('a cactus', 'a spiky green plant that stores water and grows in hot deserts'),
         ('a fire truck', 'a big red vehicle with ladders and hoses that rushes to put out fires'),
         ('a grand piano', 'a large black instrument with 88 black and white keys and a lid that opens like a wing'),
         ('a sailboat', 'something that moves across the water pushed by the wind in its white cloth'),
         ('a tiger', 'a big orange cat with black stripes that lives in the jungle'),
         ('a castle', 'a stone fortress with towers, high walls and a drawbridge where a king lived'),
         ('a surfboard', 'a long flat board people stand on to ride ocean waves'),
         ('a birthday cake', 'a sweet dessert with candles on top that people sing to once a year'),
         ('a snowy mountain', 'a huge peak of rock covered in white powder where people go skiing'),
         ('a bowl of ramen', 'a hot Japanese dish of noodles in broth with an egg and sliced pork'),
         ('a teddy bear', 'a soft stuffed toy animal that children hug at bedtime'),
         ('a red sports car', 'a fast low two-door vehicle painted the color of a fire engine'),
         ('a pizza', 'a round flat bread baked with tomato sauce and melted cheese, cut into slices'),
         ('a bicycle', 'a vehicle with two wheels and pedals that you ride without an engine'),
         ('a clock tower', 'a tall building with a big face and hands that tells the whole town the time'),
         ('a zebra', 'a horse-like African animal with black and white stripes'),
         ('a banana', 'a long curved yellow fruit that monkeys love'),
         ('an umbrella', 'something you open over your head to stay dry in the rain'),
         ('a train', 'a long line of connected carriages that runs on rails'),
         ('a penguin', 'a black and white bird that cannot fly and swims in icy water')]
THINK_T = "Think of {x}. Don't name it or describe it. Just reply OK."
RIDDLE_T = "Riddle: I am {r}. Don't say what I am. Just reply OK."


@torch.no_grad()
def explore(R, Q, f, pre, out, sig, sigt):
    names = [x for x, _ in THINK]; chars = characters()[1][:12]; (out / 'explore').mkdir(exist_ok=True)
    sets = dict(think=([THINK_T.format(x=x) for x in names], names), riddle=([RIDDLE_T.format(r=r) for _, r in THINK], names),
                char=([CHAR_T[0].format(x=x) for x in chars], chars))
    res = {}
    for k, (texts, labels) in sets.items():
        P = f(R.hold(texts)); pics = []
        for i, p in enumerate(P):
            pics.append(Q.picture(torch.cat([pre, p.to(DEV, BF)]), SEED)); pics[-1].save(out / 'explore' / f'{k}_{i}.jpg', quality=90)
        r = ranks(sig(pics), sigt([f'a photo of {l}.' for l in labels]))
        res[k] = dict(summary(r), labels=labels, prompts=texts, rank=r.tolist())
        log('explore', k, 'top1', res[k]['top1'], 'top5', res[k]['top5'], 'of', len(labels))
    return res


def ranks(q, bank):
    s = F.normalize(q.float(), dim=-1) @ F.normalize(bank.float(), dim=-1).T
    return (s > s.diagonal()[:, None]).sum(1).cpu()


def summary(r):
    r = r.float()
    return dict(n=len(r), top1=round(float((r < 1).float().mean()), 4), top5=round(float((r < 5).float().mean()), 4),
                median_rank=float(r.median()), chance_top1=round(1 / len(r), 4))


@torch.no_grad()
def collect(R, Q, df, with_hold):
    """per photo: square photo, see / inj activations, hold activation of its first caption, and Qwen3-VL-8B's states"""
    ims, caps = [], []
    with cf.ThreadPoolExecutor(64) as ex:
        for im, c in zip(ex.map(fetch, df['url'].tolist()), df['sentences']):
            if im is not None:
                ims.append(square(im)); caps.append(clean(c[0]))
    S, I, T = [], [], []; pre = None
    for b in range(0, len(ims), 32):
        a, h = R.see(to_t(ims[b:b + 32]), grab=True); S.append(a.half().cpu()); I.append(R.inject(h).half().cpu())
        for im in ims[b:b + 32]:
            e, pad, _, _ = Q.encode(TGT_T, im)
            k = int(pad[0].float().argmax())                         # text positions before the photo
            pre = e[:k].float().cpu() if pre is None else pre
            T.append(e[k:].half().cpu())
        if b % 1024 == 0:
            log('photos', b + len(ims[b:b + 32]), 'of', len(ims))
    rng = random.Random(0)
    wording = [HOLD_TR[rng.randrange(len(HOLD_TR))] if with_hold == 'tr' else HOLD_TE[0] for _ in caps]
    H = R.hold([w.format(c=c) for w, c in zip(wording, caps)])
    return dict(ims=ims, caps=caps, see=torch.cat(S), inj=torch.cat(I), hold=H, tgt=torch.stack(T), pre=pre)


def ridge(X, Y):
    """standardized X [n,d] -> Y [n,T,D]; strength chosen on the last tenth; returns (predict fn, chosen, val fve)"""
    n = len(X); v = max(n // 10, 1)
    mu, sd = X.float().mean(0), X.float().std(0) + 1e-3; ym = Y.float().mean(0)
    Xs = ((X.float() - mu) / sd).to(DEV); Yc = (Y.float() - ym).flatten(1).to(DEV)

    def fit(m):                                                       # first m rows (views, no copies)
        ev, V = torch.linalg.eigh((Xs[:m].T @ Xs[:m]).double()); ev, V = ev.float(), V.float()
        B = V.T @ (Xs[:m].T @ Yc[:m])
        return lambda lam: V @ (B / (ev + lam)[:, None])
    va = slice(n - v, n)
    f = fit(n - v); fve = {}
    for lam in LAMS:
        err = ((Xs[va] @ f(lam) - Yc[va]) ** 2).sum(); fve[lam] = 1 - float(err) / float((Yc[va] ** 2).sum())
    del f
    best = max(fve, key=fve.get); W = fit(n)(best)
    del Xs, Yc; torch.cuda.empty_cache()
    f = lambda A: (((A.float().cpu() - mu) / sd).to(DEV) @ W).view(len(A), *Y.shape[1:]).cpu() + ym
    f.parts = (W, mu, sd, ym)
    return f, best, fve


def apply_low(k, A):
    """a kept map applied to activations A -> [len(A), T, D]"""
    return (((A.float() - k['mu']) / k['sd']) @ k['U'].float() @ k['V'].float().T).view(len(A), *k['ym'].shape) + k['ym']


def low_rank(f, r=int(os.environ.get('VLENS_QIMG_RANK', 1024))):
    """the map at rank r, small enough to keep (U [d,r] and V [m,r] in fp16 plus the standardization)"""
    W, mu, sd, ym = f.parts; U, S, V = torch.svd_lowrank(W, q=r + 16, niter=4)
    return dict(U=(U[:, :r] * S[:r]).half().cpu(), V=V[:, :r].half().cpu(), mu=mu, sd=sd, ym=ym)


@torch.no_grad()
def rerank(R, Q, pre, P, first, see_z, mu, sd, sig, real, out):
    """best of K by readback, the NLA-style choice with no training: K pictures per test photo from the see map's
    states with different noise; keep the one Qwen3-VL-4B reads back closest to the source activation. Scored by SigLIP
    2 against the photos, beside one draw, the average draw, and the draw picked by the photo itself (not available
    to a reader; the most any choice among these K could add)."""
    K = int(os.environ.get('VLENS_QIMG_K', 4)); n = len(P)
    cand = [first] + [[Q.picture(torch.cat([pre, P[i].to(DEV, BF)]), SEED + 101 * k) for i in range(n)] for k in range(1, K)]
    src = F.normalize(see_z, dim=-1); sims, feats = [], []
    for c in cand:
        rb = torch.cat([R.see(to_t([square(p) for p in c[i:i + 32]])).cpu() for i in range(0, n, 32)])
        sims.append((F.normalize((rb - mu) / sd, dim=-1) * src).sum(-1)); feats.append(sig(c))
    sims, feats = torch.stack(sims), torch.stack(feats)                       # [K, n], [K, n, d]
    rows = torch.arange(n); by_rb = sims.argmax(0); by_photo = (feats * real[None]).sum(-1).argmax(0)
    for i in range(min(n, N_SHOW)):
        cand[int(by_rb[i])][i].save(out / 'pics' / f'see_rerank_{i}.jpg', quality=90)
    res = dict(k=K, one_draw=summary(ranks(feats[0], real)),
               average_draw_top1=round(sum(summary(ranks(feats[k], real))['top1'] for k in range(K)) / K, 4),
               chosen_by_readback=summary(ranks(feats[by_rb, rows], real)),
               chosen_by_the_photo=summary(ranks(feats[by_photo, rows], real)),
               readback_choice_agrees_with_photo_choice=round(float((by_rb == by_photo).float().mean()), 4))
    log('rerank', res)
    return res


def qimg(out, n_train=6000):
    out = Path(out); (out / 'pics').mkdir(parents=True, exist_ok=True)
    Q = Native(); R = Reader(); log('loaded; target text', repr(TGT_T))
    tr = collect(R, Q, coco_table('train').sample(frac=1, random_state=0).head(int(n_train)), 'tr')
    te = collect(R, Q, coco_table('test').head(int(os.environ.get('VLENS_QIMG_TEST', 100))), 'te')
    log('train photos', len(tr['ims']), 'test photos', len(te['ims']), 'target positions', tuple(tr['tgt'].shape[1:]), 'before the photo', len(tr['pre']))
    pre = tr['pre'].to(DEV, BF); res = dict(target_text=TGT_T, n_train=len(tr['ims']), n_test=len(te['ims']), maps={})
    preds = dict(ceiling=te['tgt'].float(), mean=tr['tgt'].float().mean(0).expand(len(te['ims']), -1, -1))
    keep = {}
    for kind in ('see', 'inj', 'hold'):
        f, lam, fve = ridge(tr[kind], tr['tgt'])
        preds[kind] = f(te[kind]); tf = 1 - float(((preds[kind] - te['tgt'].float()) ** 2).sum()) / float(((te['tgt'].float() - tr['tgt'].float().mean(0)) ** 2).sum())
        res['maps'][kind] = dict(strength=lam, val_fve={f'{k:g}': round(x, 4) for k, x in fve.items()}, test_fve=round(tf, 4))
        log(kind, res['maps'][kind])
        if kind == 'hold':
            fh = f
        if kind in ('see', 'hold'):
            keep[kind] = low_rank(f); lp = apply_low(keep[kind], te[kind])
            log(kind, 'kept at rank', keep[kind]['U'].shape[1], 'test fve', round(1 - float(((lp - te['tgt'].float()) ** 2).sum()) / float(((te['tgt'].float() - tr['tgt'].float().mean(0)) ** 2).sum()), 4))
        del f
    d = tr['see'].shape[1] // len(READ)                                # one layer at a time: which depth carries it
    for kind in ('see', 'hold'):
        for j, l in enumerate(READ):
            sl = slice(j * d, (j + 1) * d)
            f, lam, fve = ridge(tr[kind][:, sl], tr['tgt']); preds[f'{kind} L{l}'] = f(te[kind][:, sl]); del f
            res['maps'][f'{kind} L{l}'] = dict(strength=lam, val_fve=round(max(fve.values()), 4)); log(kind, 'layer', l, res['maps'][f'{kind} L{l}'])
    pics = {}
    for name, P in preds.items():
        pics[name] = []
        for i in range(len(P)):
            p = Q.picture(torch.cat([pre, P[i].to(DEV, BF)]), SEED); pics[name].append(p)
            p.save(out / 'pics' / f'{name.replace(" ", "_")}_{i}.jpg', quality=90 if i < N_SHOW else 80)
        log('drew', name, len(P))
    for i in range(len(te['ims'])):
        te['ims'][i].save(out / 'pics' / f'photo_{i}.jpg', quality=90)
    from transformers import AutoModel, AutoProcessor
    root = snap(SIGLIP); sm = AutoModel.from_pretrained(root, dtype=torch.float16).to(DEV).eval(); sp = AutoProcessor.from_pretrained(root)

    @torch.no_grad()
    def sig(ims):
        f = []
        for i in range(0, len(ims), 32):
            o = sm.get_image_features(pixel_values=sp(images=ims[i:i + 32], return_tensors='pt')['pixel_values'].to(DEV).half())
            f.append(F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu())
        return torch.cat(f)
    mu, sd = tr['see'].float().mean(0), tr['see'].float().std(0) + 1e-3     # readback compared standardized
    real = sig(te['ims']); see_z = (te['see'].float() - mu) / sd
    res['scores'] = {}; RB = {}
    for name, P in pics.items():
        with torch.no_grad():
            rb = torch.cat([R.see(to_t([square(p) for p in P[i:i + 32]])).cpu() for i in range(0, len(P), 32)])
        res['scores'][name] = dict(siglip=summary(ranks(sig(P), real)),
                                   readback=summary(ranks((rb - mu) / sd, see_z)))
        log(name, res['scores'][name]); RB[name] = rb.half()
    @torch.no_grad()
    def sigt(texts):
        o = sm.get_text_features(**sp(text=texts, padding='max_length', max_length=64, return_tensors='pt').to(DEV))
        return F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu()
    res['explore'] = explore(R, Q, fh, pre, out, sig, sigt)
    res['rerank'] = rerank(R, Q, pre, preds['see'], pics['see'], see_z, mu, sd, sig, real, out)
    save(dict(maps=keep, pre=tr['pre'], target_text=TGT_T, see_mu=mu, see_sd=sd), out / 'qimg_maps.pt')
    save(dict(readback=RB, see=te['see'], inj=te['inj'], hold=te['hold'], mu=mu, sd=sd, siglip_real=real), out / 'qimg_rows.pt')
    cols = ['photo'] + list(pics); S = 256
    Gd = Image.new('RGB', (len(cols) * S, N_SHOW * S), 'white')
    for r in range(N_SHOW):
        for k, im in enumerate([te['ims'][r]] + [pics[c][r] for c in pics]):
            Gd.paste(im.resize((S, S)), (k * S, r * S))
    Gd.save(out / 'qimg_grid.jpg', quality=88)
    res['columns'] = cols
    json.dump(res, open(out / 'qimg.json', 'w'), indent=1)
    log('saved', out / 'qimg_grid.jpg')
