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
Usage: python -m vlens qimg OUT [N_TRAIN]"""
from .common import *
from .models import Reader
from .native import Native, RES

TGT_T = os.environ.get('VLENS_QIMG_TEXT', SEE_T)
N_SHOW, SEED = 12, 11
LAMS = (1e1, 3e1, 1e2, 3e2, 1e3, 3e3, 1e4, 3e4, 1e5)


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

    def fit(idx):
        ev, V = torch.linalg.eigh((Xs[idx].T @ Xs[idx]).double()); ev, V = ev.float(), V.float()
        B = V.T @ (Xs[idx].T @ Yc[idx])
        return lambda lam: V @ (B / (ev + lam)[:, None])
    va = torch.arange(n - v, n)
    f = fit(torch.arange(n - v)); fve = {}
    for lam in LAMS:
        err = ((Xs[va] @ f(lam) - Yc[va]) ** 2).sum(); fve[lam] = 1 - float(err) / float((Yc[va] ** 2).sum())
    del f
    best = max(fve, key=fve.get); W = fit(torch.arange(n))(best)
    del Xs, Yc; torch.cuda.empty_cache()
    return (lambda A: (((A.float() - mu) / sd).to(DEV) @ W).view(len(A), *Y.shape[1:]).cpu() + ym), best, fve


def qimg(out, n_train=6000):
    out = Path(out); (out / 'pics').mkdir(parents=True, exist_ok=True)
    Q = Native(); R = Reader(); log('loaded; target text', repr(TGT_T))
    tr = collect(R, Q, coco_table('train').sample(frac=1, random_state=0).head(int(n_train)), 'tr')
    te = collect(R, Q, coco_table('test').head(int(os.environ.get('VLENS_QIMG_TEST', 100))), 'te')
    log('train photos', len(tr['ims']), 'test photos', len(te['ims']), 'target positions', tuple(tr['tgt'].shape[1:]), 'before the photo', len(tr['pre']))
    pre = tr['pre'].to(DEV, BF); res = dict(target_text=TGT_T, n_train=len(tr['ims']), n_test=len(te['ims']), maps={})
    preds = dict(ceiling=te['tgt'].float(), mean=tr['tgt'].float().mean(0).expand(len(te['ims']), -1, -1))
    for kind in ('see', 'inj', 'hold'):
        f, lam, fve = ridge(tr[kind], tr['tgt'])
        preds[kind] = f(te[kind]); tf = 1 - float(((preds[kind] - te['tgt'].float()) ** 2).sum()) / float(((te['tgt'].float() - tr['tgt'].float().mean(0)) ** 2).sum())
        res['maps'][kind] = dict(strength=lam, val_fve={f'{k:g}': round(x, 4) for k, x in fve.items()}, test_fve=round(tf, 4))
        log(kind, res['maps'][kind])
    pics = {}
    for name, P in preds.items():
        pics[name] = []
        for i in range(len(P)):
            p = Q.picture(torch.cat([pre, P[i].to(DEV, BF)]), SEED); pics[name].append(p)
            if i < N_SHOW:
                p.save(out / 'pics' / f'{name}_{i}.jpg', quality=90)
        log('drew', name, len(P))
    for i in range(N_SHOW):
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
    real = sig(te['ims']); see_c = te['see'].float() - tr['see'].float().mean(0)
    res['scores'] = {}
    for name, P in pics.items():
        with torch.no_grad():
            rb = torch.cat([R.see(to_t([square(p) for p in P[i:i + 32]])).cpu() for i in range(0, len(P), 32)])
        res['scores'][name] = dict(siglip=summary(ranks(sig(P), real)),
                                   readback=summary(ranks(rb - tr['see'].float().mean(0), see_c)))
        log(name, res['scores'][name])
    cols = ['photo'] + list(pics); S = 256
    Gd = Image.new('RGB', (len(cols) * S, N_SHOW * S), 'white')
    for r in range(N_SHOW):
        for k, im in enumerate([te['ims'][r]] + [pics[c][r] for c in pics]):
            Gd.paste(im.resize((S, S)), (k * S, r * S))
    Gd.save(out / 'qimg_grid.jpg', quality=88)
    res['columns'] = cols
    json.dump(res, open(out / 'qimg.json', 'w'), indent=1)
    log('saved', out / 'qimg_grid.jpg')
