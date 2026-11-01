"""Is a picture's information in what it shows, or in fine pixel patterns that only the reader decodes?

Every evaluated picture is read back in its source's context after three degradations a viewer barely notices:
  saved   the 256 px JPEG that evaluate saves (quality 82), scaled back up to 384 px
  jpeg30  that file re-encoded at JPEG quality 30
  half    that file shrunk to 128 px and scaled back up (fine detail removed)
and scored like evaluate (fraction of variance explained, standardized), next to evaluate's full-quality readback on the
same rows. Real photos degraded the same way are the reference. Pictures whose reconstruction falls much further than
the real photos' does carry their information in pixel patterns rather than in the scene.
Usage: python -m vlens robust RUN [N]   (RUN holds eval_rows.pt, pics/ and test_img/ from evaluate; N rows per set)"""
from .common import *
from .models import Reader

CTX = dict(see=('see', None), inj=('inj', None), holdT1=('hold', HOLD_TE[0]), holdT2=('hold', HOLD_TE[1]),
           char=('char', CHAR_T[0]))
KIND = dict(see='see', inj='inj', holdT1='hold', holdT2='hold', char='char')
HOW = ('saved', 'jpeg30', 'half')


def degrade(im, how):
    if how == 'jpeg30':
        b = io.BytesIO(); im.save(b, 'JPEG', quality=30); b.seek(0); im = Image.open(b).convert('RGB')
    elif how == 'half':
        im = im.resize((128, 128), Image.BICUBIC)
    return im.resize((PX, PX), Image.BICUBIC)


def score(zr, za):
    err = ((zr - za) ** 2).sum(1); tot = (za ** 2).sum(1); errp = ((zr - za.roll(1, 0)) ** 2).sum(1)
    s = F.normalize(zr, dim=-1) @ F.normalize(za, dim=-1).T
    rank = (s > s.diag()[:, None]).sum(1).float()
    return dict(fve=round(float(1 - err.sum() / tot.sum()), 4), fve_shuffled_pairs=round(float(1 - errp.sum() / tot.sum()), 4),
                median_rank=float(rank.median()), top1=round(float((rank < 1).float().mean()), 4))


def robust(run, n=300):
    run = Path(run); E = torch.load(run / 'eval_rows.pt', weights_only=False)
    mu, sd, src, full = E['mu'], E['sd'], E['src'], E['readback']
    sets = [k for k in full if k.split('_', 1)[1] in CTX and k.split('_', 1)[0] in ('B', 'C', 'N', 'real')] + ['real_see']
    R = Reader(); res = {}
    with torch.no_grad():
        for name in sets:
            s = name.split('_', 1)[1]; kd, w = CTX[s]; k = KIND[s]; m = min(n, len(src[s]))
            za = (src[s][:m].float() - mu[k]) / sd[k]
            row = dict(n=m)
            # real_see: the source activation is this very photo read in this context, so full quality is exact
            row['full'] = score(za, za) if name == 'real_see' else score((full[name][:m].float() - mu[k]) / sd[k], za)
            ims = [Image.open(run / 'test_img' / f'{i}.jpg' if name.startswith('real_') else run / 'pics' / f'{name}_{i}.jpg').convert('RGB')
                   for i in range(m)]
            for how in HOW:
                x = to_t([degrade(im, how) for im in ims])
                rb = torch.cat([R.readback(x[i:i + 16], kd, w).float().cpu() for i in range(0, m, 16)])
                row[how] = score((rb - mu[k]) / sd[k], za)
            gap = lambda r: r['fve'] - r['fve_shuffled_pairs']               # item information above the shuffled floor
            g0 = gap(row['full'])
            row['kept'] = {how: round(gap(row[how]) / g0, 3) if g0 > 0.01 else None for how in HOW}
            res[name] = row; log(name, 'fve full', row['full']['fve'], ' '.join(f"{how} {row[how]['fve']}" for how in HOW), 'kept', row['kept'])
    json.dump(dict(n=n, how=dict(saved='256 px JPEG q82 (q85 for real photos), scaled up', jpeg30='re-encoded at JPEG q30',
                                 half='shrunk to 128 px, scaled up'),
                   kept='(fve - fve of shuffled pairs) degraded / the same at full quality; None when the full-quality gap is under 0.01', results=res),
              open(run / 'robust.json', 'w'), indent=1)
    log('saved', run / 'robust.json')
