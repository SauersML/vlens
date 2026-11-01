"""Evaluation: pictures from held-out sources; reconstruction (fraction of variance explained) with each picture read back in its
source's own context, against the real photo read back the same way (ceiling) and shuffled pairs (floor); readback ranks;
SigLIP 2 image-image ranks; the text route and the random-activation prior control."""
from .common import *
from .models import Reader, Klein
from .train import load_data, make_connector, test_idx


def ranks(q, bank, truth):
    """rank of the true row (0 = best) of each query by cosine; q [N,d], bank [K,d] both already centered"""
    s = F.normalize(q.float(), dim=-1) @ F.normalize(bank.float(), dim=-1).T
    return (s > s[torch.arange(len(q)), truth][:, None]).sum(1).cpu()


def summary(r):
    r = r.float()
    return dict(n=len(r), top1=round(float((r < 1).float().mean()), 4), top10=round(float((r < 10).float().mean()), 4),
                median_rank=float(r.median()), mean_rank=round(float(r.mean()), 1))


def evaluate(W, run):
    W, run = Path(W), Path(run)
    D = load_data(W); te, tr = D['te'], D['tr']
    good = test_idx(D, 2 * N_TE); n = len(good); nte = len(te['see'])
    is_coco = (te['src'][good] == 0)
    allA = torch.cat([tr['see'][D['tr_img']], tr['inj'][D['tr_img']], tr['hold'][D['tr_hold']]])
    allA = allA[torch.randperm(len(allA), generator=torch.Generator().manual_seed(0))[:30000]].float()
    g = torch.Generator().manual_seed(11)
    NX = int(os.environ.get('VLENS_NEXTRA', 60))
    rand = allA.mean(0) + allA.std(0) * torch.randn(NX, allA.shape[1], generator=g)    # the prior control
    srcs = dict(see=te['see'][good], inj=te['inj'][good], holdT1=te['hold'][good], holdT2=te['hold'][nte + good],
                char=D['char']['A_te'][::len(CHAR_T)][:NX], random=rand.half())
    K = Klein(); pics = {}
    for tag in ('B', 'C', 'N'):
        p = W / f'con_{tag}.pt'
        if not p.exists():
            continue
        con = make_connector(D).to(DEV); con.load_state_dict(torch.load(p, weights_only=False)['state']); con.eval()
        for s, A in srcs.items():
            out = []
            with torch.no_grad():
                for i in range(0, len(A), 25):
                    out.append(((K.draw(con(A[i:i + 25].to(DEV)), list(range(i, i + len(A[i:i + 25])))) + 1) * 127.5).round().byte().cpu())
            pics[f'{tag}_{s}'] = torch.cat(out); log('drew', tag, s, len(A))
        del con
    del K; torch.cuda.empty_cache()
    K = Klein(text_encoder=True)
    for name, prompts in (('text_caption', [te['caps'][int(j)][0] for j in good]), ('text_char', D['char']['te'][:NX])):
        out = []
        with torch.no_grad():
            for i in range(0, len(prompts), 25):
                ims = K.pipe(prompt=prompts[i:i + 25], height=PX, width=PX, num_inference_steps=4, guidance_scale=1.0,
                             generator=[torch.Generator('cpu').manual_seed(i + k) for k in range(len(prompts[i:i + 25]))]).images
                out.append(to_t(ims).add(1).mul(127.5).round().byte())
        pics[name] = torch.cat(out); log('drew the text route', name, len(prompts))
    del K; torch.cuda.empty_cache()
    real = torch.stack([torch.from_numpy(np.asarray(Image.open(W / 'test_img' / f'{int(j)}.jpg').convert('RGB'))).permute(2, 0, 1) for j in good])
    (run / 'pics').mkdir(exist_ok=True); (run / 'test_img').mkdir(exist_ok=True)
    for k in range(n):
        Image.fromarray(real[k].permute(1, 2, 0).numpy()).resize((256, 256), Image.BICUBIC).save(run / 'test_img' / f'{k}.jpg', quality=85)
    for name, P in pics.items():
        for k in range(len(P)):
            Image.fromarray(P[k].permute(1, 2, 0).numpy()).resize((256, 256), Image.BICUBIC).save(run / 'pics' / f'{name}_{k}.jpg', quality=82)
    # which source each picture set came from and the context its readback uses (the picture in place of the content)
    ctx = {}
    for name in pics:
        s_ = name.split('_', 1)[1]
        s_ = dict(caption='holdT1').get(s_, s_)
        ctx[name] = (s_, dict(see=('see', None), inj=('inj', None), holdT1=('hold', HOLD_TE[0]), holdT2=('hold', HOLD_TE[1]),
                              char=('char', CHAR_T[0]), random=('see', None))[s_])
    for s_ in ('inj', 'holdT1', 'holdT2'):                                # ceiling: the real photo read back the same way
        pics[f'real_{s_}'] = real; ctx[f'real_{s_}'] = (s_, dict(inj=('inj', None), holdT1=('hold', HOLD_TE[0]), holdT2=('hold', HOLD_TE[1]))[s_])
    R = Reader(); rb_m, rb_see = {}, {}
    with torch.no_grad():
        for name, P in pics.items():
            kd, w = ctx[name][1]
            rb_m[name] = torch.cat([R.readback(P[i:i + 32].float().div(127.5).sub(1), kd, w).cpu() for i in range(0, len(P), 32)])
            if kd != 'see' and not name.startswith('real_'):
                rb_see[name] = torch.cat([R.see(P[i:i + 32].float().div(127.5).sub(1)).cpu() for i in range(0, len(P), 32)])
            else:
                rb_see[name] = rb_m[name]
    del R; torch.cuda.empty_cache()
    from .train import source_stats
    mu, sd = source_stats(D); mu['random'], sd['random'] = allA.mean(0), allA.std(0) + 1e-3

    def recon(zr, za):
        err = ((zr - za) ** 2).sum(1); tot = (za ** 2).sum(1); errp = ((zr - za.roll(1, 0)) ** 2).sum(1)
        return dict(n=len(za), fve=round(float(1 - err.sum() / tot.sum()), 4), fve_shuffled_pairs=round(float(1 - errp.sum() / tot.sum()), 4),
                    cos=round(float(F.cosine_similarity(zr, za).mean()), 4))
    res, per = {}, {}
    see_c = (srcs['see'].float() - mu['see']) / sd['see']
    for name in pics:
        s_, (kd, w) = ctx[name]; kk = dict(holdT1='hold', holdT2='hold').get(s_, s_); res[name] = {}; per[name] = {}
        zr, za = (rb_m[name] - mu[kk]) / sd[kk], (srcs[s_].float() - mu[kk]) / sd[kk]
        r = ranks(zr, za, torch.arange(len(za))); per[name]['source'] = r
        if len(za) == n:
            res[name]['reconstruction (readback in the source context, standardized)'] = dict(all=recon(zr, za), coco=recon(zr[is_coco], za[is_coco]), cc3m=recon(zr[~is_coco], za[~is_coco]))
            res[name]['readback rank among the sources'] = dict(all=summary(r), coco=summary(r[is_coco]), cc3m=summary(r[~is_coco]))
            if not name.startswith('real_'):
                r2 = ranks((rb_see[name] - mu['see']) / sd['see'], see_c, torch.arange(n)); per[name]['photo'] = r2
                res[name]['seen-picture readback rank among real-photo activations'] = dict(all=summary(r2), coco=summary(r2[is_coco]), cc3m=summary(r2[~is_coco]))
        else:
            res[name]['reconstruction (readback in the source context, standardized)'] = recon(zr, za)
            res[name]['readback rank among the sources'] = summary(r)
    readback = rb_m
    from transformers import AutoModel, AutoProcessor
    root = snap(SIGLIP); sm = AutoModel.from_pretrained(root, dtype=torch.float16).to(DEV).eval(); sp = AutoProcessor.from_pretrained(root)

    @torch.no_grad()
    def sig_f(P):
        f = []
        for i in range(0, len(P), 32):
            x = sp(images=[Image.fromarray(a.permute(1, 2, 0).numpy()) for a in P[i:i + 32]], return_tensors='pt').to(DEV)
            o = sm.get_image_features(pixel_values=x['pixel_values'].half())
            f.append(F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu())
        return torch.cat(f)
    fr = sig_f(real); feats = dict(real=fr)
    for name, P in pics.items():
        if name.startswith('real_'):
            continue
        feats[name] = sig_f(P)
        if len(P) == n:
            r = ranks(feats[name], fr, torch.arange(n)); per[name]['siglip'] = r
            res[name]['SigLIP 2 picture vs real photos'] = dict(all=summary(r), coco=summary(r[is_coco]), cc3m=summary(r[~is_coco]))
    json.dump(dict(n=n, results=res), open(run / 'eval.json', 'w'), indent=1)
    save(dict(per=per, good=good, readback=rb_m, readback_see=rb_see, src=srcs, mu=mu, sd=sd, siglip=feats, chars=D['char']['te']), run / 'eval_rows.pt')
    log(json.dumps(res, indent=1))
