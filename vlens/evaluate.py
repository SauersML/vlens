"""Evaluation: pictures from held-out sources, direct readback ranks, SigLIP 2 image-image ranks, text route and prior control."""
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
    mu = dict(see=tr['see'][D['tr_img']].float().mean(0), inj=tr['inj'][D['tr_img']].float().mean(0),
              hold=tr['hold'][D['tr_hold']].float().mean(0), char=D['char']['A_tr'].float().mean(0))
    allA = torch.cat([tr['see'][D['tr_img']], tr['inj'][D['tr_img']], tr['hold'][D['tr_hold']]])
    allA = allA[torch.randperm(len(allA), generator=torch.Generator().manual_seed(0))[:30000]].float()
    g = torch.Generator().manual_seed(11)
    NX = int(os.environ.get('VLENS_NEXTRA', 60))
    rand = allA.mean(0) + allA.std(0) * torch.randn(NX, allA.shape[1], generator=g)    # the prior control
    srcs = dict(see=te['see'][good], inj=te['inj'][good], holdT1=te['hold'][good], holdT2=te['hold'][nte + good],
                char=D['char']['A_te'][::len(CHAR_T)][:NX], random=rand.half())
    kind = dict(see='see', inj='inj', holdT1='hold', holdT2='hold', char='char', random=None)
    K = Klein(); pics = {}
    for tag in ('B', 'C'):
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
    R = Reader(); readback = {}
    with torch.no_grad():
        for name, P in pics.items():
            readback[name] = torch.cat([R.see(P[i:i + 32].float().div(127.5).sub(1)).cpu() for i in range(0, len(P), 32)])
    del R; torch.cuda.empty_cache()
    res, per = {}, {}
    see_c = srcs['see'].float() - mu['see']
    for name, rb in readback.items():
        q = rb - mu['see']; s = name.split('_', 1)[1]; res[name] = {}; per[name] = {}
        if s in ('char', 'random'):
            own = srcs[s].float() - (mu[kind[s]] if kind[s] else srcs[s].float().mean(0))
            r = ranks(q, own, torch.arange(len(q))); res[name]['readback vs own sources'] = summary(r); per[name]['source'] = r
            continue
        if name == 'text_caption':
            s = 'holdT1'
        r = ranks(q, see_c, torch.arange(n)); per[name]['photo'] = r
        res[name]['readback vs real-photo activations'] = dict(all=summary(r), coco=summary(r[is_coco]), cc3m=summary(r[~is_coco]))
        if kind[s] != 'see':
            r = ranks(q, srcs[s].float() - mu[kind[s]], torch.arange(n)); per[name]['source'] = r
            res[name]['readback vs own sources'] = dict(all=summary(r), coco=summary(r[is_coco]), cc3m=summary(r[~is_coco]))
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
        feats[name] = sig_f(P)
        if len(P) == n:
            r = ranks(feats[name], fr, torch.arange(n)); per[name]['siglip'] = r
            res[name]['SigLIP 2 picture vs real photos'] = dict(all=summary(r), coco=summary(r[is_coco]), cc3m=summary(r[~is_coco]))
    json.dump(dict(n=n, results=res), open(run / 'eval.json', 'w'), indent=1)
    save(dict(per=per, good=good, readback=readback, src=srcs, mu=mu, siglip=feats, chars=D['char']['te']), run / 'eval_rows.pt')
    log(json.dumps(res, indent=1))
