"""Stage 1: photos (COCO, CC3M) -> klein VAE latents and Qwen3-VL source activations (see, inj, hold) + characters."""
from .common import *
from .models import Reader, Klein


def coco_chunks(df, CH=256):
    """(PIL or None, captions) in order, fetched 64 at a time with the next chunk prefetching"""
    urls = df['url'].tolist(); caps = [[clean(c) for c in s] for s in df['sentences']]
    with cf.ThreadPoolExecutor(64) as ex:
        nxt = [ex.submit(fetch, u) for u in urls[:CH]]
        for a in range(0, len(urls), CH):
            cur = nxt; nxt = [ex.submit(fetch, u) for u in urls[a + CH:a + 2 * CH]]
            yield [(f.result(), caps[a + k][:2]) for k, f in enumerate(cur)]


def cc3m_chunks(shards, n, CH=256):
    if n <= 0 or TINY:
        return
    from huggingface_hub import hf_hub_download
    with cf.ThreadPoolExecutor(6) as ex:
        paths = ex.map(lambda f: hf_hub_download(CC3M, f, repo_type='dataset'), shards)
        buf, got = [], 0
        for p in paths:
            ent = {}
            with tarfile.open(p) as t:
                for mem in t:
                    if not mem.isfile():
                        continue
                    key, _, ext = mem.name.rpartition('.')
                    ent.setdefault(key, {})[ext] = t.extractfile(mem).read()
            for key, e in ent.items():
                if 'jpg' not in e or 'txt' not in e:
                    continue
                try:
                    im = Image.open(io.BytesIO(e['jpg'])).convert('RGB')
                except Exception:
                    im = None
                buf.append((im, [clean(e['txt'].decode('utf-8', 'replace'))])); got += 1
                if len(buf) == CH:
                    yield buf; buf = []
                if got >= n:
                    break
            if got >= n:
                break
        if buf:
            yield buf


def data(W, n_coco, n_cc3m):
    W = Path(W); W.mkdir(parents=True, exist_ok=True); (W / 'test_img').mkdir(exist_ok=True)
    K = Klein(text_encoder=True)
    q0 = K.pipe.encode_prompt(prompt=[''], device=DEV)[0][0, :M].float().cpu()
    log('blank conditioning', tuple(q0.shape), round(float(q0.norm(dim=-1).mean()), 1))
    K.pipe.text_encoder = None; torch.cuda.empty_cache()
    R = Reader(); out = dict(q0=q0, read=READ, px=PX, l_inj=L_INJ, see_t=SEE_T, neutral_t=NEUTRAL_T, hold_tr=HOLD_TR, hold_te=HOLD_TE, char_t=CHAR_T)
    ctr = coco_table('train').sample(frac=1, random_state=0).head(n_coco)
    cte = coco_table('test').head(N_TE)
    n_sh = math.ceil(n_cc3m / 4500) + 1
    plan = dict(te=[(0, coco_chunks(cte)), (1, cc3m_chunks(['cc3m-validation-0000.tar'], N_TE))],
                tr=[(0, coco_chunks(ctr)), (1, cc3m_chunks([f'cc3m-train-{k:04d}.tar' for k in range(n_sh)], n_cc3m))])
    for split, sources in plan.items():
        Z, S, I, ok, src, caps = [], [], [], [], [], []
        for dset, chunks in sources:
            for chunk in chunks:
                good = [k for k, (im, _) in enumerate(chunk) if im is not None]
                z = torch.zeros(len(chunk), 128, PX // 16, PX // 16, dtype=torch.float16)
                s = torch.zeros(len(chunk), 3 * R.d, dtype=torch.float16); i_ = torch.zeros_like(s)
                if good:
                    sq = [square(chunk[k][0]) for k in good]
                    if split == 'te':
                        for k, im in zip(good, sq):
                            im.save(W / 'test_img' / f'{len(ok) + k}.jpg', quality=92)
                    x = to_t(sq); gi = torch.tensor(good)
                    with torch.no_grad():
                        for b in range(0, len(x), 32):
                            sel = gi[b:b + 32]
                            z[sel] = K.encode(x[b:b + 32]).half().cpu()
                            a, h = R.see(x[b:b + 32], grab=True)
                            s[sel] = a.half().cpu(); i_[sel] = R.inject(h).half().cpu()
                Z.append(z); S.append(s); I.append(i_)
                ok += [k in set(good) for k in range(len(chunk))]; src += [dset] * len(chunk); caps += [c for _, c in chunk]
                if len(ok) % 5120 < 256:
                    log(split, 'dataset', dset, 'images', len(ok), 'ok', sum(ok))
        Z, S, I = torch.cat(Z), torch.cat(S), torch.cat(I); ok = torch.tensor(ok); src = torch.tensor(src)
        rng = random.Random(0)
        if split == 'tr':
            items = [(i, j, rng.randrange(len(HOLD_TR))) for i in range(len(caps)) for j in range(len(caps[i]))]
            H = R.hold([HOLD_TR[w].format(c=caps[i][j]) for i, j, w in items])
        else:
            items = [(i, 0, w) for w in (0, 1) for i in range(len(caps))]
            H = R.hold([HOLD_TE[w].format(c=caps[i][0]) for i, j, w in items])
        out[split] = dict(z=Z, see=S, inj=I, ok=ok, src=src, hold=H, hold_items=torch.tensor(items), caps=caps)
        log(split, 'done: images', len(ok), 'ok', int(ok.sum()), 'coco', int((src == 0).sum()), 'cc3m', int((src == 1).sum()), 'hold rows', len(H))
    ctr_, cte_ = characters()
    out['char'] = dict(tr=ctr_, te=cte_,
                       A_tr=R.hold([t.format(x=x) for x in ctr_ for t in CHAR_T]), A_te=R.hold([t.format(x=x) for x in cte_ for t in CHAR_T]))
    log('characters', len(ctr_), len(cte_))
    save(out, W / 'data.pt')
