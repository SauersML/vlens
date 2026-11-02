"""End-to-end picture adapter: Qwen3-VL-4B's activation -> Qwen-Image-2.1-Turbo's input states, trained through the
frozen generator.

It starts at qimg's ridge map (kept at rank 1024 in qimg_maps.pt), which stays fixed; on top of it a residual is
trained: an MLP of the standardized activation whose output layer starts at zero (so training starts exactly at the
ridge map), plus the mean state. Losses:
  fm      the generator's own flow-matching loss on the photo's latents, with the predicted states as its condition,
          at the Turbo sampler's 8 noise levels (oracle style: the photo is the target)
  rb      NLA style, no photo: the picture the generator would finish from step k of its own 8-step sampling (steps
          before k without gradient, then that step's estimate of the clean picture, decoded) is read back by
          Qwen3-VL-4B in the see chat and has to pick out its own source activation among the batch and BANK train
          sources (InfoNCE on standardized activations); see only
  anchor  mean squared distance of the states from the ridge map's
Kinds: see (the 4B looked at the photo) or hold (the 4B read one of its captions in a HOLD_TR wording).
Eval on qimg's 100 COCO test photos with qimg's seed and size, ridge map vs trained: SigLIP 2 rank of each picture among
the photos, readback rank (see chat) among the test photos' see activations, test flow-matching loss; for hold also the
hidden-thought sets (think, riddle, character) ranked among their labels.
Usage: python -m vlens pnla OUT MAPS [N_TRAIN]"""
from .common import *
from .models import Reader
from .qimg import ranks, summary, THINK, THINK_T, RIDDLE_T

GEN = 'Qwen/Qwen-Image-2.1-Turbo'
KIND = os.environ.get('VLENS_PNLA_KIND', 'see')
W_FM, W_RB, W_ANC = (float(os.environ.get(f'VLENS_PNLA_{k}', d)) for k, d in (('FM', 1), ('RB', 0), ('ANC', 0)))
TR_PX, EV_PX = int(os.environ.get('VLENS_PNLA_PX', PX)), int(os.environ.get('VLENS_NATIVE_RES', 512))
STEPS, MINUTES = int(os.environ.get('VLENS_PNLA_STEPS', 4000)), float(os.environ.get('VLENS_PNLA_MIN', 60))
BS, RB_BS, LR = int(os.environ.get('VLENS_PNLA_BS', 8)), int(os.environ.get('VLENS_PNLA_RBBS', 4)), float(os.environ.get('VLENS_PNLA_LR', 2e-4))
KMIN, BANK, TAU = int(os.environ.get('VLENS_PNLA_KMIN', 2)), 4096, 0.07
N_TEST, N_SHOW, SEED = int(os.environ.get('VLENS_QIMG_TEST', 100)), 12, 11
assert KIND in ('see', 'hold') and not (W_RB and KIND != 'see')


class Gen:
    """Qwen-Image-2.1-Turbo without its text encoder: the frozen transformer and VAE, the Turbo sampler written out so
    a step can carry gradient, and the pipeline itself for the eval pictures"""
    def __init__(self, proc):
        from diffusers import QwenImage21Pipeline
        if TINY:
            from diffusers import QwenImage21Transformer2DModel, AutoencoderKLQwenImage21, FlowMatchEulerDiscreteScheduler
            self.pipe = QwenImage21Pipeline(FlowMatchEulerDiscreteScheduler(), AutoencoderKLQwenImage21(base_dim=8, decoder_base_dim=8, num_res_blocks=1),
                                            None, proc, QwenImage21Transformer2DModel(num_layers=2, num_attention_heads=1),
                                            sample_sigmas=[1.0, 0.9, 0.7, 0.4]).to(DEV, BF)
        else:
            self.pipe = QwenImage21Pipeline.from_pretrained(GEN, text_encoder=None, dtype=BF).to(DEV)
        self.pipe.set_progress_bar_config(disable=True)
        self.tr, self.vae = self.pipe.transformer.eval().requires_grad_(False), self.pipe.vae.eval().requires_grad_(False)
        self.tr.enable_gradient_checkpointing()
        self.sig = [float(s) for s in self.pipe.config.sample_sigmas] + [0.0]
        c = self.vae.config; self.ch = self.tr.config.in_channels
        self.zm = torch.tensor(c.latents_mean, device=DEV).view(1, -1, 1, 1, 1).to(BF)
        self.zs = torch.tensor(c.latents_std, device=DEV).view(1, -1, 1, 1, 1).to(BF)

    @staticmethod
    def t(s, B):
        """the timestep the pipeline passes at noise level s (sigma * 1000 in bf16, / 1000)"""
        s = s.float() if torch.is_tensor(s) else torch.full((B,), s, device=DEV)
        return (s * 1000).to(BF) / 1000

    def v(self, x, s, C):
        """velocity at packed latents x [B,N,ch], noise level s (float or [B]), states C [B,P,4096]"""
        B, N = x.shape[:2]; h = math.isqrt(N)
        m = torch.cat([torch.zeros(B, C.shape[1], dtype=torch.bool, device=DEV), torch.ones(B, N // 4, dtype=torch.bool, device=DEV)], 1)
        return self.tr(hidden_states=x, timestep=self.t(s, B), encoder_hidden_states=C, img_shapes=[[(1, h, h)]] * B, img_mask=m,
                       encoder_hidden_states_mask=torch.ones(B, C.shape[1], dtype=torch.long, device=DEV), return_dict=False)[0][:, -N:]

    def noise(self, seeds, px):
        """the pipeline's starting noise for each seed (a CPU generator, bf16), packed"""
        h = 2 * (px // 32)
        z = torch.cat([torch.randn((1, 1, self.ch, h, h), generator=torch.Generator('cpu').manual_seed(int(s)), dtype=BF) for s in seeds])
        return z.to(DEV).view(len(seeds), self.ch, h * h).transpose(1, 2)

    def run(self, C, x, a, b):
        """Euler steps a .. b-1 of the Turbo sampler from x (the scheduler's update, in float32)"""
        for i in range(a, b):
            x = (x.float() + (self.sig[i + 1] - self.sig[i]) * self.v(x, self.sig[i], C).float()).to(BF)
        return x

    def decode(self, x):
        """packed normalized latents -> [B,3,H,W] in [-1,1] (the generator's alpha channel dropped)"""
        B, N, c = x.shape; h = math.isqrt(N)
        z = x.transpose(1, 2).reshape(B, c, 1, h, h).to(BF) * self.zs + self.zm
        return self.vae.decode(z, return_dict=False)[0][:, :3, 0].float().clamp(-1, 1)

    @torch.no_grad()
    def encode(self, x):
        """[B,3,H,W] in [-1,1] -> packed normalized latents, as the pipeline encodes a photo (RGBA, alpha 1, posterior mode)"""
        x4 = torch.cat([x, torch.ones_like(x[:, :1])], 1)[:, :, None].to(DEV, BF)
        z = (self.vae.encode(x4).latent_dist.mode() - self.zm) / self.zs
        return z[:, :, 0].flatten(2).transpose(1, 2)

    @torch.no_grad()
    def picture(self, e, seed, px=EV_PX, latent=False):
        out = self.pipe(prompt_embeds=e[None].to(DEV, BF), prompt_embeds_mask=torch.ones(1, len(e), dtype=torch.long, device=DEV),
                        height=px, width=px, generator=torch.Generator('cpu').manual_seed(seed), output_type='latent' if latent else 'pil')
        return out.images[0] if latent else out.images[0].convert('RGB')


class Adapter(nn.Module):
    """states = pre ++ (z U V^T + ym + MLP(z) B^T), z the standardized activation; U, V fixed (the ridge map), B starts at 0"""
    def __init__(self, k, pre, hid=1024, r=256):
        super().__init__()
        for n in ('mu', 'sd', 'U', 'V'):
            self.register_buffer(n, k[n].float())
        self.register_buffer('pre', pre.float()); self.register_buffer('ym0', k['ym'].float())
        self.ym = nn.Parameter(k['ym'].float().clone())
        self.f = nn.Sequential(nn.Linear(len(k['mu']), hid), nn.GELU(), nn.Linear(hid, r))
        self.B = nn.Parameter(torch.zeros(k['ym'].numel(), r))

    def forward(self, a, ridge=False):
        z = (a.to(self.mu.device).float() - self.mu) / self.sd
        y = (z @ self.U) @ self.V.T
        y = y.view(len(a), *self.ym0.shape) + self.ym0 if ridge else (y + self.f(z) @ self.B.T).view(len(a), *self.ym.shape) + self.ym
        return torch.cat([self.pre.expand(len(a), -1, -1), y], 1)


def fake_maps(R, T=25, D=4096, r=16):
    din = 3 * R.d
    k = dict(U=torch.randn(din, r) * 0.01, V=torch.linalg.qr(torch.randn(T * D, r))[0], mu=torch.zeros(din), sd=torch.ones(din), ym=torch.randn(T, D))
    return dict(maps=dict(see=k, hold=k), pre=torch.randn(8, D), see_mu=torch.zeros(din), see_sd=torch.ones(din))


@torch.no_grad()
def collect(R, G, df, test):
    """per photo: latents at TR_PX and the see activation; the hold sources (two captions in random HOLD_TR wordings for
    train, the first caption in HOLD_TE[0] for test, as qimg); test keeps the photos"""
    from .data import coco_chunks
    Z, S, caps, ims = [], [], [], []
    for chunk in coco_chunks(df):
        good = [(square(im), c) for im, c in chunk if im is not None]
        for b in range(0, len(good), 32):
            x = to_t([im for im, _ in good[b:b + 32]])
            S.append(R.see(x).half().cpu())
            xt = x if TR_PX == PX else F.interpolate(x, size=(TR_PX, TR_PX), mode='bicubic', antialias=True).clamp(-1, 1)
            Z.append(torch.cat([G.encode(xt[i:i + 8]).cpu() for i in range(0, len(xt), 8)]))
        caps += [c for _, c in good]; ims += [im for im, _ in good] if test else []
        log('test' if test else 'train', 'photos', len(caps))
    rng = random.Random(0)
    rows = [(i, HOLD_TE[0].format(c=c[0])) for i, c in enumerate(caps)] if test else \
           [(i, HOLD_TR[rng.randrange(len(HOLD_TR))].format(c=x)) for i, c in enumerate(caps) for x in c]
    out = dict(z=torch.cat(Z), see=torch.cat(S), caps=caps, ims=ims)
    if KIND == 'see':
        out.update(src=out['see'], idx=torch.arange(len(caps)))
    else:
        out.update(src=R.hold([t for _, t in rows]), idx=torch.tensor([i for i, _ in rows]))
    return out


def siglip():
    from transformers import AutoModel, AutoProcessor
    root = snap(SIGLIP); sm = AutoModel.from_pretrained(root, dtype=torch.float16).to(DEV).eval(); sp = AutoProcessor.from_pretrained(root)
    pool = lambda o: F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu()

    @torch.no_grad()
    def sig(ims):
        return torch.cat([pool(sm.get_image_features(pixel_values=sp(images=ims[i:i + 32], return_tensors='pt')['pixel_values'].to(DEV).half()))
                          for i in range(0, len(ims), 32)])

    @torch.no_grad()
    def sigt(texts):
        return pool(sm.get_text_features(**sp(text=texts, padding='max_length', max_length=64, return_tensors='pt').to(DEV)))
    return sig, sigt


@torch.no_grad()
def fm_eval(G, C, Z):
    """test flow-matching loss: fixed noise, noise levels cycling through the sampler's"""
    g = torch.Generator('cpu').manual_seed(5); tot = []; n = len(G.sig) - 1
    for i in range(0, len(Z), 8):
        z0 = Z[i:i + 8].to(DEV).float(); eps = torch.randn(z0.shape, generator=g).to(DEV)
        s = torch.tensor([G.sig[(i + j) % n] for j in range(len(z0))], device=DEV)
        v = G.v(((1 - s)[:, None, None] * z0 + s[:, None, None] * eps).to(BF), s, C[i:i + 8].to(DEV, BF))
        tot.append(((v.float() - (eps - z0)) ** 2).mean((1, 2)).cpu())
    return round(float(torch.cat(tot).mean()), 5)


@torch.no_grad()
def evaluate(R, G, A, te, ridge, sig, real, see_z, mu, sd, out):
    tag = 'ridge' if ridge else 'trained'
    C = torch.cat([A(te['src'][i:i + 32].to(DEV), ridge).cpu() for i in range(0, len(te['src']), 32)])
    pics = [G.picture(C[i], SEED) for i in range(len(C))]
    for i, p in enumerate(pics):
        p.save(out / 'pics' / f'{tag}_{i}.jpg', quality=90 if i < N_SHOW else 80)
    rb = torch.cat([R.see(to_t([square(p) for p in pics[i:i + 32]])).cpu() for i in range(0, len(pics), 32)])
    res = dict(siglip=summary(ranks(sig(pics), real)), readback=summary(ranks((rb - mu) / sd, see_z)), fm=fm_eval(G, C, te['z']))
    log(tag, res)
    return res, pics


@torch.no_grad()
def explore(R, G, A, sig, sigt, out):
    """hidden thoughts through the hold adapter: think / riddle / character prompts, ranked among their labels"""
    from .common import characters
    names = [x for x, _ in THINK]; chars = characters()[1][:12]
    sets = dict(think=([THINK_T.format(x=x) for x in names], names), riddle=([RIDDLE_T.format(r=r) for _, r in THINK], names),
                char=([CHAR_T[0].format(x=x) for x in chars], chars))
    res, pics = {}, {}
    for k, (texts, labels) in sets.items():
        H = R.hold(texts); bank = sigt([f'a photo of {l}.' for l in labels]); res[k] = dict(labels=labels)
        for ridge in (True, False):
            tag = 'ridge' if ridge else 'trained'; C = A(H.to(DEV), ridge).cpu()
            P = [G.picture(C[i], SEED) for i in range(len(C))]; pics[k, tag] = P
            for i, p in enumerate(P):
                p.save(out / 'pics' / f'{k}_{tag}_{i}.jpg', quality=90)
            res[k][tag] = summary(ranks(sig(P), bank))
        log('explore', k, {t: (res[k][t]['top1'], res[k][t]['top5']) for t in ('ridge', 'trained')}, 'of', len(labels))
    return res, pics


def grid(rows, path, S=256):
    Gd = Image.new('RGB', (max(len(r) for r in rows) * S, len(rows) * S), 'white')
    for r, row in enumerate(rows):
        for k, im in enumerate(row):
            Gd.paste(im.resize((S, S)), (k * S, r * S))
    Gd.save(path, quality=88)


def pnla(out, maps, n_train=12000):
    out = Path(out); (out / 'pics').mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = True
    R = Reader(); G = Gen(R.proc); log('loaded; kind', KIND, 'fm', W_FM, 'rb', W_RB, 'anchor', W_ANC, 'train px', TR_PX, 'sigmas', G.sig)
    km = fake_maps(R) if TINY else torch.load(maps, map_location='cpu')
    A = Adapter(km['maps'][KIND], km['pre']).to(DEV)
    mu, sd = km['see_mu'].float(), km['see_sd'].float(); mu_d, sd_d = mu.to(DEV), sd.to(DEV)
    te = collect(R, G, coco_table('test').head(N_TEST), True)
    with torch.no_grad():                                   # the written-out sampler against the pipeline's own
        C0 = A(te['src'][:1].to(DEV), True)[0].to(BF)
        mine = G.run(C0[None], G.noise([SEED], TR_PX), 0, len(G.sig) - 1).float(); theirs = G.picture(C0, SEED, TR_PX, latent=True).float()
        log('sampler check: relative difference from the pipeline', round(float((mine - theirs).norm() / theirs.norm()), 5))
    tr = collect(R, G, coco_table('train').sample(frac=1, random_state=0).head(int(n_train)), False)
    log('train photos', len(tr['caps']), 'sources', len(tr['src']), 'latents', tuple(tr['z'].shape), 'test photos', len(te['caps']))
    sig, sigt = siglip(); real = sig(te['ims']); see_z = (te['see'].float() - mu) / sd
    res = dict(kind=KIND, weights=dict(fm=W_FM, rb=W_RB, anchor=W_ANC), train_px=TR_PX, eval_px=EV_PX, n_train=len(tr['caps']), n_sources=len(tr['src']),
               lr=LR, batch=BS, rb_batch=RB_BS, kmin=KMIN, tau=TAU, bank=BANK)
    res['ridge'], pr = evaluate(R, G, A, te, True, sig, real, see_z, mu, sd, out)
    params = [p for p in A.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=0.0, betas=(0.9, 0.99))
    SRC, IDX, Z = tr['src'].to(DEV), tr['idx'].to(DEV), tr['z'].to(DEV)
    zb = F.normalize((tr['see'].float() - mu) / sd, dim=-1).to(DEV)            # rb's positives and bank (see sources)
    sigs = torch.tensor(G.sig[:-1], device=DEV); hist, acc = [], {}; t0 = time.monotonic(); step = 0
    for step in range(1, STEPS + 1):
        if (time.monotonic() - t0) / 60 > MINUTES:
            break
        for g in opt.param_groups:
            g['lr'] = LR * min(1.0, step / 50)
        b = torch.randint(len(SRC), (BS,), device=DEV); C = A(SRC[b]); loss = 0.0; parts = {}
        if W_ANC:
            parts['anchor'] = ((C - A(SRC[b], True)) ** 2).mean(); loss = loss + W_ANC * parts['anchor']
        if W_FM:
            z0 = Z[IDX[b]].float(); eps = torch.randn_like(z0); s = sigs[torch.randint(len(sigs), (BS,), device=DEV)]
            v = G.v(((1 - s)[:, None, None] * z0 + s[:, None, None] * eps).to(BF), s, C.to(BF))
            parts['fm'] = ((v.float() - (eps - z0)) ** 2).mean(); loss = loss + W_FM * parts['fm']
        if W_RB:
            n = RB_BS; Cb = C[:n].to(BF); k = random.randrange(min(KMIN, len(sigs) - 1), len(sigs))
            with torch.no_grad():
                x = G.run(Cb.detach(), G.noise(torch.randint(1 << 30, (n,)).tolist(), TR_PX), 0, k)
            img = G.decode(x.float() - G.sig[k] * G.v(x, G.sig[k], Cb).float())
            if TR_PX != PX:
                img = F.interpolate(img, size=(PX, PX), mode='bilinear', antialias=True)
            q = F.normalize((R.see(img) - mu_d) / sd_d, dim=-1)
            logits = q @ torch.cat([zb[IDX[b[:n]]], zb[torch.randint(len(zb), (BANK,), device=DEV)]]).T / TAU
            parts['rb'] = F.cross_entropy(logits, torch.arange(n, device=DEV)); loss = loss + W_RB * parts['rb']
            parts['rb_top1'] = (logits.argmax(1) == torch.arange(n, device=DEV)).float().mean()
        opt.zero_grad(set_to_none=True); loss.backward(); gn = torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step()
        for kk, x in dict(parts, grad_norm=gn).items():
            acc.setdefault(kk, []).append(float(x))
        if step % 25 == 0:
            row = dict(step=step, minutes=round((time.monotonic() - t0) / 60, 2), **{kk: round(sum(x) / len(x), 5) for kk, x in acc.items()})
            hist.append(row); acc = {}; log('step', row)
    res['steps'], res['history'] = step, hist
    save(dict(f={k_: v.half() for k_, v in A.f.state_dict().items()}, B=A.B.detach().half().cpu(), ym=A.ym.detach().cpu(), kind=KIND,
              hid=A.f[0].out_features, r=A.B.shape[1]), out / 'pnla_adapter.pt')
    A.eval(); res['trained'], pt = evaluate(R, G, A, te, False, sig, real, see_z, mu, sd, out)
    rows = [[te['ims'][i], pr[i], pt[i]] for i in range(min(N_SHOW, len(pt)))]
    grid(rows, out / 'pnla_grid.jpg')
    if KIND == 'hold':
        res['explore'], ep = explore(R, G, A, sig, sigt, out)
        n = len(THINK)
        grid([[ep['think', 'ridge'][i], ep['think', 'trained'][i], ep['riddle', 'ridge'][i], ep['riddle', 'trained'][i]] for i in range(n)], out / 'pnla_thoughts.jpg')
    json.dump(res, open(out / 'pnla.json', 'w'), indent=1)
    log('saved', out / 'pnla.json')
