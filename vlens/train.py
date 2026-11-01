"""Stages B (flow matching on the real photo) and C (+ direct reconstruction: the picture read back by Qwen3-VL in its source's own context, standardized mean squared error; no fitted readout)."""
from .common import *
from .models import Reader, Klein, Connector


def load_data(W):
    D = torch.load(Path(W) / 'data.pt', weights_only=False, mmap=True)
    tr = D['tr']
    D['tr_img'] = torch.nonzero(tr['ok']).squeeze(1)
    D['tr_hold'] = torch.nonzero(tr['ok'][tr['hold_items'][:, 0]]).squeeze(1)
    return D


def source_stats(D, n=30000):
    """per-dimension mean and spread of each kind's training source activations (a sample of n rows): the fixed
    standardization of the readback loss and of fraction of variance explained"""
    tr = D['tr']; g = torch.Generator().manual_seed(0); mu, sd = {}, {}
    for k, A in (('see', tr['see'][D['tr_img']]), ('inj', tr['inj'][D['tr_img']]), ('hold', tr['hold'][D['tr_hold']]), ('char', D['char']['A_tr'])):
        A = A[torch.randperm(len(A), generator=g)[:n]].float(); mu[k], sd[k] = A.mean(0), A.std(0) + 1e-3
    return mu, sd


def make_connector(D):
    tr = D['tr']; img = D['tr_img']
    A = torch.cat([tr['see'][img], tr['inj'][img], tr['hold'][D['tr_hold']]])
    A = A[torch.randperm(len(A), generator=torch.Generator().manual_seed(0))[:30000]].float()
    return Connector(A.shape[1], D['q0'], A.mean(0), A.std(0) + 1e-3)


KINDS = ('see', 'inj', 'hold', 'char')


def draw_batch(D, B, g, probs):
    """B training items: kind index, activation, image index (-1 for characters), row in the kind's table"""
    tr = D['tr']; kinds = torch.multinomial(torch.tensor(probs), B, replacement=True, generator=g)
    A, img, row = [], [], []
    for k in kinds.tolist():
        if KINDS[k] in ('see', 'inj'):
            i = int(D['tr_img'][torch.randint(len(D['tr_img']), (1,), generator=g)]); A.append(tr[KINDS[k]][i]); img.append(i); row.append(i)
        elif KINDS[k] == 'hold':
            r = int(D['tr_hold'][torch.randint(len(D['tr_hold']), (1,), generator=g)]); A.append(tr['hold'][r]); img.append(int(tr['hold_items'][r, 0])); row.append(r)
        else:
            r = int(torch.randint(len(D['char']['A_tr']), (1,), generator=g)); A.append(D['char']['A_tr'][r]); img.append(-1); row.append(r)
    return kinds, torch.stack(A), torch.tensor(img), torch.tensor(row)


def test_idx(D, n):
    te = D['te']; c = torch.nonzero(te['ok'] & (te['src'] == 0)).squeeze(1); w = torch.nonzero(te['ok'] & (te['src'] == 1)).squeeze(1)
    return torch.cat([c[:n // 2], w[:n - n // 2]])


def preview(K, con, D, W, path):
    te = D['te']; idx = test_idx(D, 8); nte = len(te['see'])
    rows = [[Image.open(W / 'test_img' / f'{int(i)}.jpg') for i in idx]]
    con.eval()
    with torch.no_grad():
        for A in (te['see'][idx], te['inj'][idx], te['hold'][idx]):
            rows.append(to_pil(K.draw(con(A.to(DEV)), idx.tolist())))
        rows.append(to_pil(K.draw(con(D['char']['A_te'][0:24:3].to(DEV)), list(range(8)))))
    con.train()
    G = Image.new('RGB', (8 * 192, len(rows) * 192), 'white')
    for r, ims in enumerate(rows):
        for c, im in enumerate(ims):
            G.paste(im.resize((192, 192)), (192 * c, 192 * r))
    G.save(path, quality=85)


def train(W, run, steps, minutes, cycle=False):
    W, run = Path(W), Path(run); run.mkdir(parents=True, exist_ok=True)
    D = load_data(W); K = Klein(base=True); tr = D['tr']
    K.tf.enable_gradient_checkpointing(); K.tf_base.enable_gradient_checkpointing()
    con = make_connector(D).to(DEV)
    PURE = cycle and os.environ.get('VLENS_PURE') == '1'   # NLA only: from the blank adapter, reconstruction loss alone, no photo
    tag = 'N' if PURE else 'C' if cycle else 'B'
    if cycle:
        if not PURE:
            con.load_state_dict(torch.load(W / 'con_B.pt', weights_only=False)['state'])
        R = Reader()
        try:
            K.vae.enable_gradient_checkpointing()
        except Exception as e:
            log('vae checkpointing unavailable', e)
        mu, sd = source_stats(D)
        mu = {k: v.to(DEV) for k, v in mu.items()}; sd = {k: v.to(DEV) for k, v in sd.items()}
        nW = len(CHAR_T)
    lr = 5e-5 if cycle and not PURE else 2e-4
    opt = torch.optim.AdamW([p for p in con.parameters() if p.requires_grad], lr=lr, weight_decay=0.01, betas=(0.9, 0.99))
    warm = 100 if cycle else 300
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / warm) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps)))))
    vg = torch.Generator().manual_seed(7); te = D['te']; vi = test_idx(D, 96); n3 = len(vi) // 3; vi = vi[:3 * n3]
    VA = torch.cat([te['see'][vi[:n3]], te['inj'][vi[n3:2 * n3]], te['hold'][vi[2 * n3:]]]); VZ = te['z'][vi]
    VS = K.sig_base[torch.tensor([0, 12, 25, 37])[torch.arange(3 * n3) % 4]]; VE = torch.randn(3 * n3, K.h * K.h, 128, generator=vg).to(DEV)

    VA_sh = torch.cat([VA[a:a + n3].roll(1, 0) for a in (0, n3, 2 * n3)])       # each photo paired with another's activation

    @torch.no_grad()
    def val():
        """held-out flow-matching loss per source (see, inj, hold) and its gain over shuffled pairs (the item information)"""
        con.eval(); out = []
        for A in (VA, VA_sh):
            v = torch.cat([K.fm_loss(VZ[i:i + 16], con(A[i:i + 16].to(DEV)), VS[i:i + 16], VE[i:i + 16]) for i in range(0, 3 * n3, 16)])
            out.append([float(v[a:a + n3].mean()) for a in (0, n3, 2 * n3)])
        con.train(); return dict(loss=[round(x, 4) for x in out[0]], gain_over_shuffled=[round(y - x, 4) for x, y in zip(*out)])

    g = torch.Generator().manual_seed(1 if not cycle else 2)
    B, BC, LAM = (16, 0, 0) if not cycle else (0, 8, 1.0) if PURE else (8, 4, 0.25)
    B, BC = int(os.environ.get('VLENS_B', B)), int(os.environ.get('VLENS_BC', BC))
    CFM = float(os.environ.get('VLENS_CFM', 0.05))                             # contrastive flow-matching weight
    hist = [dict(step=0, val=val())]; log(tag, 'step 0 val see/inj/hold', hist[0]['val'])
    preview(K, con, D, W, run / f'preview_{tag}_0000.jpg')
    run_l, t_end = [], time.monotonic() + 60 * minutes
    for step in range(1, steps + 1):
        loss = torch.zeros((), device=DEV); cyc = acc = 0.0
        if B > 0:
            _, A, img, _ = draw_batch(D, B, g, (0.4, 0.3, 0.3, 0.0))
            sig = K.sig_base[torch.randint(0, len(K.sig_base), (B,), generator=g).to(DEV)]
            eps = torch.randn(B, K.h * K.h, 128, generator=g).to(DEV)
            with torch.autocast('cuda', dtype=BF, enabled=DEV == 'cuda'):
                C = con(A.to(DEV))
            loss = K.fm_loss(tr['z'][img], C, sig, eps, CFM).mean()
        if cycle:
            kinds, A2, img2, row2 = draw_batch(D, BC, g, (0.3, 0.2, 0.3, 0.2))
            with torch.autocast('cuda', dtype=BF, enabled=DEV == 'cuda'):
                C2 = con(A2.to(DEV))
            x = K.decode(K.sample(C2, torch.randn(BC, K.h * K.h, 128, generator=g).to(DEV))).clamp(-1, 1)
            errs, fves = [], []
            for b in range(BC):                                                    # read back in the source's own context
                k = KINDS[int(kinds[b])]; t = int(row2[b])
                w = HOLD_TR[int(tr['hold_items'][t, 2])] if k == 'hold' else CHAR_T[t % nW] if k == 'char' else None
                zr = (R.readback(x[b:b + 1], k, w)[0] - mu[k]) / sd[k]; za = (A2[b].float().to(DEV) - mu[k]) / sd[k]
                e = ((zr - za) ** 2).sum(); errs.append(e / zr.numel()); fves.append(1 - float(e) / float((za ** 2).sum()))
            rec = torch.stack(errs).mean(); loss = loss + LAM * rec; cyc = float(rec); acc = float(np.mean(fves))
        opt.zero_grad(set_to_none=True); loss.backward()
        torch.nn.utils.clip_grad_norm_(con.parameters(), 1.0); opt.step(); sched.step()
        run_l.append((float(loss.detach()), cyc, acc))
        if step % 50 == 0:
            m = np.mean(run_l, 0); run_l = []
            log(tag, 'step', step, 'loss', round(m[0], 4), 'readback mse', round(m[1], 4), 'readback fve', round(m[2], 4), 'lr', f'{sched.get_last_lr()[0]:.2e}')
        last = step == steps or time.monotonic() > t_end
        if step % 500 == 0 or last:
            hist.append(dict(step=step, val=val())); log(tag, 'step', step, 'val see/inj/hold', hist[-1]['val'])
            save(dict(state={k: t.cpu() for k, t in con.state_dict().items()}, hist=hist, step=step), W / f'con_{tag}.pt')
            preview(K, con, D, W, run / f'preview_{tag}_{step:04d}.jpg')
        if last:
            break
    st = {k: t.to(torch.float16 if t.is_floating_point() else t.dtype).cpu() for k, t in con.state_dict().items()}
    save(dict(state=st, hist=hist, step=step), run / f'con_{tag}_fp16.pt')
    json.dump(hist, open(run / f'hist_{tag}.json', 'w'))
    log(tag, 'done at step', step)
