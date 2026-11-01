"""The three models: Reader (Qwen3-VL-4B: source and reader), Klein (FLUX.2 [klein] 4B, frozen) and the trained Connector."""
from .common import *


class Reader:
    """Qwen3-VL-4B activations: hooks on decoder layers 9/18/27 (outputs = hidden_states[9/18/27], pre-norm); the
    language model is cut after layer 27. see(x) is differentiable in the pixels (the processor's own patchify)."""
    def __init__(self, model=None, cut=True):
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        root = snap(VL)
        self.proc = AutoProcessor.from_pretrained(root)
        if TINY:
            from transformers import AutoConfig
            cfg = AutoConfig.from_pretrained(root); tc, vc = cfg.text_config, cfg.vision_config
            tc.hidden_size, tc.intermediate_size, tc.num_attention_heads, tc.num_key_value_heads, tc.head_dim, tc.num_hidden_layers = 128, 256, 4, 2, 32, 28
            vc.depth, vc.hidden_size, vc.intermediate_size, vc.num_heads, vc.out_hidden_size, vc.deepstack_visual_indexes = 3, 64, 128, 4, 128, [0, 1, 2]
            model = Qwen3VLForConditionalGeneration(cfg).to(BF)
        m = model if model is not None else Qwen3VLForConditionalGeneration.from_pretrained(root, dtype=BF)
        lm = m.model.language_model
        if cut:                                                   # keep every layer when the model also generates
            lm.layers = lm.layers[:max(READ)]
        self.m = m.model.to(DEV).eval().requires_grad_(False); self.layers = lm.layers
        ip = self.proc.image_processor; self.ip = ip
        self.mean = torch.tensor(ip.image_mean, device=DEV).view(1, 3, 1, 1); self.std = torch.tensor(ip.image_std, device=DEV).view(1, 3, 1, 1)
        self.ctx = {}; self.see_ids = self.image_ctx('see')[0]
        assert self.grid.tolist() == [[1, PX // ip.patch_size, PX // ip.patch_size]], self.grid
        self.neutral_ids = self.proc.tokenizer([self.chat(NEUTRAL_T)], return_tensors='pt')['input_ids'].to(DEV)
        tail = lambda ids: self.proc.tokenizer.convert_ids_to_tokens(ids[0, -TP:].tolist())
        assert tail(self.see_ids) == tail(self.neutral_ids), (tail(self.see_ids), tail(self.neutral_ids))
        self.d = m.config.text_config.hidden_size

    def _run(self, last, grab=False, patch=None, **kw):
        cap = {}

        def rd(mod, a, o, l):
            cap[l] = o[0] if isinstance(o, tuple) else o

        def pt(mod, a, o):
            t = (o[0] if isinstance(o, tuple) else o).clone()
            t[:, -TP:] = patch.to(t.dtype)
            return (t,) + tuple(o[1:]) if isinstance(o, tuple) else t
        hooks = [self.layers[l - 1].register_forward_hook(lambda mod, a, o, l=l: rd(mod, a, o, l)) for l in READ]
        if grab:
            hooks.append(self.layers[L_INJ - 1].register_forward_hook(lambda mod, a, o: rd(mod, a, o, 'g')))
        if patch is not None:
            hooks.append(self.layers[L_INJ - 1].register_forward_hook(pt))
        try:
            self.m(**kw, use_cache=False)
        finally:
            for h in hooks:
                h.remove()
        rows = torch.arange(len(last), device=DEV)
        out = torch.cat([cap[l][rows, last] for l in READ], -1).float()
        return (out, cap['g'][:, -TP:].clone()) if grab else out

    def pixels(self, x):
        """[B,3,PX,PX] in [-1,1] -> the processor's pixel_values (rescale, normalize, patchify)"""
        x = ((x + 1) / 2 - self.mean) / self.std
        p, gh, gw = self.ip.patchify(x, patch_size=self.ip.patch_size, merge_size=self.ip.merge_size, temporal_patch_size=self.ip.temporal_patch_size)
        return p.reshape(-1, p.shape[-1])

    def image_ctx(self, key):
        """token ids (+ the processor's other outputs) of a chat holding one PX-px image, cached: 'see' (SEE_T) or
        (kind, wording) for a hold or character wording with the image in place of its {c} / {x} slot"""
        if key not in self.ctx:
            if key == 'see':
                content = [{'type': 'image'}, {'type': 'text', 'text': SEE_T}]
            else:
                t = key[1]; pre, post = t.split('{c}' if '{c}' in t else '{x}')
                content = ([{'type': 'text', 'text': pre}] if pre.strip() else []) + [{'type': 'image'}] + ([{'type': 'text', 'text': post}] if post.strip() else [])
            text = self.proc.apply_chat_template([{'role': 'user', 'content': content}], tokenize=False, add_generation_prompt=True)
            enc = self.proc(text=[text], images=[Image.new('RGB', (PX, PX), (127, 127, 127))], return_tensors='pt')
            self.grid = enc['image_grid_thw']
            self.ctx[key] = (enc['input_ids'].to(DEV), {k: v.to(DEV) for k, v in enc.items() if k not in ('input_ids', 'attention_mask', 'pixel_values', 'image_grid_thw')})
        return self.ctx[key]

    def look(self, x, key='see', grab=False):
        """activation of the chat `key` with the images x [B,3,PX,PX] in [-1,1] in it (differentiable in x)"""
        ids, extra = self.image_ctx(key); B = len(x); ids = ids.expand(B, -1)
        last = torch.full((B,), ids.shape[1] - 1, device=DEV)
        return self._run(last, grab=grab, input_ids=ids, attention_mask=torch.ones_like(ids), pixel_values=self.pixels(x.to(DEV)).to(BF),
                         image_grid_thw=self.grid.expand(B, -1).to(DEV), **{k: v.expand(B, *v.shape[1:]) for k, v in extra.items()})

    def see(self, x, grab=False):
        return self.look(x, 'see', grab)

    def readback(self, x, kind, wording=None):
        """pictures read back in the context their source came from, the picture in place of the content: see -> SEE_T;
        inj -> the same layer-L_INJ reply-start transplant into NEUTRAL_T, taken from the picture; hold / char -> that
        wording with the picture in its {c} / {x} slot. Differentiable in x."""
        if kind == 'inj':
            return self.inject(self.see(x, grab=True)[1])
        if kind in ('hold', 'char'):
            return self.look(x, (kind, wording))
        return self.see(x)

    def inject(self, H):
        """H [B,TP,d]: states after layer L_INJ at the reply-start positions of a see context -> NEUTRAL_T chat"""
        B = len(H); ids = self.neutral_ids.expand(B, -1)
        return self._run(torch.full((B,), ids.shape[1] - 1, device=DEV), patch=H.to(DEV), input_ids=ids, attention_mask=torch.ones_like(ids))

    def chat(self, t):
        return self.proc.apply_chat_template([{'role': 'user', 'content': [{'type': 'text', 'text': t}]}], tokenize=False, add_generation_prompt=True)

    @torch.no_grad()
    def hold(self, texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.proc.tokenizer([self.chat(t) for t in texts[i:i + batch]], return_tensors='pt', padding=True, padding_side='right').to(DEV)
            out.append(self._run(enc['attention_mask'].sum(1) - 1, **enc).half().cpu())
        return torch.cat(out)


class Klein:
    def __init__(self, text_encoder=False):
        from diffusers import Flux2KleinPipeline
        from diffusers.pipelines.flux2.pipeline_flux2_klein import compute_empirical_mu, retrieve_timesteps
        root = snap(KLEIN, ignore=['*.jpg', 'flux-2-klein*.safetensors'])
        kw = {} if text_encoder else dict(text_encoder=None, tokenizer=None)
        self.pipe = Flux2KleinPipeline.from_pretrained(root, torch_dtype=BF, **kw).to(DEV)
        self.pipe.set_progress_bar_config(disable=True)
        self.tf = self.pipe.transformer.requires_grad_(False); self.vae = self.pipe.vae.requires_grad_(False)
        self.h = PX // 16
        retrieve_timesteps(self.pipe.scheduler, 4, DEV, sigmas=np.linspace(1.0, 0.25, 4), mu=compute_empirical_mu(image_seq_len=self.h * self.h, num_steps=4))
        self.sig = self.pipe.scheduler.sigmas.float().to(DEV)                        # 4 steps + the final 0
        assert len(self.sig) == 5 and float(self.sig[-1]) == 0, self.sig
        self.ids = self.pipe._prepare_latent_ids(torch.zeros(1, 128, self.h, self.h)).to(DEV)
        bn = self.vae.bn
        self.bn_m = bn.running_mean.view(1, -1, 1, 1).float().to(DEV)
        self.bn_s = torch.sqrt(bn.running_var.view(1, -1, 1, 1) + self.vae.config.batch_norm_eps).float().to(DEV)

    @torch.no_grad()
    def encode(self, x):
        return self.pipe._encode_vae_image(x.to(DEV, BF), None)                       # [B,128,h,h] (patchified, normalized)

    def v(self, xt, s, C):
        tids = self.pipe._prepare_text_ids(C).to(DEV)
        return self.tf(hidden_states=xt.to(BF), timestep=s.to(BF), guidance=None, encoder_hidden_states=C.to(BF),
                       txt_ids=tids, img_ids=self.ids.expand(len(xt), -1, -1), return_dict=False)[0].float()

    def pack(self, z):
        return z.float().reshape(len(z), 128, -1).permute(0, 2, 1)

    def fm_loss(self, z, C, sig, eps, lam=0.0):
        """flow-matching loss per item. lam > 0: contrastive flow matching (Stoica et al. 2025), minus lam times the
        loss against another item's target (the batch rolled by one), so a prediction gains only by being item-specific"""
        x0 = self.pack(z.to(DEV)); s = sig[:, None, None]
        pred = self.v((1 - s) * x0 + s * eps, sig, C); tgt = eps - x0
        loss = ((pred - tgt) ** 2).mean((1, 2))
        return loss - lam * ((pred - tgt.roll(1, 0)) ** 2).mean((1, 2)) if lam else loss

    def noise(self, seeds):
        return torch.stack([torch.randn(self.h * self.h, 128, generator=torch.Generator().manual_seed(int(s))) for s in seeds]).to(DEV)

    def sample(self, C, x):
        """klein's 4-step Euler sampler (the pipeline's FlowMatchEulerDiscrete steps), differentiable in C"""
        for k in range(4):
            x = x + (self.sig[k + 1] - self.sig[k]) * self.v(x, self.sig[k].expand(len(x)), C)
        return x

    def decode(self, x):
        z = x.permute(0, 2, 1).reshape(len(x), 128, self.h, self.h) * self.bn_s + self.bn_m
        z = self.pipe._unpatchify_latents(z)
        return self.vae.decode(z.to(BF), return_dict=False)[0].float()                # [B,3,PX,PX] in about [-1,1]

    @torch.no_grad()
    def draw(self, C, seeds):
        return self.decode(self.sample(C, self.noise(seeds))).clamp(-1, 1)


class Connector(nn.Module):
    """activation [B, din] -> klein conditioning [B, M, dout]. MetaQuery-style: the standardized activation becomes P
    tokens, M learned queries attend to them through a small transformer, a zero-initialized projection writes the
    result on top of klein's blank conditioning (encoding of an empty prompt, a constant), plus a zero-initialized
    per-channel gate on the raw activation (source and klein's encoder are both Qwen3 layers 9/18/27 concatenated)."""
    def __init__(self, din, q0, mu, sd, d=768, P=8, layers=4, heads=12):
        super().__init__()
        dout = q0.shape[-1]; self.P = P
        self.register_buffer('mu', mu.float()); self.register_buffer('sd', sd.float())
        self.inp = nn.Sequential(nn.Linear(din, 1536), nn.GELU(), nn.Linear(1536, P * d))
        self.pos = nn.Parameter(0.02 * torch.randn(P, d)); self.qry = nn.Parameter(0.02 * torch.randn(len(q0), d))
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d, heads, 4 * d, dropout=0.0, activation='gelu', batch_first=True, norm_first=True),
                                         layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d); self.out = nn.Linear(d, dout)
        nn.init.zeros_(self.out.weight); nn.init.zeros_(self.out.bias)
        self.q0 = nn.Parameter(q0.float().clone()); self.beta = nn.Parameter(torch.zeros(dout))
        self.drop = nn.Dropout(0.1)

    def forward(self, a):
        a = a.float(); z = self.drop((a - self.mu) / self.sd)
        t = self.inp(z).view(len(a), self.P, -1) + self.pos
        h = self.enc(torch.cat([t, self.qry.expand(len(a), -1, -1)], 1))[:, self.P:]
        return self.q0 + self.out(self.norm(h)) + (self.beta * a[:, None] if a.shape[1] == self.beta.shape[0] else 0)
