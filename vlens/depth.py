"""A picture lens across depth: what Qwen3-VL-8B's states hold at each layer, drawn by Qwen-Image-2.1.

Qwen-Image-2.1 draws from Qwen3-VL-8B's last-layer states (before the final norm). For each layer L a linear map
(ridge, fit on token pairs from N_FIT contexts: photos with the see prompt and captions in the hold wordings) carries
the layer-L state at a position to the last-layer state at the same position, and the untouched generator draws the
mapped states. Like a tuned lens with a picture as the readout. What later layers would still move between positions
is not available to it, so a layer-L picture shows what each position already holds there.
Sets:
  photos  states at the generator's positions after Qwen3-VL-8B looked at a photo (see prompt); ranked among the photos
  think   "Think of {x}. Don't name it or describe it. Just reply OK." with x an object; only the positions after
          "{x}." are drawn, so the generator never reads the words; ranked among the objects' labels
  hop     the same with x a description whose answer has to be worked out ("the tower built in Paris for the 1889
          World's Fair"); ranked among the answers' labels
  direct, twostep   the same 48 answers (PAIRS) named (direct) or described (twostep), each ranked among the 48 labels;
          per-item ranks are kept, so the first layer at which each answer is drawn can be compared within the pair
  secret  "Let's play a guessing game. Secretly choose {an animal, a vehicle, ...} whose name starts with the letter
          {B, C, ...}." + the think tail: the prompt leaves the answer open. Afterwards, in the same chat (it replied
          OK), the 8B is asked what it chose; its greedy answer is the label, and 8 sampled answers give how settled the
          choice is (agreement). Ranked among the distinct labels of the same category; a lens that always drew the
          category's most common answer is the baseline. Does the picture before the reply show the choice it later names?
  faithful  the picture against what the model itself says it thought, not our answer key: "Think of the kind of
          creature Squidward from SpongeBob SquarePants is." (CREATURES, chosen for common confusions: octopus or
          squid, warthog or boar, red panda or giant panda) and the two-step PAIRS. In the same chat, after OK, the 8B
          is asked what it was thinking of; its greedy answer is the label. Where it differs from the key, does the
          picture show its answer or the key? Judged by SigLIP 2 (which of the two is closer) and by the 8B itself
          looking at the picture (A or B, order shuffled); where they agree, the 8B picks between its answer and
          another item's answer
  raw     layer-L states drawn with no map, for L in VLENS_DEPTH_RAW (default 28; control: is the map needed?)
VLENS_DEPTH_LAYERS (comma list) and VLENS_DEPTH_SETS (default photos,think,hop) choose the layers and sets;
VLENS_DEPTH_FIT_THINK=1 makes a quarter of the fit contexts "Think of this: {caption}" + TAIL (tail positions like the
think sets', after COCO scenes, none of the eval answers).
Usage: python -m vlens depth OUT [N_FIT]"""
from .common import *
from .native import Native
from .qimg import ridge, ranks, ranks_z, summary, THINK

LAYERS = tuple(int(x) for x in os.environ.get('VLENS_DEPTH_LAYERS', '4,8,12,16,20,24,28,32').split(','))
SETS = os.environ.get('VLENS_DEPTH_SETS', 'photos,think,hop').split(',')
RAW = tuple(int(x) for x in os.environ.get('VLENS_DEPTH_RAW', '28').split(','))
FIT_THINK = os.environ.get('VLENS_DEPTH_FIT_THINK') == '1'
SEED, N_EVAL = 11, int(os.environ.get('VLENS_DEPTH_EVAL', 16))
TAIL = " Don't name it or describe it. Just reply OK."
HOP = [('the Eiffel Tower', "the tower built in Paris for the 1889 World's Fair"),
       ('a bald eagle', 'the national bird of the United States'),
       ('an apple', "the fruit said to have fallen on Isaac Newton's head"),
       ('the Titanic', 'the ship that sank in 1912 after hitting an iceberg'),
       ('an electric guitar', 'the instrument Jimi Hendrix was famous for playing'),
       ('a giant panda', 'the animal on the logo of the World Wildlife Fund'),
       ('the White House', 'the building where the president of the United States lives'),
       ('a bee', 'the insect that makes honey and lives in hives'),
       ('a clownfish', 'the kind of fish Nemo is in the Pixar film'),
       ('the planet Saturn', 'the planet with the most famous rings'),
       ('the Statue of Liberty', 'the monument France gave to the United States in 1886, standing in New York Harbor'),
       ('a pumpkin', "the vegetable that became Cinderella's coach"),
       ('a kangaroo', 'the animal on the Australian coat of arms that hops'),
       ('the Great Pyramid of Giza', 'the largest tomb built for the pharaoh Khufu'),
       ('the Mona Lisa', 'the most famous painting by Leonardo da Vinci'),
       ('a polar bear', 'the largest land predator of the Arctic')]
PAIRS = [('the Eiffel Tower', "the iron tower built in Paris for the 1889 World's Fair"),
         ('the Statue of Liberty', 'the monument France gave to the United States in 1886'),
         ('the Mona Lisa', 'the most famous painting by Leonardo da Vinci'),
         ('a clownfish', 'the kind of fish Nemo is in the Pixar film'),
         ('an electric guitar', 'the instrument Jimi Hendrix was famous for playing'),
         ('a giant panda', 'the animal on the logo of the World Wildlife Fund'),
         ('the Great Pyramid of Giza', 'the largest tomb built for the pharaoh Khufu'),
         ('the planet Saturn', 'the sixth planet from the Sun'),
         ('a bald eagle', 'the national bird of the United States'),
         ('a pumpkin', "the vegetable the fairy godmother turned into Cinderella's coach"),
         ('a kangaroo', 'the animal on the Australian coat of arms that carries its baby in a pouch'),
         ('the Taj Mahal', 'the white marble tomb Shah Jahan built for his wife in Agra'),
         ('the Golden Gate Bridge', 'the red suspension bridge at the entrance to San Francisco Bay'),
         ('the Colosseum', 'the ancient amphitheater in Rome where gladiators fought'),
         ('the Sydney Opera House', "the building with white sail-shaped roofs on the harbor of Australia's largest city"),
         ('a lion', 'the animal called the king of the jungle'),
         ('an owl', 'the kind of bird that delivers mail at Hogwarts'),
         ('a penguin', 'the flightless bird of Antarctica that looks like it wears a tuxedo'),
         ('a koala', 'the Australian animal that sleeps in eucalyptus trees'),
         ('a camel', 'the animal called the ship of the desert'),
         ('a flamingo', 'the pink bird that stands on one leg'),
         ('a strawberry', 'the red fruit with its seeds on the outside'),
         ('a pineapple', 'the fruit SpongeBob lives in'),
         ('a carrot', 'the vegetable Bugs Bunny is always eating'),
         ('a violin', 'the string instrument played with a bow and held under the chin'),
         ('a saxophone', 'the instrument Lisa Simpson plays'),
         ('the Titanic', 'the ship that sank in 1912 after hitting an iceberg'),
         ('a submarine', 'the yellow vessel the Beatles sang about, which travels under the sea'),
         ('a hot air balloon', 'the aircraft lifted by a big bag of heated air above a basket'),
         ('a rocket', 'the kind of vehicle the Saturn V was'),
         ('a volcano', 'the kind of mountain Vesuvius is'),
         ('a snowman', 'what Olaf from Frozen is'),
         ('a pirate ship', 'the vessel Captain Hook sails in Peter Pan'),
         ('a dragon', 'the fire-breathing creature Daenerys Targaryen rides'),
         ('a unicorn', 'the national animal of Scotland'),
         ('Mount Fuji', 'the highest mountain in Japan'),
         ('the Leaning Tower of Pisa', 'the bell tower in Italy famous for tilting'),
         ('a sunflower', 'the flower Vincent van Gogh painted in a famous series of vases'),
         ('a lighthouse', 'the tower on a coast whose light guides ships at night'),
         ('a teddy bear', 'the stuffed toy named after President Theodore Roosevelt'),
         ('a pizza', 'the dish whose Margherita version was named after a queen of Italy'),
         ('a cactus', 'the spiky plant that stores water in the desert'),
         ('a crocodile', 'the reptile that swallowed a ticking clock in Peter Pan'),
         ('a polar bear', 'the largest land predator of the Arctic'),
         ('a bee', 'the insect that makes honey and lives in hives'),
         ('a giraffe', 'the tallest animal in the world'),
         ('a tiger', 'the largest cat species, native to Asia'),
         ('the White House', 'the building where the president of the United States lives')]


SECRET_CATS = ['an animal', 'a fruit or vegetable', 'a musical instrument', 'a vehicle', 'a famous landmark', 'a dish of food']
SECRET_LETTERS = 'BCDFGHKLMPRSTW'
SECRET_T = "Let's play a guessing game. Secretly choose {c} whose name starts with the letter {l}."
REVEAL = "OK<|im_end|>\n<|im_start|>user\nWhat did you choose? Reply with just its name.<|im_end|>\n<|im_start|>assistant\n"


CREATURES = [('an octopus', 'Squidward from SpongeBob SquarePants'), ('a starfish', 'Patrick Star from SpongeBob SquarePants'),
             ('a squirrel', 'Sandy Cheeks from SpongeBob SquarePants'), ('a snail', 'Gary from SpongeBob SquarePants'),
             ('a crab', 'Mr. Krabs from SpongeBob SquarePants'), ('a warthog', 'Pumbaa from The Lion King'), ('a mandrill', 'Rafiki from The Lion King'),
             ('a hornbill', 'Zazu from The Lion King'), ('a meerkat', 'Timon from The Lion King'), ('a seagull', 'Scuttle from The Little Mermaid'),
             ('a parrot', 'Iago from Aladdin'), ('a monkey', 'Abu from Aladdin'), ('a raccoon', 'Meeko from Pocahontas'), ('a chameleon', 'Pascal from Tangled'),
             ('a reindeer', 'Sven from Frozen'), ('a rooster', 'Heihei from Moana'), ('a blue tang', 'Dory from Finding Nemo'),
             ('a sea turtle', 'Crush from Finding Nemo'), ('a shark', 'Bruce from Finding Nemo'), ('a rat', 'Remy from Ratatouille'),
             ('a golden retriever', 'Dug from Up'), ('a pig', 'Hamm from Toy Story'), ('a dinosaur', 'Rex from Toy Story'), ('a beagle', 'Snoopy from Peanuts'),
             ('a Great Dane', 'Scooby-Doo'), ('a canary', 'Tweety from Looney Tunes'), ('a roadrunner', 'Road Runner from Looney Tunes'),
             ('a coyote', 'Wile E. Coyote from Looney Tunes'), ('a frog', 'Kermit from The Muppets'), ('a donkey', 'Eeyore from Winnie the Pooh'),
             ('a black panther', 'Bagheera from The Jungle Book'), ('a python', 'Kaa from The Jungle Book'), ('an orangutan', 'King Louie from The Jungle Book'),
             ('a rabbit', 'Thumper from Bambi'), ('a skunk', 'Flower from Bambi'), ('a snowy owl', 'Hedwig from Harry Potter'),
             ('a hippogriff', 'Buckbeak from Harry Potter'), ('a phoenix', 'Fawkes from Harry Potter'), ('a giant spider', 'Aragog from Harry Potter'),
             ('a dragon', 'Smaug from The Hobbit'), ('a horse', 'Shadowfax from The Lord of the Rings'), ('a giant panda', 'Po from Kung Fu Panda'),
             ('a red panda', 'Master Shifu from Kung Fu Panda'), ('a snow leopard', 'Tai Lung from Kung Fu Panda'), ('a sloth', 'Sid from Ice Age'),
             ('a woolly mammoth', 'Manny from Ice Age'), ('a sabre-toothed tiger', 'Diego from Ice Age'), ('a mouse', 'Jerry from Tom and Jerry')]
ASK_CREATURE = "OK<|im_end|>\n<|im_start|>user\nWhat kind of creature were you thinking of? Answer in one or two words.<|im_end|>\n<|im_start|>assistant\n"
ASK_THING = "OK<|im_end|>\n<|im_start|>user\nWhat were you thinking of? Answer with just its name.<|im_end|>\n<|im_start|>assistant\n"
JUDGE_COLS = ('raw L28', 'raw L30', 'raw L32', 'L30', 'L36')


def norm(x):
    x = x.lower().strip().strip('.!"').strip()
    for a in ('a ', 'an ', 'the '):
        x = x[len(a):] if x.startswith(a) else x
    return x


def same(a, b):
    a, b = norm(a), norm(b)
    return a == b or a in b or b in a


@torch.no_grad()
def judge(Q, im, a, b):
    """the 8B looks at the picture and picks A or B: 0 for a, 1 for b, -1 for neither"""
    msgs = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': f'Which of these does the picture show? A) {a} B) {b} Answer with A or B only.'}]}]
    t = Q.proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    enc = Q.proc(text=[t], images=[im], return_tensors='pt').to(DEV)
    o = Q.enc.generate(**enc, max_new_tokens=3, do_sample=False)
    ans = Q.proc.tokenizer.decode(o[0, enc['input_ids'].shape[1]:], skip_special_tokens=True).strip().upper()
    return 0 if ans.startswith('A') else 1 if ans.startswith('B') else -1


@torch.no_grad()
def reveal(Q, text, n_sample=8, ask=None):
    """the 8B's greedy answer when asked afterwards what it chose (same template, it replied OK), and the share of
    n_sample sampled answers (temperature 1) that match it"""
    ids = Q.proc.tokenizer([Q.pipe.prompt_template_t2i.format(text) + (ask or REVEAL)], return_tensors='pt').to(DEV); n = ids['input_ids'].shape[1]
    dec = lambda o: [Q.proc.tokenizer.decode(x[n:], skip_special_tokens=True).strip().strip('.!"').strip() for x in o]
    g = dec(Q.enc.generate(**ids, max_new_tokens=12, do_sample=False))[0]
    if not n_sample:
        return g, None
    smp = dec(Q.enc.generate(**ids, max_new_tokens=12, do_sample=True, temperature=1.0, top_p=1.0, top_k=0, num_return_sequences=n_sample))
    return g, sum(x.lower() == g.lower() for x in smp) / n_sample


@torch.no_grad()
def states(Q, text, im=None):
    """{layer: [P, D]} at the positions the generator reads (after the system prompt, image pads removed); key
    Q.last holds the last-layer states the pipeline itself returns"""
    cap = {}
    hs = [Q.layers[l - 1].register_forward_hook(lambda m, a, o, l=l: cap.__setitem__(l, (o[0] if isinstance(o, tuple) else o)[0]))
          for l in LAYERS]
    try:
        e, _, pad = Q.pipe._get_qwen_prompt_embeds([text], None if im is None else [im], DEV)
    finally:
        for h in hs:
            h.remove()
    keep = ~pad[0].bool(); out = {l: cap[l][Q.drop:][keep] for l in LAYERS}; out[Q.last] = e[0][keep]
    assert all(len(v) == len(out[Q.last]) for v in out.values())
    return out


def tail_len(Q, text):
    """positions from the end of the generator's input that come after the item (TAIL + the template's end)"""
    tok = lambda s: Q.proc.tokenizer(s)['input_ids']
    full = Q.pipe.prompt_template_t2i.format(text); a, b = tok(full), tok(full[:full.index(TAIL)])
    k = next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), len(b))   # a token spanning the boundary stays out
    return len(a) - k


def fit(Q, n_fit):
    df = coco_table('train').sample(frac=1, random_state=1).head(int(n_fit)); rng = random.Random(0)
    with cf.ThreadPoolExecutor(64) as ex:
        ims = list(ex.map(fetch, df['url'].tolist()))
    X, Y = {l: [] for l in LAYERS}, []
    for i, (im, caps) in enumerate(zip(ims, df['sentences'])):
        if i % 2 == 0 and im is not None:
            s = states(Q, SEE_T, square(im))
        elif FIT_THINK and i % 4 == 1:                       # tail positions after a scene the model is told to keep
            s = states(Q, f'Think of this: {clean(caps[0])}' + TAIL)
        else:
            s = states(Q, HOLD_TR[rng.randrange(len(HOLD_TR))].format(c=clean(caps[0])))
        for l in LAYERS:
            X[l].append(s[l].half().cpu())
        Y.append(s[Q.last].half().cpu())
        if i % 1000 == 0:
            log('fit contexts', i, 'of', len(df))
    Y = torch.cat(Y)[:, None]; maps, res = {}, {}
    for l in LAYERS:
        f, lam, fve = ridge(torch.cat(X[l]), Y); maps[l] = f
        res[l] = dict(strength=lam, val_fve=round(max(fve.values()), 4)); log('layer', l, res[l])
    log('token pairs', len(Y))
    return maps, res


def depth(out, n_fit=4000):
    out = Path(out); (out / 'pics').mkdir(parents=True, exist_ok=True)
    Q = Native(); Q.last = len(Q.layers); log('loaded; layers', Q.last)
    maps, fits = fit(Q, n_fit)
    te = coco_table('test').head(int(N_EVAL * 1.5))
    with cf.ThreadPoolExecutor(32) as ex:
        tims = [square(im) for im in ex.map(fetch, te['url'].tolist()) if im is not None][:N_EVAL]
    think = THINK[:N_EVAL]
    sets = dict(photos=[(SEE_T, im, None) for im in tims],
                think=[(f'Think of {x}.' + TAIL, None, x) for x, _ in think],
                hop=[(f'Think of {d}.' + TAIL, None, x) for x, d in HOP[:N_EVAL]],
                direct=[(f'Think of {x}.' + TAIL, None, x) for x, _ in PAIRS],
                twostep=[(f'Think of {d}.' + TAIL, None, x) for x, d in PAIRS])
    if 'secret' in SETS:
        meta = []
        for c in SECRET_CATS:
            for l in SECRET_LETTERS:
                t = SECRET_T.format(c=c, l=l) + TAIL; g, agree = reveal(Q, t); meta.append(dict(cat=c, letter=l, label=g, agreement=agree, letter_ok=g[:1].upper() == l))
        sets['secret'] = [(SECRET_T.format(c=m['cat'], l=m['letter']) + TAIL, None, m['label']) for m in meta]
        log('secret choices', [(m['cat'], m['letter'], m['label'], m['agreement']) for m in meta])
    if 'faithful' in SETS:
        fmeta = []
        for key, who in CREATURES:
            t = f'Think of the kind of creature {who} is.' + TAIL; fmeta.append(dict(text=t, key=key, report=reveal(Q, t, 0, ASK_CREATURE)[0], kind='creature'))
        for key, d in PAIRS:
            t = f'Think of {d}.' + TAIL; fmeta.append(dict(text=t, key=key, report=reveal(Q, t, 0, ASK_THING)[0], kind='twostep'))
        for m in fmeta:
            m['agree'] = same(m['key'], m['report'])
        sets['faithful'] = [(m['text'], None, m['report']) for m in fmeta]
        log('faithful reports', [(m['key'], m['report']) for m in fmeta if not m['agree']], 'disagree of', len(fmeta))
    sets = {k: v for k, v in sets.items() if k in SETS}
    assert all(l in LAYERS for l in RAW), (RAW, LAYERS)
    cols = [f'L{l}' for l in LAYERS] + [f'L{Q.last}'] + [f'raw L{l}' for l in RAW]
    pics, res = {}, dict(layers=list(LAYERS) + [Q.last], fits={str(l): v for l, v in fits.items()}, columns=cols, scores={})
    for name, items in sets.items():
        pics[name] = {c: [] for c in cols}
        for i, (text, im, _) in enumerate(items):
            s = states(Q, text, im); pos = slice(None) if im is not None else slice(-tail_len(Q, text), None)
            for c in cols:
                if c.startswith('raw'):
                    e = s[int(c[5:])][pos]
                elif c == f'L{Q.last}':
                    e = s[Q.last][pos]
                else:
                    e = maps[int(c[1:])](s[int(c[1:])][pos].float())[:, 0]
                p = Q.picture(e.to(DEV, BF), SEED); pics[name][c].append(p)
                p.save(out / 'pics' / f'{name}_{i}_{c.replace(" ", "")}.jpg', quality=85)
        log('drew', name, len(items))
    for i, im in enumerate(tims):
        im.save(out / 'pics' / f'photo_{i}.jpg', quality=90)
    from transformers import AutoModel, AutoProcessor
    root = snap(SIGLIP); sm = AutoModel.from_pretrained(root, dtype=torch.float16).to(DEV).eval(); sp = AutoProcessor.from_pretrained(root)

    @torch.no_grad()
    def feats(ims=None, texts=None):
        if texts is not None:
            o = sm.get_text_features(**sp(text=texts, padding='max_length', max_length=64, return_tensors='pt').to(DEV))
            return F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu()
        f = []
        for i in range(0, len(ims), 32):
            o = sm.get_image_features(pixel_values=sp(images=ims[i:i + 32], return_tensors='pt')['pixel_values'].to(DEV).half())
            f.append(F.normalize((o if torch.is_tensor(o) else o.pooler_output).float(), dim=-1).cpu())
        return torch.cat(f)
    for name, items in sets.items():
        bank = feats(ims=tims) if name == 'photos' else None if name in ('secret', 'faithful') else feats(texts=[f'a photo of {x}.' for _, _, x in items])
        if name == 'faithful':                            # picture vs the model's own report, and vs the key where they differ
            labs = sorted({m['report'] for m in fmeta}); T = feats(texts=[f'a photo of {x}.' for x in labs]); K = feats(texts=[f'a photo of {m["key"]}.' for m in fmeta])
            rng = random.Random(0); out_f = dict(items=fmeta, columns={})
            for c in cols:
                f = feats(ims=pics[name][c]); s_ = f @ T.T; z = (s_ - s_.mean(0)) / (s_.std(0) + 1e-6)
                own = torch.tensor([labs.index(m['report']) for m in fmeta])
                row = dict(report_top1=round(float((s_.argmax(1) == own).float().mean()), 4), report_top1_z=round(float((z.argmax(1) == own).float().mean()), 4))
                dis = [i for i, m in enumerate(fmeta) if not m['agree']]
                if dis:
                    rep = s_[dis, own[dis]]; key = (f[dis] * K[dis]).sum(-1)
                    row['siglip_report_over_key'] = round(float((rep > key).float().mean()), 4); row['n_disagree'] = len(dis)
                if c in JUDGE_COLS:
                    picks_dis, picks_ag = [], []
                    for i, m in enumerate(fmeta):
                        if m['agree']:
                            foil = rng.choice([x for x in labs if not same(x, m['report'])]); a, b = (m['report'], foil)
                        else:
                            a, b = m['report'], m['key']
                        flip = rng.random() < 0.5; j = judge(Q, pics[name][c][i], *((b, a) if flip else (a, b)))
                        pick_report = (j == 1) if flip else (j == 0)
                        (picks_ag if m['agree'] else picks_dis).append(pick_report)
                    row['judge_report_over_key'] = round(sum(picks_dis) / max(len(picks_dis), 1), 4)
                    row['judge_report_over_foil'] = round(sum(picks_ag) / max(len(picks_ag), 1), 4)
                out_f['columns'][c] = row
            res['faithful'] = out_f
            log('faithful', {c: v for c, v in out_f['columns'].items() if c in JUDGE_COLS or c == f'L{Q.last}'})
            rk = {c: ranks(feats(ims=pics[name][c]), T[own]) for c in cols}
        elif name == 'secret':                              # rank among the distinct labels of the item's own category
            rk = {}
            for c in cols:
                f = feats(ims=pics[name][c]); r = torch.zeros(len(items), dtype=torch.long)
                for cat in SECRET_CATS:
                    idx = [i for i, m in enumerate(meta) if m['cat'] == cat]; labs = sorted({meta[i]['label'] for i in idx})
                    sim = f[idx] @ feats(texts=[f'a photo of {x}.' for x in labs]).T
                    for j, i in enumerate(idx):
                        r[i] = int((sim[j] > sim[j, labs.index(meta[i]['label'])]).sum())
                rk[c] = r
            from collections import Counter
            mode = sum(Counter(m['label'] for m in meta if m['cat'] == cat).most_common(1)[0][1] for cat in SECRET_CATS)
            res['secret'] = dict(items=meta, n_labels={cat: len({m['label'] for m in meta if m['cat'] == cat}) for cat in SECRET_CATS},
                                 most_common_answer_top1=round(mode / len(meta), 4), letter_ok=round(sum(m['letter_ok'] for m in meta) / len(meta), 4),
                                 chance_top1=round(sum(1 / len({x['label'] for x in meta if x['cat'] == m['cat']}) for m in meta) / len(meta), 4))
            log('secret baselines', {k: v for k, v in res['secret'].items() if k != 'items'})
        else:
            fs = {c: feats(ims=pics[name][c]) for c in cols}
            rk = {c: ranks(fs[c], bank) for c in cols}
            if name != 'photos':                          # also with each label's offset removed (ranks_z)
                res.setdefault('scores_z', {})[name] = {c: summary(ranks_z(fs[c], bank)) for c in cols}
        res['scores'][name] = {c: summary(r) for c, r in rk.items()}
        res.setdefault('ranks', {})[name] = {c: r.tolist() for c, r in rk.items()}
        if name != 'photos':
            res['scores'][name]['labels'] = [x for _, _, x in items]
        log(name, {c: (v['top1'], v['top5']) for c, v in res['scores'][name].items() if c != 'labels'})
    S, n = 192, min(8, N_EVAL)
    for name in sets:
        lead = [tims[i] for i in range(n)] if name == 'photos' else [Image.new('RGB', (S, S), 'white')] * n
        Gd = Image.new('RGB', ((len(cols) + 1) * S, n * S), 'white')
        for r in range(n):
            for k, im in enumerate([lead[r]] + [pics[name][c][r] for c in cols]):
                Gd.paste(im.resize((S, S)), (k * S, r * S))
        Gd.save(out / f'depth_{name}.jpg', quality=88)
    json.dump(res, open(out / 'depth.json', 'w'), indent=1)
    log('saved', out / 'depth.json')
