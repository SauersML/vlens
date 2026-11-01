# vlens

vlens turns one hidden activation of a language model into a picture, with no text anywhere in the image path. The same
model then looks at the picture, and its new activation is compared directly with the original one.

The source and reader is Qwen3-VL-4B-Instruct, a vision-language model. The image generator is FLUX.2 [klein] 4B, a
4-step distilled rectified-flow model that normally reads the hidden states of a Qwen3 text encoder. In vlens klein
gets no prompt and no word embeddings. A small trained adapter computes klein's entire conditioning from the activation.

## Pipeline

1. **Activation.** Qwen3-VL's residual stream after decoder layers 9, 18 and 27 (pre-norm), taken at the last
   position of the reply start and concatenated: 3 x 2560 = 7,680 numbers.
2. **Adapter.** A MetaQuery-style connector maps that vector to 64 conditioning tokens of size 7,680:
   - the standardized activation becomes 8 tokens;
   - 64 learned queries attend to them through a 4-layer transformer;
   - a zero-initialized projection writes the result on top of klein's blank conditioning (the encoding of an empty
     prompt, one constant);
   - a zero-initialized per-channel gate also passes the raw activation through.
3. **Picture.** Klein (frozen) runs its own 4-step Euler sampler at 384 px from that conditioning.
4. **Readback.** Qwen3-VL is shown the picture with the same instruction it used when it saw a real photo. Its
   activation is compared with the source activation by cosine similarity, after each context's mean activation is
   subtracted (one fixed vector per context). No probe or readout is fitted.

## Sources of activations

| source | what Qwen3-VL gets | ground truth |
|---|---|---|
| `see` | a real photo (COCO or CC3M) and "keep it in mind, just reply OK"; no caption anywhere | the photo |
| `inj` | a text-only chat with no image and no description. The see context's states at the 5 reply-start positions after layer 8 are copied in, bypassing tokens, and the model computes on from there | the photo |
| `hold` | the photo's caption in a text-only chat, under 7 training wordings and 2 test wordings | the photo |
| `char` | "You are <character>" chats (300 characters, 60 held out) | none |

## Training

- **Stage B (oracle style, supervised).** Klein's own flow-matching loss, with the real photo as the target:
  1. encode the photo with klein's VAE (latent x0);
  2. mix in noise at one of klein's 4 sampling levels, x_t = (1 - s) x0 + s * noise;
  3. klein predicts the velocity (noise - x0) from x_t, s and the adapter's conditioning;
  4. the loss is the mean squared error. Only the adapter is trained.

  Klein can only denoise well if the conditioning says what the photo is, so the adapter must carry the photo's
  content from the activation.
- **Stage C (NLA style, unsupervised readback added).** The adapter makes a picture through klein's 4 sampling
  steps and the VAE decoder, and Qwen3-VL reads it; gradients flow through all of these.
  - A contrastive loss (temperature 0.05) asks the readback to be nearest its own source among all training sources
    of that kind (90k photos, 120k captions, 720 character chats).
  - This loss needs no photo, so character activations train on it alone.
  - The stage B loss stays on, to keep pictures natural and to limit hiding information in pixel patterns that only
    the reader decodes.

## Evaluation

- **Test sets.** 1,000 held-out photos (500 COCO, 500 CC3M) under 4 sources, plus 60 held-out characters.
- **Readback rank.** The source's rank among all 1,000 sources, using the reader's own activations.
- **SigLIP 2 rank.** The real photo's rank among the 1,000 real photos by SigLIP 2 (so400m/16-512) image-image
  similarity. No stage trains on this scorer.
- **Text route.** Klein given the caption as an ordinary prompt: the text-bottleneck reference.
- **Prior control.** Random activations with the same per-dimension mean and spread as real ones. If these also
  produce confident, meaningful scenes, the pictures reflect klein's prior rather than Qwen's state.

## Run

```
pip install -e .
scripts/test_tiny.sh                      # every stage in minutes, small random Qwen3-VL (plumbing only)
scripts/run_all.sh /data/work /data/run   # full run on one 80 GB GPU, about 4 hours on an A100
```

All models and data are public: Qwen/Qwen3-VL-4B-Instruct, black-forest-labs/FLUX.2-klein-4B (Apache 2.0),
google/siglip2-so400m-patch16-512, yerevann/coco-karpathy (COCO image URLs), pixparse/cc3m-wds.

## Status

The first full run is in progress. Results will be added here.
