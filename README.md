# vlens

vlens turns one hidden activation of a language model into a picture, with no text anywhere in the image path. The same
model then looks at the picture, and its new activation is compared directly with the original one.

The source and reader is Qwen3-VL-4B-Instruct, a vision-language model. The image generator is FLUX.2 [klein] 4B, a
4-step distilled rectified-flow model that normally reads the hidden states of a Qwen3 text encoder. In vlens klein
gets no prompt and no word embeddings. A small trained adapter computes klein's entire conditioning from the activation.

## Pipeline

1. **Activation.** Qwen3-VL's residual stream after decoder layers 9, 18 and 27 (pre-norm), taken at the last
   position of the reply start and concatenated: 3 x 2560 = 7,680 numbers.
2. **Adapter.** It maps that vector to 64 conditioning tokens of size 7,680, written on top of klein's blank
   conditioning (the encoding of an empty prompt, one constant), as the sum of three parts:
   - a free bias, which carries whatever all pictures share;
   - a direct path that is linear in the standardized activation with no biases (activation -> 1,024 -> 64 tokens of
     256 -> 7,680 per token, last layer zero-initialized). It averages exactly zero over training items, so it can only
     carry what differs between them and every gradient it gets is item-specific;
   - a MetaQuery-style path for nonlinear structure: the activation becomes 8 tokens, 64 learned queries attend to them
     through a 4-layer transformer, and a zero-initialized projection writes the result.

   The first version had only the MetaQuery path. Its outputs for different activations pointed 92% in the same
   direction, and with every loss tried it learned one shared conditioning that ignored the activation.
3. **Picture.** Klein (frozen) runs its own 4-step Euler sampler at 384 px from that conditioning.
4. **Readback.** Qwen3-VL reads the picture in the same context the source activation came from, with the picture in
   place of the content:
   - a caption chat "Picture this scene in your mind: {caption} ..." becomes "... in your mind: [picture] ...";
   - a character chat "You are {x}. ..." becomes "You are [picture]. ...".

   The new activation is compared with the source activation directly, after standardizing each dimension by its
   training mean and spread (fixed constants). No probe or readout is fitted.

## Sources of activations

| source | what Qwen3-VL gets | ground truth |
|---|---|---|
| `see` | a real photo (COCO or CC3M) and "keep it in mind, just reply OK"; no caption anywhere | the photo |
| `inj` | a text-only chat with no image and no description. The see context's states at the 5 reply-start positions after layer 8 are copied in, bypassing tokens, and the model computes on from there | the photo |
| `hold` | the photo's caption in a text-only chat, under 7 training wordings and 2 test wordings | the photo |
| `char` | "You are <character>" chats (264 characters, 53 held out) | none |

## Training

- **Stage B (oracle style, supervised), end to end.** The adapter's conditioning goes through klein's 4 sampling
  steps and the VAE decoder with gradients on; the picture is compared with the real photo in DINOv2-base features
  (1 - cosine of the class tokens, plus the mean over the 256 patch tokens of 1 - cosine, so both content and layout
  count). Klein is frozen and only its conditioning is trained, so every picture is a klein picture.
  - A flow-matching prior on the undistilled klein base 4B (same architecture and conditioning space) keeps the
    conditioning in the region klein draws photos from.
  - Flow matching alone did not work. On the step-distilled klein its error against a real photo is lowest for
    conditionings that make it draw a blurry average (at the first step the blank conditioning scored 1.10, the true
    caption 1.58), and the adapter learned exactly that, one average-photo conditioning that ignored the activation.
    On klein base the true caption beats the blank conditioning at every noise level, but the per-sample signal is
    small (true caption vs a shuffled one: 0.037), and in 500 steps the adapter's gain over shuffled pairs was 0.0001.
  - Held-out checks: the flow-matching loss and the DINOv2 loss of 8 test items per source, each with its gain over
    shuffled activation-photo pairs.
- **Stage C (NLA style, unsupervised reconstruction added).** The adapter makes a picture through klein's 4 sampling
  steps and the VAE decoder, and Qwen3-VL reads it back in the source's own context; gradients flow through all of
  these.
  - The loss is the mean squared error between the standardized readback and source activations: the picture must
    let the model rebuild its own activation.
  - This loss needs no photo or label, so character activations train on it alone.
  - The stage B loss stays on, to keep pictures natural and to limit hiding information in pixel patterns that only
    the reader decodes.

## Evaluation

- **Test sets.** 1,000 held-out photos (500 COCO, 500 CC3M) under 4 sources, plus 53 held-out characters.
- **Reconstruction.** Fraction of variance explained between the readback and the source activation (standardized).
  Two references: the real photo read back in the same context (how much any picture could explain), and shuffled
  pairs (no item information).
- **Readback rank.** The source's rank among all 1,000 sources.
- **SigLIP 2 rank.** The real photo's rank among the 1,000 real photos by SigLIP 2 (so400m/16-512) image-image
  similarity. No stage trains on this scorer.
- **Text route.** Klein given the caption as an ordinary prompt: the text-bottleneck reference.
- **Robustness.** Each picture is read back again after degradations a viewer barely notices (256 px JPEG, JPEG
  quality 30, half resolution), with real photos degraded the same way as the reference. If a picture's
  reconstruction falls much further than the real photos' does, its information sits in pixel patterns that only the
  reader decodes, not in the scene.
- **Prior control.** Random activations with the same per-dimension mean and spread as real ones. If these also
  produce confident, meaningful scenes, the pictures reflect klein's prior rather than Qwen's state.

## A real LLM's own states, no training (`vlens native`)

Qwen-Image-2.1 (Alibaba, September 2026) generates pictures from the last-layer states of Qwen3-VL-8B, and its encoder
is the public Qwen3-VL-8B-Instruct checkpoint (checked tensor by tensor). So no adapter is needed:
Qwen3-VL-8B reads a blank prompt while its states are changed from the inside, and Qwen-Image-2.1-Turbo renders them.
- **Concept steering:** + a multiple of a concept direction after layer 18 (mean state of photos whose captions name
  the concept minus the mean over all photos).
- **Neuron steering:** + 10, 20 or 40 standard deviations on the single MLP neuron of layer 18 that best separates
  those photos; random neurons at the same strength are the control.
- **Transplant:** the states of the last 5 positions while Qwen3-VL-8B looks at a real photo, copied into the blank
  prompt after layer 8 or 18.

Every picture in a row uses the same noise. Qwen3-VL-8B is also asked, in the same steered state, what it is thinking
about.

## Run

```
pip install -e .
scripts/test_tiny.sh                      # every stage in minutes, small random Qwen3-VL (plumbing only)
scripts/run_all.sh /data/work /data/run   # full run on one 80 GB GPU, about 4 hours on an A100
```

All models and data are public: Qwen/Qwen3-VL-4B-Instruct, black-forest-labs/FLUX.2-klein-4B and FLUX.2-klein-base-4B (Apache 2.0), facebook/dinov2-base,
google/siglip2-so400m-patch16-512, yerevann/coco-karpathy (COCO image URLs), pixparse/cc3m-wds.

## Status

The first full run is in progress. Results will be added here.
