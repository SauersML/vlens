"""Activation -> picture with no text anywhere in the image path, read back directly by the same model.

Source and reader: Qwen3-VL-4B-Instruct. Activation = its residual stream after decoder layers 9, 18 and 27 (raw,
pre-norm) at the last position of the reply start, concatenated (3 x 2560 = 7,680 numbers). Photos: COCO (scenes) and
CC3M (web images: products, art, illustrations, people, text). Sources:
  see    Qwen3-VL is shown the photo (384 px) with SEE_T; no caption anywhere.                      ground truth: photo
  inj    bypassing tokens: the see context's states at the 5 reply-start positions after layer 8 are copied into a
         text-only chat with no image and no description (NEUTRAL_T); Qwen3-VL computes on; read 9/18/27.  photo
  hold   the photo's caption in a text-only chat (HOLD_TR wordings in training, HOLD_TE at test).            photo
  char   'You are <character>' chats (characters.txt): no photo, used only by the unsupervised loss.          none
Generator: FLUX.2 [klein] 4B, frozen, 4 sampling steps at 384 px. No prompt and no word vectors go to it: its whole
conditioning (M=64 tokens of 7,680) is computed from the activation by a trained adapter ("connector").
  stage B (oracle style)  klein's own flow-matching loss on the real photo (see, inj, hold sources).
  stage C (NLA style)     + an autoencoder loss: activation -> picture (4 differentiable sampling steps) -> Qwen3-VL reads
                          the picture in the source's own context, the picture in place of the content -> activation;
                          mean squared error against the source after standardizing each dimension (fixed constants).
                          No probe or readout is fitted; needs no photo (characters train on it alone).
  eval  1,000 held-out photos (500 COCO, 500 CC3M) x 4 sources, 53 held-out characters, 60 random activations (same
        per-dimension mean and spread: the prior control); fraction of variance explained with the real photo read back
        the same way as the ceiling and shuffled pairs as the floor; readback ranks; SigLIP 2 image-image ranks against
        the real photos (a scorer no stage trained on); the text route (klein given the caption) as the reference.
Usage: python -m vlens ... (vlens/__main__.py, scripts/run_all.sh)"""
__version__ = "0.1.0"
