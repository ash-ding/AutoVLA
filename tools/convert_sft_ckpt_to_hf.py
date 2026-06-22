"""
Convert AutoVLA SFT PyTorch-Lightning checkpoint to HuggingFace safetensors
format, so vLLM (which only ingests HF-format dirs) can load the model.

Key transformations:
  1. Strip Lightning prefix: keys `autovla.vlm.X` -> `X`
  2. Extend tokenizer with 2048 action tokens (matches SFT-time vocab)
  3. resize_token_embeddings on the HF model to match
  4. Load remaining (stripped) state_dict
  5. Save model + tokenizer to <out_dir> in HF format

Usage:
  python tools/vllm_explore/convert_sft_ckpt_to_hf.py \
      --sft_ckpt /data/ckpt_cache/4v90.ckpt \
      --base_model_path /backup/autovla_models/Qwen2.5-VL-3B-Instruct \
      --codebook_path codebook_cache/agent_vocab.pkl \
      --out_dir /data/hf_ckpt/4v90 \
      [--verify]

After conversion:
  python -c "
  from transformers import Qwen2_5_VLForConditionalGeneration, AutoTokenizer
  m = Qwen2_5_VLForConditionalGeneration.from_pretrained('/data/hf_ckpt/4v90')
  t = AutoTokenizer.from_pretrained('/data/hf_ckpt/4v90')
  print('vocab:', len(t))  # should be 153713 = 151665 + 2048
  print('lm_head:', m.lm_head.weight.shape)  # should be (153713, hidden)
  "
"""
import argparse
import pickle
import sys
from pathlib import Path

import torch
from transformers import (
    Qwen2_5_VLForConditionalGeneration,
    AutoTokenizer,
    AutoProcessor,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sft_ckpt", required=True,
                   help="PyTorch Lightning .ckpt path (e.g. /data/ckpt_cache/4v90.ckpt)")
    p.add_argument("--base_model_path", required=True,
                   help="Base HF model path (e.g. /backup/autovla_models/Qwen2.5-VL-3B-Instruct)")
    p.add_argument("--codebook_path", required=True,
                   help="Action codebook .pkl (e.g. codebook_cache/agent_vocab.pkl); "
                        "used to derive number of action tokens (typically 2048)")
    p.add_argument("--out_dir", required=True,
                   help="HF output dir (e.g. /data/hf_ckpt/4v90)")
    p.add_argument("--verify", action="store_true",
                   help="After save, reload and assert no missing/unexpected keys")
    return p.parse_args()


def main():
    args = parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Loading SFT ckpt {args.sft_ckpt}")
    sft = torch.load(args.sft_ckpt, map_location="cpu", weights_only=False)
    sd = sft["state_dict"]
    print(f"      state_dict has {len(sd)} keys, "
          f"first key: {next(iter(sd.keys()))}")

    print(f"[2/6] Stripping `autovla.vlm.` prefix -> bare HF keys")
    # The original SFT ckpt was saved from SFTAutoVLA, which contains
    #   self.autovla = AutoVLA(config)        # AutoVLA wraps self.vlm
    # so saved keys look like `autovla.vlm.model.X`. We want the keys
    # that Qwen2_5_VLForConditionalGeneration directly accepts:
    # `model.X` and `lm_head.weight`. Strip `autovla.vlm.` from each.
    hf_sd = {}
    for k, v in sd.items():
        if k.startswith("autovla.vlm."):
            hf_sd[k[len("autovla.vlm."):]] = v
        else:
            print(f"      WARN: unexpected key prefix dropped: {k}")
    print(f"      stripped state_dict: {len(hf_sd)} keys, "
          f"first key: {next(iter(hf_sd.keys()))}")

    print(f"[3/6] Loading base model from {args.base_model_path}")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        args.base_model_path,
        dtype=torch.bfloat16,
    )
    # NOTE: model.config.vocab_size (e.g. 151936) is padded for matmul
    # alignment; the actual tokenizer vocab is smaller (e.g. 151665).
    # SFT-time `resize_token_embeddings(tokenizer_vocab + 2048)` truncated the
    # padding away. We must mirror that here, not extend from the padded size.
    embed_rows = model.get_input_embeddings().weight.shape[0]
    print(f"      base config.vocab_size: {model.config.vocab_size} "
          f"(embedding rows: {embed_rows}; padded for alignment)")

    print(f"[4/6] Extending tokenizer + embeddings with action tokens")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    base_tok_size = len(tokenizer)
    print(f"      base tokenizer vocab: {base_tok_size}")
    with open(args.codebook_path, "rb") as f:
        cb = pickle.load(f)
    n_action = cb["token_all"]["veh"].shape[0]
    print(f"      codebook action_len: {n_action}")
    tokenizer.add_tokens([f"<action_{i}>" for i in range(n_action)],
                         special_tokens=False)
    print(f"      tokenizer after add_tokens: {len(tokenizer)} "
          f"(= {base_tok_size} + {n_action})")
    assert len(tokenizer) == base_tok_size + n_action, \
        f"unexpected tokenizer size: {len(tokenizer)} != {base_tok_size}+{n_action}"

    # Resize to EXACT tokenizer size (matches SFT-time resize behavior, which
    # truncates the padded embedding to len(tokenizer) and re-allocates).
    model.resize_token_embeddings(len(tokenizer))
    print(f"      resized embed_tokens to {model.get_input_embeddings().weight.shape}")
    print(f"      resized lm_head     to {model.lm_head.weight.shape}")
    # Sanity: shapes should match the SFT state_dict's stored shapes
    sft_embed_shape = hf_sd.get("model.language_model.embed_tokens.weight",
                                hf_sd.get("model.embed_tokens.weight", None))
    if sft_embed_shape is not None:
        sft_embed_shape = sft_embed_shape.shape
        cur_embed_shape = model.get_input_embeddings().weight.shape
        print(f"      SFT ckpt embed shape: {tuple(sft_embed_shape)}  "
              f"vs current model: {tuple(cur_embed_shape)}")
        assert tuple(sft_embed_shape) == tuple(cur_embed_shape), \
            "embedding shape mismatch — resize did not match SFT-time vocab"

    print(f"[5/6] Loading stripped SFT state_dict (must be 0/0)")
    missing, unexpected = model.load_state_dict(hf_sd, strict=False)
    print(f"      missing: {len(missing)}, unexpected: {len(unexpected)}")
    if missing:
        print(f"      first missing: {missing[:5]}")
    if unexpected:
        print(f"      first unexpected: {unexpected[:5]}")
    assert not missing and not unexpected, \
        "ckpt -> HF model load is not exact; refusing to save bogus model"

    print(f"[6/6] Saving to {out_dir}")
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    # Also save processor (vLLM may need it for image/video preprocessing)
    try:
        processor = AutoProcessor.from_pretrained(args.base_model_path)
        # Need to update processor's tokenizer too
        processor.tokenizer = tokenizer
        processor.save_pretrained(out_dir)
        print(f"      processor saved")
    except Exception as e:
        print(f"      WARN: processor save failed: {e}")
    print(f"\n✓ Done. HF model at: {out_dir}")
    print(f"  - {sum(1 for _ in out_dir.glob('*.safetensors'))} safetensors shards")
    print(f"  - tokenizer.json present: {(out_dir/'tokenizer.json').exists()}")

    if args.verify:
        print("\n[verify] Reloading from disk + asserting cleanliness...")
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        model2 = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            out_dir, dtype=torch.bfloat16,
        )
        tok2 = AutoTokenizer.from_pretrained(out_dir)
        expected = base_tok_size + n_action
        assert len(tok2) == expected, f"reloaded tokenizer: {len(tok2)} != {expected}"
        assert model2.lm_head.weight.shape[0] == expected, \
            f"reloaded lm_head: {model2.lm_head.weight.shape[0]} != {expected}"
        print(f"      ✓ reload OK; vocab={len(tok2)}, "
              f"lm_head shape={model2.lm_head.weight.shape}")


if __name__ == "__main__":
    main()
