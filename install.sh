# NOTE: flash-attn is intentionally NOT installed.
#   - The AutoVLA repo has zero `flash_attn` imports; Qwen2.5-VL is loaded
#     without an attn_implementation kwarg, so transformers defaults to SDPA
#     (built into torch 2.10 — no compile needed) and falls back to eager.
#   - vLLM ships its own pre-compiled flash-attn internally (vllm_flash_attn);
#     the repo doesn't import it either.
#   - Building flash-attn==2.7.4.post1 from source reliably OOMs nvcc on this
#     box (no matching prebuilt wheel for torch 2.10 + cu128) and drops SSH.
# pip install flash-attn==2.7.4.post1

# Only needed for Waymo preprocessing; skip until that path is actually used.
# pip install waymo-open-dataset-tf-2-12-0==1.6.7

pip install --upgrade typing_extensions
pip install autoawq==0.2.8 --no-deps
