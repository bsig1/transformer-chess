# Transformer Chess Engine Parameters

This project trains a small transformer language model on chess move sequences.

Top-level parameters in `main()` are defined at:
- [main.py:135](/home/blake/ai-lab2/main.py:135) to [main.py:139](/home/blake/ai-lab2/main.py:139)

## `prompt`
- What it does: Inference-only seed text passed to `predict_next_move`.
- Training impact: None.
- Notes:
  - Should be legal/meaningful move text.
  - If length exceeds `block_size`, it is truncated from the left in [main.py:121](/home/blake/ai-lab2/main.py:121).
- Good range: Any opening prefix, ideally `<= block_size` tokens.

## `steps`
- What it does: Number of optimizer updates.
- Training impact: Controls total compute and data exposure.
- Rule of thumb:
  - Total tokens seen is roughly `steps * batch_size * block_size`.
- Good range:
  - Debug/toy runs: `1k-20k`
  - Real training: `100k-2M+` (depends on dataset size and budget)

## `batch_size`
- What it does: Number of sampled sequences per optimizer step.
- Training impact:
  - Larger batch: smoother gradients, better throughput, more memory.
  - Smaller batch: noisier gradients, lower memory.
- Good range:
  - CPU/small GPU: `16-64`
  - Larger GPUs: `128-1024` (often with gradient accumulation to emulate large effective batch)

## `block_size`
- What it does: Context window length used for training samples; also sets model `max_len` in [main.py:158](/home/blake/ai-lab2/main.py:158).
- Training impact:
  - Larger values let the model condition on longer move history.
  - Memory and compute increase with sequence length.
- Good range for chess move modeling:
  - Minimal: `16-32`
  - Practical: `64-256`
  - Large-scale: `256-512` if memory allows

## `lr`
- What it does: AdamW learning rate (set at [main.py:160](/home/blake/ai-lab2/main.py:160)).
- Training impact: Main stability/speed knob.
- Guidance:
  - Current `3e-3` is aggressive for scaling.
  - Toy runs: `1e-3` to `3e-4`
  - Larger runs: `3e-4` to `1e-4` (prefer warmup + decay schedule)

## Other scaling-critical model params (currently defaults in class ctor)

Defined in [main.py:52](/home/blake/ai-lab2/main.py:52):
- `num_hiddens` (`d_model`): currently `64`; large-scale common range `256-1024`
- `num_layers`: currently `2`; large-scale common range `6-24`
- `num_heads`: currently `4`; common heuristic `num_heads ~= d_model / 64` (for example `8, 12, 16`)

## Large-scale starter recipe

A practical first serious setup:
- `block_size=128`
- `batch_size=256` (or equivalent via gradient accumulation)
- `steps=300k+`
- `lr=3e-4` with warmup and decay
- Model:
  - `num_hiddens=512`
  - `num_layers=8`
  - `num_heads=8`

This is a strong baseline before scaling to longer contexts or deeper models.
# transformer-chess
