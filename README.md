# smoLLM500M

A small-scale causal language model pretraining setup built around fixed-length token packing, streaming datasets, and step-based training using Accelerate.

This repository contains:

- a script to build packed Parquet shards from multiple text sources,
- a pretraining script that streams those shards efficiently across GPUs,
- a minimal custom model implementation (smollm3),
- and supporting configs/utilities used during development and debugging.

The overall goal was to keep the pipeline simple, reproducible, and easy to reason about, even if there are more “correct” or sophisticated ways to do parts of this.

### Pretrained weights

The pretrained model weights are available on Hugging Face:

**Hugging Face Hub:** https://huggingface.co/Xv-yn/smollm3-500m

This repository contains the model checkpoints produced by the training
pipeline in this repo, along with the corresponding tokenizer files.

The weights were trained using the scripts in this repository
(`build_packed_shards.py` + `pretrain_smollm3.py`) on ~19.8B tokens
of packed text data.

### Repository Structure

```
.
├── build_packed_shards.py      # Build fixed-length token shards (Parquet)
├── pretrain_smollm3.py         # Multi-GPU pretraining script (Accelerate)
├── configuration_smollm3.py    # Model config (HF-compatible)
├── modeling_smollm3.py         # Model definition
├── modular_smollm3.py          # Modularized model components
├── config.json                 # Example model/config JSON
├── tokenizer.json              # Tokenizer files
├── tokenizer_config.json
├── special_tokens_map.json
├── poc_sanity_check.py         # Small sanity/debug script
├── print_token_ids.py          # Utility for inspecting tokenizer IDs
└── README.md
```

### Overview

#### Data pipeline (high level)

1. Stream text datasets from Hugging Face (streaming=True)
2. Sample across sources using weighted probabilities and per-source token caps
3. Tokenize text, append EOS between documents
4. Concatenate tokens into a rolling buffer
5. Slice fixed-length blocks (seq_len, e.g. 4096)
6. Write Parquet shards (shard-000000.parquet, …)
7. Track progress via:
   - `manifest.jsonl` (checksums, row counts, timestamps)
   - `state.json` (for safe resume)

Each Parquet row corresponds to exactly one packed sequence.

### Building packed shards

```bash
export SOURCES='[
  {"name":"fineweb_edu","dataset":"HuggingFaceFW/fineweb-edu","config":null,"text_col":"text","weight":0.88,"token_cap":17600000000},
  {"name":"dclm","dataset":"mlfoundations/dclm-baseline-1.0","config":null,"text_col":"text","weight":0.07,"token_cap":1400000000},
  {"name":"stackexchange","dataset":"allenai/dolmino-mix-1124","config":"stackexchange","text_col":"text","weight":0.03,"token_cap":600000000},
  {"name":"wiki","dataset":"allenai/dolmino-mix-1124","config":"wiki","text_col":"text","weight":0.01,"token_cap":200000000}
]'

python build_packed_shards.py \
  --model_dir . \
  --out_dir ./packed/seq4096 \
  --seq_len 4096 \
  --total_tokens 20000000000 \
  --blocks_per_shard 8192 \
  --shuffle_buffer 100000 \
  --seed 1234 \
  --sources_json "$SOURCES"
```

#### Key design notes

- Fixed-length only
  - All output rows are exactly seq_len tokens. Anything else is skipped upstream.

- EOS-separated documents
  - Documents are concatenated with EOS between them before packing.

- Token-budget driven
  - Packing stops after total_tokens is reached (not by epochs).

- Weighted multi-source sampling
  - Sources are sampled proportionally to weight, with optional token_cap.

- Resumable
  - RNG state, partial token buffer, and counters are saved so interrupted runs can resume deterministically.

#### Dataset access note

One of the datasets originally used during experimentation is **not included**
in the `SOURCES` example above because it required a Hugging Face login /
special credentials to access.

I intentionally left it out here so that anyone can reproduce the data
packing and training pipeline **without needing to authenticate or request
access**.

If you have access to additional private or gated datasets, they can be added
to `SOURCES` using the same schema:

```json
{
  "name": "...",
  "dataset": "...",
  "config": "...",
  "text_col": "text",
  "weight": ...,
  "token_cap": ...
}
```

### Pretraining

```bash
accelerate launch --num_processes 4 pretrain_smollm3.py \
  --model_dir . \
  --packed_dir ./packed/seq4096 \
  --output_dir ./runs/full_20B_1epoch_4gpu \
  --seq_len 4096 \
  --micro_batch_size 4 \
  --grad_accum_steps 1 \
  --learning_rate 1e-4 \
  --weight_decay 0.01 \
  --num_train_steps 610352 \
  --warmup_steps 2000 \
  --log_every 10 \
  --save_every 50000 \
  --num_workers 0 \
  --mixed_precision bf16 \
  --attn_impl sdpa
```

#### Training loop design

- Step-based, not epoch-based
  - Since the dataset is streamed from Parquet shards, there is no fixed notion of an epoch.

- IterableDataset + file-level sharding
  - Each distributed rank gets a disjoint subset of shard files (files[rank::world_size]).

- Gradient accumulation
  - Used to control effective batch size.

- Explicit safety checks
  - Out-of-range token IDs
  - NaN / Inf loss detection

- Main-process-only logging & checkpointing

The number of training steps was chosen based on dataset size (~19.8B tokens) and effective batch configuration.
I don’t remember the exact calculation used at the time, but it can be reverse-derived roughly as:

```
tokens_per_step ≈ seq_len × micro_batch_size × num_processes × grad_accum_steps
steps ≈ total_tokens / tokens_per_step
```

There’s probably a cleaner or more canonical way to structure this, but this approach was the easiest for me to reason about and worked reliably in practice.

### Model

The smollm3 model is implemented in a Hugging Face–compatible way:

- `configuration_smollm3.py` — config class
- `modeling_smollm3.py` — main model
- `modular_smollm3.py` — modular components / experimentation

The model is instantiated from config (not from_pretrained) to support full pretraining from scratch.
