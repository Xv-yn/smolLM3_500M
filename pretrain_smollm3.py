"""
Launch multi-GPU pretraining with Accelerate.
- Spawns 4 processes (1 per GPU, in my case I used 4 GPUs).
- Trains using pre-packed Parquet shards of fixed-length token sequences.


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

The number of training steps was chosen based on the dataset size
(~19.8B tokens) and the effective batch configuration, aiming for
roughly one full pass over the data. I don’t remember the exact
step calculation I used at the time, but it can be reverse-derived
from the dataset packing (something along the lines of 
tokens per sequence × batch size × number of processes) to recover 
the expected number of steps.

I ran with num_workers=0 to avoid any weird parquet/multiprocessing issues.

Notes on data pipeline & design choices
---------------------------------------

- Fixed-length sequences only
  This training setup assumes that all samples are already packed to a fixed
  length (`seq_len`). If a row doesn’t match that length, it’s skipped.
  Variable-length sequences would complicate batching and attention masks,
  so this keeps things simple and predictable.

- Padding labels set to -100
  Labels at padding positions are set to -100 so they’re ignored by the loss.
  This follows the usual Hugging Face causal LM convention.

- File-level shard partitioning
  In distributed runs, shard files are split across ranks at the file level
  (files[rank::world_size]). This avoids cross-rank duplication without needing
  a shared sampler and works cleanly with streaming IterableDatasets.

I set it up this way mainly because it felt more universal and easier for me
to reason about — at least based on my current understanding — especially
when scaling to multiple GPUs or different training setups.
"""

import argparse
import glob
import os
import random
import time

import pyarrow.parquet as pq
import torch
from accelerate import Accelerator
from torch.utils.data import DataLoader, IterableDataset
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, get_scheduler

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class ParquetPackedIterable(IterableDataset):
    """
    Iterable dataset that streams fixed-length token sequences from Parquet shards.

    Expected Parquet format:
      - Files named: shard-*.parquet inside `packed_dir`
      - Each file contains a column: "input_ids"
      - Each row is a list[int] of length exactly `seq_len`

    Distributed behavior:
      - If torch.distributed is initialized, each process (rank) receives a disjoint
        subset of shard files using strided partitioning: files[rank::world_size]
      - Optional shard-level shuffling is done per-rank using seed + rank for
        deterministic-yet-different ordering across workers.

    Output per example (dict[str, torch.Tensor]):
      - input_ids: (seq_len,) int64
      - attention_mask: (seq_len,) int64, 1 for non-pad, 0 for pad
      - labels: (seq_len,) int64, same as input_ids but pad positions are set to -100
        so they are ignored by standard causal LM loss functions.

    Notes:
      - This is an IterableDataset (streaming). It does not support random access.
      - It reads Parquet row groups sequentially to keep memory usage bounded.
      - Any row whose token length != seq_len is skipped (data integrity guard).
    """

    def __init__(
        self,
        packed_dir: str,
        seq_len: int,
        shuffle_shards: bool,
        seed: int,
        pad_id: int,
    ):
        super().__init__()
        self.packed_dir = packed_dir
        self.seq_len = seq_len
        self.shuffle_shards = shuffle_shards
        self.seed = seed
        self.pad_id = pad_id

        # Discover parquet shards. Requires at least one match.
        self.files = sorted(glob.glob(os.path.join(packed_dir, "shard-*.parquet")))
        if not self.files:
            raise FileNotFoundError(f"No shard-*.parquet in {packed_dir}")

    def __iter__(self):
        """
        Stream examples from rank-local shard files.

        For each row:
          1) Load `input_ids` list[int]
          2) Validate length == seq_len
          3) Build attention_mask (pad_id => 0)
          4) Build labels where pad positions are masked with -100
          5) Yield a training-ready dict
        """
        # Determine distributed rank/world if running under torch.distributed.
        try:
            import torch.distributed as dist

            if dist.is_available() and dist.is_initialized():
                rank = dist.get_rank()
                world = dist.get_world_size()
            else:
                rank, world = 0, 1
        except Exception:
            # Fallback: treat as single-process.
            rank, world = 0, 1

        files = self.files[:]

        # OPTIONAL deterministic per-rank shuffle of shard order.
        if self.shuffle_shards:
            rng = random.Random(self.seed + rank)
            rng.shuffle(files)

        # Shard partitioning across ranks (each rank gets a disjoint subset of files).
        files = files[rank::world]

        for path in files:
            pf = pq.ParquetFile(path)

            # Read row groups one at a time (better memory behavior than reading whole file).
            for rg in range(pf.num_row_groups):
                table = pf.read_row_group(rg, columns=["input_ids"])
                col = table["input_ids"]

                for i in range(table.num_rows):
                    ids = col[i].as_py()

                    # Data integrity guard: training expects fixed-length packed sequences.
                    if len(ids) != self.seq_len:
                        continue

                    input_ids = torch.tensor(ids, dtype=torch.long)

                    # Mask is 1 for tokens, 0 for padding.
                    attention_mask = (input_ids != self.pad_id).long()

                    # Labels match input_ids, but padded positions are ignored by loss.
                    labels = input_ids.clone()
                    labels[attention_mask == 0] = -100

                    yield {
                        "input_ids": input_ids,
                        "labels": labels,
                        "attention_mask": attention_mask,
                    }


def collate_packed(examples):
    """
    Collate function for already-packed, fixed-length sequences.

    This assumes that each example in `examples` already contains:
      - input_ids: (seq_len,)
      - attention_mask: (seq_len,)
      - labels: (seq_len,)

    Since all sequences are pre-packed to the same length, collation is
    just a straightforward stack along the batch dimension with no
    padding or dynamic shape handling required.

    Returns a batch dict compatible with Hugging Face causal LM training:
      - input_ids: (batch_size, seq_len)
      - attention_mask: (batch_size, seq_len)
      - labels: (batch_size, seq_len)
    """
    input_ids = torch.stack([ex["input_ids"] for ex in examples], dim=0)
    attention_mask = torch.stack([ex["attention_mask"] for ex in examples], dim=0)
    labels = torch.stack([ex["labels"] for ex in examples], dim=0)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def main():
    """
    Main training entrypoint.

    This function wires together argument parsing, distributed setup via
    Accelerate, model/tokenizer initialization, dataset loading, optimizer
    and scheduler configuration, and the main training loop.

    The training loop is step-based (num_train_steps) rather than epoch-based,
    since the dataset is streamed from pre-packed Parquet shards and does not
    have a fixed length in terms of batches. There’s probably a cleaner or more
    “correct” way to structure this, but this approach made the most sense to
    me and was the easiest for me to reason about at the time.

    Design notes:
    - Assumes input data is already packed to fixed-length sequences.
    - Uses gradient accumulation to control effective batch size.
    - Explicitly guards against invalid token IDs and NaN/Inf losses.
    - Checkpoints and logging are only performed on the main process.
    """

    # -------------------------
    # Argument parsing
    # -------------------------

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--packed_dir", type=str, required=True)
    ap.add_argument("--output_dir", type=str, default="./outputs")
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument("--micro_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum_steps", type=int, default=8)
    ap.add_argument("--learning_rate", type=float, default=2e-4)
    ap.add_argument("--weight_decay", type=float, default=0.1)
    ap.add_argument("--num_train_steps", type=int, default=2000)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--log_every", type=int, default=10)
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num_workers", type=int, default=2)
    ap.add_argument("--shuffle", action="store_true")

    ap.add_argument(
        "--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"]
    )
    ap.add_argument(
        "--attn_impl",
        type=str,
        default="sdpa",
        choices=["auto", "sdpa", "flash_attention_2"],
    )
    ap.add_argument(
        "--compile", action="store_true", help="torch.compile the model (PyTorch 2.x)"
    )

    args = ap.parse_args()

    # -------------------------
    # Accelerator / reproducibility setup
    # -------------------------
    # Accelerate handles device placement, DDP, gradient accumulation,
    # and mixed precision in a backend-agnostic way.

    accelerator = Accelerator(
        gradient_accumulation_steps=args.grad_accum_steps,
        mixed_precision=args.mixed_precision,
    )
    torch.manual_seed(args.seed)

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # -------------------------
    # Tokenizer setup
    # -------------------------
    # Ensure a pad token exists so attention masks and label masking work
    # consistently for packed sequences.

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id

    if accelerator.is_main_process:
        print("\n=== Token IDs ===")
        print("tokenizer.bos_token_id:", tokenizer.bos_token_id)
        print("tokenizer.eos_token_id:", eos_id)
        print("tokenizer.pad_token_id:", pad_id)
        print("=================\n")

    accelerator.wait_for_everyone()

    config = AutoConfig.from_pretrained(args.model_dir, trust_remote_code=True)

    # -------------------------
    # Model construction
    # -------------------------
    # Model is created from config rather than from_pretrained so this script
    # can be used for full pretraining, not just fine-tuning.

    model_kwargs = dict(trust_remote_code=True)
    if args.attn_impl != "auto":
        model_kwargs["attn_implementation"] = args.attn_impl

    model = AutoModelForCausalLM.from_config(config, **model_kwargs)

    if model.config.vocab_size != len(tokenizer):
        model.resize_token_embeddings(len(tokenizer))

    # Disable KV cache during training to reduce memory usage.
    model.config.use_cache = False

    if accelerator.is_main_process:
        print("model vocab_size:", model.config.vocab_size)
        print("tokenizer vocab_size:", len(tokenizer))
        print("attn_impl:", args.attn_impl)
        print("mixed_precision:", args.mixed_precision)

    # torch.compile can speed things up on PyTorch 2.x (not too sure, but it did for me).
    if args.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    # -------------------------
    # Dataset & DataLoader
    # -------------------------
    # Uses a streaming IterableDataset backed by Parquet shards.
    # Shuffling (if enabled) happens at the shard level.

    packed = ParquetPackedIterable(
        args.packed_dir, args.seq_len, args.shuffle, args.seed, pad_id=pad_id
    )
    train_loader = DataLoader(
        packed,
        batch_size=args.micro_batch_size,
        shuffle=False,  # shuffle=False is required for IterableDataset
        collate_fn=collate_packed,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=(args.num_workers > 0),
    )

    adamw_kwargs = dict(
        lr=args.learning_rate,
        betas=(0.9, 0.95),
        eps=1e-6,
        weight_decay=args.weight_decay,
    )

    # -------------------------
    # Optimizer & LR scheduler
    # -------------------------
    # Try fused AdamW when available for better performance.

    try:
        optimizer = torch.optim.AdamW(model.parameters(), fused=True, **adamw_kwargs)
        if accelerator.is_main_process:
            print("Using fused AdamW")
    except TypeError:
        optimizer = torch.optim.AdamW(model.parameters(), **adamw_kwargs)
        if accelerator.is_main_process:
            print("Using standard AdamW (fused not available)")

    lr_scheduler = get_scheduler(
        name="cosine",
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_train_steps,
    )

    model, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_loader, lr_scheduler
    )
    model.train()
    os.makedirs(args.output_dir, exist_ok=True)

    base_model = accelerator.unwrap_model(model)
    vocab_size = base_model.config.vocab_size

    data_iter = iter(train_loader)

    micro_step = 0
    opt_step = 0

    t0 = time.time()
    last_log_t = t0
    last_log_micro = 0

    tokens_per_micro = args.micro_batch_size * args.seq_len

    # Was used at some point but I don't remember for what
    tokens_per_opt = tokens_per_micro * args.grad_accum_steps

    # -------------------------
    # Main training loop
    # -------------------------
    # Runs until num_train_steps optimizer updates are completed.
    # The data loader is re-cycled as needed since this is not epoch-based.

    while opt_step < args.num_train_steps:
        # Fetch next batch; re-create the iterator if end is hit.
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        micro_step += 1

        # Sanity check, ensure all token IDs are within vocab range.
        ids = batch["input_ids"]
        if ids.min().item() < 0 or ids.max().item() >= vocab_size:
            if accelerator.is_main_process:
                print("vocab_size:", vocab_size)
                print("min token:", ids.min().item())
                print("max token:", ids.max().item())
            raise RuntimeError("Found out-of-range token id")

        if (batch["labels"] != -100).sum().item() == 0:
            continue

        with accelerator.accumulate(model):
            outputs = model(**batch)
            loss = outputs.loss

            # NaN/Inf guard
            if torch.isnan(loss) or torch.isinf(loss):
                if accelerator.is_main_process:
                    am = batch["attention_mask"]
                    valid = (batch["labels"] != -100).sum().item()
                    print(
                        f"\nNaN/Inf loss at micro_step {micro_step} opt_step {opt_step}"
                    )
                    print("valid label tokens:", valid)
                    print("attention_mask sum:", am.sum().item(), "of", am.numel())
                    print("input_ids min/max:", ids.min().item(), ids.max().item())
                raise RuntimeError("NaN/Inf loss encountered")

            accelerator.backward(loss)

            # Only update weights when gradients are synchronized
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                opt_step += 1

                # Periodic logging
                if accelerator.is_main_process and (opt_step % args.log_every == 0):
                    now = time.time()
                    dt = now - last_log_t
                    dmicro = micro_step - last_log_micro
                    tok = dmicro * tokens_per_micro
                    tps = tok / max(dt, 1e-9)

                    lr = lr_scheduler.get_last_lr()[0]
                    remaining = args.num_train_steps - opt_step
                    eta_sec = remaining * (dt / max(args.log_every, 1e-9))
                    eta_hr = eta_sec / 3600.0

                    print(
                        f"opt {opt_step:7d}/{args.num_train_steps} | "
                        f"micro {micro_step:9d} | "
                        f"loss {loss.item():.4f} | lr {lr:.3e} | "
                        f"{tps:,.0f} tok/s | "
                        f"ETA {eta_hr:.2f} h"
                    )

                    last_log_t = now
                    last_log_micro = micro_step

                # Periodic checkpointing
                if accelerator.is_main_process and (opt_step % args.save_every == 0):
                    ckpt = os.path.join(args.output_dir, f"ckpt-opt-{opt_step}")
                    os.makedirs(ckpt, exist_ok=True)
                    unwrapped = accelerator.unwrap_model(model)
                    unwrapped.save_pretrained(ckpt, save_function=accelerator.save)
                    tokenizer.save_pretrained(ckpt)
                    print(f"Saved checkpoint: {ckpt}")

    if accelerator.is_main_process:
        final_dir = os.path.join(args.output_dir, "final")
        os.makedirs(final_dir, exist_ok=True)
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(final_dir, save_function=accelerator.save)
        tokenizer.save_pretrained(final_dir)
        print(f"Saved final model: {final_dir}")


if __name__ == "__main__":
    main()
