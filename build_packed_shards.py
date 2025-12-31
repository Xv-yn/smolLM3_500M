"""
Build packed Parquet shards for pretraining (fixed-length token blocks).

This script streams text from multiple Hugging Face datasets, tokenizes it,
and writes out fixed-length blocks of token IDs into Parquet shards:

- Each output row is exactly `seq_len` tokens (a "packed block").
- Blocks are formed by concatenating documents together with an EOS token
  between docs, then slicing into seq_len chunks.
- Shards are written as Parquet files containing:
    - input_ids: List[int] of length seq_len
    - source: string label of which dataset the tokens came from

How I ran it:

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
  --tokenize_batch_size 256 \
  --seed 1234 \
  --sources_json "$SOURCES"

Notes / assumptions:
- This is “token budget” driven: it stops after writing `total_tokens`.
- Sampling across sources is weighted, but also capped per source via token_cap.
- It saves a resumable state (rng state + carry buffer + counts) so you can
  restart without throwing away progress.
"""

import argparse
import hashlib
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"


@dataclass
class SourceSpec:
    name: str
    # HF dataset id, e.g. "HuggingFaceFW/fineweb-edu"
    dataset: str
    # Optional config name if dataset has configs
    config: Optional[str]
    # column containing text
    text_col: str
    # sampling weight (relative)
    weight: float
    # token cap for this source (None means no cap)
    token_cap: Optional[int]


def sha256_file(path: str, chunk_size: int = 1024 * 1024) -> str:
    """Compute a SHA-256 checksum for a file (used for manifest integrity)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def atomic_write_json(path: str, obj: dict):
    """Write JSON atomically by writing to a temp file then renaming."""
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def weighted_choice(rng, items: List[SourceSpec]) -> SourceSpec:
    """
    Pick one SourceSpec using its `.weight` as a relative sampling weight.
    Assumes all weights are non-negative and at least one item exists.
    """
    total = sum(s.weight for s in items)
    r = rng.random() * total
    acc = 0.0
    for s in items:
        acc += s.weight
        if r <= acc:
            return s
    return items[-1]


def main():
    # -------------------------
    # Argument parsing
    # -------------------------
    # This script builds a packed token dataset:
    # - streams text from HF datasets
    # - tokenizes
    # - concatenates with EOS separators
    # - slices into fixed seq_len blocks
    # - writes Parquet shards + a manifest + resumable state

    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--seq_len", type=int, default=4096)
    ap.add_argument(
        "--total_tokens",
        type=int,
        required=True,
        help="Stop after writing this many tokens total",
    )
    ap.add_argument("--blocks_per_shard", type=int, default=8192)

    # Shuffle for streaming datasets is approximate; buffer_size controls
    # how much data is held in memory to randomize order.
    ap.add_argument("--shuffle_buffer", type=int, default=100_000)

    # Tokenization batch size is currently not used in this script directly
    # (left here for easy future batching improvements).
    ap.add_argument("--tokenize_batch_size", type=int, default=256)

    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--resume", action="store_true")

    # Sources config is passed in as JSON for convenience.
    # Each source has a weight (sampling probability) and optional token cap.
    ap.add_argument(
        "--sources_json",
        type=str,
        required=True,
        help="JSON list of sources: [{name,dataset,config,text_col,weight,token_cap}, ...]",
    )

    args = ap.parse_args()

    # -------------------------
    # Output files / bookkeeping
    # -------------------------
    os.makedirs(args.out_dir, exist_ok=True)
    state_path = os.path.join(args.out_dir, "state.json")
    manifest_path = os.path.join(args.out_dir, "manifest.jsonl")

    # -------------------------
    # Tokenizer setup
    # -------------------------
    # We tokenize documents and insert EOS between docs.
    # If pad token is missing, we just reuse EOS as pad (common for LMs).
    tok = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, fix_mistral_regex=True
    )
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_id = tok.eos_token_id

    # -------------------------
    # Parse and validate sources
    # -------------------------
    src_dicts = json.loads(args.sources_json)
    sources = [SourceSpec(**d) for d in src_dicts]

    # token caps sanity
    for s in sources:
        if s.token_cap is not None and s.token_cap <= 0:
            raise ValueError(f"Bad token_cap for {s.name}: {s.token_cap}")

    # -------------------------
    # RNG + resume state
    # -------------------------
    # Store RNG state + carry buffer so resuming keeps packing deterministic.
    rng = random.Random(args.seed)

    if args.resume and os.path.exists(state_path):
        with open(state_path, "r") as f:
            state = json.load(f)
        shard_idx = state["shard_idx"]
        total_tokens_written = state["total_tokens_written"]
        per_source_tokens = state["per_source_tokens"]

        # carry holds leftover tokens that didn't fill a full seq_len block yet
        carry = state["carry"]

        # random.getstate() contains tuples; JSON stores lists, so must rebuild tuples
        def to_tuple(x):
            if isinstance(x, list):
                return tuple(to_tuple(i) for i in x)
            return x

        rng.setstate(to_tuple(state["rng_state"]))

        print(f"[resume] shard_idx={shard_idx} total_tokens={total_tokens_written}")
    else:
        shard_idx = 0
        total_tokens_written = 0
        per_source_tokens = {s.name: 0 for s in sources}
        carry = []
        print("[start] fresh build")

    # -------------------------
    # Create one streaming iterator per source
    # -------------------------
    # Each source is streamed from HF + shuffled with a buffer
    iters = {}
    for s in sources:
        ds = load_dataset(
            s.dataset,
            s.config,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(buffer_size=args.shuffle_buffer, seed=args.seed)
        iters[s.name] = iter(ds)

    # -------------------------
    # Shard buffers (in-memory)
    # -------------------------
    # Accumulate blocks until blocks_per_shard then write a Parquet file.
    shard_input_ids = []
    shard_sources = []

    def flush_shard():
        """
        Write the current shard buffers to disk as a Parquet file, append an entry
        to manifest.jsonl, then reset buffers for the next shard.
        """
        nonlocal shard_idx, shard_input_ids, shard_sources

        if not shard_input_ids:
            return

        filename = f"shard-{shard_idx:06d}.parquet"
        out_path = os.path.join(args.out_dir, filename)

        table = pa.table(
            {
                "input_ids": pa.array(shard_input_ids, type=pa.list_(pa.int32())),
                "source": pa.array(shard_sources, type=pa.string()),
            }
        )

        # Write to a temp path then atomically rename (safer for crashes).
        tmp_path = out_path + ".tmp"
        pq.write_table(
            table,
            tmp_path,
            compression="zstd",
            compression_level=3,
            use_dictionary=True,
            data_page_size=1024 * 1024,
        )
        os.replace(tmp_path, out_path)

        rows = len(shard_input_ids)
        tokens = rows * args.seq_len
        digest = sha256_file(out_path)

        # Append one JSON record per shard for future audit / verify later.
        rec = {
            "file": filename,
            "rows": rows,
            "tokens": tokens,
            "sha256": digest,
            "time": time.time(),
        }
        with open(manifest_path, "a") as mf:
            mf.write(json.dumps(rec) + "\n")

        print(f"[write] {filename} rows={rows} tokens={tokens} sha256={digest[:12]}...")

        shard_idx += 1
        shard_input_ids = []
        shard_sources = []

    def save_state():
        """
        Save resumable state:
        - shard index
        - total + per-source token counters
        - carry buffer (partial block tokens)
        - RNG state for deterministic sampling on resume
        """
        st = rng.getstate()

        # random.getstate() returns tuples, which aren’t JSON serializable.
        # Convert tuples -> lists recursively to dump to JSON.
        def to_list(x):
            if isinstance(x, tuple):
                return [to_list(i) for i in x]
            return x

        rng_state_list = to_list(st)

        atomic_write_json(
            state_path,
            {
                "shard_idx": shard_idx,
                "total_tokens_written": total_tokens_written,
                "per_source_tokens": per_source_tokens,
                "carry": carry,
                "rng_state": rng_state_list,
                "seq_len": args.seq_len,
                "blocks_per_shard": args.blocks_per_shard,
            },
        )

    def source_available(s: SourceSpec) -> bool:
        """Return True if this source still has remaining token budget."""
        cap = s.token_cap
        if cap is None:
            return True
        return per_source_tokens[s.name] < cap

    # -------------------------
    # Main packing loop
    # -------------------------
    # Repeatedly:
    #  1) choose an available source (weighted, respecting token caps)
    #  2) stream one record of text
    #  3) tokenize + append EOS
    #  4) append tokens to carry buffer
    #  5) emit full seq_len blocks into shard buffers
    # until total_tokens is reached (or all sources hit caps).
    while total_tokens_written < args.total_tokens:
        alive = [s for s in sources if source_available(s)]
        if not alive:
            print("[done] all sources hit token caps before reaching total_tokens")
            break

        s = weighted_choice(rng, alive)
        it = iters[s.name]

        # get a record, if iterator ends, recreate with a new shuffle seed
        try:
            rec = next(it)
        except StopIteration:
            ds = load_dataset(s.dataset, s.config, split="train", streaming=True)
            ds = ds.shuffle(
                buffer_size=args.shuffle_buffer, seed=rng.randint(0, 2**31 - 1)
            )
            iters[s.name] = iter(ds)
            it = iters[s.name]
            rec = next(it)

        # Pull text; skip invalid records.
        text = rec.get(s.text_col, None)
        if not text or not isinstance(text, str):
            continue

        # tokenize (no special tokens)
        ids = tok(text, add_special_tokens=False, truncation=False)["input_ids"]
        if not ids:
            continue

        # append EOS between docs
        if eos_id is not None:
            ids = ids + [eos_id]

        # extend carry buffer, and emit full blocks
        carry.extend(ids)

        while len(carry) >= args.seq_len:
            block = carry[: args.seq_len]
            carry = carry[args.seq_len :]

            shard_input_ids.append(block)
            shard_sources.append(s.name)

            per_source_tokens[s.name] += args.seq_len
            total_tokens_written += args.seq_len

            # shard full? then write
            if len(shard_input_ids) >= args.blocks_per_shard:
                flush_shard()
                save_state()

            # stop exactly at token budget
            if total_tokens_written >= args.total_tokens:
                break

    # flush leftovers
    flush_shard()
    save_state()
    print(f"[done] total_tokens_written={total_tokens_written} out_dir={args.out_dir}")


if __name__ == "__main__":
    main()
