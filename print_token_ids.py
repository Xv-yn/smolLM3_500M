import argparse

import torch
from accelerate import Accelerator
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_dir", type=str, required=True)
    args = ap.parse_args()

    accelerator = Accelerator()
    torch.manual_seed(42)

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)

    # Only print once (rank 0)
    if accelerator.is_main_process:
        print("\n=== Token IDs ===")
        print("tokenizer.bos_token_id:", tokenizer.bos_token_id)
        print("tokenizer.eos_token_id:", tokenizer.eos_token_id)
        print("tokenizer.pad_token_id:", tokenizer.pad_token_id)
        print("=================\n")

    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
