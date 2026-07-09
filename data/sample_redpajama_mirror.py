import argparse
import os
import random

import torch
from datasets import load_dataset
from transformers import AutoTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default="michelangelo-engs/RedPajama-Data-1T-1024Sample",
    )
    parser.add_argument("--tokenizer", default="/run/z84450661/Qwen3-8B")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--num-seq", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default="redpajama_1024x2048.pt")
    args = parser.parse_args()

    # Force Hugging Face traffic through mirror.
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    random.seed(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=True,
    )

    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        eos_id = tokenizer.pad_token_id
    if eos_id is None:
        raise ValueError("Tokenizer has no eos_token_id or pad_token_id.")

    ds = load_dataset(
        args.dataset,
        split="train",
        trust_remote_code=True,
    )

    # Shuffle rows locally after mirror download.
    ds = ds.shuffle(seed=args.seed)

    sequences = []
    token_buffer = []

    for ex in ds:
        text = ex.get("text", "")
        if not text:
            continue

        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            continue

        token_buffer.extend(ids)
        token_buffer.append(eos_id)

        while len(token_buffer) >= args.seq_len:
            sequences.append(token_buffer[: args.seq_len])
            token_buffer = token_buffer[args.seq_len :]

            if len(sequences) >= args.num_seq:
                input_ids = torch.tensor(sequences, dtype=torch.long)

                torch.save(
                    {
                        "input_ids": input_ids,
                        "dataset": args.dataset,
                        "tokenizer": args.tokenizer,
                        "seq_len": args.seq_len,
                        "num_seq": args.num_seq,
                        "seed": args.seed,
                    },
                    args.out,
                )

                print(f"Saved {input_ids.shape} to {args.out}")
                return

    raise RuntimeError(
        f"Only collected {len(sequences)} sequences. "
        f"Need {args.num_seq}. Try a bigger sample dataset."
    )


if __name__ == "__main__":
    main()
