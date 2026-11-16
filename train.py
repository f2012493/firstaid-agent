"""LoRA SFT on teacher episodes: loss only on the agent's tool-call turns.

usage: python train.py --model CohereLabs/tiny-aya-fire --out out/lora --minutes 30
"""
import argparse
import json
import random
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def _ids(tok, msgs, gen=False):
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=gen)
    return tok(text, add_special_tokens=False)["input_ids"]


def encode(tok, msgs, max_len):
    """Token ids with labels masked to -100 everywhere except assistant turns."""
    ids = _ids(tok, msgs)
    labels = [-100] * len(ids)
    for k, m in enumerate(msgs):
        if m["role"] != "assistant":
            continue
        pre, upto = _ids(tok, msgs[:k], gen=True), _ids(tok, msgs[:k + 1])
        for j in range(len(pre), min(len(upto), len(ids))):
            labels[j] = ids[j]
    return ids[:max_len], labels[:max_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="artifacts/sft_train.jsonl")
    ap.add_argument("--out", default="out/lora")
    ap.add_argument("--minutes", type=float, default=30)
    ap.add_argument("--max_len", type=int, default=1600)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--accum", type=int, default=4)
    a = ap.parse_args()

    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float16).to(dev)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
                                             target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]))
    model.print_trainable_parameters()

    rows = [json.loads(l) for l in open(a.data)]
    random.Random(0).shuffle(rows)
    data = [encode(tok, r["messages"], a.max_len) for r in rows]
    print("examples", len(data), "max tokens", max(len(d[0]) for d in data), flush=True)

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr)
    model.train()
    t0, step, log = time.time(), 0, []
    for i, (ids, labels) in enumerate(data):
        x = torch.tensor([ids], device=dev)
        y = torch.tensor([labels], device=dev)
        loss = model(input_ids=x, labels=y).loss / a.accum
        loss.backward()
        if (i + 1) % a.accum == 0:
            opt.step(); opt.zero_grad(); step += 1
            log.append({"step": step, "examples": i + 1, "loss": loss.item() * a.accum,
                        "elapsed_s": time.time() - t0})
            print(json.dumps(log[-1]), flush=True)
        if time.time() - t0 > a.minutes * 60:
            print("time cap reached", flush=True)
            break
    model.save_pretrained(a.out)
    tok.save_pretrained(a.out)
    json.dump({"args": vars(a), "log": log, "examples_seen": i + 1}, open(f"{a.out}/train_log.json", "w"), indent=2)
    print("saved", a.out)


if __name__ == "__main__":
    main()
