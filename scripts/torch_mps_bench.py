#!/usr/bin/env python3

import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def choose_device(force_cpu: bool) -> str:
    if not force_cpu and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_bench(
    model_id: str, prompt: str, gen_tokens: int, device: str, repeats: int
) -> None:
    dtype = torch.float16 if device == "mps" else torch.float32

    t_load = time.time()
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device)
    load_s = time.time() - t_load

    enc = tok(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    attention_mask = enc.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)
    prompt_tokens = int(input_ids.shape[-1])

    with torch.no_grad():
        o = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=True)
        n = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        _ = model(input_ids=n, use_cache=True, past_key_values=o.past_key_values)

    pp_tps_all = []
    tg_tps_all = []

    for _ in range(repeats):
        t0 = time.time()
        with torch.no_grad():
            o = model(
                input_ids=input_ids, attention_mask=attention_mask, use_cache=True
            )
        pp_s = time.time() - t0
        pp_tps_all.append(prompt_tokens / pp_s)

        next_id = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        past = o.past_key_values
        t1 = time.time()
        for _ in range(gen_tokens):
            with torch.no_grad():
                o = model(input_ids=next_id, use_cache=True, past_key_values=past)
            next_id = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            past = o.past_key_values
        tg_s = time.time() - t1
        tg_tps_all.append(gen_tokens / tg_s)

    pp_avg = sum(pp_tps_all) / len(pp_tps_all)
    tg_avg = sum(tg_tps_all) / len(tg_tps_all)

    print(f"model={model_id}")
    print(f"device={device} dtype={dtype}")
    print(f"load_s={load_s:.2f}")
    print(
        f"prompt_tokens={prompt_tokens} pp_tok_s_avg={pp_avg:.2f} pp_tok_s_runs={[round(x, 2) for x in pp_tps_all]}"
    )
    print(
        f"gen_tokens={gen_tokens} tg_tok_s_avg={tg_avg:.2f} tg_tok_s_runs={[round(x, 2) for x in tg_tps_all]}"
    )
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MPS prompt-processing and token-generation speeds for Qwen models."
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"],
        help="One or more model IDs to benchmark.",
    )
    parser.add_argument(
        "--gen-tokens",
        type=int,
        default=128,
        help="Number of generated tokens for decode benchmark.",
    )
    parser.add_argument(
        "--prompt-repeat",
        type=int,
        default=64,
        help="Repeat factor for long prompt template.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Number of timed benchmark runs per model.",
    )
    parser.add_argument(
        "--force-cpu", action="store_true", help="Force CPU benchmark instead of MPS."
    )
    args = parser.parse_args()

    device = choose_device(args.force_cpu)
    print(f"torch={torch.__version__}")
    print(f"mps_built={torch.backends.mps.is_built()}")
    print(f"mps_available={torch.backends.mps.is_available()}")
    print()

    prompt_base = (
        "Local inference keeps data on-device and reduces network dependence. "
    )
    prompt = prompt_base * args.prompt_repeat

    for model_id in args.models:
        run_bench(
            model_id=model_id,
            prompt=prompt,
            gen_tokens=args.gen_tokens,
            device=device,
            repeats=args.repeats,
        )


if __name__ == "__main__":
    main()
