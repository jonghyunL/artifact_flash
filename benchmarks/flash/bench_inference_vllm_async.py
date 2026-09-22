"""
bench_inference_vllm_async.py — In-process Poisson load generator for vLLM.

Drives an in-process AsyncLLMEngine with Poisson-arrival prompts. Same
front-door behavior as bench_inference_vllm_poisson.py (which targets a
remote `vllm serve` over HTTP), but everything runs in one Python process
— so LD_PRELOAD'd shims (phos / gcr) and our libckpt_vllm see the worker's
CUDA calls directly without HTTP/uvicorn/parent-vs-worker mess.

Why in-process: for the checkpoint-then-restore eval, restore semantics are
crisp — dump at request K, restart in a fresh process with the ckpt image,
replay the deck from K+1 and verify byte-identical outputs. There's no
analogue for HTTP serving.

Usage:
    LD_PRELOAD=$PWD/../phos_emul/libphos_intercept.so \\
    PHOS_CKPT_DELAY=120 PHOS_CKPT_INTERVAL_MS=10000 PHOS_CKPT_ROUNDS=10 \\
    python bench_inference_vllm_async.py \\
        --model ~/huggingface/Llama-3.2-1B-Instruct/ \\
        --dataset sharegpt --rate 5 --duration 400 \\
        --min-tokens 256 --max-tokens 256 --ignore-eos \\
        --num-prompts 1024 --max-concurrent 32 \\
        --out-jsonl results/llama_1b_FS_phos.jsonl

Args mirror bench_inference_vllm_poisson.py with HTTP-specific flags
(--host, --port) replaced by engine-init flags (--max-model-len,
--gpu-memory-utilization, --enforce-eager).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any


# -------------------------------------------------------------------------
# Prompt
# -------------------------------------------------------------------------
@dataclass
class Prompt:
    text: str                          # chat-templated text the model sees
    image: Any | None = None           # PIL.Image, or None for text-only
    max_tokens: int = 256
    min_tokens: int = 0
    ignore_eos: bool = False


def _load_docvqa(num: int, model_name: str,
                 max_tokens: int, min_tokens: int, ignore_eos: bool) -> list[Prompt]:
    from datasets import load_dataset
    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation",
                      streaming=False)
    is_qwen = "qwen" in model_name.lower()
    prompts: list[Prompt] = []
    for i, sample in enumerate(ds):
        if i >= num:
            break
        q = sample["question"]
        if is_qwen:
            text = (f"<|im_start|>user\n"
                    f"<|vision_start|><|image_pad|><|vision_end|>{q}"
                    f"<|im_end|>\n<|im_start|>assistant\n")
        else:
            text = f"USER: <image>\n{q}\nASSISTANT:"
        prompts.append(Prompt(text=text, image=sample["image"],
                              max_tokens=max_tokens, min_tokens=min_tokens,
                              ignore_eos=ignore_eos))
    return prompts

def _load_sharegpt(num_samples, sharegpt_path=None):
    """Load conversation prompts from ShareGPT dataset."""
    import json

    # Try local file first, then download
    paths = [
        sharegpt_path,
        "ShareGPT_V3_unfiltered_cleaned_split.json",
        os.path.expanduser("~/ShareGPT_V3_unfiltered_cleaned_split.json"),
    ]

    data = None
    for p in paths:
        if p and os.path.exists(p):
            print(f"Loading ShareGPT from {p}...")
            with open(p) as f:
                data = json.load(f)
            break

    if data is None:
        print("ShareGPT JSON not found locally, downloading...")
        try:
            import urllib.request
            url = "https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
            local = "ShareGPT_V3_unfiltered_cleaned_split.json"
            urllib.request.urlretrieve(url, local)
            with open(local) as f:
                data = json.load(f)
        except Exception as e:
            print(f"Failed to download ShareGPT: {e}")
            return None

    # Extract human turns as prompts
    samples = []
    for conv in data:
        if len(samples) >= num_samples:
            break
        turns = conv.get("conversations", [])
        # Find first human message
        for turn in turns:
            if turn.get("from") == "human" and len(turn.get("value", "")) > 20 and len(turn.get("value", "")) < 3000:
                samples.append(turn["value"])
                break

    print(f"Loaded {len(samples)} ShareGPT prompts")
    return samples

def _load_longbench(num: int, model_name: str,
                    max_tokens: int, min_tokens: int, ignore_eos: bool,
                    subset: str, max_prompt_tokens: int) -> list[Prompt]:
    """Long-context prompts from THUDM/LongBench. Truncates the document context
    (not the trailing question) to fit max_prompt_tokens, then applies the
    model's chat template.

    Bypasses datasets-4.x's removal of dataset-script support by fetching the
    raw per-subset JSONL from the HF Hub directly.
    """
    import json, os, zipfile
    from transformers import AutoTokenizer

    print(f"Loading LongBench subset '{subset}'...", flush=True)

    # LongBench ships as a single data.zip at the repo root. After extraction
    # it lays out as data/<subset>.jsonl. Try local paths first (in case the
    # user already extracted), then download+extract the zip on demand.
    search_dirs = [
        "data",
        os.path.expanduser("~/longbench_data/data"),
        os.path.join(os.path.dirname(__file__), "data"),
    ]
    local_path = None
    for d in search_dirs:
        cand = os.path.join(d, f"{subset}.jsonl")
        if os.path.exists(cand):
            local_path = cand
            print(f"  -> using local {cand}", flush=True)
            break

    if local_path is None:
        from huggingface_hub import hf_hub_download
        print("  -> not found locally, downloading data.zip from "
              "zai-org/LongBench...", flush=True)
        zip_path = hf_hub_download(repo_id="zai-org/LongBench",
                                   filename="data.zip", repo_type="dataset")
        extract_dir = os.path.join(os.path.dirname(zip_path), "extracted")
        os.makedirs(extract_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(extract_dir)
        local_path = os.path.join(extract_dir, "data", f"{subset}.jsonl")
        if not os.path.exists(local_path):
            raise RuntimeError(
                f"After extracting data.zip, '{subset}.jsonl' not found at "
                f"{local_path}. Valid subsets: narrativeqa, qasper, "
                f"multifieldqa_en, hotpotqa, 2wikimqa, musique, gov_report, "
                f"qmsum, multi_news, trec, triviaqa, samsum, passage_count, "
                f"passage_retrieval_en, lcc, repobench-p, etc.")
        print(f"  -> extracted to {local_path}", flush=True)

    samples = []
    with open(local_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    tok = AutoTokenizer.from_pretrained(model_name)

    prompts: list[Prompt] = []
    for sample in samples:
        if len(prompts) >= num:
            break
        context = sample.get("context", "") or ""
        question = sample.get("input", "") or ""
        if not context:
            continue

        q_tail = f"\n\nQuestion: {question}\nAnswer:"
        q_ids = tok.encode(q_tail, add_special_tokens=False)
        room = max_prompt_tokens - len(q_ids) - 16  # margin for chat template
        if room <= 0:
            continue
        ctx_ids = tok.encode(context, add_special_tokens=False)
        if len(ctx_ids) > room:
            ctx_ids = ctx_ids[:room]
            context = tok.decode(ctx_ids, skip_special_tokens=True)
        raw = f"{context}{q_tail}"

        text = tok.apply_chat_template(
            [{"role": "user", "content": raw}],
            tokenize=False, add_generation_prompt=True)
        prompts.append(Prompt(text=text, max_tokens=max_tokens,
                              min_tokens=min_tokens, ignore_eos=ignore_eos))

    print(f"Loaded {len(prompts)} LongBench prompts "
          f"(subset={subset}, max_prompt_tokens={max_prompt_tokens})", flush=True)
    return prompts


def load_prompts(dataset: str, num: int, model_name: str,
                 max_tokens: int, min_tokens: int, ignore_eos: bool,
                 longbench_subset: str = "narrativeqa",
                 max_prompt_tokens: int = 3500) -> list[Prompt]:
    if dataset == "docvqa":
        return _load_docvqa(num, model_name, max_tokens, min_tokens, ignore_eos)
    if dataset == "sharegpt":
        raw = _load_sharegpt(num, None)
        if not raw:
            raise RuntimeError("ShareGPT load returned empty")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_name)
        out: list[Prompt] = []
        for s in raw:
            text = tok.apply_chat_template(
                [{"role": "user", "content": s}],
                tokenize=False, add_generation_prompt=True)
            out.append(Prompt(text=text, max_tokens=max_tokens,
                              min_tokens=min_tokens, ignore_eos=ignore_eos))
        return out
    if dataset == "longbench":
        return _load_longbench(num, model_name, max_tokens, min_tokens,
                               ignore_eos, longbench_subset, max_prompt_tokens)
    raise ValueError(f"unknown dataset: {dataset}")


# -------------------------------------------------------------------------
# Per-request bookkeeping
# -------------------------------------------------------------------------
@dataclass
class RequestResult:
    request_id: int
    submit_time: float
    first_token_time: float | None
    end_time: float | None
    output_tokens: int = 0
    error: str | None = None
    inter_token_ms: list[float] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float | None:
        if self.first_token_time is None:
            return None
        return (self.first_token_time - self.submit_time) * 1000.0

    @property
    def total_ms(self) -> float | None:
        if self.end_time is None:
            return None
        return (self.end_time - self.submit_time) * 1000.0

    @property
    def median_tbt_ms(self) -> float | None:
        if not self.inter_token_ms:
            return None
        return statistics.median(self.inter_token_ms)


# -------------------------------------------------------------------------
# Engine driver
# -------------------------------------------------------------------------
async def submit_one(engine, prompt: Prompt, request_id: int,
                     sampling_params) -> RequestResult:
    res = RequestResult(
        request_id=request_id,
        submit_time=time.time(),
        first_token_time=None,
        end_time=None,
    )
    last_token_time: float | None = None
    last_seen_tokens = 0

    if prompt.image is not None:
        inputs = {"prompt": prompt.text,
                  "multi_modal_data": {"image": prompt.image}}
    else:
        inputs = prompt.text

    try:
        gen = engine.generate(inputs, sampling_params,
                              request_id=str(request_id))
        async for output in gen:
            n = len(output.outputs[0].token_ids)
            now = time.time()
            if n > last_seen_tokens:
                if res.first_token_time is None:
                    res.first_token_time = now
                if last_token_time is not None:
                    # gap = (now - last_token_time) / new_tokens_in_chunk
                    new_toks = n - last_seen_tokens
                    gap_ms = (now - last_token_time) * 1000.0 / max(1, new_toks)
                    for _ in range(new_toks):
                        res.inter_token_ms.append(gap_ms)
                last_token_time = now
                last_seen_tokens = n
            res.output_tokens = n
        res.end_time = time.time()
    except Exception as e:
        res.error = f"{type(e).__name__}: {e}"
        res.end_time = time.time()
    return res


async def run_submitter(*, engine, prompts: list[Prompt],
                        rate: float, duration: float,
                        max_concurrent: int, sampling_params_for) -> list[RequestResult]:
    sem = asyncio.Semaphore(max_concurrent)
    tasks: list[asyncio.Task] = []
    results: list[RequestResult] = []
    request_id = 0

    async def fire(prompt: Prompt, rid: int) -> None:
        async with sem:
            sp = sampling_params_for(prompt)
            r = await submit_one(engine, prompt, rid, sp)
            results.append(r)
            ttft = f"{r.ttft_ms:7.1f}" if r.ttft_ms is not None else "    n/a"
            total = f"{r.total_ms:7.1f}" if r.total_ms is not None else "    n/a"
            err = f"  ERROR: {r.error}" if r.error else ""
            print(f"  [req {rid:4d}] ttft={ttft} ms  total={total} ms"
                  f"  out_toks={r.output_tokens:4d}{err}",
                  flush=True)

    start = time.time()
    deadline = start + duration

    while True:
        now = time.time()
        if now >= deadline:
            break
        prompt = prompts[request_id % len(prompts)]
        tasks.append(asyncio.create_task(fire(prompt, request_id)))
        request_id += 1
        gap = random.expovariate(rate)
        sleep_for = min(gap, max(0.0, deadline - time.time()))
        await asyncio.sleep(sleep_for)

    print(f"\n[submitter] submission window ended after {duration:.1f}s "
          f"({len(tasks)} requests fired) — draining in-flight...",
          flush=True)
    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)
    return results


# -------------------------------------------------------------------------
# Reporting (same shape as bench_inference_vllm_poisson.py)
# -------------------------------------------------------------------------
def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def report(results: list[RequestResult], duration_actual: float) -> None:
    n_ok    = sum(1 for r in results if r.error is None)
    n_err   = sum(1 for r in results if r.error is not None)
    ttfts   = [r.ttft_ms for r in results if r.ttft_ms is not None]
    totals  = [r.total_ms for r in results if r.total_ms is not None]
    tbts    = [t for r in results for t in r.inter_token_ms]
    out_tok = sum(r.output_tokens for r in results)

    print("\n=== async submitter results ===")
    print(f"requests submitted   : {len(results)}")
    print(f"  OK                 : {n_ok}")
    print(f"  failed             : {n_err}")
    print(f"actual run duration  : {duration_actual:.1f} s")
    print(f"effective request/s  : {n_ok / duration_actual:.2f}")
    print(f"output tokens total  : {out_tok}")
    print(f"output tok/s         : {out_tok / duration_actual:.1f}")

    def fmt_dist(label: str, xs: list[float], unit: str = "ms") -> None:
        if not xs:
            print(f"  {label:24s} no data")
            return
        print(f"  {label:24s} "
              f"p50={percentile(xs, 0.50):.1f} {unit}  "
              f"p95={percentile(xs, 0.95):.1f} {unit}  "
              f"p99={percentile(xs, 0.99):.1f} {unit}  "
              f"mean={statistics.mean(xs):.1f} {unit}")

    print("\nlatency distributions:")
    fmt_dist("TTFT", ttfts)
    fmt_dist("total request time", totals)
    fmt_dist("inter-token (TBT)", tbts)

    if n_err:
        print(f"\nfirst {min(n_err, 5)} errors:")
        shown = 0
        for r in results:
            if r.error and shown < 5:
                print(f"  req {r.request_id}: {r.error}")
                shown += 1


def write_jsonl(results: list[RequestResult], path: str) -> None:
    with open(path, "w") as f:
        for r in results:
            f.write(json.dumps({
                "request_id":       r.request_id,
                "submit_time":      r.submit_time,
                "first_token_time": r.first_token_time,
                "end_time":         r.end_time,
                "output_tokens":    r.output_tokens,
                "ttft_ms":          r.ttft_ms,
                "total_ms":         r.total_ms,
                "median_tbt_ms":    r.median_tbt_ms,
                "inter_token_ms":   r.inter_token_ms,
                "error":            r.error,
            }) + "\n")
    print(f"\nper-request log written to {path}")


# -------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------
def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model",       required=True)
    p.add_argument("--dataset",     default="docvqa",
                   choices=["docvqa", "sharegpt", "longbench"])
    p.add_argument("--longbench-subset", default="narrativeqa",
                   help="LongBench subset (narrativeqa, multifieldqa_en, "
                        "gov_report, qasper, etc.); only used when --dataset longbench")
    p.add_argument("--max-prompt-tokens", default=3500, type=int,
                   help="Truncate context to fit this many input tokens "
                        "(leave room for --max-tokens generation)")
    p.add_argument("--num-prompts", default=1024, type=int)
    p.add_argument("--max-tokens",  default=256, type=int)
    p.add_argument("--min-tokens",  default=0, type=int)
    p.add_argument("--ignore-eos",  action="store_true")
    p.add_argument("--rate",        default=5.0, type=float,
                   help="Poisson arrival rate (req/s)")
    p.add_argument("--duration",    default=200.0, type=float,
                   help="Submission window in seconds")
    p.add_argument("--max-concurrent", default=64, type=int,
                   help="Cap on in-flight requests at any moment")
    p.add_argument("--seed",        default=0, type=int)
    p.add_argument("--out-jsonl",   default=None)
    # Engine init flags
    p.add_argument("--max-model-len",          default=32768, type=int)
    p.add_argument("--gpu-memory-utilization", default=0.9, type=float)
    p.add_argument("--enforce-eager",          action="store_true", default=True)
    p.add_argument("--no-enforce-eager",       dest="enforce_eager", action="store_false")
    p.add_argument("--block-size",             default=None, type=int,
                   choices=[1, 8, 16, 32, 64, 128],
                   help="vLLM paged-KV block size (tokens per block). "
                        "Default = vLLM auto. Larger blocks reduce 2MB-tracking "
                        "amplification on small-KV models (Llama-3.2-1B, Qwen-2.5-7B).")
    args = p.parse_args()

    random.seed(args.seed)

    print("=== async in-process Poisson submitter ===")
    print(f"model               : {args.model}")
    print(f"dataset             : {args.dataset} ({args.num_prompts} prompts)")
    print(f"rate                : {args.rate} req/s (Poisson)")
    print(f"duration            : {args.duration} s")
    print(f"min/max tokens      : {args.min_tokens} / {args.max_tokens}")
    print(f"ignore_eos          : {args.ignore_eos}")
    print(f"max concurrent      : {args.max_concurrent}")
    print(f"seed                : {args.seed}")
    print()

    print(f"Loading {args.num_prompts} prompts from '{args.dataset}'...",
          flush=True)
    prompts = load_prompts(args.dataset, args.num_prompts, args.model,
                           args.max_tokens, args.min_tokens, args.ignore_eos,
                           longbench_subset=args.longbench_subset,
                           max_prompt_tokens=args.max_prompt_tokens)
    print(f"  -> {len(prompts)} prompts loaded\n", flush=True)

    print("Starting AsyncLLMEngine...", flush=True)
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.sampling_params import SamplingParams

    engine_kwargs = dict(
        model=args.model,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enforce_eager=args.enforce_eager,
        tensor_parallel_size=1,
        disable_log_requests=True,
        enable_prefix_caching=False,
        max_num_seqs=args.max_concurrent,
    )
    if args.block_size is not None:
        engine_kwargs["block_size"] = args.block_size
        print(f"block_size           : {args.block_size} tokens")
    engine_args = AsyncEngineArgs(**engine_kwargs)
    def sampling_params_for(prompt: Prompt) -> SamplingParams:
        kw = dict(temperature=0.0, max_tokens=prompt.max_tokens)
        if prompt.ignore_eos:
            kw["ignore_eos"] = True
        if prompt.min_tokens > 0:
            kw["min_tokens"] = prompt.min_tokens
        return SamplingParams(**kw)

    async def amain():
        # Engine MUST be built inside the running event loop. Under V1,
        # AsyncLLMEngine binds its output handler to
        # asyncio.get_running_loop() at from_engine_args time. If created
        # on the main thread before asyncio.run(...), generate() never
        # delivers tokens — the run silently completes 0 requests.
        engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Engine ready.\n", flush=True)
        return await run_submitter(
            engine=engine, prompts=prompts, rate=args.rate,
            duration=args.duration, max_concurrent=args.max_concurrent,
            sampling_params_for=sampling_params_for,
        )

    t_start = time.time()
    results = asyncio.run(amain())
    duration_actual = time.time() - t_start

    report(results, duration_actual)

    if args.out_jsonl:
        os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
        write_jsonl(results, args.out_jsonl)


if __name__ == "__main__":
    main()