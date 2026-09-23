"""Offline vLLM harness: correctness (prompt logprobs vs HF ref) + decode timing.

usage: python run_llm.py --model DIR [--ref REF.pt] [--ids JSON] [--gen N]
       [--bench-bs 1,4] [--bench-tokens 32] [--extra JSON-of-LLM-kwargs]
"""
import argparse
import json
import os
import time

import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--ref")
    p.add_argument("--ids", default="[785, 6722, 315, 9621, 374]")
    p.add_argument("--gen", type=int, default=8)
    p.add_argument("--bench-bs", default="")
    p.add_argument("--bench-tokens", type=int, default=32)
    p.add_argument("--max-model-len", type=int, default=256)
    p.add_argument("--max-num-seqs", type=int, default=4)
    p.add_argument("--eager", action="store_true")
    p.add_argument("--extra", default="{}")
    p.add_argument("--out", default="")
    p.add_argument("--profile-bs", type=int, default=0)
    p.add_argument("--spin-tokens", type=int, default=0)
    p.add_argument("--spin-warm", type=int, default=0)
    p.add_argument("--spy", action="store_true")
    p.add_argument("--prompt-len", type=int, default=0)
    p.add_argument("--no-lp", action="store_true")
    args = p.parse_args()

    from vllm import LLM, SamplingParams  # noqa: E402
    from vllm.inputs import TokensPrompt  # noqa: E402

    kw = dict(
        model=args.model,
        tensor_parallel_size=8,
        max_model_len=args.max_model_len,
        max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=max(args.max_model_len, 256),
        gpu_memory_utilization=0.5,
        enforce_eager=args.eager,
        limit_mm_per_prompt={"image": 0, "video": 0},
    )
    kw.update(json.loads(args.extra))
    t0 = time.time()
    llm = LLM(**kw)
    print(f"[harness] load {time.time() - t0:.1f}s", flush=True)
    result = {"load_s": time.time() - t0}

    ids = json.loads(args.ids)
    if args.prompt_len:
        g = torch.Generator().manual_seed(0)
        ids = torch.randint(1000, 100000, (args.prompt_len,), generator=g).tolist()
    if args.no_lp:
        sp = SamplingParams(max_tokens=args.gen, temperature=0.0)
    else:
        sp = SamplingParams(max_tokens=args.gen, temperature=0.0, logprobs=5, prompt_logprobs=5)
    t0 = time.time()
    out = llm.generate([TokensPrompt(prompt_token_ids=ids)], sp)[0]
    result["first_gen_s"] = time.time() - t0
    gen_ids = list(out.outputs[0].token_ids)
    print(f"[harness] generated {gen_ids} text={out.outputs[0].text!r} in {result['first_gen_s']:.1f}s")
    result["gen_ids"] = gen_ids
    result["gen_text"] = out.outputs[0].text
    if args.no_lp:
        args.ref = None

    if not args.no_lp:
        # decode-vs-prefill self check: logprob of each generated token under the
        # decode path vs under a single prefill over prompt+generated.
        dec_lp = [lp[t].logprob for lp, t in zip(out.outputs[0].logprobs, gen_ids)]
        chk = llm.generate([TokensPrompt(prompt_token_ids=ids + gen_ids)],
                           SamplingParams(max_tokens=1, temperature=0.0, prompt_logprobs=1))[0]
        pre_lp = [chk.prompt_logprobs[len(ids) + i][t].logprob for i, t in enumerate(gen_ids)]
        diffs = [abs(a - b) for a, b in zip(dec_lp, pre_lp)]
        print(f"[harness] decode-vs-prefill logprob max|diff|={max(diffs):.3f} "
              f"dec={[round(x, 2) for x in dec_lp]} pre={[round(x, 2) for x in pre_lp]}")
        result["dec_vs_pre_maxdiff"] = max(diffs)

        # prompt_logprobs[i] is the distribution for token i given tokens < i; we
        # compare position i (predicting token i+1) against the reference logits.
        vl_top = []
        for i in range(1, len(ids)):
            d = out.prompt_logprobs[i]
            vl_top.append(sorted(d.items(), key=lambda kv: -kv[1].logprob)[:5])
        last = out.outputs[0].logprobs[0]
        vl_top.append(sorted(last.items(), key=lambda kv: -kv[1].logprob)[:5])
        if args.ref:
            ref = torch.load(args.ref)
            lg = torch.log_softmax(ref["logits"].float(), -1)
            agree = 0
            for pos, top in enumerate(vl_top):
                r_top = lg[pos].topk(5)
                vtok = [t for t, _ in top]
                vlp = [round(x.logprob, 3) for _, x in top]
                rtok = r_top.indices.tolist()
                rlp = [round(x, 3) for x in r_top.values.tolist()]
                # logprob the ref assigns to vLLM's top token
                agree += int(vtok[0] == rtok[0])
                print(f"pos{pos}: vllm {vtok} {vlp}\n       ref  {rtok} {rlp}  ref_lp(vllm_top)={lg[pos, vtok[0]].item():.3f}")
            print(f"[harness] top1 agreement {agree}/{len(vl_top)}")
            result["top1_agree"] = agree
            result["positions"] = len(vl_top)

    for bs in [int(b) for b in args.bench_bs.split(",") if b]:
        n = args.bench_tokens
        sp = SamplingParams(max_tokens=n, min_tokens=n, temperature=0.0, ignore_eos=True)
        prompts = [TokensPrompt(prompt_token_ids=ids) for _ in range(bs)]
        llm.generate(prompts, SamplingParams(max_tokens=4, min_tokens=4, ignore_eos=True))  # warm
        # time prefill-only (1 token) and full run; decode = difference.
        t0 = time.time(); llm.generate(prompts, SamplingParams(max_tokens=1, ignore_eos=True)); t1 = time.time() - t0
        t0 = time.time(); llm.generate(prompts, sp); tn = time.time() - t0
        per_step = (tn - t1) / (n - 1)
        print(f"[harness] bs={bs}: prefill+1={t1*1e3:.1f}ms  {n} tok={tn:.2f}s  decode step={per_step*1e3:.2f}ms  "
              f"-> {bs/per_step:.1f} tok/s")
        result[f"bs{bs}_step_ms"] = per_step * 1e3

    if args.spin_tokens:
        if args.spin_warm:
            # compile every shape the spin will hit before timing it
            llm.generate([TokensPrompt(prompt_token_ids=ids)],
                         SamplingParams(max_tokens=args.spin_warm, min_tokens=args.spin_warm,
                                        ignore_eos=True, temperature=0.0))
        print("[harness] spin start", flush=True)
        spies = []
        if args.spy:
            import subprocess, psutil
            me = psutil.Process()
            procs = {"front": me.pid}
            for c in me.children(recursive=True):
                try:
                    nm = " ".join(c.cmdline())
                except Exception:
                    continue
                if nm.startswith("VLLM::EngineCore"):
                    procs["engine"] = c.pid
                elif nm.startswith("VLLM::Worker_TP0"):
                    procs["worker"] = c.pid
            for k, pid in procs.items():
                spies.append(subprocess.Popen(["py-spy", "record", "--pid", str(pid), "-d", "12", "-r", "200",
                                               "--idle", "-f", "raw", "-o", f"/work/logs/spy_{k}.txt"],
                                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
            print(f"[harness] spying {procs}", flush=True)
        t0 = time.time()
        llm.generate([TokensPrompt(prompt_token_ids=ids)],
                     SamplingParams(max_tokens=args.spin_tokens, min_tokens=args.spin_tokens,
                                    ignore_eos=True, temperature=0.0))
        dt = time.time() - t0
        print(f"[harness] spin {args.spin_tokens} tok in {dt:.1f}s -> {dt/args.spin_tokens*1e3:.2f} ms/tok", flush=True)
        for sp in spies:
            sp.wait()

    if args.profile_bs:
        prompts = [TokensPrompt(prompt_token_ids=ids) for _ in range(args.profile_bs)]
        llm.generate(prompts, SamplingParams(max_tokens=2, ignore_eos=True, temperature=0.0))
        llm.start_profile()
        llm.generate(prompts, SamplingParams(max_tokens=6, min_tokens=6, ignore_eos=True, temperature=0.0))
        llm.stop_profile()
        print("[harness] profile captured", flush=True)
        time.sleep(20)

    if args.out:
        json.dump(result, open(args.out, "w"), indent=1)
    os._exit(0)


if __name__ == '__main__':
    main()
