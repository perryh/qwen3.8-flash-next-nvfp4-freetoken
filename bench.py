#!/usr/bin/env python3
"""FreeToken bench v4 — computes tok/s from wall clock (FreeToken usage block
doesn't expose per-phase timings on this build). Scenarios sized to the
~8.2K-token KV budget."""
import json, os, sys, time, urllib.request, urllib.error

BASE = os.environ.get("BASE", "http://localhost:1919")
MODEL = os.environ.get("MODEL", "Qwen3.8-Flash-Next-NVFP4")

def chat(payload):
    payload = {"model": MODEL, **payload}
    t0 = time.time()
    req = urllib.request.Request(f"{BASE}/v1/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        d = json.load(urllib.request.urlopen(req, timeout=1500))
    except urllib.error.HTTPError as e:
        return {"ERROR": f"HTTP {e.code}: {e.read().decode(errors='ignore')[:250]}", "wall_s": round(time.time()-t0, 2)}
    wall = time.time() - t0
    u = dict(d.get("usage", {}))
    ptoks = u.get("prompt_tokens", 0)
    ctoks = u.get("completion_tokens", 0)
    return {
        "wall_s": round(wall, 2),
        "prompt_tokens": ptoks,
        "completion_tokens": ctoks,
        # wall includes prefill+decode+overhead; mark as effective throughput
        "eff_gen_tok_s_incl_prefill": round(ctoks / wall, 1) if wall else None,
        "prefill_tok_s_est": round(ptoks / max(wall - ctoks / 20.0, 0.01), 1) if ptoks > 2000 else None,
    }

def long_prompt(n_words):
    seq = ("The lighthouse keeper logged the weather, wind speed, barometric pressure, "
           "tide tables, cargo manifests, gull sightings, and the occasional ship's bell. ")
    reps = n_words // len(seq.split())
    return "Summarize the following log:\n\n" + seq * reps

results = {}

print("=== 1. SHORT PROMPT: baseline decode (400 out) ===")
r = chat({"messages":[{"role":"user","content":"Write a detailed 300-word explanation of how a bicycle drivetrain works."}],"max_tokens":400})
results["1_short_decode"] = r
print(json.dumps(r, indent=2))

print("\n=== 2. ~4K-TOKEN PROMPT: prefill (2 runs, take best) ===")
runs = []
for i in range(2):
    r = chat({"messages":[{"role":"user","content":long_prompt(3000)}],"max_tokens":8})
    runs.append(r)
    print(json.dumps(r, indent=2))
results["2_prefill_4k"] = min(runs, key=lambda x: x["wall_s"])

print("\n=== 3. ~7K-TOKEN PROMPT: prefill near KV budget ===")
runs = []
for i in range(2):
    r = chat({"messages":[{"role":"user","content":long_prompt(5300)}],"max_tokens":8})
    runs.append(r)
    print(json.dumps(r, indent=2))
results["3_prefill_7k"] = min(runs, key=lambda x: x["wall_s"])

print("\n=== 4. LONG CTX -> SUSTAINED GEN ===")
r = chat({"messages":[{"role":"user","content":long_prompt(5300) + "\n\nNow write a 400-word story about the keeper's cat."}],"max_tokens":500})
results["4_longctx_gen"] = r
print(json.dumps(r, indent=2))

json.dump(results, open("/tmp/ft_bench_results.json", "w"), indent=2)
print("\nSaved /tmp/ft_bench_results.json")
