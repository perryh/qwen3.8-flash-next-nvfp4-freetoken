#!/usr/bin/env python3
"""FreeToken bench v5 (final) — KV budget now 129 pages * 64 = 8256 tokens KV,
but with --memory-ratio 1 the cache budget is 19.5GB. Scenarios: baseline
decode, 4K + 8K prefill, long-ctx gen. Prefill tok/s measured via wall clock
on prompt-only runs; decode from short-prompt wall minus prefill overhead.
"""
import json, os, time, urllib.request, urllib.error

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
    return {"wall_s": round(wall, 2),
            "prompt_tokens": u.get("prompt_tokens", 0),
            "completion_tokens": u.get("completion_tokens", 0)}

def long_prompt(n_words):
    seq = ("The lighthouse keeper logged the weather, wind speed, barometric pressure, "
           "tide tables, cargo manifests, gull sightings, and the occasional ship's bell. ")
    reps = n_words // len(seq.split())
    return "Summarize the following log:\n\n" + seq * reps

results = {}

# --- pure decode rate: short prompt, 400 tokens out
r = chat({"messages":[{"role":"user","content":"Say OK then count from 1 to 100."}],"max_tokens":400})
decode_rate = r["completion_tokens"] / r["wall_s"]
results["decode"] = {"completion_tokens": r["completion_tokens"], "wall_s": r["wall_s"], "tok_s": round(decode_rate, 1)}
print("DECODE:", results["decode"])

# --- prefill: big prompt, 1 token out, measure wall
r = chat({"messages":[{"role":"user","content":long_prompt(3000)}],"max_tokens":1})
p4k = {"prompt_tokens": r["prompt_tokens"], "wall_s": r["wall_s"], "prefill_tok_s": round(r["prompt_tokens"]/r["wall_s"], 1)}
results["prefill_4k"] = p4k
print("PREFILL 4K:", p4k)

r = chat({"messages":[{"role":"user","content":long_prompt(5300)}],"max_tokens":1})
p8k = {"prompt_tokens": r["prompt_tokens"], "wall_s": r["wall_s"], "prefill_tok_s": round(r["prompt_tokens"]/r["wall_s"], 1)}
results["prefill_8k"] = p8k
print("PREFILL 8K:", p8k)

# --- long ctx -> gen
r = chat({"messages":[{"role":"user","content":long_prompt(5300) + "\n\nNow write a 400-word story about the keeper's cat."}],"max_tokens":500})
gen_rate = r["completion_tokens"] / max(r["wall_s"] - r["prompt_tokens"]/p8k["prefill_tok_s"], 0.01)
results["longctx_gen"] = {"prompt_tokens": r["prompt_tokens"], "completion_tokens": r["completion_tokens"],
                          "wall_s": r["wall_s"], "gen_tok_s_est": round(gen_rate, 1)}
print("LONGCTX GEN:", results["longctx_gen"])

json.dump(results, open("/tmp/ft_bench_final.json","w"), indent=2)
print("\nSaved /tmp/ft_bench_final.json")
