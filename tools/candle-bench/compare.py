#!/usr/bin/env python3
"""Compare llamacpp vs candle benchmark run JSONs."""
import json
import sys

a_path, b_path = sys.argv[1], sys.argv[2]
a = json.load(open(a_path))
b = json.load(open(b_path))

assert len(a["items"]) == len(b["items"])
n = len(a["items"])
match = 0
mismatches = []
for ia, ib in zip(a["items"], b["items"]):
    assert ia["input"] == ib["input"]
    if ia["prediction"] == ib["prediction"]:
        match += 1
    else:
        mismatches.append((ia["input"], ia["prediction"], ib["prediction"]))

print(f"engines: {a['engine']} vs {b['engine']}")
print(f"output match: {match}/{n} ({100.0*match/n:.1f}%)")
print()
for name, r in (("A:" + a["engine"], a), ("B:" + b["engine"], b)):
    print(f"{name:12s} mean {r['mean_ms']:7.1f}ms  median {r['median_ms']:7.1f}ms  "
          f"p95 {r['p95_ms']:7.1f}ms  total {r['total_ms']/1000:6.2f}s  "
          f"tok/s {r['tokens_per_s']:6.1f}  [{r['threads']}]")
print()
print(f"mean latency ratio (B/A): {b['mean_ms']/a['mean_ms']:.3f}  "
      f"(B is {100*(b['mean_ms']/a['mean_ms']-1):+.1f}% vs A)")
print(f"median latency ratio (B/A): {b['median_ms']/a['median_ms']:.3f}")
print()
if mismatches:
    print(f"mismatch examples (up to 3 of {len(mismatches)}):")
    for inp, pa, pb in mismatches[:3]:
        print(f"  input : {inp}")
        print(f"    {a['engine']:9s}: {pa}")
        print(f"    {b['engine']:9s}: {pb}")
