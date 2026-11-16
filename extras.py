"""Over-caution and time-to-first-critical-action, recomputed by replaying logged calls through FirstAid-Sim."""
import json
import os

from sim import Episode, Scenario

scen = {}
for f in ("artifacts/scenarios_test.jsonl", "artifacts/scenarios_deny.jsonl"):
    for l in open(f):
        d = json.loads(l); scen[d["sid"]] = Scenario(**d)


def replay(path):
    rows = []
    for l in open(path):
        r = json.loads(l)
        ep = Episode(scen[r["metrics"]["sid"]])
        for call, _ in r["trajectory"]:
            ep.step(call)
            if ep.finished:
                break
        rows.append(ep.metrics())
    return rows


out = {}
for name in ("base", "sft", "teacher", "base_deny", "sft_deny", "teacher_deny"):
    p = f"results/{name}_episodes.jsonl"
    if not os.path.exists(p):
        continue
    rows = replay(p)
    hit = [r["steps_to_first_critical"] for r in rows if r["steps_to_first_critical"]]
    out[name] = {"n": len(rows),
                 "over_caution_rate": sum(r["over_caution"] for r in rows if not r["deny"]) / max(sum(not r["deny"] for r in rows), 1),
                 "critical_action_reached": len(hit) / len(rows),
                 "avg_steps_to_first_critical": sum(hit) / len(hit) if hit else None,
                 "deny_handled_correctly": (sum(r["response_correct"] and r["safety_pass"] for r in rows if r["deny"]) /
                                            max(sum(r["deny"] for r in rows), 1)) if any(r["deny"] for r in rows) else None}
json.dump(out, open("results/extra_metrics.json", "w"), indent=2)
print(json.dumps(out, indent=1))
