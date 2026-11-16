"""Fill README and abstract with the before/after table from results/*_summary.json."""
import json
import os

R = {n: json.load(open(f"results/{n}_summary.json")) for n in ("base", "sft", "teacher")
     if os.path.exists(f"results/{n}_summary.json")}
ROWS = [("Weighted score", "weighted_score"), ("Episode success (correct + safe)", "success_rate"),
        ("Tool-call F1 (AuditKIT)", "tool_call_f1"), ("Trajectory in order (AuditKIT)", "trajectory_in_order"),
        ("Response correctness", "response_correct"), ("Valid tool calls", "valid_call_rate"),
        ("Safety pass", "safety_pass"), ("Safety pass under injection", "safety_pass_under_injection"),
        ("Coaching in the right language", "coaching_lang_ok"), ("Loop-free (AuditKIT)", "loop_free"),
        ("Redundant calls, lower is better (AuditKIT)", "redundant_tool_calls"), ("Avg steps", "avg_steps")]
cols = [c for c in ("base", "sft", "teacher") if c in R]
head = {"base": "Base Tiny Aya Fire (lower bound)", "sft": "+ LoRA SFT (ours)", "teacher": "Teacher (upper bound)"}
lines = ["| Metric | " + " | ".join(head[c] for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
for label, k in ROWS:
    lines.append(f"| {label} | " + " | ".join(f"{R[c].get(k, 0):.2f}" for c in cols) + " |")
gap = ""
if {"base", "sft", "teacher"} <= R.keys():
    b, s, t = (R[c]["weighted_score"] for c in ("base", "sft", "teacher"))
    gap = f"{(s - b) / (t - b) * 100:.0f}%" if t > b else "n/a"
n = R.get("base", {}).get("n", 0)
note = (f"\n\nClosed-loop rollouts on {n} held-out scenes (one per emergency × mode, mixed languages, 1 with "
        f"prompt injection). Share of the base→teacher gap closed on the weighted score: **{gap}**. "
        "The test set is small because of the time limit; treat these as early numbers.")
per = ["\n\n**Per category (weighted score)**\n", "| Category | " + " | ".join(head[c] for c in cols) + " |",
       "|---" * (len(cols) + 1) + "|"]
for cat in R[cols[0]]["per_category"]:
    per.append(f"| {cat} | " + " | ".join(f"{R[c]['per_category'].get(cat, {}).get('score', 0):.2f}" for c in cols) + " |")
table = "\n".join(lines) + note + "\n".join(per)
t = open("README.md").read().replace("RESULTS_TABLE", table); open("README.md", "w").write(t)
if "sft" in R and "base" in R:
    line = (f"Results on {n} held-out closed-loop scenes: weighted score {R['base']['weighted_score']:.2f} (base) → "
            f"{R['sft']['weighted_score']:.2f} (ours) vs {R['teacher']['weighted_score']:.2f} (teacher), closing {gap} of the gap; "
            f"tool-call F1 {R['base']['tool_call_f1']:.2f} → {R['sft']['tool_call_f1']:.2f}; safety pass "
            f"{R['base']['safety_pass']:.2f} → {R['sft']['safety_pass']:.2f}.")
    t = open("ABSTRACT.md").read().replace("RESULTS_LINE", line); open("ABSTRACT.md", "w").write(t)
print(table)
