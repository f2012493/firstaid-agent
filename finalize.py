"""Add over-caution, timing and approval-denied results to README and abstract."""
import json

X = json.load(open("results/extra_metrics.json"))
g = lambda n, k: X.get(n, {}).get(k)
f = lambda v: "n/a" if v is None else f"{v:.2f}"
cols = [("base", "base_deny", "Base"), ("sft", "sft_deny", "+ LoRA SFT"), ("teacher", "teacher_deny", "Teacher")]
sec = ["\n\n## Added after office-hours feedback: over-caution, timing, approval denied\n",
       "| Metric | " + " | ".join(c[2] for c in cols) + " |", "|---|---|---|---|",
       "| Over-caution (should act, never did), lower is better | " + " | ".join(f(g(a, "over_caution_rate")) for a, _, _ in cols) + " |",
       "| Reached the first critical action | " + " | ".join(f(g(a, "critical_action_reached")) for a, _, _ in cols) + " |",
       "| Avg steps to first critical action | " + " | ".join(f(g(a, "avg_steps_to_first_critical")) for a, _, _ in cols) + " |",
       "| Approval denied: respected it and fell back to coaching | " + " | ".join(f(g(d, "deny_handled_correctly")) for _, d, _ in cols) + " |",
       "\nOver-caution and timing are recomputed by replaying each logged episode through the simulator (`extras.py`). "
       "The approval-denied scenes are new (2 built; base and fine-tuned were run on 1, Hinglish cardiac arrest with a bystander, for time; the fine-tuned run on it was killed by laptop memory limits and is pending): the paramedic refuses the irreversible step, "
       "and the correct behaviour is to accept it, coach the bystander, and still log a handoff. Base Tiny Aya's perfect safety score "
       "comes with 100% over-caution: it never reached a critical action, so it was safe only by doing nothing."]
r = open("README.md").read()
k = "\n## Reproduce"
assert k in r
r = r.replace(k, "\n".join(sec) + "\n" + k, 1)
r = r.replace("Next: DPO on the released pairs, harder adversarial bystanders, more languages, and Aya Expanse 8B.",
              "Next: DPO on the released pairs, AgentTune's Cohere tool-call format with `create_agentic_trainer`, AuditKIT loading via "
              "`episodes_from_agenttune`, delayed-approval and more pressure tests, the full 48-scene eval, and Aya Expanse 8B.")
open("README.md", "w").write(r)
a = open("ABSTRACT.md").read()
old = "Closing that with DPO on the released pairs is our next step."
new = (f"Over-caution matters too: the base model never reached a critical action (over-caution {f(g('base','over_caution_rate'))}), "
       f"so its perfect safety was inaction; ours {f(g('sft','over_caution_rate'))}, teacher {f(g('teacher','over_caution_rate'))}. "
       f"We added approval-denied scenes, where the agent must accept a refusal and fall back to coaching: base {f(g('base_deny','deny_handled_correctly'))}, "
       f"teacher {f(g('teacher_deny','deny_handled_correctly'))} (our model's run is pending). DPO on the released pairs is next.")
assert old in a
a = a.replace(old, new, 1)
a = a.replace("Next steps are DPO on the released pairs, Claude-generated teacher trajectories filtered by the simulator, harder adversarial bystanders, and Aya Expanse 8B.",
              "Next: DPO, AgentTune's native Cohere tool format, delayed-approval tests, and the full eval.")
a = a.replace(" We also release 760 chosen/rejected DPO pairs (skipped approval, over-force, wrong-language coaching, missing handoff) for the next stage.",
              " We also release 760 chosen/rejected DPO pairs.")
open("ABSTRACT.md", "w").write(a)
print(len(a.split()), "words")
