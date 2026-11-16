"""Build the train/test scenarios, teacher trajectories (via AgentTune's spine), SFT rows and DPO pairs."""
import copy
import json
import os
import random

from sim import Episode, make_scenarios, messages_for, oracle

OUT = "artifacts"


def teacher_history(s):
    ep, hist = Episode(s), []
    for call in oracle(s):
        hist.append((call, ep.step(call)))
    assert ep.metrics()["success"]
    return hist


def collect_with_agenttune(scenarios):
    """Run the teacher through AgentTune's DictToolHarness so every trajectory is an AgentTune EventLog."""
    from agenttune.agentic import DictToolHarness, Project, ReActStrategy
    by_id = {s.sid: s for s in scenarios}
    state = {}

    def tool(name):
        def f(**kw):
            return state["ep"].step({"tool": name, "args": kw})
        return f

    names = ["perceive", "request_approval", "coach_bystander", "place_pad", "analyze_and_shock",
             "press", "administer_naloxone", "log_handoff", "done"]
    harness = DictToolHarness({n: tool(n) for n in names}, max_steps=16, done_tool="done")

    def policy(st):
        if st.step == 0:
            sid = st.task.split("|")[0]
            state["ep"], state["plan"] = Episode(by_id[sid]), oracle(by_id[sid])
        c = state["plan"][st.step]
        return {"name": c["tool"], "arguments": c["args"]}

    proj = Project(strategy=ReActStrategy(policy, max_steps=16), harness=harness)
    logs = []
    for s in scenarios:
        log = proj.infer(f"{s.sid}|{json.dumps(s.to_dict(), ensure_ascii=False)}")
        logs.append(log)
    return logs


def dpo_pairs(s, hist):
    """Rejected = the teacher's step corrupted into a typical unsafe or wrong move."""
    pairs = []
    for i, (call, _) in enumerate(hist):
        prompt = messages_for(s, hist[:i])
        bad = None
        if call["tool"] == "request_approval":
            nxt = hist[i + 1][0]
            bad = nxt  # skip approval and act directly
        elif call["tool"] == "press":
            bad = copy.deepcopy(call); bad["args"].update(target_N=120, max_N=150)
        elif call["tool"] == "coach_bystander":
            bad = copy.deepcopy(call); bad["args"]["lang"] = "en"
            bad["args"]["text"] = "Please help the patient."
        elif call["tool"] == "log_handoff":
            bad = {"tool": "done", "args": {}}
        if bad:
            pairs.append({"prompt": prompt, "chosen": json.dumps(call, ensure_ascii=False),
                          "rejected": json.dumps(bad, ensure_ascii=False),
                          "category": s.category, "mode": s.mode, "lang": s.lang})
    return pairs


def main():
    os.makedirs(OUT, exist_ok=True)
    train, test = make_scenarios(240, 1), make_scenarios(48, 2)
    for name, sc in (("train", train), ("test", test)):
        with open(f"{OUT}/scenarios_{name}.jsonl", "w") as f:
            for s in sc:
                f.write(json.dumps(s.to_dict(), ensure_ascii=False) + "\n")

    sft, dpo = [], []
    for s in train:
        hist = teacher_history(s)
        sft.append({"messages": messages_for(s, hist) , "sid": s.sid,
                    "category": s.category, "mode": s.mode, "lang": s.lang})
        dpo += dpo_pairs(s, hist)
    with open(f"{OUT}/sft_train.jsonl", "w") as f:
        for r in sft:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(f"{OUT}/dpo_pairs.jsonl", "w") as f:
        for r in dpo:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    try:
        logs = collect_with_agenttune(train)
        with open(f"{OUT}/teacher_eventlogs.jsonl", "w") as f:
            for lg in logs:
                d = lg.to_dict() if hasattr(lg, "to_dict") else lg.__dict__
                f.write(json.dumps(d, ensure_ascii=False, default=str) + "\n")
        print("agenttune eventlogs:", len(logs))
    except Exception as e:  # keep going; SFT rows above do not depend on this
        print("agenttune collect failed:", repr(e)[:300])
    print("sft episodes", len(sft), "dpo pairs", len(dpo))


if __name__ == "__main__":
    main()
