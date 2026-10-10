"""
analyse_corrected_arm.py  --  same tests as the original arm, both arms side by side. (5 Sep 2026)

Reads corrected_penalty_arm.json (job ids, ideal predictions, bit conventions),
retrieves finished jobs, and for each computes:
  monolithic p=1 : valid shots, |V|, x* shots, x*/uniform-within-valid, top hardware state and its validity
  subproblem p=3 : single-state and pair-state shot masses, top state, whether it is the ground state
Original-arm numbers are read from valid_subspace_test.json and job_records.json
so both arms print in one table. Pending jobs are listed and skipped; rerun later.
Writes corrected_penalty_results.json.
"""
import json, itertools
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

arm = json.load(open("corrected_penalty_arm.json"))
service = QiskitRuntimeService()


def counts_of(job):
    data = job.result()[0].data
    for name in dir(data):
        if not name.startswith("_") and hasattr(getattr(data, name), "get_counts"):
            return getattr(data, name).get_counts()
    raise RuntimeError("no counts register")


def valid_states(n, nonedges):
    V = []
    for x in range(1, 2 ** n):
        b = [(x >> k) & 1 for k in range(n)]
        if not any(b[i] and b[j] for i, j in nonedges):
            V.append(x)
    return V


results = {"monolithic": [], "subproblems": [], "pending": []}
for j in arm["jobs"]:
    job = service.job(j["job_id"])
    status = str(job.status())
    if "DONE" not in status.upper():
        results["pending"].append({**j, "status": status}); continue
    counts = counts_of(job); total = sum(counts.values())
    t = arm["targets"][j["target"]]
    if j["arm"] == "monolithic_p1":
        n = 11; ne = [tuple(e) for e in t["nonedges"]]
        V = valid_states(n, ne); Vs = set(V)
        xs = t["xstar_clbit"]; obs = counts.get(xs, 0)
        vshots = sum(c for k, c in counts.items() if int(k, 2) in Vs)
        top = max(counts, key=counts.get)
        m = t["monolithic"]
        results["monolithic"].append({**j, "shots": total, "valid_shots": vshots, "valid_frac": vshots / total,
            "n_valid": len(V), "xstar_shots": obs, "xstar_over_uniform": obs / (vshots / len(V)) if vshots else None,
            "top_state": top, "top_state_valid": int(top, 2) in Vs, "top_state_count": counts[top],
            "ideal_valid_frac": m["ideal_valid_frac"], "ideal_xstar_prob": m["ideal_xstar_prob"],
            "ideal_xstar_over_uniform": m["ideal_xstar_over_uniform"], "ideal_top_state": m["ideal_top_state"]})
    else:
        g = j["arm"].split("sub_p3_")[1]; s = t["subproblems"][g]
        singles = {format(1 << k, "03b"): counts.get(format(1 << k, "03b"), 0) for k in range(3)}
        pairs = {format(7 - (1 << k), "03b"): counts.get(format(7 - (1 << k), "03b"), 0) for k in range(3)}
        top = max(counts, key=counts.get)
        results["subproblems"].append({**j, "shots": total, "singles": singles, "pairs": pairs,
            "single_mass": sum(singles.values()) / total, "pair_mass": sum(pairs.values()) / total,
            "empty": counts.get("000", 0) / total, "triple": counts.get("111", 0) / total,
            "top_state": top, "top_is_ground_state": top == s["ground_state_bits"],
            "ground_state_bits": s["ground_state_bits"], "ideal_singles": s["singles_ideal"], "ideal_pairs": s["pairs_ideal"],
            "bit_to_node": s["nodes"], "weights": s["weights"]})

json.dump(results, open("corrected_penalty_results.json", "w"), indent=1)

# ---------------------------------------------------------------- side-by-side
print(f"pending: {len(results['pending'])}")
try:
    orig = json.load(open("valid_subspace_test.json"))
except FileNotFoundError:
    orig = []
print("\nMONOLITHIC p=1   (hardware valid frac | x*/uniform | top state valid?)   ideal: valid | x*/uniform | top valid?")
for label, rows in (("ORIGINAL P=0.5wmax", orig), ("CORRECTED P=1.1wmax", results["monolithic"])):
    for r in rows:
        vf = r.get("valid_frac", r.get("valid_shots", 0) / r.get("shots", 4096))
        ratio = r.get("xstar_over_uniform", r.get("obs_over_uniform"))
        topv = r.get("top_state_valid", "?")
        iv = r.get("ideal_valid_frac", float("nan")); ir = r.get("ideal_xstar_over_uniform", float("nan"))
        itv = r.get("ideal_top_state_valid", r.get("ideal_top_state", "?") not in ("00000000000",))
        print(f"  {label:20s} {r['backend']:14s} {vf:6.3f} | {ratio:5.2f} | {str(topv):5s}     {iv:6.3f} | {ir:5.2f} | {itv}")
print("\nSUBPROBLEMS p=3 corrected   (single mass | pair mass | top state | is ground state?)   ideal singles/pairs")
for r in sorted(results["subproblems"], key=lambda r: (r["arm"], r["run"])):
    print(f"  {r['arm']:12s} run {r['run']}: {r['single_mass']:.3f} | {r['pair_mass']:.3f} | {r['top_state']} | {r['top_is_ground_state']}   "
          f"ideal singles {sum(r['ideal_singles'].values()):.3f} pairs {sum(r['ideal_pairs'].values()):.3f}")
print("\nFor the ORIGINAL subproblems, ideal pair masses were: P76 0.573, P79 0.753, P85 0.704 (extracted_hamiltonians.json).")
print("written corrected_penalty_results.json")
