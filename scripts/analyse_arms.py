"""
analyse_arms.py  --  one analysis for corrected_penalty_arm.json, depth_penalty_sweep.json,
size_test.json.  (5 Sep 2026)

Per job: valid fraction, x* shots, x*/uniform-within-valid, hardware mode and its
validity; for 3-node cases also single/pair mass. Ideal values are read from the
experiment JSON (computed at build time). Pending jobs are skipped; rerun later.
Writes analysis_<experiment>.json and prints per-case five-run means +- sample s.d.

    python analyse_arms.py depth_penalty_sweep.json
    python analyse_arms.py size_test.json
    python analyse_arms.py corrected_penalty_arm.json
"""
import json, sys, itertools
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

fn = sys.argv[1]; exp = json.load(open(fn)); service = QiskitRuntimeService()


def counts_of(job):
    data = job.result()[0].data
    for name in dir(data):
        if not name.startswith("_") and hasattr(getattr(data, name), "get_counts"):
            return getattr(data, name).get_counts()
    raise RuntimeError("no counts register")


def case_lookup(j):
    """Return (case_dict, n, nonedges, xstar_bits) for a job record."""
    if "cases" in exp:                                     # sweep or size test
        for c in exp["cases"]:
            if fn.startswith("depth") and (c["group"], c["penalty_factor"], c["p"]) == (j["group"], j["penalty_factor"], j["p"]):
                w = np.array(c["weights"]); ne = list(itertools.combinations(range(3), 2))
                return c, 3, ne, c["ground_state_bits"] if c["ground_state_valid"] else max(
                    (format(1 << k, "03b") for k in range(3)), key=lambda s: w[int(np.log2(int(s, 2)))])
            if fn.startswith("size") and (c["union"], c["penalty_factor"]) == (j["union"], j["penalty_factor"]):
                jid_inst = "d96imekqp3as739qlehg" if "1hsg" in fn else "d95ti2otcv6s73dj4f7g"
                inst = json.load(open("extracted_hamiltonians.json"))[jid_inst]["monolithic"]
                idx = {v: k for k, v in enumerate(c["nodes"])}
                ne = [(idx[a], idx[b]) for a, b in inst["nonedges"] if a in idx and b in idx]
                return c, c["n"], ne, c["xstar_bits"]
    t = exp["targets"][j["target"]]                        # corrected arm
    if j["arm"] == "monolithic_p1":
        return t["monolithic"], 11, [tuple(e) for e in t["nonedges"]], t["xstar_clbit"]
    g = j["arm"].split("sub_p3_")[1]; s = t["subproblems"][g]
    w = np.array(s["weights"])
    return s, 3, list(itertools.combinations(range(3), 2)), format(1 << int(np.argmax(w)), "03b")


import os
noisy = json.load(open("noise_sim.json")) if os.path.exists("noise_sim.json") else {}
rows, pending = [], 0
for j in exp["jobs"]:
    job = service.job(j["job_id"])
    if "DONE" not in str(job.status()).upper():
        pending += 1; continue
    case, n, ne, xstar = case_lookup(j)
    counts = counts_of(job); total = sum(counts.values())

    def bits(k): return [int(k[n - 1 - q]) for q in range(n)]
    def valid(k):
        b = bits(k); return sum(b) > 0 and not any(b[i] and b[j2] for i, j2 in ne)
    V = [k for k in (format(x, f"0{n}b") for x in range(1, 2 ** n)) if valid(k)]
    vshots = sum(counts.get(k, 0) for k in V)
    obs = counts.get(xstar, 0); mode = max(counts, key=counts.get)
    row = {k: v for k, v in j.items() if k in ("group", "union", "arm", "target", "n", "p", "penalty_factor", "run", "job_id", "depth", "ops")}
    row.update({"shots": total, "valid_frac": vshots / total, "xstar_shots": obs,
                "xstar_over_uniform": obs / (vshots / len(V)) if vshots else None,
                "mode": mode, "mode_valid": valid(mode), "mode_frac": counts[mode] / total,
                "empty_frac": counts.get("0" * n, 0) / total,
                "ideal_valid_frac": case.get("ideal_valid_frac"), "ideal_xstar_over_uniform": case.get("ideal_xstar_over_uniform"),
                "ideal_mode": case.get("ideal_mode", case.get("ideal_top_state")),
                "ideal_mode_valid": case.get("ideal_mode_valid", case.get("ideal_top_state_valid"))})
    nc = noisy.get(j["job_id"], {}).get("counts_noisy")
    if nc:
        nt = sum(nc.values()); nv = sum(nc.get(k, 0) for k in V); nmode = max(nc, key=nc.get)
        row.update({"noisy_valid_frac": nv / nt, "noisy_xstar_over_uniform": nc.get(xstar, 0) / (nv / len(V)) if nv else None,
                    "noisy_mode_valid": valid(nmode), "noisy_empty_frac": nc.get("0" * n, 0) / nt})
        if n == 3:
            row["noisy_pair_mass"] = sum(nc.get(format(7 - (1 << k), "03b"), 0) for k in range(3)) / nt
        row["exec_running"] = noisy[j["job_id"]].get("timestamps", {}).get("running")
    if n == 3:
        row.update({"single_mass": sum(counts.get(format(1 << k, "03b"), 0) for k in range(3)) / total,
                    "pair_mass": sum(counts.get(format(7 - (1 << k), "03b"), 0) for k in range(3)) / total,
                    "ideal_single_mass": case.get("ideal_single_mass", sum(case.get("singles_ideal", {}).values())),
                    "ideal_pair_mass": case.get("ideal_pair_mass", sum(case.get("pairs_ideal", {}).values()))})
    rows.append(row)

json.dump(rows, open("analysis_" + fn, "w"), indent=1)
print(f"pending {pending}, analysed {len(rows)}")

# ------------------------------------------------------------ per-case summary
def key(r):
    return tuple(r.get(k) for k in ("target", "arm", "group", "union", "n", "p", "penalty_factor") if r.get(k) is not None)
groups = {}
for r in rows:
    groups.setdefault(key(r), []).append(r)
print(f"\n{'case':44s} {'valid(hw)':>16s} {'valid(id)':>9s} {'x*/unif(hw)':>16s} {'x*/unif(id)':>11s} {'mode valid hw/id':>16s} {'pair(hw)':>14s} {'pair(id)':>8s}")
for k, rs in sorted(groups.items(), key=lambda kv: [str(x) for x in kv[0]]):
    vf = np.array([r["valid_frac"] for r in rs]); ru = np.array([r["xstar_over_uniform"] or 0 for r in rs])
    mv = sum(r["mode_valid"] for r in rs); r0 = rs[0]
    pair = f"{np.mean([r['pair_mass'] for r in rs]):.3f}+-{np.std([r['pair_mass'] for r in rs], ddof=1):.3f}" if "pair_mass" in r0 else ""
    pid = (f"{r0['ideal_pair_mass']:.3f}" if r0.get("ideal_pair_mass") else "n/a") if "pair_mass" in r0 else ""
    nvf = np.array([r["noisy_valid_frac"] for r in rs if "noisy_valid_frac" in r])
    npm = np.array([r["noisy_pair_mass"] for r in rs if "noisy_pair_mass" in r])
    nef = np.array([r["noisy_empty_frac"] for r in rs if "noisy_empty_frac" in r]); hef = np.array([r["empty_frac"] for r in rs])
    noisy_str = (f"  noisy-sim: valid {nvf.mean():.3f}" + (f" pair {npm.mean():.3f}" if len(npm) else "") + f"  empty hw {hef.mean():.3f} sim {nef.mean():.3f}") if len(nvf) else f"  empty hw {hef.mean():.3f}"
    ivf = r0["ideal_valid_frac"]; ivf = f"{ivf:9.3f}" if ivf is not None else f"{'n/a':>9s}"
    iru = r0["ideal_xstar_over_uniform"]; iru = f"{iru:11.2f}" if iru is not None else f"{'n/a':>11s}"
    print(f"{' '.join(map(str, k)):44s} {vf.mean():7.3f}+-{vf.std(ddof=1):5.3f} {ivf} "
          f"{ru.mean():8.2f}+-{ru.std(ddof=1):5.2f} {iru} {mv:2d}/{len(rs)} / {str(r0['ideal_mode_valid']):5s}   {pair:>14s} {pid:>8s}{noisy_str}")
print(f"written analysis_{fn}")
