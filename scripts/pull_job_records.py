"""
pull_job_records.py  --  Quantum-Vina job-record pull (4 Sep 2026)

Run inside qpyenv (qiskit 2.5.2, qiskit-ibm-runtime 0.49.0). Reads the job-ID
files, retrieves every job, and writes job_records.json with:

  jobs[<job_id>]            backend, creation date, counts, layout (if a PUB
                            circuit is attached), depth and gate counts of the
                            submitted circuit
  support_check[...]        for each subproblem job: which single-assignment
                            states appear, whether ALL of them appear, and the
                            probability mass on each single-assignment state
  monolithic[...]           for each monolithic job: counts of the optimum bit
                            string and top-10 states

It does not decide which job is which experiment. The job-ID files are
expected to carry that mapping; if they are flat lists, the script tags each
job by qubit count (11 = monolithic, 3 = subproblem, 1 = trivial) and by
backend, and you match runs by creation time afterwards.

Nothing here modifies any job or account state.
"""
import json, sys, glob, datetime
from collections import defaultdict

from qiskit_ibm_runtime import QiskitRuntimeService

OUT = "job_records.json"
OPTIMA = {"2bm2": "01010001000", "1hsg": "00010001100"}  # for reference only


def load_job_ids():
    ids = []
    for fn in ("all_job_ids.json", "job_ids.json", "job_ids_1hsg.json"):
        try:
            data = json.load(open(fn))
        except FileNotFoundError:
            continue
        # accept list, dict of lists, or dict of dicts
        if isinstance(data, list):
            ids += [(fn, None, j) for j in data]
        elif isinstance(data, dict):
            for k, v in data.items():
                if isinstance(v, list):
                    ids += [(fn, k, j) for j in v]
                elif isinstance(v, str):
                    ids.append((fn, k, v))
                elif isinstance(v, dict):
                    for k2, v2 in v.items():
                        if isinstance(v2, list):
                            ids += [(fn, f"{k}/{k2}", j) for j in v2]
                        elif isinstance(v2, str):
                            ids.append((fn, f"{k}/{k2}", v2))
    seen, uniq = set(), []
    for fn, tag, j in ids:
        if j not in seen:
            seen.add(j)
            uniq.append((fn, tag, j))
    return uniq


def counts_from_result(result):
    """SamplerV2 result -> dict of bitstring -> count, first PUB, first classical register."""
    pub = result[0]
    data = pub.data
    for name in dir(data):
        if name.startswith("_"):
            continue
        reg = getattr(data, name)
        if hasattr(reg, "get_counts"):
            return reg.get_counts()
    raise RuntimeError("no classical register with get_counts() found")


def single_assignment_states(n):
    return ["".join("1" if i == k else "0" for i in range(n)) for k in range(n)]


def main():
    service = QiskitRuntimeService()
    job_ids = load_job_ids()
    print(f"{len(job_ids)} unique job ids")

    records, support, mono = {}, {}, {}
    for fn, tag, jid in job_ids:
        try:
            job = service.job(jid)
        except Exception as e:  # noqa
            records[jid] = {"error": f"retrieve failed: {e}", "source": fn, "tag": tag}
            continue
        rec = {"source": fn, "tag": tag,
               "backend": job.backend().name,
               "created": str(job.creation_date)}
        try:
            res = job.result()
            counts = counts_from_result(res)
            total = sum(counts.values())
            n = len(next(iter(counts)))
            rec.update({"n_qubits": n, "shots": total, "counts": counts})
        except Exception as e:  # noqa
            rec["error"] = f"result failed: {e}"
            records[jid] = rec
            continue

        # submitted circuit, if retrievable
        try:
            circ = job.inputs["pubs"][0][0]
            rec["depth"] = circ.depth()
            rec["ops"] = dict(circ.count_ops())
            lay = getattr(circ, "layout", None)
            if lay is not None and lay.initial_layout is not None:
                rec["initial_layout"] = [lay.initial_layout[q] for q in circ.qubits] \
                    if hasattr(lay.initial_layout, "__getitem__") else str(lay.initial_layout)
        except Exception as e:  # noqa
            rec["circuit_note"] = f"submitted circuit not retrievable: {e}"

        # analysis by qubit count
        if n == 11:
            top10 = sorted(counts.items(), key=lambda kv: -kv[1])[:10]
            mono[jid] = {"backend": rec["backend"], "created": rec["created"],
                         "top10": top10,
                         "optimum_counts": {k: counts.get(v, 0) for k, v in OPTIMA.items()}}
        else:
            sas = single_assignment_states(n)
            present = {s: counts.get(s, 0) for s in sas}
            empty = counts.get("0" * n, 0)
            valid = empty + sum(present.values())
            support[jid] = {"backend": rec["backend"], "created": rec["created"],
                            "n_qubits": n, "shots": total,
                            "single_assignment_counts": present,
                            "all_single_states_present": all(c > 0 for c in present.values()),
                            "single_assignment_mass": {s: c / total for s, c in present.items()},
                            "valid_fraction": valid / total,
                            "most_probable_single": max(present, key=present.get)}
        records[jid] = rec

    out = {"pulled": datetime.datetime.now().isoformat(),
           "n_jobs": len(records), "jobs": records,
           "support_check": support, "monolithic": mono}
    json.dump(out, open(OUT, "w"), indent=1)

    # console summary
    print("\nSUBPROBLEM SUPPORT CHECK")
    for jid, s in sorted(support.items(), key=lambda kv: kv[1]["created"]):
        flag = "FULL" if s["all_single_states_present"] else "MISSING"
        print(f"{s['created'][:19]} {s['backend']:15s} n={s['n_qubits']} {flag} "
              f"valid={s['valid_fraction']:.3f} single={s['single_assignment_counts']}")
    missing = [j for j, s in support.items() if not s["all_single_states_present"]]
    print(f"\n{len(support)} subproblem jobs, {len(missing)} with a missing single-assignment state")
    print(f"written {OUT}")


if __name__ == "__main__":
    main()
