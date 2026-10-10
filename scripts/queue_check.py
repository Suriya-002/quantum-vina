import json, os, time
from collections import Counter, OrderedDict
from qiskit_ibm_runtime import QiskitRuntimeService

def load(fn):
    if os.path.exists(fn):
        with open(fn, encoding="utf-8") as f:
            return json.load(f)
    return None

def ids_from(fn):
    d = load(fn)
    return [j["job_id"] for j in d["jobs"]] if d and d.get("jobs") else []

jr = load("job_records.json")
corr, src = ids_from("corrected_penalty_arm.json"), "corrected_penalty_arm.json"
if not corr:
    corr, src = ids_from("corrected_penalty_arm_executed.json"), "corrected_penalty_arm_executed.json"
if not corr:
    cr = load("corrected_penalty_results.json")
    if cr:
        corr = [r["job_id"] for r in cr.get("monolithic", []) + cr.get("subproblems", [])]
        src = "corrected_penalty_results.json (corrected_penalty_arm.json was overwritten by a dry run)"
ledger = OrderedDict([
    ("July originals",     (60, list(jr["jobs"]) if jr else [])),
    ("Corrected arm",      (25, corr)),
    ("1HSG corrected p=3", (15, ids_from("corrected_penalty_arm_1hsg-subs.json"))),
    ("Depth sweep",        (90, ids_from("depth_penalty_sweep.json"))),
    ("Size chain 2BM2",    (70, ids_from("size_test.json"))),
    ("Size chain 1HSG",    (80, ids_from("size_test_1hsg.json"))),
])
print("Corrected-arm IDs read from:", src)

BK = "results_backup.json"
backup = load(BK) or {}
if jr:
    for jid, v in jr["jobs"].items():
        if v.get("counts") and jid not in backup:
            backup[jid] = {"experiment": "July originals", "backend": v.get("backend"), "counts": v["counts"],
                           "shots": sum(v["counts"].values()), "saved": "from job_records.json"}
try:
    svc = QiskitRuntimeService(channel="ibm_cloud")
except Exception:
    svc = QiskitRuntimeService()

def counts_of(job):
    data = job.result()[0].data
    for name in dir(data):
        if not name.startswith("_") and hasattr(getattr(data, name), "get_counts"):
            return getattr(data, name).get_counts()
    return None

rows, problems = [], []
for exp, (expected, ids) in ledger.items():
    c, saved = Counter(), 0
    for i, jid in enumerate(ids, 1):
        print(f"\r  {exp:20s} {i:3d}/{len(ids)}", end="", flush=True)
        try:
            job = svc.job(jid)
            st = str(job.status())
        except Exception as e:
            c["LOOKUP_FAILED"] += 1
            problems.append((exp, jid, "LOOKUP_FAILED", str(e)[:90]))
            continue
        c[st] += 1
        if st == "DONE":
            if backup.get(jid, {}).get("counts"):
                saved += 1
            else:
                try:
                    cnt = counts_of(job)
                    backup[jid] = {"experiment": exp, "backend": job.backend().name, "counts": cnt,
                                   "shots": sum(cnt.values()), "saved": time.strftime("%Y-%m-%d %H:%M")}
                    saved += 1
                except Exception as e:
                    problems.append((exp, jid, "RESULT_NOT_RETRIEVABLE", str(e)[:90]))
        elif st not in ("QUEUED", "RUNNING", "INITIALIZING", "VALIDATING"):
            problems.append((exp, jid, st, ""))
    print()
    rows.append((exp, expected, len(ids), c, saved))
    with open(BK, "w") as f:
        json.dump(backup, f)

print("\n%-20s %8s %6s %6s %7s %6s %14s" % ("experiment", "expected", "found", "DONE", "QUEUED", "other", "results saved"))
tot = Counter()
for exp, expected, found, c, saved in rows:
    other = found - c.get("DONE", 0) - c.get("QUEUED", 0)
    flag = "" if found == expected else "   <-- ledger count differs"
    print("%-20s %8d %6d %6d %7d %6d %14d%s" % (exp, expected, found, c.get("DONE", 0), c.get("QUEUED", 0), other, saved, flag))
    for k, v in (("expected", expected), ("found", found), ("DONE", c.get("DONE", 0)), ("QUEUED", c.get("QUEUED", 0)), ("other", other), ("saved", saved)):
        tot[k] += v
print("%-20s %8d %6d %6d %7d %6d %14d" % ("TOTAL", tot["expected"], tot["found"], tot["DONE"], tot["QUEUED"], tot["other"], tot["saved"]))
if problems:
    print("\nJobs that are not DONE or QUEUED, or whose results could not be retrieved:")
    for exp, jid, st, msg in problems:
        print(f"  {exp:20s} {jid}  {st}  {msg}")
print("\nLocal files:")
for fn in ["job_records.json", "extracted_hamiltonians.json", "valid_subspace_test.json", "corrected_penalty_results.json",
           "corrected_penalty_arm.json", "corrected_penalty_arm_1hsg-subs.json", "depth_penalty_sweep.json", "size_test.json",
           "size_test_1hsg.json", "noise_sim.json", "snapshot_medians.json", "analysis_depth_penalty_sweep.json",
           "analysis_size_test.json", "analysis_corrected_penalty_arm_1hsg-subs.json", BK]:
    if os.path.exists(fn):
        print(f"  OK       {fn:48s} {os.path.getsize(fn)/1024:9.0f} KB  {time.strftime('%d %b %H:%M', time.localtime(os.path.getmtime(fn)))}")
    else:
        print(f"  MISSING  {fn}")
print(f"\nAll retrieved counts backed up in {BK} ({sum(1 for v in backup.values() if v.get('counts'))} jobs).")
