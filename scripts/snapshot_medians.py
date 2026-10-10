"""
snapshot_medians.py -- device medians (cz error, T1, T2, readout) for the calibration
snapshot in force at EXECUTION, queried at the exact running time recorded in
noise_sim.json. Skips qubits missing a property. Run in qpyenv AFTER noise_sim.py.
Writes snapshot_medians.json.

Timezones (10 Oct 2026): qiskit-ibm-runtime returns BackendProperties.last_update_date
in the LOCAL timezone of the querying machine (utils/backend_decoder.py applies
utc_to_local_all), so the same calibration prints 3 h apart on a machine moved from
EDT to PDT. Job running timestamps are UTC ('Z'). This script prints and stores both
the raw local string and the UTC conversion; tab:circuits uses UTC throughout.
For a ledger-based set the medians are computed for EVERY distinct calibration the
set executed against (the 1HSG size chain ran on two).
"""
import json, os, datetime, numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

ns = json.load(open("noise_sim.json"))
REP = {  # label: representative job id, or ("ledger file",) -> one entry per distinct calibration
    "2BM2 originals mono":      "d95ti2otcv6s73dj4f7g",
    "2BM2 originals subs":      "d95u74d2su3c739gap90",
    "1HSG originals mono":      "d96imekqp3as739qlehg",
    "1HSG originals subs":      "d96imm52su3c739h0ju0",
    "corrected arm Marrakesh":  "dae331bdd5gc73d82bm0",
    "corrected arm Fez":        "dae335tnj4cs73aejv20",
    "sweep + size test":        "dae5vc51ierc738lmvg0",
    "1HSG corrected subs":      ("corrected_penalty_arm_1hsg-subs.json",),
    "1HSG size test":           ("size_test_1hsg.json",),
}
service = QiskitRuntimeService(); out = {}; backends = {}
UTC = datetime.timezone.utc


def med(vals): return float(np.median([v for v in vals if v is not None]))


def to_utc(s):
    d = datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    return d.astimezone(UTC) if d.tzinfo else None


def utc_label(s):
    d = to_utc(s); return f"{d.day} {d:%b %H:%M}" if d else "?"


def ledger_ids(fn):
    return [j["job_id"] for j in json.load(open(fn)).get("jobs", [])] if os.path.exists(fn) else []


def medians(label, jid, extra=None):
    rec = ns[jid]; bname = rec["backend"]; running = rec["timestamps"]["running"]
    b = backends.setdefault(bname, service.backend(bname)); p = b.properties(datetime=to_utc(running))
    cz = [g.parameters[0].value for g in p.gates if g.gate == "cz"]

    def q(name):
        vals = []
        for i in range(b.num_qubits):
            try: vals.append(p.qubit_property(i, name)[0])
            except Exception: pass
        return vals
    t1 = [v * 1e6 for v in q("T1")]; t2 = [v * 1e6 for v in q("T2")]; ro = q("readout_error")
    lu = str(p.last_update_date)
    out[label] = {"backend": bname, "representative_job": jid, "queried_at_running": running, "properties_last_update": lu,
                  "calibration_utc": utc_label(lu), "n_qubits_with_T2": len(t2), "median_cz_error": med(cz), "median_T1_us": med(t1),
                  "median_T2_us": med(t2), "median_readout_error": med(ro), **(extra or {})}
    print(f"{label:30s} {bname:14s} run {utc_label(running)} UTC  calibration {utc_label(lu)} UTC (local {lu[:19]})  "
          f"cz {med(cz):.3e}  T1 {med(t1):6.1f}  T2 {med(t2):6.1f} (n={len(t2)})  RO {med(ro):.3e}")


for label, ref in REP.items():
    if isinstance(ref, tuple):
        ids = ledger_ids(ref[0])
        if not ids:
            print(f"{label:30s} ledger {ref[0]} missing or empty, skipped"); continue
        by_cal = {}
        for j in ids:
            r = ns.get(j, {})
            if (r.get("timestamps") or {}).get("running") and r.get("properties_last_update"):
                by_cal.setdefault(r["properties_last_update"], []).append((r["timestamps"]["running"], j))
        if not by_cal:
            print(f"{label:30s} no running timestamps in noise_sim.json for {ref[0]} (rerun noise_sim.py), skipped"); continue
        print(f"{label:30s} {len(ids)} jobs on {len(by_cal)} calibration(s): " + "; ".join(
            f"{utc_label(c)} UTC x{len(v)} (executed {utc_label(min(v)[0])}--{utc_label(max(v)[0])[-5:]} UTC)" for c, v in sorted(by_cal.items())))
        for k, (c, v) in enumerate(sorted(by_cal.items()), 1):
            v.sort()
            sub = label if len(by_cal) == 1 else f"{label} (cal. {k})"
            medians(sub, v[0][1], {"n_jobs_on_this_calibration": len(v), "executed_utc": [v[0][0], v[-1][0]]})
    else:
        medians(label, ref)
json.dump(out, open("snapshot_medians.json", "w"), indent=1); print("written snapshot_medians.json")
