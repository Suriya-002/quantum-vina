"""
noise_sim.py  --  expected-under-noise companion for every hardware job. (5 Sep 2026)

For each job in job_records.json (originals), corrected_penalty_arm.json,
depth_penalty_sweep.json, size_test.json:
  * execution timestamps from job.metrics()  (created / running / finished)
  * backend properties in force at the RUNNING time (historical query)
  * Aer noise model from those properties (from_backend_properties: gate errors,
    readout errors, T1/T2 thermal relaxation)
  * 4096-shot AerSimulator run of the exact submitted ISA circuit
Writes noise_sim.json: {job_id: {timestamps, properties_last_update, counts_noisy}}.
Skips jobs already present (rerunnable). Run in qpyenv; needs qiskit-aer:
    pip install qiskit-aer
"""
import json, os, datetime
import numpy as np
from qiskit_ibm_runtime import QiskitRuntimeService

try:
    from qiskit_aer import AerSimulator
    from qiskit_aer.noise import NoiseModel
except ImportError:
    raise SystemExit("qiskit-aer not installed in this environment: pip install qiskit-aer")

SHOTS = 4096; SEED = 7
OUT = "noise_sim.json"
service = QiskitRuntimeService()
done = json.load(open(OUT)) if os.path.exists(OUT) else {}


def job_ids():
    ids = []
    if os.path.exists("job_records.json"):
        ids += [(j, "original") for j, r in json.load(open("job_records.json"))["jobs"].items() if "counts" in r]
    for fn, tag in (("corrected_penalty_arm.json", "corrected"), ("corrected_penalty_arm_executed.json", "corrected"), ("corrected_penalty_arm_1hsg-subs.json", "corrected_1hsg"),
                    ("depth_penalty_sweep.json", "sweep"), ("size_test.json", "size"), ("size_test_1hsg.json", "size_1hsg")):
        if os.path.exists(fn):
            d = json.load(open(fn))
            ids += [(j["job_id"], tag) for j in d.get("jobs", [])]
    return ids


def noise_model_for(backend, when):
    try:
        props = backend.properties(datetime=when)
    except Exception:
        props = backend.properties()
    try:
        nm = NoiseModel.from_backend_properties(props, basis_gates=list(backend.configuration().basis_gates),
                                                 gate_lengths=None, temperature=0)
    except Exception:
        nm = NoiseModel.from_backend(backend)
    return nm, str(props.last_update_date)


backends = {}
for jid, tag in job_ids():
    if jid in done:
        continue
    try:
        job = service.job(jid)
        if "DONE" not in str(job.status()).upper():
            print(f"{jid} {tag}: not executed yet ({job.status()}), skipped until it runs")
            continue
        metrics = job.metrics()
        ts = metrics.get("timestamps", {})
        running = ts.get("running") or ts.get("created")
        when = datetime.datetime.fromisoformat(running.replace("Z", "+00:00")) if running else None
        bname = job.backend().name
        if bname not in backends:
            backends[bname] = service.backend(bname)
        backend = backends[bname]
        nm, props_date = noise_model_for(backend, when)
        circ = job.inputs["pubs"][0][0]
        sim = AerSimulator(noise_model=nm, seed_simulator=SEED)
        counts = sim.run(circ, shots=SHOTS).result().get_counts()
        done[jid] = {"experiment": tag, "backend": bname, "timestamps": ts, "properties_last_update": props_date,
                     "counts_noisy": counts, "shots": SHOTS}
        print(f"{jid} {tag:9s} {bname:14s} running {running}  props {props_date[:19]}  noisy states {len(counts)}")
    except Exception as e:  # noqa
        done[jid] = {"experiment": tag, "error": str(e)}
        print(f"{jid} {tag}: ERROR {e}")
    json.dump(done, open(OUT, "w"), indent=1)

n_ok = sum(1 for v in done.values() if "counts_noisy" in v)
print(f"written {OUT}: {n_ok} simulated, {len(done) - n_ok} errors")
