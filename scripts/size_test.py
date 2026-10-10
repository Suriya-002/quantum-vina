"""
size_test.py  --  at fixed depth p=1, when does the wrong penalty show in the output?
(5 Sep 2026)

Induced subgraphs of the executed 2BM2 graph that are UNIONS OF WHOLE PROTEIN
GROUPS (structure-preserving). For every non-empty union (30 cases, sizes 1..11)
and both penalties (P = 0.5 w_max, 1.1 w_max; w_max = global max, as in the
original code) the dry run reports:
  n, groups, brute-force optimum (recomputed), whether the global ground state
  is infeasible, fresh p=1 (gamma,beta) grid optimum, ideal mode and validity,
  ideal valid fraction, ideal x*/uniform-within-valid.

    python size_test.py --dry-run                      # 2BM2; add --1hsg for the 1HSG chain on ibm_fez
    python size_test.py --submit "P76,P79" "P76+P79,P38"   # submit selected cases (by group union), both penalties

Bit convention: subgraph bit k <-> node order[k] (order printed and stored).
Writes size_test.json.
"""
import json, sys, itertools, datetime
import numpy as np
from qiskit import transpile
from qiskit.quantum_info import Statevector
from corrected_penalty_arm import load, energies, qaoa_circuit, SHOTS, N_RUNS, SEED_TRANSPILER

TARGET = "1hsg" if "--1hsg" in sys.argv else "2bm2"
if TARGET == "2bm2":
    JOB = "d95ti2otcv6s73dj4f7g"; BACKEND = "ibm_marrakesh"
    GROUPS = {"P38": [0], "P75": [1], "P76": [2, 3, 4], "P79": [5, 6, 7], "P85": [8, 9, 10]}
else:
    JOB = "d96imekqp3as739qlehg"; BACKEND = "ibm_fez"
    GROUPS = {"P136": [0], "P139": [1, 2, 3], "P147": [4], "P51": [5, 6, 7], "P52": [8, 9, 10]}
PENALTY_FACTORS = (0.5, 1.1)
GRID = 32


def subgraph(inst, nodes):
    idx = {v: k for k, v in enumerate(nodes)}
    w = inst["w"][nodes]
    ne = [(idx[i], idx[j]) for i, j in inst["nonedges"] if i in idx and j in idx]
    return w, ne


def analyse_case(nodes, w, ne, P):
    n = len(nodes); E, xs = energies(n, w, P, ne)
    valid = np.array([xs[b].sum() > 0 and not any(xs[b, i] and xs[b, j] for i, j in ne) for b in range(2 ** n)])
    xstar = int(np.argmin(np.where(valid, E, np.inf)))
    gs = int(np.argmin(E))
    best = None
    for g in np.linspace(-np.pi, np.pi, GRID):
        for b in np.linspace(-np.pi / 2, np.pi / 2, GRID):
            qc = qaoa_circuit(n, w, P, ne, [g], [b]).remove_final_measurements(inplace=False)
            val = float((np.abs(Statevector(qc).data) ** 2 * E).sum())
            if best is None or val < best[0]:
                best = (val, g, b)
    qc = qaoa_circuit(n, w, P, ne, [best[1]], [best[2]])
    pr = np.abs(Statevector(qc.remove_final_measurements(inplace=False)).data) ** 2
    pv = float(pr[valid].sum()); nV = int(valid.sum()); top = int(np.argmax(pr))
    return {"n": n, "P": float(P), "gamma": float(best[1]), "beta": float(best[2]),
            "xstar_bits": format(xstar, f"0{n}b"), "xstar_weight": float(-E[xstar]),
            "ground_state_bits": format(gs, f"0{n}b"), "ground_state_infeasible": bool(not valid[gs]),
            "n_valid": nV, "ideal_valid_frac": pv, "ideal_xstar_prob": float(pr[xstar]),
            "ideal_xstar_over_uniform": float(pr[xstar] / (pv / nV)) if pv > 0 else None,
            "ideal_mode": format(top, f"0{n}b"), "ideal_mode_valid": bool(valid[top]), "ideal_mode_prob": float(pr[top])}, qc


def main():
    ext = json.load(open("extracted_hamiltonians.json")); inst = load(ext, JOB); wmax = inst["w"].max()
    submit = "--submit" in sys.argv
    selected = [a for a in sys.argv[1:] if not a.startswith("--")] if submit else None
    outname = f"size_test_{TARGET}.json"
    out = {"built": datetime.datetime.now().isoformat(), "backend": BACKEND, "w_max_global": float(wmax), "GRID": GRID,
           "bit_convention": "subgraph bit k <-> node order[k]", "cases": []}
    circuits = []
    names = list(GROUPS)
    print(f"{'union':22s} {'n':>2s} {'Pf':>4s} {'x*':>12s} {'w(x*)':>6s} {'gs infeas':>9s} {'|V|':>4s} {'valid':>6s} {'x*/unif':>7s} {'mode valid':>10s} {'mode p':>6s}")
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            label = "+".join(combo)
            if submit and label not in selected:
                continue
            nodes = sorted(v for g in combo for v in GROUPS[g])
            w, ne = subgraph(inst, nodes)
            for pf in PENALTY_FACTORS:
                res, qc = analyse_case(nodes, w, ne, pf * wmax)
                res.update({"union": label, "nodes": nodes, "penalty_factor": pf})
                qc.metadata = {"arm": f"size_{label}_P{pf}", "bit_to_node": nodes, "penalty_factor": pf}
                out["cases"].append(res); circuits.append((res, qc))
                print(f"{label:22s} {res['n']:2d} {pf:4.1f} {res['xstar_bits']:>12s} {res['xstar_weight']:6.3f} {str(res['ground_state_infeasible']):>9s} "
                      f"{res['n_valid']:4d} {res['ideal_valid_frac']:6.3f} {res['ideal_xstar_over_uniform'] if res['ideal_xstar_over_uniform'] is None else round(res['ideal_xstar_over_uniform'],2):>7} "
                      f"{str(res['ideal_mode_valid']):>10s} {res['ideal_mode_prob']:6.3f}")
    if not submit:
        json.dump(out, open(outname, "w"), indent=1); print("dry run: nothing submitted"); return
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService(); backend = service.backend(BACKEND); jobs = []
    for res, qc in circuits:
        isa = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=SEED_TRANSPILER)
        props = backend.properties()
        snap = {"last_update": str(props.last_update_date), "submitted": datetime.datetime.now().isoformat(),
                "depth": isa.depth(), "ops": dict(isa.count_ops()), "final_index_layout": isa.layout.final_index_layout() if isa.layout else None}
        sampler = SamplerV2(mode=backend)
        for run in range(N_RUNS):
            job = sampler.run([isa], shots=SHOTS)
            jobs.append({"union": res["union"], "n": res["n"], "penalty_factor": res["penalty_factor"], "run": run + 1, "job_id": job.job_id(), **snap})
            print(f"submitted {res['union']} n={res['n']} P={res['penalty_factor']} run {run+1}: {job.job_id()} depth {isa.depth()} cz {snap['ops'].get('cz',0)}")
    out["jobs"] = jobs
    json.dump(out, open(outname, "w"), indent=1); print(f"wrote {outname} with {len(jobs)} job ids")


if __name__ == "__main__":
    main()
