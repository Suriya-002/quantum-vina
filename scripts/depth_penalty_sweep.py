"""
depth_penalty_sweep.py  --  where does the penalty error become visible as a function of depth?
(5 Sep 2026)

Three-node 2BM2 subproblems (P76, P79, P85) at p = 1, 2, 3 with P = 0.5 w_max
(original, infeasible ground state) and P = 1.1 w_max (corrected), parameters
optimised per (group, depth, penalty) on the statevector, 5 runs x 4096 shots
each on ibm_marrakesh.  3 x 3 x 2 x 5 = 90 jobs.

    python depth_penalty_sweep.py --dry-run     # ideal predictions only
    python depth_penalty_sweep.py               # submit

Bit convention as in corrected_penalty_arm.py: subproblem bit k <-> node group[k].
w_max is the GLOBAL maximum node weight of the 2BM2 instance (1.0 in relative
units), as in the original code (P_PEN = max(node_weights) * 0.5), not the
subproblem maximum. Writes depth_penalty_sweep.json.
"""
import json, sys, itertools, datetime
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize
from corrected_penalty_arm import load, energies, qaoa_circuit, SHOTS, N_RUNS, SEED_TRANSPILER

TARGET = "2bm2"; JOB = "d95ti2otcv6s73dj4f7g"; BACKEND = "ibm_marrakesh"
GROUPS = {"P76": [2, 3, 4], "P79": [5, 6, 7], "P85": [8, 9, 10]}
DEPTHS = (1, 2, 3)
PENALTY_FACTORS = (0.5, 1.1)
DRY = "--dry-run" in sys.argv


def optimise(w3, P, p, restarts=25):
    ne = list(itertools.combinations(range(3), 2)); E, _ = energies(3, w3, P, ne)

    def f(x):
        qc = qaoa_circuit(3, w3, P, ne, x[:p], x[p:]).remove_final_measurements(inplace=False)
        return float((np.abs(Statevector(qc).data) ** 2 * E).sum())
    rng = np.random.default_rng(0); best = None
    for _ in range(restarts):
        r = minimize(f, rng.uniform(-1.5, 1.5, 2 * p), method="COBYLA", options={"maxiter": 800, "rhobeg": 0.4})
        if best is None or r.fun < best.fun:
            best = r
    return best.x[:p], best.x[p:], best.fun


def main():
    ext = json.load(open("extracted_hamiltonians.json"))
    inst = load(ext, JOB); w = inst["w"]; wmax = w.max()
    out = {"built": datetime.datetime.now().isoformat(), "target": TARGET, "backend": BACKEND, "dry_run": DRY,
           "w_max_global": float(wmax), "SHOTS": SHOTS, "N_RUNS": N_RUNS, "seed_transpiler": SEED_TRANSPILER,
           "bit_convention": "subproblem bit k <-> node group[k]", "cases": []}
    circuits = []
    for g, nodes in GROUPS.items():
        w3 = w[nodes]; ne = list(itertools.combinations(range(3), 2))
        for pf in PENALTY_FACTORS:
            P = pf * wmax; E, xs = energies(3, w3, P, ne); gs = int(np.argmin(E))
            for p in DEPTHS:
                gam, bet, fun = optimise(w3, P, p)
                qc = qaoa_circuit(3, w3, P, ne, gam, bet)
                qc.metadata = {"target": TARGET, "arm": f"sweep_{g}_p{p}_P{pf}", "bit_to_node": nodes, "penalty_factor": pf}
                pr = np.abs(Statevector(qc.remove_final_measurements(inplace=False)).data) ** 2
                singles = {format(1 << k, "03b"): float(pr[1 << k]) for k in range(3)}
                pairs = {format(7 - (1 << k), "03b"): float(pr[7 - (1 << k)]) for k in range(3)}
                case = {"group": g, "nodes": nodes, "weights": w3.tolist(), "penalty_factor": pf, "P": float(P), "p": p,
                        "gammas": list(map(float, gam)), "betas": list(map(float, bet)), "expected_E": fun,
                        "ground_state_bits": format(gs, "03b"), "ground_state_valid": bool(xs[gs].sum() == 1),
                        "ideal_singles": singles, "ideal_pairs": pairs, "ideal_single_mass": sum(singles.values()),
                        "ideal_pair_mass": sum(pairs.values()), "ideal_top_state": format(int(np.argmax(pr)), "03b")}
                out["cases"].append(case); circuits.append((case, qc))
                print(f"{g} P={pf}w_max p={p}: ground {case['ground_state_bits']} valid={case['ground_state_valid']}  "
                      f"ideal single mass {case['ideal_single_mass']:.3f} pair mass {case['ideal_pair_mass']:.3f}  top {case['ideal_top_state']}")
    if DRY:
        json.dump(out, open("depth_penalty_sweep.json", "w"), indent=1); print("dry run: nothing submitted"); return

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService(); backend = service.backend(BACKEND)
    jobs = []
    for case, qc in circuits:
        isa = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=SEED_TRANSPILER)
        props = backend.properties()
        snap = {"last_update": str(props.last_update_date), "submitted": datetime.datetime.now().isoformat(),
                "depth": isa.depth(), "ops": dict(isa.count_ops()),
                "final_index_layout": isa.layout.final_index_layout() if isa.layout else None}
        sampler = SamplerV2(mode=backend)
        for run in range(N_RUNS):
            job = sampler.run([isa], shots=SHOTS)
            jobs.append({"group": case["group"], "penalty_factor": case["penalty_factor"], "p": case["p"],
                         "run": run + 1, "job_id": job.job_id(), **snap})
            print(f"submitted {case['group']} P={case['penalty_factor']} p={case['p']} run {run+1}: {job.job_id()} depth {isa.depth()} cz {snap['ops'].get('cz',0)}")
    out["jobs"] = jobs
    json.dump(out, open("depth_penalty_sweep.json", "w"), indent=1)
    print(f"wrote depth_penalty_sweep.json with {len(jobs)} job ids")


if __name__ == "__main__":
    main()
