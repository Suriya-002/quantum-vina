"""
corrected_penalty_arm.py  --  QAOA on the executed Quantum-Vina instances with a
correct penalty (P = PENALTY_FACTOR * w_max, > w_max), p=1 monolithic and p=3
subproblems, both targets.  (5 Sep 2026)

    python corrected_penalty_arm.py --dry-run     # build + simulate, print ideal predictions, no submission
    python corrected_penalty_arm.py               # same, then submit 5 runs x 4096 shots per circuit
    python corrected_penalty_arm.py --only=1hsg-subs   # submit only the 1HSG p=3 subproblems (15 jobs)

Conventions, written into every circuit's metadata and into the output JSON:
  node i  <->  qubit i  <->  classical bit i      (get_counts key: clbit n-1 leftmost)
  subproblem bit k  <->  node group[k]
Instances come from extracted_hamiltonians.json (executed graphs, clbit order).
Outputs: corrected_penalty_arm.json (parameters, ideal predictions, layouts, job ids,
backend properties snapshot per submission).  Existing job-id files are not touched.
"""
import json, sys, math, itertools, datetime
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Statevector
from scipy.optimize import minimize

PENALTY_FACTOR = 1.1
GRID = 30                   # original convention: linspace(-pi,pi,30) x linspace(-pi/2,pi/2,30)
SHOTS = 4096
N_RUNS = 5
SEED_TRANSPILER = 2026
TARGETS = {"2bm2": {"job": "d95ti2otcv6s73dj4f7g", "backend": "ibm_marrakesh",
                    "groups": {"P38": [0], "P75": [1], "P76": [2, 3, 4], "P79": [5, 6, 7], "P85": [8, 9, 10]}},
           "1hsg": {"job": "d96imekqp3as739qlehg", "backend": "ibm_fez",
                    "groups": {"P136": [0], "P139": [1, 2, 3], "P147": [4], "P51": [5, 6, 7], "P52": [8, 9, 10]}}}  # 1HSG: protein-sorted contiguous blocks, verified 5 Sep
DRY = "--dry-run" in sys.argv
ONLY = None
for a in sys.argv:
    if a.startswith("--only="):
        ONLY = a.split("=", 1)[1]        # e.g. --only=1hsg-subs  (target-subs | target-mono | target)


# ----------------------------------------------------------------- instance
def load(ext, jid):
    m = ext[jid]["monolithic"]; n = 11
    w = np.array([m["w_relative"][str(i)] for i in range(n)])
    ne = [tuple(e) for e in m["nonedges"]]
    P = PENALTY_FACTOR * w.max()
    return {"n": n, "w": w, "P": P, "nonedges": ne, "xstar_clbit": m["argmin_bits"]}


def energies(n, w, P, nonedges):
    """E(x) = -sum w x + P sum_{ne} x x  for every basis state (little-endian index)."""
    xs = np.array([[(b >> k) & 1 for k in range(n)] for b in range(2 ** n)])
    E = -xs @ w
    for i, j in nonedges:
        E = E + P * xs[:, i] * xs[:, j]
    return E, xs


def qaoa_circuit(n, w, P, nonedges, gammas, betas):
    qc = QuantumCircuit(n, n)
    qc.h(range(n))
    lin = w / 2.0; quad = {}
    for i, j in nonedges:
        quad[(i, j)] = P / 4.0; lin[i] -= P / 4.0; lin[j] -= P / 4.0
    for g, b in zip(gammas, betas):
        for i in range(n):
            if abs(lin[i]) > 1e-12:
                qc.rz(2 * g * lin[i], i)
        for (i, j), c in quad.items():
            qc.rzz(2 * g * c, i, j)
        qc.rx(2 * b, range(n))
    qc.measure(range(n), range(n))
    return qc


def ideal_stats(qc, E, valid_mask, xstar_idx):
    pr = np.abs(Statevector(qc.remove_final_measurements(inplace=False)).data) ** 2
    pv = float(pr[valid_mask].sum()); nV = int(valid_mask.sum())
    top = int(np.argmax(pr))
    return {"ideal_valid_frac": pv, "ideal_xstar_prob": float(pr[xstar_idx]),
            "ideal_xstar_over_uniform": float(pr[xstar_idx] / (pv / nV)),
            "ideal_top_state": format(top, f"0{qc.num_qubits}b"), "ideal_top_state_valid": bool(valid_mask[top]),
            "ideal_expected_E": float((pr * E).sum())}


# ----------------------------------------------------------------- p=1 grid
def grid_p1(n, w, P, nonedges):
    E, _ = energies(n, w, P, nonedges)
    pts = []
    for g in np.linspace(-np.pi, np.pi, GRID):
        for b in np.linspace(-np.pi / 2, np.pi / 2, GRID):
            qc = qaoa_circuit(n, w, P, nonedges, [g], [b]).remove_final_measurements(inplace=False)
            pr = np.abs(Statevector(qc).data) ** 2
            pts.append((float((pr * E).sum()), g, b))
    pts.sort()
    print("   top-3 grid points (expected E, gamma, beta):", [(round(v, 4), round(g, 4), round(b, 4)) for v, g, b in pts[:3]])
    return pts[0]


# ----------------------------------------------------------------- p=3 subproblem optimisation
def optimise_p3(w3, P):
    n = 3; ne = list(itertools.combinations(range(n), 2))
    E, _ = energies(n, w3, P, ne)

    def f(p):
        qc = qaoa_circuit(n, w3, P, ne, p[:3], p[3:]).remove_final_measurements(inplace=False)
        pr = np.abs(Statevector(qc).data) ** 2
        return float((pr * E).sum())
    rng = np.random.default_rng(0); best = None
    for _ in range(25):
        r = minimize(f, rng.uniform(-1.5, 1.5, 6), method="COBYLA", options={"maxiter": 600, "rhobeg": 0.4})
        if best is None or r.fun < best.fun:
            best = r
    return best.x[:3], best.x[3:], best.fun


# ----------------------------------------------------------------- 1HSG group inference
def infer_groups(n, nonedges, sizes=(1, 3, 1, 3, 3)):
    """Partition nodes into groups that are pairwise non-edges (same protein pharmacophore)
    with the given size multiset. Returns the unique partition or None."""
    ne = set(nonedges) | {(j, i) for i, j in nonedges}
    nodes = list(range(n)); found = set()

    def rec(remaining, groups):
        if not remaining:
            if sorted(len(g) for g in groups) == sorted(sizes):
                found.add(tuple(sorted(tuple(sorted(g)) for g in groups)))
            return
        if len(groups) >= len(sizes):
            return
        first = remaining[0]
        for size in set(sizes):
            for extra in itertools.combinations(remaining[1:], size - 1):
                g = (first,) + extra
                if all((a, b) in ne for a, b in itertools.combinations(g, 2)):
                    rec([x for x in remaining if x not in g], groups + [list(g)])
    rec(nodes, [])
    if len(found) == 1:
        return {f"G{k}": list(g) for k, g in enumerate(next(iter(found)))}
    print(f"  1HSG group inference: {len(found)} partitions found; skipping 1HSG subproblems until the rebuild names them")
    return None


# ----------------------------------------------------------------- main
def main():
    ext = json.load(open("extracted_hamiltonians.json"))
    out = {"built": datetime.datetime.now().isoformat(), "PENALTY_FACTOR": PENALTY_FACTOR, "GRID": GRID,
           "SHOTS": SHOTS, "N_RUNS": N_RUNS, "seed_transpiler": SEED_TRANSPILER, "dry_run": DRY,
           "bit_convention": "node i <-> qubit i <-> classical bit i; subproblem bit k <-> node group[k]", "targets": {}}
    circuits = []   # (target, label, circuit, backend)
    for name, cfg in TARGETS.items():
        inst = load(ext, cfg["job"]); n, w, P, ne = inst["n"], inst["w"], inst["P"], inst["nonedges"]
        E, xs = energies(n, w, P, ne)
        valid = np.array([not any(xs[b, i] and xs[b, j] for i, j in ne) and xs[b].sum() > 0 for b in range(2 ** n)])
        xstar_idx = int(np.argmin(np.where(valid, E, np.inf)))
        assert format(xstar_idx, f"0{n}b") == inst["xstar_clbit"], "argmin mismatch with extracted instance"
        assert int(np.argmin(E)) == xstar_idx, "corrected penalty must make the global ground state valid"
        print(f"== {name}: P = {P:.3f} (w_max {w.max():.3f}); global ground state is the valid optimum: OK")
        tinfo = {"P": float(P), "w": w.tolist(), "nonedges": ne, "xstar_clbit": inst["xstar_clbit"], "monolithic": {}, "subproblems": {}}

        val, g, b = grid_p1(n, w, P, ne)
        qc = qaoa_circuit(n, w, P, ne, [g], [b]); qc.metadata = {"target": name, "arm": "monolithic_p1", "bit_to_node": list(range(n))}
        st = ideal_stats(qc, E, valid, xstar_idx)
        tinfo["monolithic"] = {"gamma": float(g), "beta": float(b), "grid_min_expected_E": val, **st}
        print(f"  p=1: gamma {g:.3f} beta {b:.3f}  ideal valid {st['ideal_valid_frac']:.3f}  x* {st['ideal_xstar_prob']:.4f} "
              f"({st['ideal_xstar_over_uniform']:.2f}x uniform)  top valid? {st['ideal_top_state_valid']}")
        circuits.append((name, "monolithic_p1", qc, cfg["backend"]))

        groups = cfg["groups"] or infer_groups(n, ne)
        if groups:
            for gname, nodes in groups.items():
                if len(nodes) != 3:
                    continue
                w3 = w[nodes]; gam, bet, fun = optimise_p3(w3, P)
                ne3 = list(itertools.combinations(range(3), 2))
                E3, xs3 = energies(3, w3, P, ne3)
                valid3 = np.array([xs3[b].sum() == 1 for b in range(8)])
                xs3_idx = int(np.argmin(E3))
                qc3 = qaoa_circuit(3, w3, P, ne3, gam, bet)
                qc3.metadata = {"target": name, "arm": f"sub_p3_{gname}", "bit_to_node": nodes}
                st3 = ideal_stats(qc3, E3, valid3, xs3_idx)
                pr = np.abs(Statevector(qc3.remove_final_measurements(inplace=False)).data) ** 2
                singles = {format(1 << k, "03b"): float(pr[1 << k]) for k in range(3)}
                pairs = {format(7 - (1 << k), "03b"): float(pr[7 - (1 << k)]) for k in range(3)}
                tinfo["subproblems"][gname] = {"nodes": nodes, "weights": w3.tolist(), "gammas": gam.tolist(), "betas": bet.tolist(),
                                              "ground_state_bits": format(xs3_idx, "03b"), "singles_ideal": singles, "pairs_ideal": pairs, **st3}
                print(f"  {gname} nodes {nodes} w {np.round(w3, 3)}: ground state {format(xs3_idx, '03b')} valid? {st3['ideal_top_state_valid']}; "
                      f"ideal singles {dict((k, round(v, 3)) for k, v in singles.items())} pairs {dict((k, round(v, 3)) for k, v in pairs.items())}")
                circuits.append((name, f"sub_p3_{gname}", qc3, cfg["backend"]))
        out["targets"][name] = tinfo

    def selected(name, label):
        if not ONLY:
            return True
        tgt, _, kind = ONLY.partition("-")
        return name == tgt and not (kind == "subs" and not label.startswith("sub_")) and not (kind == "mono" and label != "monolithic_p1")
    to_submit = [(name, label) for name, label, _, _ in circuits if selected(name, label)]
    print(f"filter {ONLY or 'none'}: would submit {len(to_submit) * N_RUNS} jobs: {to_submit}")
    if DRY:
        json.dump(out, open("corrected_penalty_arm.json", "w"), indent=1)
        print("dry run: wrote corrected_penalty_arm.json, nothing submitted"); return

    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService()
    jobs = []
    out_name = "corrected_penalty_arm.json" if ONLY is None else f"corrected_penalty_arm_{ONLY}.json"
    for name, label, qc, bname in circuits:
        if not selected(name, label):
            continue
        backend = service.backend(bname)
        isa = transpile(qc, backend=backend, optimization_level=3, seed_transpiler=SEED_TRANSPILER)
        props = backend.properties()
        snapshot = {"backend": bname, "last_update": str(props.last_update_date),
                    "submitted": datetime.datetime.now().isoformat(), "depth": isa.depth(), "ops": dict(isa.count_ops()),
                    "final_index_layout": isa.layout.final_index_layout() if isa.layout else None}
        sampler = SamplerV2(mode=backend)
        for run in range(N_RUNS):
            job = sampler.run([isa], shots=SHOTS)
            jobs.append({"target": name, "arm": label, "run": run + 1, "job_id": job.job_id(), **snapshot})
            print(f"submitted {name} {label} run {run + 1}: {job.job_id()}  depth {isa.depth()} cz {snapshot['ops'].get('cz', 0)}")
    out["jobs"] = jobs
    json.dump(out, open(out_name, "w"), indent=1)
    print(f"wrote {out_name} with {len(jobs)} job ids")


if __name__ == "__main__":
    main()
