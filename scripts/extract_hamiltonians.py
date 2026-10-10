"""
extract_hamiltonians.py  --  recover the executed instance from the submitted circuits

Run inside qpyenv after pull_job_records.py. For every job whose PUB circuit is
retrievable:

  11-qubit (monolithic, p=1):
      statevector of the reduced circuit  ->  scan beta so |amp| is uniform after
      undoing the mixer  ->  phases = -gamma*E(x) mod 2pi  ->  node weights
      (relative), edge/non-edge classification, argmin state.
  3-qubit (subproblem, p=3):
      IDEAL noiseless distribution of the reduced circuit in clbit order
      (exact, no fitting): which single-assignment state the circuit was
      designed to concentrate on, to compare with the hardware counts.
      A Hamiltonian fit exists but is DISABLED by default: it was shown
      unreliable on synthetic data (5 Sep 2026). FIT_SUBPROBLEMS=1 enables it.
  1-qubit: recorded as trivial.

Bitstring convention: same as get_counts() (classical bit n-1 leftmost), so the
argmin printed here is directly comparable to the JSON's "01010001000".

Writes extracted_hamiltonians.json and prints a summary. Nothing is modified
on the IBM side.
"""
import json, sys, os, itertools
import numpy as np
from scipy.optimize import least_squares

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, Operator
from qiskit_ibm_runtime import QiskitRuntimeService

OUT = "extracted_hamiltonians.json"


# ----------------------------------------------------------------------------
def reduce_circuit(circ):
    """Keep only qubits that carry gates; return (reduced circuit without
    measurements, list mapping clbit index -> reduced qubit index)."""
    active = []
    meas_map = {}
    for inst in circ.data:
        name = inst.operation.name
        qs = [circ.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            c = circ.find_bit(inst.clbits[0]).index
            meas_map[c] = qs[0]
            continue
        for q in qs:
            if q not in active:
                active.append(q)
    active = sorted(active)
    idx = {q: i for i, q in enumerate(active)}
    red = QuantumCircuit(len(active))
    for inst in circ.data:
        name = inst.operation.name
        if name in ("barrier", "measure"):
            continue
        qs = [idx[circ.find_bit(q).index] for q in inst.qubits]
        red.append(inst.operation, qs)
    n_cl = len(meas_map)
    cl2red = [idx[meas_map[c]] for c in range(n_cl)]
    red.phys2red = idx
    return red, cl2red


def permute_to_clbit_order(vec_or_mat, cl2red, n):
    """Reorder a statevector (or unitary, on the output side) so that qubit k
    (little-endian) is the qubit measured into classical bit k."""
    # build permutation of basis indices
    perm = np.zeros(2 ** n, dtype=int)
    for src in range(2 ** n):
        bits = [(src >> r) & 1 for r in range(n)]  # reduced-qubit little endian
        dst = 0
        for k, r in enumerate(cl2red):
            dst |= bits[r] << k
        perm[dst] = src
    if vec_or_mat.ndim == 1:
        return vec_or_mat[perm]
    return vec_or_mat[perm, :]


def bitstr(x, n):
    return format(x, f"0{n}b")  # clbit n-1 leftmost, matches get_counts


# ----------------------------------------------------------------------------
def analyse_monolithic(red, cl2red):
    n = red.num_qubits
    psi = Statevector(red).data
    psi = permute_to_clbit_order(psi, cl2red, n)

    def undo_mixer(beta):
        c, s = np.cos(beta), np.sin(beta)
        rx_dag = np.array([[c, 1j * s], [1j * s, c]])  # Rx(-2beta)
        v = psi.reshape([2] * n)
        for q in range(n):
            v = np.moveaxis(np.tensordot(rx_dag, v, axes=([1], [n - 1 - q])), 0, n - 1 - q)
        return v.reshape(-1)

    # The uniformity condition has TWO exact minima per period, beta0 and beta0 - pi/2 (10 Oct 2026):
    # undoing with the wrong one equals applying X to every qubit and reads the phases of the
    # bit-complemented Hamiltonian C(x-bar) = const + sum_i (w_i - P d_i) x_i + P sum_NE x_i x_j,
    # itself a QUBO on the same non-edges. Both candidates are kept; the reading is selected by
    # membership of the 30-point grid -pi/2 + k pi/29 that the July angles were drawn from
    # (pi/2 is not a multiple of pi/29, so only one candidate can lie on it).
    betas = np.linspace(0, np.pi, 3601)
    spread = np.array([np.std(np.abs(undo_mixer(b))) for b in betas])
    thr = 0.05 * float(spread.max())
    idx = [i for i in range(len(betas)) if spread[i] < thr and spread[i] <= spread[max(i - 1, 0)] and spread[i] <= spread[min(i + 1, len(betas) - 1)]]
    modpi = lambda b: (b + np.pi / 2) % np.pi - np.pi / 2  # noqa
    cands = []
    for i in idx:
        bf = np.linspace(betas[i] - 0.002, betas[i] + 0.002, 401)
        b = float(bf[int(np.argmin([np.std(np.abs(undo_mixer(x))) for x in bf]))])
        if any(abs(modpi(b - c)) < 0.01 for c in cands):
            continue
        cands.append(b)
    grid30 = [-np.pi / 2 + k * np.pi / 29 for k in range(30)]
    gdist = {b: min(abs(modpi(b - g)) for g in grid30) for b in cands}
    beta = min(cands, key=lambda b: gdist[b])
    beta_candidates = [{"beta": float(modpi(b)), "grid30_distance": float(gdist[b]), "selected": b == beta} for b in cands]
    phi = undo_mixer(beta)
    uniformity = float(np.std(np.abs(phi)) / np.mean(np.abs(phi)))

    theta = -np.angle(phi)               # = gamma*E(x) + const  (mod 2pi)
    theta = theta - theta[0]
    single = {i: float(np.angle(np.exp(1j * theta[1 << i]))) for i in range(n)}
    pair = {}
    for i, j in itertools.combinations(range(n), 2):
        d = theta[(1 << i) | (1 << j)] - single[i] - single[j]
        pair[(i, j)] = float(np.angle(np.exp(1j * d)))
    # classify pairs: edges ~0, non-edges ~ gamma*P (common value)
    vals = np.array(list(pair.values()))
    edges, nonedges = [], []
    for (i, j), v in pair.items():
        (edges if abs(v) < 0.05 else nonedges).append((i, j))
    ne_vals = [pair[e] for e in nonedges]
    gammaP = float(np.median(ne_vals)) if ne_vals else 0.0
    gammaP_spread = float(np.std(ne_vals)) if ne_vals else 0.0
    # weights: single-bit phase = gamma*(-w_i)  -> w_i proportional to -single
    gw = {i: -single[i] for i in range(n)}
    wmax = max(abs(v) for v in gw.values())
    w_rel = {i: gw[i] / wmax for i in range(n)}
    # argmin over valid (non-empty) states using recovered structure
    best, bestE = None, None
    valid = 0
    for x in range(1, 2 ** n):
        bits = [(x >> i) & 1 for i in range(n)]
        if any(bits[i] and bits[j] for (i, j) in nonedges):
            continue
        valid += 1
        E = -sum(gw[i] * bits[i] for i in range(n))
        if bestE is None or E < bestE:
            best, bestE = x, E
    return {"beta": float(beta), "uniformity": uniformity, "beta_candidates": beta_candidates,
            "n_edges": len(edges), "n_nonedges": len(nonedges),
            "edges": edges, "nonedges": nonedges,
            "gammaP": gammaP, "gammaP_spread": gammaP_spread,
            "gamma_w": gw, "w_relative": w_rel,
            "valid_nonempty_cliques": valid,
            "argmin_bits": bitstr(best, n), "argmin_gamma_weight": float(-bestE)}


# ----------------------------------------------------------------------------
def analyse_subproblem(red, cl2red, in_order=None):
    """Two-stage fit of prod_k Rx(2 beta_k) D(gamma_k) . H^3, E = -sum w x + P sum x x.
    Stage 1: fit the |0> column (cheap, non-unique) many times.
    Stage 2: refine each stage-1 solution against the FULL 8x8 unitary.
    in_order: reduced-qubit index of virtual qubit k at the input (from the
    circuit's initial layout); if None, all 6 permutations are tried."""
    n = red.num_qubits
    U_full = Operator(red).data
    H1 = np.array([[1, 1], [1, -1]]) / np.sqrt(2)

    def kron_all(ms):
        out = np.array([[1.0 + 0j]])
        for m in ms:
            out = np.kron(m, out)
        return out

    Hn = kron_all([H1] * n)
    xs = np.array([[(x >> i) & 1 for i in range(n)] for x in range(2 ** n)])
    pairs = list(itertools.combinations(range(n), 2))
    quad = sum(xs[:, i] * xs[:, j] for i, j in pairs)

    def model(p):
        w = p[0:n]; P = p[n]; g = p[n + 1:n + 4]; b = p[n + 4:n + 7]
        E = -xs @ w + P * quad
        M = Hn.copy()
        for k in range(3):
            M = np.exp(-1j * g[k] * E)[:, None] * M
            c, s_ = np.cos(b[k]), np.sin(b[k])
            rx = np.array([[c, -1j * s_], [-1j * s_, c]])
            M = kron_all([rx] * n) @ M
        return M

    def perm_matrix(order):
        Pm = np.zeros((2 ** n, 2 ** n))
        for src in range(2 ** n):
            bits = [(src >> r) & 1 for r in range(n)]
            dst = sum(bits[order[k]] << k for k in range(n))
            Pm[dst, src] = 1
        return Pm

    def phased_resid(M, T):
        ph = np.vdot(M.ravel(), T.ravel())
        ph = ph / abs(ph) if abs(ph) > 1e-12 else 1.0
        d = (M * ph - T).ravel()
        return np.concatenate([d.real, d.imag])

    Pout = perm_matrix(cl2red)
    rng = np.random.default_rng(0)
    col_target = (Pout @ U_full)[:, 0]

    # stage 1: column fits
    stage1 = []
    for _ in range(80):
        p0 = np.concatenate([rng.uniform(0.05, 1.0, n), [rng.uniform(0.5, 6)],
                             rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)])
        r = least_squares(lambda p: phased_resid(model(p)[:, 0], col_target), p0, max_nfev=4000)
        if r.cost < 1e-8:
            stage1.append(r.x)
        if len(stage1) >= 12:
            break
    orders = [list(in_order)] if in_order is not None else [list(o) for o in itertools.permutations(range(n))]
    best_all = None
    for order in orders:
        target = Pout @ U_full @ perm_matrix(order).T
        for p0 in stage1[:12] + [np.concatenate([rng.uniform(0.05, 1.0, n), [rng.uniform(0.5, 6)],
                                                rng.uniform(-2, 2, 3), rng.uniform(-2, 2, 3)]) for _ in range(5)]:
            r = least_squares(lambda p: phased_resid(model(p), target), p0, max_nfev=2000)
            if best_all is None or r.cost < best_all[0].cost:
                best_all = (r, order)
            if best_all[0].cost < 1e-14:
                break
        if best_all[0].cost < 1e-14:
            break
    best, order = best_all
    p = best.x.copy()
    w = p[0:n]
    if np.sum(w) < 0:
        w = -w; p[n] = -p[n]; p[n + 1:n + 4] = -p[n + 1:n + 4]
    return {"fit_residual": float(np.sqrt(2 * best.cost)),
            "w_fitted": [float(v) for v in w], "P_fitted": float(p[n]),
            "gamma": [float(v) for v in p[n + 1:n + 4]], "beta": [float(v) for v in p[n + 4:n + 7]],
            "input_order": order, "n_stage1_solutions": len(stage1),
            "note": "w and P up to the scale absorbed by gamma; ratios meaningful; trust only if residual < 1e-6"}


def noiseless_distribution(red, cl2red):
    """Ideal statevector probabilities of the reduced circuit in clbit order:
    what the circuit was designed to concentrate on."""
    n = red.num_qubits
    psi = permute_to_clbit_order(Statevector(red).data, cl2red, n)
    pr = np.abs(psi) ** 2
    probs = {bitstr(x, n): float(pr[x]) for x in range(2 ** n)}
    singles = {bitstr(1 << k, n): float(pr[1 << k]) for k in range(n)}
    return {"probs": probs, "single_assignment_probs": singles,
            "most_probable_single": max(singles, key=singles.get),
            "valid_fraction_ideal": float(pr[0] + sum(singles.values()))}


def initial_order(circ, cl2red, red_active_map):
    """Reduced-qubit index of virtual qubit k at the INPUT, from the stored
    initial layout. Returns None if the layout is not available."""
    try:
        lay = circ.layout
        init = lay.initial_index_layout(filter_ancillas=True)  # virtual k -> physical
        return [red_active_map[init[k]] for k in range(len(cl2red))]
    except Exception:
        return None


# ----------------------------------------------------------------------------
def main():
    rec = json.load(open("job_records.json"))["jobs"]
    service = QiskitRuntimeService()
    out = {}
    for jid, r in rec.items():
        if "error" in r and "counts" not in r:
            continue
        try:
            job = service.job(jid)
            circ = job.inputs["pubs"][0][0]
        except Exception as e:  # noqa
            out[jid] = {"error": f"no PUB circuit: {e}"}
            continue
        red, cl2red = reduce_circuit(circ)
        n = red.num_qubits
        entry = {"backend": r.get("backend"), "created": r.get("created"),
                 "n_qubits": n, "depth": red.depth(), "ops": dict(red.count_ops())}
        try:
            if n == 11:
                entry["monolithic"] = analyse_monolithic(red, cl2red)
            elif n == 3:
                entry["ideal"] = noiseless_distribution(red, cl2red)
                if os.environ.get("FIT_SUBPROBLEMS") == "1":
                    io = initial_order(circ, cl2red, red.phys2red)
                    entry["subproblem"] = analyse_subproblem(red, cl2red, io)
                else:
                    entry["subproblem"] = {"note": "Hamiltonian fit disabled: the nonlinear fit was shown unreliable on synthetic data (5 Sep 2026). Set FIT_SUBPROBLEMS=1 to attempt it; trust only residual < 1e-6."}
            else:
                entry["trivial"] = True
        except Exception as e:  # noqa
            entry["error"] = f"analysis failed: {e}"
        out[jid] = entry
        print(jid, entry.get("backend"), f"n={n}",
              entry.get("monolithic", {}).get("argmin_bits", ""),
              entry.get("monolithic", {}).get("n_edges", ""),
              entry.get("monolithic", {}).get("n_nonedges", ""),
              entry.get("ideal", {}).get("most_probable_single", ""),
              {k: round(v, 3) for k, v in entry.get("ideal", {}).get("single_assignment_probs", {}).items()},
              [round(v, 3) for v in entry.get("subproblem", {}).get("w_fitted", [])],
              entry.get("subproblem", {}).get("fit_residual", ""),
              entry.get("error", ""))
    json.dump(out, open(OUT, "w"), indent=1)
    print("written", OUT)


if __name__ == "__main__":
    main()
