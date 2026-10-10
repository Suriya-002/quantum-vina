"""
valid_subspace_test.py  --  is the optimum sampled more or less often than uniform
over the feasible set?  (5 Sep 2026)

Inputs: job_records.json (raw hardware counts, clbit order) and
        extracted_hamiltonians.json (edges/non-edges/argmin in clbit index).
Optional: if qiskit_ibm_runtime is importable and the PUBs are retrievable, the
ideal noiseless distribution of each circuit is computed for comparison.

Per monolithic job it reports
  valid_shots, |V| (non-empty valid cliques), uniform_per_state = valid_shots/|V|,
  observed x* count, ratio = observed/uniform, and for the ideal circuit the same,
  plus P(never observe x*) under uniform-within-valid sampling.
Everything is in clbit order end to end; no node-order step is involved.
"""
import json, math, sys
import numpy as np

rec = json.load(open("job_records.json"))["jobs"]
ext = json.load(open("extracted_hamiltonians.json"))

try:
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit.quantum_info import Statevector
    sys.path.insert(0, ".")
    from extract_hamiltonians import reduce_circuit, permute_to_clbit_order
    service = QiskitRuntimeService()
    HAVE_IDEAL = True
except Exception as e:  # noqa
    print("ideal distributions unavailable:", e)
    HAVE_IDEAL = False


def bits_of(key, n):
    """clbit k value from a get_counts key (clbit n-1 leftmost)."""
    return [int(key[n - 1 - k]) for k in range(n)]


def valid_set(n, nonedges):
    V = []
    for x in range(1, 2 ** n):
        b = [(x >> k) & 1 for k in range(n)]
        if not any(b[i] and b[j] for i, j in nonedges):
            V.append(x)
    return V


rows = []
for jid, r in rec.items():
    if r.get("n_qubits") != 11 or "counts" not in r:
        continue
    m = ext[jid]["monolithic"]
    n = 11
    nonedges = [tuple(e) for e in m["nonedges"]]
    V = valid_set(n, nonedges); Vset = set(V)
    xstar_key = m["argmin_bits"]                       # clbit-order string
    xstar_int = int(xstar_key, 2)
    counts = r["counts"]; total = sum(counts.values())
    valid_shots = sum(c for k, c in counts.items() if int(k, 2) in Vset)
    obs = counts.get(xstar_key, 0)
    uniform = valid_shots / len(V)
    p_never = (1 - 1 / len(V)) ** valid_shots
    row = {"job": jid, "backend": r["backend"], "created": r["created"][:19],
           "shots": total, "valid_shots": valid_shots, "valid_frac": valid_shots / total,
           "n_valid_states": len(V), "uniform_per_state": uniform,
           "xstar_obs": obs, "obs_over_uniform": obs / uniform if uniform else None,
           "p_never_xstar_uniform": p_never,
           "top_valid_state": max((k for k in counts if int(k, 2) in Vset), key=counts.get),
           "top_valid_count": max(c for k, c in counts.items() if int(k, 2) in Vset)}
    if HAVE_IDEAL:
        try:
            circ = service.job(jid).inputs["pubs"][0][0]
            red, cl2red = reduce_circuit(circ)
            psi = permute_to_clbit_order(Statevector(red).data, cl2red, n)
            pr = np.abs(psi) ** 2
            pv = sum(pr[x] for x in V)
            row.update({"ideal_valid_frac": float(pv), "ideal_xstar_prob": float(pr[xstar_int]),
                        "ideal_xstar_over_uniform": float(pr[xstar_int] / (pv / len(V))),
                        "ideal_top_state": format(int(np.argmax(pr)), f"0{n}b"),
                        "ideal_top_state_valid": bool(int(np.argmax(pr)) in Vset)})
        except Exception as e:  # noqa
            row["ideal_error"] = str(e)
    rows.append(row)

json.dump(rows, open("valid_subspace_test.json", "w"), indent=1)
print(f"{'backend':14s} {'valid':>6s} {'|V|':>4s} {'unif':>6s} {'x*obs':>6s} {'ratio':>6s} {'P(never)':>9s}  ideal: valid  x*prob  ratio  top_valid?")
for row in rows:
    ideal = (f"{row.get('ideal_valid_frac', float('nan')):.3f}  {row.get('ideal_xstar_prob', float('nan')):.4f}  "
             f"{row.get('ideal_xstar_over_uniform', float('nan')):.2f}  {row.get('ideal_top_state_valid', '?')}")
    print(f"{row['backend']:14s} {row['valid_shots']:6d} {row['n_valid_states']:4d} {row['uniform_per_state']:6.1f} "
          f"{row['xstar_obs']:6d} {row['obs_over_uniform']:6.2f} {row['p_never_xstar_uniform']:9.2e}  {ideal}")
print("written valid_subspace_test.json")
