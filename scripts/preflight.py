"""
preflight.py  --  Quantum-Vina consistency check  (10 Oct 2026)

Run from Downloads in qpyenv:
    python preflight.py                 full check (needs IBM access, ~5-10 min)
    python preflight.py --offline       no network: files, ledgers, numbers vs paper, TeX, bib
    python preflight.py --repair        also write corrected_penalty_arm_executed.json
    python preflight.py --no-status     online, but skip the 340 job-status lookups
    python preflight.py --tex main.tex --bib references.bib

What it checks
  1 Files     every record the paper depends on is present
  2 Ledgers   340 job IDs over six experiment files, no duplicates, no overwritten ledger
  3 Numbers   tab:hw, tab:corrected, tab:sweep, tab:size, tab:nodes, tab:circuits and the
              Section 3/6/7 prose recomputed from the raw records and matched to the .tex
  4 TeX/bib   citations, references, labels, stale phrases, open TODOs, BibTeX syntax
  5 Online    job status; for every depth-one circuit the executed penalty, graph, weights,
              bit order and angles recovered from the submitted circuit; ideal pair mass of
              every depth-2/3 subproblem circuit; July runtime options; the values that are
              still TODOs in the paper (P51 mode, empty-string masses, 1HSG subs timing)
Output: [PASS] / [FAIL] / [WARN] / [INFO] lines, and preflight_report.json.
ASCII only, so it prints cleanly in PowerShell.
"""
import json, os, re, sys, math, itertools, datetime
from collections import Counter
import numpy as np

A = sys.argv[1:]
OFFLINE = "--offline" in A
REPAIR = "--repair" in A
NO_STATUS = "--no-status" in A


def argval(flag, default=None):
    return A[A.index(flag) + 1] if flag in A and A.index(flag) + 1 < len(A) else default


TEX = argval("--tex") or next((f for f in ("quantum_vina_qst.tex", "main.tex") if os.path.exists(f)), "quantum_vina_qst.tex")
BIB = argval("--bib", "references.bib")

TALLY = Counter()
REPORT = {"generated": datetime.datetime.now().isoformat(timespec="seconds"), "fail": [], "warn": [], "values_to_patch": {}}
WORD = {0: "zero", 1: "one", 2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
MONO = {"2bm2": "d95ti2otcv6s73dj4f7g", "1hsg": "d96imekqp3as739qlehg"}
BACKEND = {"2bm2": "ibm_marrakesh", "1hsg": "ibm_fez"}
SUBNAMES = {"ibm_marrakesh": ["P76", "P79", "P85"], "ibm_fez": ["P139", "P51", "P52"]}
GROUPS = {"2bm2": {"P76": [2, 3, 4], "P79": [5, 6, 7], "P85": [8, 9, 10]},
          "1hsg": {"P139": [1, 2, 3], "P51": [5, 6, 7], "P52": [8, 9, 10]}}
EXEC_CORRECTED = {"gamma": -math.pi + 17 * (2 * math.pi / 39), "beta": -math.pi / 2 + 26 * (math.pi / 39), "grid": "40x40"}
PAIR_IDX = (3, 5, 6)   # 3-qubit basis states with two bits set


def out(level, msg):
    TALLY[level] += 1
    print(f"[{level}] {msg}")
    if level in ("FAIL", "WARN"):
        REPORT[level.lower()].append(msg)


def section(title):
    print("\n" + "=" * 78 + "\n" + title + "\n" + "=" * 78)


def load(fn):
    if not os.path.exists(fn):
        return None
    try:
        with open(fn, encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:  # noqa
        out("FAIL", f"{fn}: unreadable JSON ({e})")
        return None


def sd(x):
    x = np.asarray(x, float)
    return float(np.std(x, ddof=1)) if len(x) > 1 else 0.0


def mean(x):
    return float(np.mean(x))


def pm(vals, d, pct=False):
    """'$m \\pm s$' (or with \\,\\% for percentages) exactly as the tables write it."""
    f = 100.0 if pct else 1.0
    m, s = f * mean(vals), f * sd(vals)
    return f"${m:.{d}f} \\pm {s:.{d}f}\\,\\%$" if pct else f"${m:.{d}f} \\pm {s:.{d}f}$"


def to_utc(s):
    """ISO string with offset (as stored by noise_sim.py / snapshot_medians.py) -> aware UTC datetime, or None."""
    try:
        d = datetime.datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:  # noqa
        return None
    if d.tzinfo is None:
        return None
    return d.astimezone(datetime.timezone.utc)


def utc_label(s):
    """'7 Jul 02:57' as tab:circuits writes it (UTC); '?' when the string has no offset."""
    d = to_utc(s)
    return f"{d.day} {d:%b %H:%M}" if d else "?"


def sci(x, d=3):
    e = int(math.floor(math.log10(abs(x))))
    return f"${x / 10 ** e:.{d}f}\\times10^{{{e}}}$"


# ------------------------------------------------------------------ TeX text
def strip_comments(s):
    keep = []
    for line in s.split("\n"):
        m = re.search(r"(?<!\\)%", line)
        keep.append(line[:m.start()] if m else line)
    return "\n".join(keep)


TEX_RAW = open(TEX, encoding="utf-8").read() if os.path.exists(TEX) else ""
TEXT = strip_comments(TEX_RAW)
IS_PAPER = "\\label{tab:size}" in TEXT and "\\documentclass{iopjournal}" in TEXT
if not IS_PAPER:
    print(f"[FAIL] {TEX!r} is not the Quantum-Vina manuscript (no \\label{{tab:size}} / iopjournal): every number-vs-paper check "
          f"is skipped. Put quantum_vina_qst.tex (and references.bib) in this folder or pass --tex <path> --bib <path>.")


def block(label):
    i = TEXT.find("\\label{" + label + "}")
    if i < 0:
        return ""
    j = TEXT.rfind("\\begin{table}", 0, i)
    k = TEXT.find("\\end{table}", i)
    return TEXT[j:k]


def claim(desc, s, where=None):
    if not IS_PAPER:
        TALLY["SKIP"] += 1
        return False
    hay = TEXT if where is None else where
    if s in hay:
        out("PASS", f"{desc}: {s}")
        return True
    out("FAIL", f"{desc}: not found in the paper -> {s}")
    return False


# ------------------------------------------------------------------ circuit helpers (as in extract_hamiltonians.py)
def reduce_circuit(circ):
    from qiskit import QuantumCircuit
    active, meas = [], {}
    for inst in circ.data:
        name = inst.operation.name
        qs = [circ.find_bit(q).index for q in inst.qubits]
        if name == "barrier":
            continue
        if name == "measure":
            meas[circ.find_bit(inst.clbits[0]).index] = qs[0]
            continue
        for q in qs:
            if q not in active:
                active.append(q)
    active = sorted(active)
    for c, q in meas.items():               # a measured qubit with no gates (idle) still counts
        if q not in active:
            active.append(q)
    active = sorted(active)
    idx = {q: i for i, q in enumerate(active)}
    red = QuantumCircuit(len(active))
    for inst in circ.data:
        if inst.operation.name in ("barrier", "measure"):
            continue
        red.append(inst.operation, [idx[circ.find_bit(q).index] for q in inst.qubits])
    return red, [idx[meas[c]] for c in range(len(meas))]


def probs_clbit_order(circ):
    from qiskit.quantum_info import Statevector
    red, cl2red = reduce_circuit(circ)
    n = red.num_qubits
    psi = Statevector(red).data
    perm = np.zeros(2 ** n, dtype=int)
    for src in range(2 ** n):
        bits = [(src >> r) & 1 for r in range(n)]
        perm[sum(bits[cl2red[k]] << k for k in range(n))] = src
    return psi[perm], n


GRIDS = (30, 32, 40)   # beta grids used in this paper: July/sweep/corrected subs 30, size study 32, corrected monolithic 40


def grid_distance(beta):
    """Distance of beta (mod pi) to the nearest point of each linspace(-pi/2, pi/2, G) grid."""
    return {G: min(abs(modpi(beta - (-math.pi / 2 + k * math.pi / (G - 1)))) for k in range(G)) for G in GRIDS}


def extract_p1(circ, b_exp=None):
    """Depth-one QAOA circuit -> mixer angle, single-node phases gamma*w_i, pair phases.

    The uniformity criterion has TWO exact minima per period, beta0 and beta0 - pi/2: undoing the mixer
    with the wrong one is the same as applying X to every qubit, which reads the phases of the
    bit-complemented Hamiltonian C(x-bar) = const + sum_i (w_i - P d_i) x_i + P sum_NE x_i x_j (itself
    a QUBO with the same non-edges). Both candidates are returned; the reading is selected by the
    declared beta when one is given (the alternative is reported), and the distance of each candidate
    to the beta grids used in this paper is reported, since only one of the two can lie on a grid
    (pi/2 is not a multiple of pi/(G-1) for G = 30, 32, 40)."""
    psi, n = probs_clbit_order(circ)

    def undo(beta):
        c, s = np.cos(beta), np.sin(beta)
        rx_dag = np.array([[c, 1j * s], [1j * s, c]])
        v = psi.reshape([2] * n)
        for q in range(n):
            v = np.moveaxis(np.tensordot(rx_dag, v, axes=([1], [n - 1 - q])), 0, n - 1 - q)
        return v.reshape(-1)

    grid = np.linspace(0, np.pi, 1801)
    spread = np.array([np.std(np.abs(undo(b))) for b in grid])
    thr = 0.05 * float(spread.max())
    idx = [i for i in range(len(grid)) if spread[i] < thr and spread[i] <= spread[max(i - 1, 0)] and spread[i] <= spread[min(i + 1, len(grid) - 1)]]
    wrap = lambda x: float(np.angle(np.exp(1j * x)))  # noqa
    cands = []
    for i in idx:
        fine = np.linspace(grid[i] - 0.002, grid[i] + 0.002, 401)
        beta = float(fine[int(np.argmin([np.std(np.abs(undo(b))) for b in fine]))])
        if any(abs(modpi(beta - c["beta"])) < 0.01 for c in cands):
            continue
        phi = undo(beta)
        unif = float(np.std(np.abs(phi)) / np.mean(np.abs(phi)))
        theta = -np.angle(phi)
        theta = theta - theta[0]
        single = np.array([wrap(theta[1 << i]) for i in range(n)])
        pair = {(a, b): wrap(theta[(1 << a) | (1 << b)] - single[a] - single[b]) for a, b in itertools.combinations(range(n), 2)}
        cands.append({"n": n, "beta": modpi(beta), "uniformity": unif, "gw": -single, "pair": pair, "grid_distance": grid_distance(beta)})
    if not cands:
        raise RuntimeError("no uniformity minimum found (not a depth-one QAOA circuit?)")
    if b_exp is not None:
        sel = min(cands, key=lambda c: abs(modpi(c["beta"] - b_exp)))
    else:
        sel = min(cands, key=lambda c: min(c["grid_distance"].values()))
    sel = dict(sel)
    sel["candidates"] = cands
    return sel


def best_valid(w, nonedges, n):
    ne = set(nonedges)
    best = None
    for x in range(1, 2 ** n):
        b = [(x >> k) & 1 for k in range(n)]
        if any(b[i] and b[j] for i, j in ne):
            continue
        v = float(np.dot(w, b))
        if best is None or v > best[0]:
            best = (v, format(x, f"0{n}b"))
    return best[1]


def modpi(b):
    b = (b + math.pi / 2) % math.pi - math.pi / 2
    return b


# ================================================================== 1 FILES
section("1  FILES")
FILES = [("job_records.json", "July ledger + raw counts", True),
         ("extracted_hamiltonians.json", "executed instances from the July circuits", True),
         ("valid_subspace_test.json", "July recomputed statistics incl. ideal columns", True),
         ("corrected_penalty_results.json", "corrected-arm analysis (holds the 25 corrected job IDs)", True),
         ("corrected_penalty_arm.json", "corrected-arm build file", False),
         ("corrected_penalty_arm_1hsg-subs.json", "1HSG corrected p=3 ledger (15 jobs)", True),
         ("depth_penalty_sweep.json", "depth-sweep ledger (90 jobs)", True),
         ("size_test.json", "2BM2 size-chain ledger (70 jobs)", True),
         ("size_test_1hsg.json", "1HSG size-chain ledger (80 jobs)", True),
         ("noise_sim.json", "noise companions + execution timestamps", True),
         ("snapshot_medians.json", "calibration medians (tab:circuits)", True),
         ("analysis_depth_penalty_sweep.json", "sweep analysis", True),
         ("analysis_size_test.json", "2BM2 size analysis", True),
         ("analysis_corrected_penalty_arm_1hsg-subs.json", "1HSG corrected p=3 analysis", True),
         ("analysis_size_test_1hsg.json", "1HSG size analysis (80 jobs, executed Oct 2026)", True),
         (TEX, "paper", True), (BIB, "bibliography", True)]
for fn, what, needed in FILES:
    if os.path.exists(fn):
        ts = datetime.datetime.fromtimestamp(os.path.getmtime(fn)).strftime("%d %b %H:%M")
        out("PASS", f"{fn:48s} {what}  (modified {ts})")
    else:
        out("FAIL" if needed else "WARN", f"{fn:48s} MISSING - {what}")
for fn in ("size_test_2bm2.json", "size_test_1hsg_dryrun.json"):
    if os.path.exists(fn):
        out("INFO", f"{fn} present")

jr, ext, vst = load("job_records.json"), load("extracted_hamiltonians.json"), load("valid_subspace_test.json")
cp, cres = load("corrected_penalty_arm.json"), load("corrected_penalty_results.json")
c15, sweep, size2, size1 = (load(f) for f in ("corrected_penalty_arm_1hsg-subs.json", "depth_penalty_sweep.json", "size_test.json", "size_test_1hsg.json"))
ns = load("noise_sim.json") or {}
snap = load("snapshot_medians.json") or {}
an_sweep, an_size, an_c15, an_size1 = (load(f) for f in ("analysis_depth_penalty_sweep.json", "analysis_size_test.json", "analysis_corrected_penalty_arm_1hsg-subs.json", "analysis_size_test_1hsg.json"))

# ================================================================== 2 LEDGERS
section("2  JOB LEDGERS")
ledger = {}
if jr:
    ledger["july"] = list(jr["jobs"].keys())

corrected_rows = None
if cp and cp.get("jobs"):
    corrected_rows = cp["jobs"]
    out("PASS", "corrected_penalty_arm.json holds its job list")
elif os.path.exists("corrected_penalty_arm_executed.json"):
    corrected_rows = load("corrected_penalty_arm_executed.json").get("jobs")
    out("PASS", "corrected_penalty_arm_executed.json in use (the original build file was overwritten by a dry run on 5 Sep)")
else:
    if cp is not None:
        out("FAIL", "corrected_penalty_arm.json has NO job list: a later --dry-run overwrote it (the dry run wrote to the same "
                    "file). Its monolithic block now describes an unexecuted grid point. The 25 IDs are recovered below; run "
                    "with --repair to write corrected_penalty_arm_executed.json and use that file from now on.")
    if cres:
        corrected_rows = cres.get("monolithic", []) + cres.get("subproblems", [])
        out("INFO", f"recovered {len(corrected_rows)} corrected-arm job IDs from corrected_penalty_results.json")
    elif ns:
        corrected_rows = [{"job_id": k, **v} for k, v in ns.items() if v.get("experiment") == "corrected"]
        out("WARN", f"recovered {len(corrected_rows)} corrected-arm job IDs from noise_sim.json only")
if corrected_rows:
    ledger["corrected"] = [r["job_id"] for r in corrected_rows]
if cp and cp.get("targets", {}).get("2bm2", {}).get("monolithic"):
    g = cp["targets"]["2bm2"]["monolithic"].get("gamma")
    if g is not None and abs(g - EXEC_CORRECTED["gamma"]) > 0.01:
        out("WARN", f"corrected_penalty_arm.json monolithic gamma = {g:.4f}, but the executed circuits used "
                    f"{EXEC_CORRECTED['gamma']:.4f} (40x40 grid): the file describes the unexecuted 30x30 point")

for key, d, fn in (("corrected_1hsg_subs", c15, "corrected_penalty_arm_1hsg-subs.json"), ("sweep", sweep, "depth_penalty_sweep.json"),
                   ("size_2bm2", size2, "size_test.json"), ("size_1hsg", size1, "size_test_1hsg.json")):
    if d is None:
        continue
    if not d.get("jobs"):
        out("FAIL", f"{fn} has no job list (overwritten by a dry run?)")
        continue
    ledger[key] = [j["job_id"] for j in d["jobs"]]

EXPECT = {"july": 60, "corrected": 25, "corrected_1hsg_subs": 15, "sweep": 90, "size_2bm2": 70, "size_1hsg": 80}
for k, n in EXPECT.items():
    got = len(ledger.get(k, []))
    out("PASS" if got == n else "FAIL", f"ledger {k:20s} {got:3d} job IDs (expected {n})")
allids = [j for v in ledger.values() for j in v]
dup = [j for j, c in Counter(allids).items() if c > 1]
out("PASS" if len(allids) == 340 and not dup else "FAIL", f"total {len(allids)} job IDs (paper states 340), duplicates across ledgers: {len(dup)}")

odd = {}
if jr:
    by_n = Counter(v.get("n_qubits") for v in jr["jobs"].values() if "counts" in v)
    out("INFO", f"July records with counts by qubit count: {dict(by_n)}  (expect 11:10, 1:20, 3:30)")
    odd = {k: v for k, v in jr["jobs"].items() if "counts" not in v or v.get("n_qubits") not in (1, 3, 11)}
    out("PASS" if not odd else "FAIL", f"{len(odd)} July records outside the 60 analysed: {sorted(odd)} (the two backend-name entries were removed on 21 Sep)")
    REPORT["values_to_patch"]["unaccounted_july_ids"] = {k: {"backend": v.get("backend"), "created": v.get("created"),
                                                              "n_qubits": v.get("n_qubits"), "error": v.get("error")} for k, v in odd.items()}

executed = set(allids) - set(odd)   # all 340 executed (1HSG size chain ran in October 2026)
missing_ns = sorted(executed - set(ns))
bad_ns = sorted(k for k, v in ns.items() if "counts_noisy" in v and not (v.get("timestamps") or {}).get("running"))
out("PASS" if not missing_ns else "WARN", f"noise_sim.json covers {len(executed) - len(missing_ns)} of {len(executed)} executed jobs"
    + (f"; missing {len(missing_ns)} (rerun the new noise_sim.py): {missing_ns[:5]}{' ...' if len(missing_ns) > 5 else ''}" if missing_ns else ""))
if bad_ns:
    out("FAIL", f"{len(bad_ns)} noise_sim entries have no execution timestamp (recorded before the job ran); delete them and rerun: {bad_ns[:5]}")

if REPAIR and cres:
    fixed = json.loads(json.dumps(cp)) if cp else {"targets": {}}
    fixed["note"] = ("Repaired " + datetime.date.today().isoformat() + " by preflight.py: the original file was overwritten by a later "
                     "dry run. Jobs and ideal values are those of the executed 5 Sep build (from corrected_penalty_results.json); "
                     "monolithic angles are the executed 40x40 grid point.")
    fixed["GRID"] = 40
    keep = ("target", "arm", "run", "job_id", "backend", "last_update", "submitted", "depth", "ops", "final_index_layout")
    fixed["jobs"] = [{k: r[k] for k in keep if k in r} for r in cres.get("monolithic", []) + cres.get("subproblems", [])]
    for t in ("2bm2", "1hsg"):
        mono = next((r for r in cres.get("monolithic", []) if r.get("target") == t), None)
        if mono and t in fixed.get("targets", {}):
            fixed["targets"][t]["monolithic"] = {**EXEC_CORRECTED, **{k: mono[k] for k in ("ideal_valid_frac", "ideal_xstar_prob",
                                                  "ideal_xstar_over_uniform", "ideal_top_state") if k in mono},
                                                  "ideal_top_state_valid": mono.get("ideal_top_state", "0" * 11) != "0" * 11}
        subs = {}
        for r in cres.get("subproblems", []):
            if r.get("target") == t:
                subs[r["arm"].split("sub_p3_")[1]] = {"nodes": r.get("bit_to_node"), "weights": r.get("weights"),
                                                     "ground_state_bits": r.get("ground_state_bits"),
                                                     "singles_ideal": r.get("ideal_singles"), "pairs_ideal": r.get("ideal_pairs")}
        if t in fixed.get("targets", {}):
            fixed["targets"][t]["subproblems"] = subs
    json.dump(fixed, open("corrected_penalty_arm_executed.json", "w"), indent=1)
    out("INFO", f"wrote corrected_penalty_arm_executed.json ({len(fixed['jobs'])} jobs). Use it with analyse_arms.py.")

# ================================================================== 3 NUMBERS vs PAPER
section("3  NUMBERS: recomputed from raw records and matched to the paper")
inst = {}
if ext:
    for t, jid in MONO.items():
        m = ext[jid]["monolithic"]
        w = np.array([m["w_relative"][str(i)] for i in range(11)])
        gw = np.array([m["gamma_w"][str(i)] for i in range(11)])
        inst[t] = {"ne": [tuple(e) for e in m["nonedges"]], "edges": [tuple(e) for e in m["edges"]], "w": w, "xs": m["argmin_bits"],
                   "nvalid": m["valid_nonempty_cliques"], "n_edges": m["n_edges"], "n_ne": m["n_nonedges"],
                   "ratio": m["gammaP"] / gw[int(np.argmax(np.abs(gw)))], "beta": m["beta"]}
        same = [k for k, v in ext.items() if v.get("backend") == BACKEND[t] and v.get("n_qubits") == 11]
        consistent = all(ext[k]["monolithic"]["argmin_bits"] == m["argmin_bits"] and ext[k]["monolithic"]["beta"] == m["beta"] for k in same)
        out("PASS" if consistent else "FAIL", f"{t}: all {len(same)} July monolithic circuits carry the identical instance and angles")

    I2, I1 = inst["2bm2"], inst["1hsg"]
    claim("2BM2 graph", f"{I2['n_edges']} edges and {I2['n_ne']} non-edges")
    claim("1HSG graph", f"{I1['n_edges']} edges and {I1['n_ne']} non-edges")
    hw_block = block("tab:hw")
    for t, I in inst.items():
        claim(f"{t} feasible cliques", f"{I['nvalid']} / 2048", hw_block)
        claim(f"{t} optimum, node order", I["xs"][::-1], hw_block)
        nodes = sorted(10 - i for i, c in enumerate(I["xs"]) if c == "1")
        claim(f"{t} optimum node set", "$\\{" + ",".join(map(str, nodes)) + "\\}$")
        out("PASS" if abs(I["ratio"] - 0.5) < 1e-6 else "FAIL", f"{t}: executed P / w_max = {I['ratio']:.7f} (paper: 0.5)")
    claim("penalty ratio in Section 6", f"= {inst['2bm2']['ratio']:.4f}$")
    bmod = modpi(inst["2bm2"]["beta"])
    gridpt = -math.pi / 2 + 11 * (math.pi / 29)
    mant, ex = f"{abs(bmod - gridpt):.1e}".split("e")
    claim("beta vs 30-point grid", f"${mant}\\times10^{{{int(ex)}}}$")
    nb = block("tab:nodes")
    rows = re.findall(r"^\s*(\d+) & ([^&]+) & ([0-9.]+) & ([^&]+) & ([0-9.]+) \\\\", nb, re.M)
    if len(rows) == 11:
        bad = [r[0] for r in rows if f"{inst['2bm2']['w'][int(r[0])]:.3f}" != r[2] or f"{inst['1hsg']['w'][int(r[0])]:.3f}" != r[4]]
        out("PASS" if not bad else "FAIL", "tab:nodes relative weights match the executed circuits" + (f" except nodes {bad}" if bad else ""))
    else:
        out("FAIL", f"tab:nodes: parsed {len(rows)} rows, expected 11")

# ---- tab:hw from raw counts
if jr and inst:
    stats = {}
    for t in ("2bm2", "1hsg"):
        I = inst[t]
        V = set(int(best, 2) for best in [format(x, "011b") for x in range(1, 2048)
                                          if not any(((x >> i) & 1) and ((x >> j) & 1) for i, j in I["ne"])])
        runs = sorted([k for k, v in jr["jobs"].items() if v.get("n_qubits") == 11 and v.get("backend") == BACKEND[t] and "counts" in v],
                      key=lambda k: jr["jobs"][k]["created"])
        vs, xs, rho, vf = [], [], [], []
        for k in runs:
            c = jr["jobs"][k]["counts"]
            tot = sum(c.values())
            v = sum(n for s, n in c.items() if int(s, 2) in V)
            x = c.get(I["xs"], 0)
            vs.append(v); xs.append(x); rho.append(x / (v / len(V))); vf.append(v / tot)
        stats[t] = {"V": len(V), "vs": vs, "xs": xs, "rho": rho, "vf": vf}
    hw = block("tab:hw")
    S2, S1 = stats["2bm2"], stats["1hsg"]
    for t, S in (("2bm2", S2), ("1hsg", S1)):
        claim(f"{t} per-run valid shots", ", ".join(map(str, S["vs"])), hw)
        claim(f"{t} per-run optimum shots", ", ".join(map(str, S["xs"])), hw)
        claim(f"{t} per-run rho", ", ".join(f"{r:.2f}" for r in S["rho"]), hw)
        claim(f"{t} valid fraction", pm(S["vf"], 1, pct=True), hw)
        claim(f"{t} Pr(x*)", pm([x / 4096 for x in S["xs"]], 2, pct=True), hw)
    claim("s.d. row", f"Sample s.d. of $\\mathbf{{x}}^{{\\ast}}$ shots & {sd(S2['xs']):.2f} & {sd(S1['xs']):.2f}", hw)
    claim("Poisson row", f"Poisson $\\sqrt{{\\text{{mean}}}}$ & {math.sqrt(mean(S2['xs'])):.2f} & {math.sqrt(mean(S1['xs'])):.2f}", hw)
    claim("7.1 rho", f"$\\rho = {mean(S2['rho']):.2f} \\pm {sd(S2['rho']):.2f}$ for 2BM2 and ${mean(S1['rho']):.2f} \\pm {sd(S1['rho']):.2f}$ for 1HSG")
    claim("7.1 scatter", f"sample standard deviations of {sd(S2['xs']):.2f} and {sd(S1['xs']):.2f} shots against Poisson expectations of "
                         f"{math.sqrt(mean(S2['xs'])):.2f} and {math.sqrt(mean(S1['xs'])):.2f}")
    claim("7.1 valid fraction", f"The valid fraction is {100 * mean(S2['vf']):.1f}\\% for 2BM2 and {100 * mean(S1['vf']):.1f}\\% for 1HSG")
    claim("7.1 feasible shots", f"on average {mean(S2['vs']):.0f} and {mean(S1['vs']):.0f} feasible shots per run")
    claim("7.1 optimum shots", f"({mean(S2['xs']):.0f} and {mean(S1['xs']):.0f} per run on average)")
    def e1(p):
        m_, e_ = f"{p:.0e}".split("e")
        return f"{m_}\\times10^{{{int(e_)}}}"
    for t, S in (("2bm2", S2), ("1hsg", S1)):
        p = (1 - 1 / S["V"]) ** mean(S["vs"])
        claim(f"7.1 {t} never-observe at the mean", f"(1-1/{S['V']})^{{{mean(S['vs']):.0f}}} \\approx {e1(p)}$")
        per = [(1 - 1 / S["V"]) ** v for v in S["vs"]]
        claim(f"tab:hw {t} never-observe range", f"${e1(min(per))}$ to ${e1(max(per))}$", hw)
    claim("7.1 worst run", f"at most ${e1(max((1 - 1 / S['V']) ** v for S in (S2, S1) for v in S['vs']))}$ in any of the ten runs")
    if vst:
        iv = {r["backend"]: r for r in vst}
        r2, r1 = iv.get("ibm_marrakesh", {}), iv.get("ibm_fez", {})
        if r2.get("ideal_valid_frac") is not None:
            claim("7.1 ideal vs measured 2BM2", f"{100 * r2['ideal_valid_frac']:.1f}\\% ideal, {100 * mean(S2['vf']):.1f}\\% measured")
            claim("7.1 ideal vs measured 1HSG", f"{100 * r1['ideal_valid_frac']:.1f}\\% ideal, {100 * mean(S1['vf']):.1f}\\% measured")
            claim("7.1 ideal rho 2BM2", f"${r2['ideal_xstar_over_uniform']:.2f}\\times$ the uniform reference")
        else:
            out("WARN", "valid_subspace_test.json has no ideal columns (rerun valid_subspace_test.py in qpyenv)")

    # ---- original-arm subproblems
    orig = {}
    for b, names in SUBNAMES.items():
        ids = sorted([k for k, v in jr["jobs"].items() if v.get("n_qubits") == 3 and v.get("backend") == b and "counts" in v],
                     key=lambda k: jr["jobs"][k]["created"])
        for i, nm in enumerate(names):
            chunk = ids[5 * i: 5 * i + 5]
            pairs = [sum(n for s, n in jr["jobs"][k]["counts"].items() if s.count("1") == 2) / sum(jr["jobs"][k]["counts"].values()) for k in chunk]
            ideal, infeas, top = None, None, None
            if ext and chunk and "ideal" in ext.get(chunk[0], {}):
                pr = ext[chunk[0]]["ideal"]["probs"]
                ideal = sum(pr[s] for s in ("011", "101", "110"))
                infeas = ideal + pr["111"]
                top = max(pr, key=pr.get)
            orig[nm] = {"hw": pairs, "ideal": ideal, "infeas": infeas, "top": top}
    cb = block("tab:corrected")
    if all(orig[n]["top"] is not None for n in orig):
        for nm in SUBNAMES["ibm_marrakesh"]:
            out("PASS" if orig[nm]["top"].count("1") == 2 else "FAIL", f"July {nm}: ideal mode {orig[nm]['top']} (paper: two-contact string)")
        kind = {0: "the empty string", 1: "a single contact", 2: "a two-contact string", 3: "the three-contact string"}
        claim("7.1 1HSG July modes", f"{kind[orig['P52']['top'].count('1')]} for P52, {kind[orig['P139']['top'].count('1')]} for P139 and "
                                     f"{kind[orig['P51']['top'].count('1')]} for P51")
        inf = [orig[n]["infeas"] for n in orig]
        claim("7.1 infeasible mass range", f"between {min(inf):.2f} and {max(inf):.2f} of the ideal probability in all six circuits")
    if all(orig[n]["ideal"] is not None for n in orig):
        claim("orig ideal pair 2BM2", ", ".join(f"{n} {orig[n]['ideal']:.3f}" for n in SUBNAMES["ibm_marrakesh"]), cb)
        claim("orig ideal pair 1HSG", ", ".join(f"{n} {orig[n]['ideal']:.3f}" for n in SUBNAMES["ibm_fez"]), cb)
        claim("7.1 orig ideal 2BM2", "{:.2f}, {:.2f} and {:.2f}".format(*[orig[n]["ideal"] for n in SUBNAMES["ibm_marrakesh"]]))
        claim("7.1 orig ideal 1HSG", "{:.2f}, {:.2f} and {:.2f}".format(*[orig[n]["ideal"] for n in SUBNAMES["ibm_fez"]]))
    claim("orig hw pair 2BM2", ", ".join(f"{n} {pm(orig[n]['hw'], 3)}" for n in SUBNAMES["ibm_marrakesh"]), cb)
    claim("orig hw pair 1HSG", ", ".join(f"{n} ${mean(orig[n]['hw']):.3f}$" for n in SUBNAMES["ibm_fez"]), cb)
    claim("7.1 orig hw 2BM2", "{:.2f}, {:.2f} and {:.2f}".format(*[mean(orig[n]["hw"]) for n in SUBNAMES["ibm_marrakesh"]]))
    claim("7.1 orig hw 1HSG", "{:.2f}, {:.2f} and {:.2f}".format(*[mean(orig[n]["hw"]) for n in SUBNAMES["ibm_fez"]]))

# ---- corrected arm
if cres:
    cb = block("tab:corrected")
    mono = {t: [r for r in cres.get("monolithic", []) if r.get("target") == t] for t in ("2bm2", "1hsg")}
    if all(len(v) == 5 for v in mono.values()):
        m2, m1 = mono["2bm2"], mono["1hsg"]
        claim("corrected depth/cz", f"Transpiled depth / \\texttt{{cz}} & {m2[0]['depth']} / {m2[0]['ops'].get('cz')} & {m1[0]['depth']} / {m1[0]['ops'].get('cz')} \\\\", cb)
        claim("corrected ideal valid", f"Ideal valid fraction & {100 * m2[0]['ideal_valid_frac']:.1f}\\,\\% & {100 * m1[0]['ideal_valid_frac']:.1f}\\,\\% \\\\", cb)
        claim("corrected hw valid", f"Hardware valid fraction & {pm([r['valid_frac'] for r in m2], 1, True)} & {pm([r['valid_frac'] for r in m1], 1, True)} \\\\", cb)
        claim("corrected ideal rho", f"Ideal $\\rho$ & {m2[0]['ideal_xstar_over_uniform']:.2f} & {m1[0]['ideal_xstar_over_uniform']:.2f} \\\\", cb)
        claim("corrected hw rho", f"Hardware $\\rho$ & {pm([r['xstar_over_uniform'] for r in m2], 2)} & {pm([r['xstar_over_uniform'] for r in m1], 2)} \\\\", cb)
        e2 = sum(r["top_state"] == "0" * 11 for r in m2)
        e1 = sum(r["top_state"] == "0" * 11 for r in m1)
        for tname, mm in (("2BM2", m2), ("1HSG", m1)):
            out("INFO", f"corrected {tname} monolithic per-run top states: {[r['top_state'] for r in mm]} (empty = {'0' * 11})")
        other2 = [sorted(10 - i for i, ch in enumerate(r["top_state"]) if ch == "1") for r in m2 if r["top_state"] != "0" * 11]
        if sorted(other2, key=len) == [[4], [3, 4]]:
            claim("7.2 other 2BM2 modes", "a single contact (node 4) and an infeasible two-contact string (nodes 3 and 4, both P76 contacts)")
        elif other2:
            out("FAIL", f"7.2: the non-empty 2BM2 corrected monolithic modes are nodes {other2}; the paper names node 4 and nodes 3+4")
        claim("corrected 2BM2 empty mode", f"empty in {e2} of 5 runs", cb)
        claim("corrected 1HSG empty mode", f"empty in {e1} of 5 runs", cb)
        claim("7.2 empty mode count", f"empty string in {WORD[e2 + e1]} of ten runs")
        claim("7.2 valid rise", f"from {100 * mean(S2['vf']):.1f}\\% to {100 * mean([r['valid_frac'] for r in m2]):.1f}\\% (2BM2) and from "
                                f"{100 * mean(S1['vf']):.1f}\\% to {100 * mean([r['valid_frac'] for r in m1]):.1f}\\% (1HSG)") if jr and inst else None
    else:
        out("FAIL", f"corrected_penalty_results.json: monolithic runs per target {[len(v) for v in mono.values()]} (expected 5, 5)")
    subs = {}
    for r in cres.get("subproblems", []):
        subs.setdefault(r["arm"].split("sub_p3_")[1], []).append(r)
    if set(subs) == {"P76", "P79", "P85"}:
        claim("corrected 2BM2 hw pair", ", ".join(f"{g} {pm([r['pair_mass'] for r in subs[g]], 3)}" for g in ("P76", "P79", "P85")), cb)
        claim("corrected 2BM2 ideal pair", ", ".join(f"{g} {sum(subs[g][0]['ideal_pairs'].values()):.3f}" for g in ("P76", "P79", "P85")), cb)
        gs_hits = {g: sum(r["top_is_ground_state"] for r in subs[g]) for g in subs}
        out("PASS" if gs_hits["P76"] == 5 and gs_hits["P79"] == 5 else "FAIL", f"corrected P76/P79 mode = ground state: {gs_hits['P76']}/5, {gs_hits['P79']}/5")
        p85 = Counter(r["top_state"] for r in subs["P85"])
        top85, n85 = p85.most_common(1)[0]
        node85 = GROUPS["2bm2"]["P85"][2 - top85.index("1")] if top85.count("1") == 1 else None
        out("INFO", f"corrected P85 mode {top85} in {n85}/5 runs -> node {node85}")
        if inst and node85 is not None:
            claim("P85 near-tie weights", f"weights {inst['2bm2']['w'][node85]:.3f} vs {inst['2bm2']['w'][10]:.3f}", cb)
            si = subs["P85"][0]["ideal_singles"]
            claim("P85 near-tie ideal", f"ideal {si[top85]:.3f} vs {si[subs['P85'][0]['ground_state_bits']]:.3f}", cb)
    else:
        out("FAIL", f"corrected_penalty_results.json subproblems: {sorted(subs)}")

if an_c15:
    cb = block("tab:corrected")
    s15 = {}
    for r in an_c15:
        s15.setdefault(r["arm"].split("sub_p3_")[1], []).append(r)
    if set(s15) == {"P139", "P51", "P52"}:
        claim("1HSG corrected hw pair", ", ".join(f"{g} {pm([r['pair_mass'] for r in s15[g]], 3)}" for g in ("P139", "P51", "P52")), cb)
        claim("1HSG corrected ideal pair", ", ".join(f"{g} {s15[g][0]['ideal_pair_mass']:.3f}" for g in ("P139", "P51", "P52")), cb)
        modes = {g: Counter(r["mode"] for r in s15[g]) for g in s15}
        sp15 = c15["targets"]["1hsg"]["subproblems"] if c15 else {}
        gs15 = {g: sp15[g]["ground_state_bits"] for g in s15} if sp15 else {}

        def node_of(g, bits):
            return GROUPS["1hsg"][g][2 - bits.index("1")] if bits.count("1") == 1 else None
        for g in ("P139", "P51", "P52"):
            per_run = [r["mode"] for r in s15[g]]
            top, n = modes[g].most_common(1)[0]
            nd, ngs = node_of(g, top), node_of(g, gs15.get(g, ""))
            si = sp15.get(g, {}).get("singles_ideal") or sp15.get(g, {}).get("ideal_singles") or {}
            wtop = inst["1hsg"]["w"][nd] if inst and nd is not None else float("nan")
            wgs = inst["1hsg"]["w"][ngs] if inst and ngs is not None else float("nan")
            out("INFO", f"1HSG corrected {g}: per-run modes {per_run}; top {top} in {n}/5 -> node {nd} (w {wtop:.3f}); ground state {gs15.get(g)} -> node {ngs} (w {wgs:.3f}); "
                        f"ideal singles {{{', '.join(f'{k}: {v:.3f}' for k, v in sorted(si.items()))}}}")
            REPORT["values_to_patch"][f"1hsg_corrected_{g}_modes"] = {"per_run": per_run, "top": top, "node": nd, "w_top": wtop, "ground_state": gs15.get(g), "w_gs": wgs, "ideal_singles": si}
        top52 = modes["P52"].most_common(1)[0][0]
        out("PASS" if top52 == "010" and modes["P52"]["010"] == 5 else "FAIL", f"1HSG corrected P52: mode {top52} in {modes['P52'][top52]}/5 (paper: second-heaviest option, node 9, in 5 of 5; ground state {gs15.get('P52')})")
        top139 = modes["P139"].most_common(1)[0][0]
        out("PASS" if top139 == gs15.get("P139") and modes["P139"][top139] == 5 else "FAIL", f"1HSG corrected P139: mode {top139} in {modes['P139'][top139]}/5 (paper: ground state in 5 of 5)")
        if inst:
            claim("P52 second option weights", f"weights {inst['1hsg']['w'][9]:.3f} vs {inst['1hsg']['w'][10]:.3f}", cb)
            si52 = sp15.get("P52", {}).get("singles_ideal") or sp15.get("P52", {}).get("ideal_singles") or {}
            if "010" in si52 and "100" in si52:
                claim("P52 second option ideal", f"ideal {si52['010']:.3f} vs {si52['100']:.3f}", cb)
        top51, n51 = modes["P51"].most_common(1)[0]
        node51 = node_of("P51", top51)
        c51 = Counter(node_of("P51", r["mode"]) for r in s15["P51"])
        if len(c51) == 2 and None not in c51:
            (a, na), (b, nb) = c51.most_common(2)
            claim("7.2/8.4 P51 split", f"node {a} in {WORD[na]} runs and node {b} in {WORD[nb]}")
            claim("tab:corrected P51 split", f"node {a} in {na} runs and node {b} in {nb}", cb)
        si51 = sp15.get("P51", {}).get("singles_ideal") or sp15.get("P51", {}).get("ideal_singles") or {}
        if "010" in si51 and "100" in si51 and f"{si51['010']:.3f}" == f"{si51['100']:.3f}":
            claim("8.4 P51 ideal tie", f"({si51['010']:.3f} each)")
        si52 = sp15.get("P52", {}).get("singles_ideal") or sp15.get("P52", {}).get("ideal_singles") or {}
        if "010" in si52 and "100" in si52:
            claim("7.2 P52 ideal gap", f"by only {abs(round(si52['100'], 3) - round(si52['010'], 3)):.3f} in ideal probability ({si52['010']:.3f} against {si52['100']:.3f})")
        REPORT["values_to_patch"]["P51_mode"] = {"bits": top51, "runs": n51, "node": node51, "per_run": [r["mode"] for r in s15["P51"]]}
        out("INFO", f"P51 (exactly degenerate) mode {top51} in {n51}/5 runs -> node {node51}; per-run {[r['mode'] for r in s15['P51']]}   [Section 8 near-tie paragraph]")
        if cres and subs:
            all_means = [mean([r["pair_mass"] for r in subs[g]]) for g in subs] + [mean([r["pair_mass"] for r in s15[g]]) for g in s15]
            ideals = [sum(subs[g][0]["ideal_pairs"].values()) for g in subs] + [s15[g][0]["ideal_pair_mass"] for g in s15]
            m_m = [mean([r["pair_mass"] for r in subs[g]]) for g in subs]
            f_m = [mean([r["pair_mass"] for r in s15[g]]) for g in s15]
            claim("7.2 pair range marrakesh", f"{min(m_m):.2f}--{max(m_m):.2f} on \\texttt{{ibm\\_marrakesh}}")
            claim("7.2 pair range fez", f"{min(f_m):.2f}--{max(f_m):.2f} on \\texttt{{ibm\\_fez}}")
            claim("7.2 ideal range", f"ideals of {min(ideals):.2f}--{max(ideals):.2f}")

# ---- depth sweep
if an_sweep and sweep:
    sb = block("tab:sweep")
    cases = {(c["group"], c["penalty_factor"], c["p"]): c for c in sweep["cases"]}
    grp = {}
    for r in an_sweep:
        grp.setdefault((r["group"], r["penalty_factor"], r["p"]), []).append(r)
    n_ok = 0
    for key in sorted(grp, key=lambda k: (k[0], k[1], k[2])):
        rs, c = grp[key], cases[key]
        g, pf, p = key
        k_single = sum(r["mode_valid"] for r in rs)
        row = (f"{g} & ${pf}\\,w_{{\\max}}$ & {p} & {rs[0]['depth']} / {rs[0]['ops'].get('cz')} & {c['ideal_pair_mass']:.3f} & "
               f"{pm([r['pair_mass'] for r in rs], 3)} & {k_single}/5 \\\\")
        n_ok += claim(f"tab:sweep {g} P={pf} p={p}", row, sb)
    out("INFO", f"tab:sweep: {n_ok}/{len(grp)} rows match")
    n05 = sum(not r["mode_valid"] for r in an_sweep if r["penalty_factor"] == 0.5)
    n11 = sum(r["mode_valid"] for r in an_sweep if r["penalty_factor"] == 1.1)
    claim("7.3 executed-penalty count", f"all {n05} executed-penalty runs")
    claim("7.3 corrected-penalty count", f"all {n11} corrected-penalty runs")
    sep = lambda g, p: mean([r["pair_mass"] for r in grp[(g, 0.5, p)]]) - mean([r["pair_mass"] for r in grp[(g, 1.1, p)]])  # noqa
    claim("7.3 hardware separation", "{:.2f}, {:.2f} and {:.2f} at $p=1$ to {:.2f}, {:.2f} and {:.2f} at $p=3$".format(
        *[sep(g, 1) for g in ("P76", "P79", "P85")], *[sep(g, 3) for g in ("P76", "P79", "P85")]))
    claim("7.3 ideal P76", f"executed-penalty pair mass {cases[('P76', 0.5, 1)]['ideal_pair_mass']:.2f} to {cases[('P76', 0.5, 3)]['ideal_pair_mass']:.2f}, "
                           f"corrected {cases[('P76', 1.1, 1)]['ideal_pair_mass']:.2f} to {cases[('P76', 1.1, 3)]['ideal_pair_mass']:.2f} for P76")

# ---- size study (both targets; the 1HSG block carries two off-chain controls marked with a dagger)
CONTROLS_1HSG = ("P136+P147+P51", "P136+P139+P147+P52")
size_grp = {}
for tname, an_s, d_s in (("2bm2", an_size, size2), ("1hsg", an_size1, size1)):
    if not (an_s and d_s):
        out("FAIL", f"size study {tname}: analysis or ledger missing (run analyse_arms.py on the size ledger)")
        continue
    zb = block("tab:size")
    cases = {(c["union"], c["penalty_factor"]): c for c in d_s["cases"]}
    grp = {}
    for r in an_s:
        grp.setdefault((r["union"], r["penalty_factor"]), []).append(r)
    size_grp[tname] = (grp, cases)
    n_ok = 0
    orphan = [k for k in grp if k not in cases]
    if orphan:
        out("FAIL", f"size study {tname}: analysis rows with no case in the ledger: {orphan}")
    for key in sorted((k for k in grp if k in cases), key=lambda k: (cases[k]["n"], k[1])):
        rs, c = grp[key], cases[key]
        label = "all five groups" if key[0].count("+") == 4 else key[0]
        if tname == "1hsg" and key[0] in CONTROLS_1HSG:
            label += "$^{\\dagger}$"
        k_feas = sum(r["mode_valid"] for r in rs)
        row = (f"{label} & {c['n']} & {key[1]} & {'yes' if c['ground_state_infeasible'] else 'no'} & {c['ideal_valid_frac']:.3f} & "
               f"{pm([r['valid_frac'] for r in rs], 3)} & {pm([r['xstar_over_uniform'] for r in rs], 2)} & {k_feas}/5 ({'yes' if c['ideal_mode_valid'] else 'no'}) \\\\")
        n_ok += claim(f"tab:size {tname} n={c['n']} P={key[1]}", row, zb)
        if len(rs) != 5:
            out("FAIL", f"tab:size {tname} {key}: {len(rs)} runs analysed, expected 5")
    out("INFO", f"tab:size {tname}: {n_ok}/{len(grp)} rows match")

if "2bm2" in size_grp:
    grp, cases = size_grp["2bm2"]
    top = "P38+P75+P76+P79+P85"
    if (top, 0.5) in grp:
        v11 = [r["valid_frac"] for r in grp[(top, 0.5)]]
        claim("7.4 n=11 noise", f"({100 * mean(v11):.1f}\\% against {100 * cases[(top, 0.5)]['ideal_valid_frac']:.1f}\\% ideal)")
        if vst and jr and inst:
            r2 = next(r for r in vst if r["backend"] == "ibm_marrakesh")
            claim("7.4 transfer control ideal", f"{cases[(top, 0.5)]['ideal_valid_frac']:.3f} against {r2['ideal_valid_frac']:.3f} for the July angles")
            claim("7.4 transfer control hw", f"on hardware, {mean(v11):.3f} against {mean(S2['vf']):.3f} and "
                                             f"${mean([r['xstar_over_uniform'] for r in grp[(top, 0.5)]]):.2f}\\times$ against ${mean(S2['rho']):.2f}\\times$")

NEEDED_1HSG = [(u, pf) for u in ("P52", "P136+P147+P51", "P51+P52", "P136+P51+P52", "P136+P139+P147+P52", "P139+P51+P52",
                                 "P136+P139+P51+P52", "P136+P139+P147+P51+P52") for pf in (0.5, 1.1)]
if "1hsg" in size_grp and all(k in size_grp["1hsg"][0] and k in size_grp["1hsg"][1] for k in NEEDED_1HSG):
    grp, cases = size_grp["1hsg"]
    # prose of section 7.4 (1HSG paragraph), recomputed from the analysis file
    def kf(u, pf):
        return sum(r["mode_valid"] for r in grp[(u, pf)])
    small = [u for u in ("P52", "P136+P147+P51", "P51+P52", "P136+P51+P52", "P136+P139+P147+P52")]
    out("PASS" if all(kf(u, 0.5) == 0 for u in small) and all(kf(u, 1.1) == 5 for u in small) else "FAIL",
        "7.4 1HSG: mode infeasible in all 25 executed-penalty runs and feasible in all 25 corrected-penalty runs at <= 8 nodes")
    c5 = grp[("P136+P147+P51", 0.5)]
    out("PASS" if not cases[("P136+P147+P51", 0.5)]["ground_state_infeasible"] and kf("P136+P147+P51", 0.5) == 0 else "FAIL",
        "7.4 1HSG five-node control: ground state feasible under P=0.5 w_max, hardware mode infeasible in 5/5")
    claim("7.4 1HSG five-node rho", f"${mean([r['xstar_over_uniform'] for r in c5]):.2f} \\pm {sd([r['xstar_over_uniform'] for r in c5]):.2f}$ times relative to uniform (noiseless {cases[('P136+P147+P51', 0.5)]['ideal_xstar_over_uniform']:.2f})")
    big = [u for u in ("P139+P51+P52", "P136+P139+P51+P52", "P136+P139+P147+P51+P52")]
    out("PASS" if all(kf(u, 1.1) == 0 for u in big) and all(not cases[(u, 1.1)]["ideal_mode_valid"] for u in big) else "FAIL",
        "7.4 1HSG corrected: mode infeasible in all 15 runs at 9-11 nodes, noiseless agrees")
    out("PASS" if [kf(u, 0.5) for u in big] == [2, 2, 5] else "FAIL",
        f"7.4 1HSG executed: mode feasible in {[kf(u, 0.5) for u in big]} of 5 runs at 9, 10, 11 nodes (paper: two, two, five)")
    claim("7.4 1HSG mode probabilities", "probability {:.3f}, {:.3f} and {:.3f}".format(*[cases[(u, 0.5)]["ideal_mode_prob"] for u in big]))
    c8 = cases[("P136+P139+P147+P52", 1.1)]
    out("PASS" if (not c8["ideal_mode_valid"]) and kf("P136+P139+P147+P52", 1.1) == 5 else "FAIL",
        f"7.4 1HSG eight-node corrected control: noiseless mode infeasible (p={c8['ideal_mode_prob']:.3f}), hardware feasible in {kf('P136+P139+P147+P52', 1.1)}/5")
    claim("7.4 1HSG eight-node mode prob", f"with probability {c8['ideal_mode_prob']:.3f}")
    REPORT["values_to_patch"]["hsg_eight_node_control_modes"] = {
        "ideal_mode": c8.get("ideal_mode"), "hardware_modes": [r["mode"] for r in grp[("P136+P139+P147+P52", 1.1)]],
        "noisy_sim_mode_valid": [r.get("noisy_mode_valid") for r in grp[("P136+P139+P147+P52", 1.1)]]}
    out("INFO", f"7.4 1HSG eight-node corrected control: ideal mode {c8.get('ideal_mode')}, hardware modes {[r['mode'] for r in grp[('P136+P139+P147+P52', 1.1)]]}"
                f", noisy-sim mode valid {[r.get('noisy_mode_valid') for r in grp[('P136+P139+P147+P52', 1.1)]]}  (section 7.4 TODO)")
    dv = {(u, pf): mean([r["valid_frac"] for r in grp[(u, pf)]]) - cases[(u, pf)]["ideal_valid_frac"] for (u, pf) in grp}
    d05 = max(abs(v) for (u, pf), v in dv.items() if pf == 0.5)
    out("PASS" if d05 <= 0.04 else "FAIL", f"7.4 1HSG executed penalty: |hardware - noiseless| valid fraction at most {d05:.3f} (paper: within 0.04)")
    d67 = [dv[(u, 1.1)] for u in ("P51+P52", "P136+P51+P52")]
    d811 = [dv[(u, 1.1)] for u in ("P136+P139+P147+P52", "P139+P51+P52", "P136+P139+P51+P52", "P136+P139+P147+P51+P52")]
    out("PASS" if -0.075 <= min(d67) and max(d67) <= -0.055 else "FAIL", f"7.4 1HSG corrected deficit at 6-7 nodes {[round(x, 3) for x in d67]} (paper: 0.06 to 0.07)")
    out("PASS" if -0.165 <= min(d811) and max(d811) <= -0.105 else "FAIL", f"7.4 1HSG corrected deficit at 8-11 nodes {[round(x, 3) for x in d811]} (paper: 0.11 to 0.16)")
    sdmax = max(sd([r["valid_frac"] for r in rs]) for rs in grp.values())
    claim("fig:size caption 1HSG sd", f"at most 0.010 for 2BM2 and {sdmax:.3f} for 1HSG")
    if "2bm2" in size_grp:
        sd2 = max(sd([r["valid_frac"] for r in rs]) for rs in size_grp["2bm2"][0].values())
        out("PASS" if sd2 <= 0.010 else "FAIL", f"fig:size caption: 2BM2 max valid-fraction sd {sd2:.3f} (paper: at most 0.010)")
    n_noisy = sum(1 for rs in grp.values() for r in rs if "noisy_valid_frac" in r)
    out("PASS" if n_noisy == 80 else "WARN", f"1HSG size analysis carries noisy-sim columns for {n_noisy}/80 runs (rerun noise_sim.py, then analyse_arms.py size_test_1hsg.json)")
    # ---- section 7.5: calibration-based simulation against the 1HSG size chain
    if n_noisy == 80:
        def nm(u, pf, key):
            return mean([r[key] for r in grp[(u, pf)]])
        c8u, top1 = "P136+P139+P147+P52", "P136+P139+P147+P51+P52"
        claim("7.5 1HSG five-node sim vs hw", f"{nm('P136+P147+P51', 0.5, 'noisy_valid_frac'):.3f} against {nm('P136+P147+P51', 0.5, 'valid_frac'):.3f} for the five-node control")
        claim("7.5 1HSG P52 pair sim vs hw", f"{nm('P52', 0.5, 'noisy_pair_mass'):.3f} against {nm('P52', 0.5, 'pair_mass'):.3f} in pair mass")
        claim("7.5 1HSG n=11 corrected sim vs hw", f"({nm(top1, 1.1, 'noisy_valid_frac'):.3f} against {nm(top1, 1.1, 'valid_frac'):.3f} at eleven nodes")
        claim("7.5 1HSG n=8 corrected sim vs hw", f"{nm(c8u, 1.1, 'valid_frac'):.3f}, is 0.08 below the simulated {nm(c8u, 1.1, 'noisy_valid_frac'):.3f}")
        claim("7.4 n=8 empty-string mass", f"{nm(c8u, 1.1, 'empty_frac'):.3f} of its shots fell on the empty string, against {nm(c8u, 1.1, 'noisy_empty_frac'):.3f}")
        claim("7.5 n=8 empty-string mass", f"{nm(c8u, 1.1, 'noisy_empty_frac'):.3f} of its shots on the empty string, the device {nm(c8u, 1.1, 'empty_frac'):.3f}")
        dsim = {k: abs(nm(k[0], k[1], "noisy_valid_frac") - nm(k[0], k[1], "valid_frac")) for k in grp}
        v911 = max(dsim[(u, pf)] for u in big for pf in (0.5, 1.1))
        out("PASS" if v911 <= 0.03 else "FAIL", f"7.5 1HSG 9-11 nodes: |sim - hw| valid fraction at most {v911:.3f} (paper: within 0.03)")
        v1011 = max(dsim[(u, pf)] for u in big[1:] for pf in (0.5, 1.1))
        out("PASS" if v1011 <= 0.015 else "FAIL", f"7.5 1HSG 10-11 nodes: |sim - hw| at most {v1011:.3f} (paper: within 0.015)")
        vsmall = max(dsim[(u, pf)] for u in small for pf in (0.5, 1.1) if not (u == c8u and pf == 1.1))
        out("PASS" if vsmall <= 0.03 else "FAIL", f"7.5 1HSG <= 8 nodes: |sim - hw| at most {vsmall:.3f} excluding the n=8 corrected control (paper: about 0.03)")
        out("PASS" if 0.075 <= dsim[(c8u, 1.1)] <= 0.085 else "FAIL", f"7.5 n=8 corrected control: |sim - hw| = {dsim[(c8u, 1.1)]:.3f} (paper: 0.08)")
        out("PASS" if all(r.get("noisy_mode_valid") is False for r in grp[(c8u, 1.1)]) else "FAIL",
            "7.5 n=8 corrected control: noisy-sim mode infeasible in all five simulated runs")
        hwm = [r["mode"] for r in grp[(c8u, 1.1)]]
        out("PASS" if hwm.count("00010001") == 4 and sum(m.count("1") == 1 for m in hwm) == 1 else "FAIL",
            f"7.4 n=8 corrected control: hardware modes {hwm} (paper: the two-contact clique P136+P147 in four runs, a single contact in one)")
        if an_c15:
            g15 = {}
            for r in an_c15:
                g15.setdefault(r["arm"].split("sub_p3_")[1], []).append(r)
            if all(g in g15 and all("noisy_pair_mass" in r for r in g15[g]) for g in ("P139", "P51", "P52")):
                sims = ", ".join(f"{mean([r['noisy_pair_mass'] for r in g15[g]]):.3f}" for g in ("P139", "P51"))
                sims += f" and {mean([r['noisy_pair_mass'] for r in g15['P52']]):.3f}"
                hws = ", ".join(f"{mean([r['pair_mass'] for r in g15[g]]):.3f}" for g in ("P139", "P51"))
                hws += f" and {mean([r['pair_mass'] for r in g15['P52']]):.3f}"
                claim("7.5 1HSG corrected subs sim vs hw pair", f"the simulated pair masses, {sims}, are within 0.015 of the measured {hws}")
                gap = max(abs(mean([r['noisy_pair_mass'] for r in g15[g]]) - mean([r['pair_mass'] for r in g15[g]])) for g in ("P139", "P51", "P52"))
                out("PASS" if gap <= 0.015 else "FAIL", f"7.5 1HSG corrected subs: |sim - hw| pair mass at most {gap:.3f} (paper: within 0.015)")
            else:
                out("WARN", "1HSG corrected subs analysis lacks noisy-sim columns: rerun analyse_arms.py corrected_penalty_arm_1hsg-subs.json")
elif "1hsg" in size_grp:
    out("FAIL", "1HSG size study: not all 16 cases present in ledger + analysis; section 7.4 prose checks skipped")

# ---- run-to-run scatter against the binomial expectation (sections 7.1 and 8.4)
def scatter_ratio(vals):
    vals = [float(v) for v in vals]
    m = mean(vals)
    b = math.sqrt(max(m * (1 - m), 1e-12) / 4096)
    return sd(vals) / b


SC = {}
if "stats" in globals() and stats.get("2bm2") and stats.get("1hsg"):
    SC["July 2BM2 monolithic"] = scatter_ratio(stats["2bm2"]["vf"])
    SC["July 1HSG monolithic"] = scatter_ratio(stats["1hsg"]["vf"])
    vb2 = math.sqrt(mean(stats["2bm2"]["vf"]) * (1 - mean(stats["2bm2"]["vf"])) / 4096) * 100
    vb1 = math.sqrt(mean(stats["1hsg"]["vf"]) * (1 - mean(stats["1hsg"]["vf"])) / 4096) * 100
    claim("7.1 valid-fraction scatter", f"{100 * sd(stats['2bm2']['vf']):.1f} percentage points against a binomial {vb2:.1f} for 2BM2")
    claim("7.1 valid-fraction scatter 1HSG", f"{100 * sd(stats['1hsg']['vf']):.1f} against {vb1:.1f} for 1HSG")
    vs2 = sorted(stats["2bm2"]["vs"])
    claim("7.1 odd run", f"one run with {vs2[0]} feasible shots against {vs2[1]} to {vs2[-1]} in the other four")
if "size_grp" in globals():
    for tname, label in (("2bm2", "September 2BM2 size chain"), ("1hsg", "October 1HSG size chain")):
        if tname in size_grp:
            SC[label] = max(scatter_ratio([r["valid_frac"] for r in rs]) for rs in size_grp[tname][0].values())
if an_sweep:
    gsw = {}
    for r in an_sweep:
        gsw.setdefault((r["group"], r["penalty_factor"], r["p"]), []).append(r["pair_mass"])
    SC["sweep (pair mass)"] = max(scatter_ratio(v) for v in gsw.values() if len(v) == 5)
if cres and cres.get("monolithic"):
    for tname in ("2bm2", "1hsg"):
        rows_m = [r["valid_frac"] for r in cres["monolithic"] if r.get("target") == tname]
        if len(rows_m) == 5:
            SC[f"corrected {tname} monolithic"] = scatter_ratio(rows_m)
if an_c15:
    g15s = {}
    for r in an_c15:
        g15s.setdefault(r["arm"], []).append(r["valid_frac"])
    SC["corrected 1HSG subproblems"] = max(scatter_ratio(v) for v in g15s.values() if len(v) == 5)
for k, v in SC.items():
    out("INFO", f"five-run scatter / binomial s.d. for {k}: {v:.2f}")
REPORT["values_to_patch"]["scatter_ratios"] = SC
if SC:
    stated = {"September 2BM2 size chain": 1.3, "October 1HSG size chain": 3.6, "July 2BM2 monolithic": 4.0, "corrected 1HSG subproblems": 4.4}
    for k, s in stated.items():
        if k in SC:
            out("PASS" if abs(SC[k] - s) <= 0.15 else "FAIL", f"8.4 scatter ratio {k}: {SC[k]:.2f} (paper: about {s})")
    two = [SC[k] for k in ("sweep (pair mass)", "corrected 2bm2 monolithic", "corrected 1hsg monolithic") if k in SC]
    if two:
        out("PASS" if max(two) <= 2.15 and max(two) >= 1.85 else "FAIL", f"8.4 scatter ratio sweep and corrected monolithic: max {max(two):.2f} (paper: about 2)")

# ---- tab:circuits medians and calibration times (UTC)
if snap and IS_PAPER:
    tb = block("tab:circuits")
    for lab, v in snap.items():
        ok = all(s in tb for s in (sci(v["median_cz_error"]), f"{v['median_T1_us']:.1f}", f"{v['median_T2_us']:.1f}", sci(v["median_readout_error"])))
        cal = utc_label(v["properties_last_update"])
        out("PASS" if ok else "FAIL", f"tab:circuits medians for '{lab}' ({v['backend']}, calibration {cal} UTC)")
        if cal == "?":
            out("WARN", f"snapshot '{lab}': properties_last_update has no timezone offset ({v['properties_last_update']}); rerun snapshot_medians.py")
        else:
            day, hm = cal.rsplit(" ", 1)
            ok_cal = cal in tb or (day in tb and re.search(re.escape(day) + r"[^\\]*" + re.escape(hm), tb) is not None)
            out("PASS" if ok_cal else "FAIL", f"tab:circuits calibration time (UTC) for '{lab}': {cal}")
        ex = v.get("queried_at_running", "")
        if ex:
            d = to_utc(ex)
            if d and f"{d.day} {d:%b}" not in tb:
                out("FAIL", f"tab:circuits execution date for '{lab}' ({d.day} {d:%b}) not in the table")
elif snap:
    out("INFO", "tab:circuits checks skipped (manuscript not found)")

# ---- every five-run set should sit inside one calibration (offline, from noise_sim.json)
if ns:
    sets = {}
    for r in (corrected_rows or []):
        sets.setdefault(("corrected", r.get("target"), r.get("arm")), []).append(r["job_id"])
    for j in ((c15 or {}).get("jobs") or []):
        sets.setdefault(("corrected_1hsg_subs", j.get("target"), j.get("arm")), []).append(j["job_id"])
    for j in ((sweep or {}).get("jobs") or []):
        sets.setdefault(("sweep", j["group"], j["penalty_factor"], j["p"]), []).append(j["job_id"])
    for d, tname in ((size2, "size_2bm2"), (size1, "size_1hsg")):
        for j in ((d or {}).get("jobs") or []):
            sets.setdefault((tname, j["union"], j["penalty_factor"]), []).append(j["job_id"])
    modes = {}
    for an in (an_size, an_size1, an_sweep, an_c15):
        for r in (an or []):
            modes[r["job_id"]] = r
    n_split = 0
    for key, ids in sets.items():
        cals = {}
        for jid in ids:
            cals.setdefault(ns.get(jid, {}).get("properties_last_update", "missing"), []).append(jid)
        if len(cals) > 1:
            n_split += 1
            out("WARN", f"five-run set {key} spans {len(cals)} calibrations: " + "; ".join(f"{utc_label(c)} UTC x{len(v)}" for c, v in cals.items()))
            for jid in ids:
                m = modes.get(jid, {})
                run = ns.get(jid, {}).get("timestamps", {}).get("running", "")
                print(f"         {jid}  run {utc_label(run)} UTC  calibration {utc_label(ns.get(jid, {}).get('properties_last_update', ''))} UTC"
                      + (f"  valid {m['valid_frac']:.3f}  mode {m['mode']} feasible {m['mode_valid']}" if m else ""))
    out("PASS" if n_split == 0 else "WARN", f"five-run sets spanning more than one calibration: {n_split} (the paper must name each)")
    # execution batches of the 1HSG size chain (section 6 / tab:circuits)
    if ledger.get("size_1hsg"):
        bat = {}
        for jid in ledger["size_1hsg"]:
            bat.setdefault(ns.get(jid, {}).get("properties_last_update", "missing"), []).append(ns.get(jid, {}).get("timestamps", {}).get("running", ""))
        for c, runs in sorted(bat.items()):
            runs = sorted(r for r in runs if r)
            out("INFO", f"1HSG size chain: {len(runs)} jobs on calibration {utc_label(c)} UTC, executed {utc_label(runs[0]) if runs else '?'}--{utc_label(runs[-1])[-5:] if runs else '?'} UTC")

# ================================================================== 4 TeX / BIB
section("4  TeX AND BIBLIOGRAPHY")
if TEXT:
    out("PASS" if "\\documentclass{iopjournal}" in TEXT else "FAIL", "document class iopjournal")
    cites = set(k.strip() for grp in re.findall(r"\\cite[pt]?\{([^}]*)\}", TEXT) for k in grp.split(","))
    labels = re.findall(r"\\label\{([^}]*)\}", TEXT)
    refs = set(re.findall(r"\\(?:eq)?ref\{([^}]*)\}", TEXT))
    dl = [l for l, c in Counter(labels).items() if c > 1]
    out("PASS" if not dl else "FAIL", f"duplicate labels: {dl or 'none'}")
    miss = sorted(refs - set(labels))
    out("PASS" if not miss else "FAIL", f"unresolved \\ref/\\eqref: {miss or 'none'}")
    STALE = ["YAP-TEAD", "Hippo", "Krasniqi", "42.1", "41--45", "any sampling distribution", "any distribution over the feasible",
             "for any feasible-set distribution", "past two months", "within about 0.02", "0.992", "fewer than 0.4",
             "five-run statistical validation", "\\documentclass{iopart}", "32.2\\%", "33.3\\%", "28.7\\%", "28.1\\%",
             "is flat near 0.67", "largest on hardware at $p=1$ to 2"]
    hits = [s for s in STALE if s in TEXT] + (["'ecr' gate name"] if re.search(r"\becr\b", TEXT) else [])
    out("PASS" if not hits else "FAIL", f"stale or retracted phrases in the paper: {hits or 'none'}")
    todos = []
    for m in re.finditer(r"\\todo\{", TEXT):
        depth, i = 1, m.end()
        while depth and i < len(TEXT):
            depth += {"{": 1, "}": -1}.get(TEXT[i], 0)
            i += 1
        todos.append((TEXT.count("\n", 0, m.start()) + 1, re.sub(r"\s+", " ", TEXT[m.end():i - 1])[:110]))
    out("WARN" if todos else "PASS", f"open \\todo items: {len(todos)}")
    for ln, t in todos:
        print(f"         line {ln:4d}: {t}")
    REPORT["values_to_patch"]["open_todos"] = [f"line {ln}: {t}" for ln, t in todos]
    nfig = len(re.findall(r"\\placeholderfig\{", TEXT))
    out("WARN" if nfig else "PASS", f"figure placeholders still in the paper: {nfig}")
    if "TODO title" in TEX_RAW:
        out("WARN", "title is still marked as a candidate awaiting approval")
if os.path.exists(BIB):
    braw = open(BIB, encoding="utf-8").read()
    entries = re.findall(r"^@(\w+)\{([^,\s]+),(.*?)^\}", braw, re.M | re.S)
    keys = [k for _, k, _ in entries]
    out("INFO", f"{len(keys)} bib entries")
    dk = [k for k, c in Counter(keys).items() if c > 1]
    out("PASS" if not dk else "FAIL", f"duplicate bib keys: {dk or 'none'}")
    incomment = [k for _, k, body in entries for line in body.split("\n") if "=" in line and re.search(r"(?<!\\)%", line) and not line.strip().startswith("%")]
    out("PASS" if not incomment else "FAIL", f"entries with a % comment inside a field line (breaks BibTeX): {incomment or 'none'}")
    noyear = [k for _, k, body in entries if not re.search(r"^\s*year\s*=", body, re.M)]
    nojour = [k for typ, k, body in entries if typ.lower() == "article" and not re.search(r"^\s*journal\s*=", body, re.M)]
    out("PASS" if not noyear else "FAIL", f"entries without year: {noyear or 'none'}")
    out("PASS" if not nojour else "FAIL", f"@article entries without journal: {nojour or 'none'}")
    if TEXT:
        missing = sorted(cites - set(keys))
        out("PASS" if not missing else "FAIL", f"citations with no bib entry: {missing or 'none'}")
        out("INFO", f"uncited bib entries (harmless under iopart-num): {sorted(set(keys) - cites)}")
    out("INFO", f"open VERIFY notes in the bib: {len(re.findall(r'^% VERIFY(?!ED)', braw, re.M))}")

# ================================================================== 5 ONLINE
if OFFLINE:
    section("5  ONLINE CHECKS SKIPPED (--offline)")
else:
    section("5  ONLINE: job status, circuit audits, runtime options, TODO values")
    svc = None
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        try:
            svc = QiskitRuntimeService()
        except Exception:
            svc = QiskitRuntimeService(channel="ibm_cloud")
        out("PASS", "IBM Quantum account loaded")
    except Exception as e:  # noqa
        out("FAIL", f"cannot load the IBM Quantum account ({e}); rerun with --offline or re-authenticate")
    circ_cache = {}

    def circuit(jid):
        if jid not in circ_cache:
            circ_cache[jid] = svc.job(jid).inputs["pubs"][0][0]
        return circ_cache[jid]

    def counts(jid):
        data = svc.job(jid).result()[0].data
        for name in dir(data):
            if not name.startswith("_") and hasattr(getattr(data, name), "get_counts"):
                return getattr(data, name).get_counts()
        raise RuntimeError("no counts register")

    if svc and not NO_STATUS:
        for key, ids in ledger.items():
            c = Counter()
            for jid in ids:
                try:
                    c[str(svc.job(jid).status())] += 1
                except Exception:  # noqa
                    c["LOOKUP_FAILED"] += 1
            expect_done = True   # all 340 jobs executed (1HSG size chain ran after the 3 Oct 2026 window reset)
            ok = c.get("DONE", 0) == len(ids) - (len(odd) if key == "july" else 0)
            level = "PASS" if ok or (not expect_done and set(c) <= {"QUEUED", "DONE", "RUNNING"}) else "FAIL"
            out(level, f"status {key:20s} {dict(c)}")

    if svc:
        # ---- depth-one circuit audit: penalty, graph, weights, bit order, angles
        audits = []
        if inst:
            for t in ("2bm2", "1hsg"):
                I = inst[t]
                ivf = next((r.get("ideal_valid_frac") for r in (vst or []) if r.get("backend") == BACKEND[t]), None)
                audits.append((f"July {t} monolithic", MONO[t], I["w"], I["ne"], I["edges"], I["xs"], 0.5, None, modpi(I["beta"]), list(range(11)), ivf))
            if corrected_rows:
                for t in ("2bm2", "1hsg"):
                    row = next((r for r in corrected_rows if r.get("target") == t and r.get("arm") == "monolithic_p1"), None)
                    if row:
                        I = inst[t]
                        ivf = next((r.get("ideal_valid_frac") for r in (cres or {}).get("monolithic", []) if r.get("target") == t), None)
                        audits.append((f"corrected {t} monolithic", row["job_id"], I["w"], I["ne"], I["edges"], I["xs"], 1.1,
                                       EXEC_CORRECTED["gamma"], EXEC_CORRECTED["beta"], list(range(11)), ivf))
            if sweep:
                for c in sweep["cases"]:
                    if c["p"] != 1:
                        continue
                    jid = next(j["job_id"] for j in sweep["jobs"] if (j["group"], j["penalty_factor"], j["p"]) == (c["group"], c["penalty_factor"], 1))
                    w3 = np.array(c["weights"])
                    audits.append((f"sweep {c['group']} P={c['penalty_factor']} p=1", jid, w3, list(itertools.combinations(range(3), 2)), [],
                                   format(1 << int(np.argmax(w3)), "03b"), c["penalty_factor"], c["gammas"][0], modpi(c["betas"][0]), c["nodes"], None))
            for d, t in ((size2, "2bm2"), (size1, "1hsg")):
                if not d or not d.get("jobs"):
                    continue
                I = inst[t]
                for c in d["cases"]:
                    jid = next((j["job_id"] for j in d["jobs"] if (j["union"], j["penalty_factor"]) == (c["union"], c["penalty_factor"])), None)
                    if not jid:
                        continue
                    idx = {v: k for k, v in enumerate(c["nodes"])}
                    ne = [(idx[a], idx[b]) for a, b in I["ne"] if a in idx and b in idx]
                    ed = [(idx[a], idx[b]) for a, b in I["edges"] if a in idx and b in idx]
                    audits.append((f"size {t} n={c['n']} P={c['penalty_factor']}", jid, I["w"][c["nodes"]], ne, ed, c["xstar_bits"],
                                   c["penalty_factor"], c["gamma"], modpi(c["beta"]), c["nodes"], c.get("ideal_valid_frac")))
        print(f"  auditing {len(audits)} depth-one circuits from their submitted form ...")
        GRIDSTAT = {"ok": 0, "bad": 0, "max_on": 0.0, "min_off": 1.0}
        for label, jid, w, ne, ed, xs, pf, g_exp, b_exp, nodes, ivf_decl in audits:
            try:
                circ = circuit(jid)
                X = extract_p1(circ, b_exp)
                gw, pair = X["gw"], X["pair"]
                cand_txt = "; ".join(f"beta={c['beta']:+.4f} (grid dist {min(c['grid_distance'].values()):.1e} at G={min(c['grid_distance'], key=c['grid_distance'].get)})" for c in X["candidates"])
                on_grid = [c for c in X["candidates"] if min(c["grid_distance"].values()) < 1e-4]
                if len(X["candidates"]) != 2:
                    out("WARN", f"{label}: {len(X['candidates'])} uniformity minima found (expected 2, pi/2 apart): {cand_txt}")
                if label.startswith("sweep"):
                    out("INFO", f"{label}: angles from COBYLA (continuous), grid membership not applicable; reading selected by the recorded angle")
                elif len(on_grid) != 1 or abs(modpi(on_grid[0]["beta"] - X["beta"])) > 1e-3:
                    out("FAIL", f"{label}: grid membership does not single out the selected reading: {cand_txt}")
                    GRIDSTAT["bad"] += 1
                else:
                    GRIDSTAT["ok"] += 1
                    GRIDSTAT["max_on"] = max(GRIDSTAT["max_on"], min(on_grid[0]["grid_distance"].values()))
                    GRIDSTAT["min_off"] = min(GRIDSTAT["min_off"], min(min(c["grid_distance"].values()) for c in X["candidates"] if c is not on_grid[0]))
                k = int(np.argmax(np.abs(w)))
                gamma = gw[k] / w[k]
                w_ext = gw / gamma
                ne_ph = [pair[e] for e in ne]
                P_ext = float(np.median(ne_ph)) / gamma if ne_ph else float("nan")
                problems = []
                if X["uniformity"] > 1e-3:
                    problems.append(f"mixer not isolated (uniformity {X['uniformity']:.1e})")
                if ed and max(abs(pair[e]) for e in ed) > 1e-4:
                    problems.append(f"edge phases up to {max(abs(pair[e]) for e in ed):.1e} (should be 0)")
                if ne_ph and (max(ne_ph) - min(ne_ph)) > 1e-4:
                    problems.append(f"non-edge phases spread {max(ne_ph) - min(ne_ph):.1e}")
                if np.max(np.abs(w_ext - w)) > 1e-3:
                    problems.append(f"weights differ by {np.max(np.abs(w_ext - w)):.1e}")
                if abs(P_ext - pf) > 1e-3:
                    problems.append(f"penalty {P_ext:.4f} x w_max(global), declared {pf}")
                if best_valid(w_ext, ne, len(w)) != xs:
                    problems.append(f"optimum {best_valid(w_ext, ne, len(w))} vs declared {xs}")
                if b_exp is not None and abs(modpi(X["beta"] - b_exp)) > 1e-3:
                    problems.append(f"beta {X['beta']:.4f} vs declared {b_exp:.4f}")
                if g_exp is not None and abs(gamma - g_exp) > 1e-3:
                    problems.append(f"gamma {gamma:.4f} vs declared {g_exp:.4f}")
                if ivf_decl is not None:
                    pr, nn = probs_clbit_order(circ)
                    nes = set(ne)
                    vfrac = float(sum(abs(pr[x]) ** 2 for x in range(1, 2 ** nn)
                                      if not any(((x >> i) & 1) and ((x >> j) & 1) for i, j in nes)))
                    if abs(vfrac - ivf_decl) > 1e-3:
                        problems.append(f"noiseless valid fraction {vfrac:.4f} vs declared {ivf_decl:.4f}")
                md = (circ.metadata or {}).get("bit_to_node")
                if md is not None and list(md) != list(nodes):
                    problems.append(f"metadata bit_to_node {md} vs {nodes}")
                out("PASS" if not problems else "FAIL", f"{label:34s} P={P_ext:.4f}  gamma={gamma:+.4f}  beta={X['beta']:+.4f}  "
                    + ("graph, weights, bit order and angles as declared" if not problems else "; ".join(problems))
                    + f"  [uniformity minima: {cand_txt}; selected by declared beta, the other is the bit-complement reading]")
                REPORT["values_to_patch"].setdefault("p1_audit", {})[label] = {
                    "selected_beta": X["beta"], "candidates": [{"beta": c["beta"], "grid_distance": c["grid_distance"]} for c in X["candidates"]],
                    "P_over_wmax": P_ext, "gamma": gamma, "problems": problems}
            except Exception as e:  # noqa
                out("WARN", f"{label}: circuit audit not possible ({e})")

        if GRIDSTAT["ok"] + GRIDSTAT["bad"]:
            out("PASS" if GRIDSTAT["bad"] == 0 and GRIDSTAT["max_on"] < 5e-6 and GRIDSTAT["min_off"] > 6e-3 else "FAIL",
                f"grid membership singles out the reading for {GRIDSTAT['ok']}/{GRIDSTAT['ok'] + GRIDSTAT['bad']} grid-based circuits "
                f"(on-grid candidate within {GRIDSTAT['max_on']:.1e}, other candidate no closer than {GRIDSTAT['min_off']:.1e}; paper: 5e-6 and 6e-3)")

        # ---- ideal pair mass of every depth-2/3 subproblem circuit
        pair_checks = []
        if sweep:
            for c in sweep["cases"]:
                if c["p"] >= 1:
                    jid = next(j["job_id"] for j in sweep["jobs"] if (j["group"], j["penalty_factor"], j["p"]) == (c["group"], c["penalty_factor"], c["p"]))
                    pair_checks.append((f"sweep {c['group']} P={c['penalty_factor']} p={c['p']}", jid, c["ideal_pair_mass"]))
        if cres:
            for g in ("P76", "P79", "P85"):
                r = next((r for r in cres.get("subproblems", []) if r["arm"] == f"sub_p3_{g}"), None)
                if r:
                    pair_checks.append((f"corrected 2BM2 {g} p=3", r["job_id"], sum(r["ideal_pairs"].values())))
        if an_c15:
            for g in ("P139", "P51", "P52"):
                r = next((r for r in an_c15 if r["arm"] == f"sub_p3_{g}"), None)
                if r:
                    pair_checks.append((f"corrected 1HSG {g} p=3", r["job_id"], r["ideal_pair_mass"]))
        for label, jid, declared in pair_checks:
            try:
                psi, n = probs_clbit_order(circuit(jid))
                pmass = float(sum(abs(psi[i]) ** 2 for i in PAIR_IDX))
                out("PASS" if abs(pmass - declared) < 1e-6 else "FAIL", f"{label:34s} ideal pair mass of submitted circuit {pmass:.6f} (declared {declared:.6f})")
            except Exception as e:  # noqa
                out("WARN", f"{label}: {e}")

        # ---- runtime options: July (two jobs) and the first job of every September ledger.
        #      SamplerV2 defaults in qiskit-ibm-runtime 0.47.0 / 0.49.0 (options/*.py): dynamical_decoupling.enable False,
        #      twirling.enable_gates False, twirling.enable_measure False. Empty options = none of them applied.
        try:
            opts = {}
            probe = [("July", MONO["2bm2"]), ("July", next(k for k, v in (jr or {"jobs": {}})["jobs"].items() if v.get("n_qubits") == 3))]
            probe += [(key, ids[0]) for key, ids in ledger.items() if key != "july" and ids]
            for tag, jid in probe:
                inp = svc.job(jid).inputs
                opts[jid] = (tag, inp.get("options", {}) if isinstance(inp, dict) else {})
            REPORT["values_to_patch"]["runtime_options"] = {k: v[1] for k, v in opts.items()}
            clean = True
            for jid, (tag, o) in opts.items():
                dd = (o.get("dynamical_decoupling") or {}).get("enable", "not set (default False)")
                tw = o.get("twirling") or "not set (default: gates False, measure False)"
                if (o.get("dynamical_decoupling") or {}).get("enable") or (o.get("twirling") or {}).get("enable_gates") or (o.get("twirling") or {}).get("enable_measure"):
                    clean = False
                out("INFO", f"{tag:20s} job {jid}: dynamical_decoupling.enable = {dd}; twirling = {tw}; raw options: {json.dumps(o, default=str)[:160]}")
            out("PASS" if clean else "FAIL", "no dynamical decoupling or twirling in any probed job (paper: none in either campaign)")
        except Exception as e:  # noqa
            out("WARN", f"runtime options not readable ({e})")

        # ---- empty-string mass: hardware vs noisy vs ideal (Section 8 TODO)
        emp = {}
        groups_e = []
        if corrected_rows:
            for t in ("2bm2", "1hsg"):
                groups_e.append((f"corrected {t} monolithic", [r["job_id"] for r in corrected_rows if r.get("target") == t and r.get("arm") == "monolithic_p1"], 11))
        for d, t in ((size2, "2bm2"), (size1, "1hsg")):
            if d and d.get("jobs"):
                for pf in (0.5, 1.1):
                    ids = [j["job_id"] for j in d["jobs"] if j["union"].count("+") == 4 and j["penalty_factor"] == pf]
                    if ids:
                        groups_e.append((f"size {t} n=11 P={pf}", ids, 11))
        for label, ids, n in groups_e:
            try:
                if str(svc.job(ids[0]).status()) != "DONE":
                    out("INFO", f"{label}: not executed yet")
                    continue
                hwv = []
                for jid in ids:
                    c = counts(jid)
                    hwv.append(c.get("0" * n, 0) / sum(c.values()))
                nv = [ns[j]["counts_noisy"].get("0" * n, 0) / ns[j]["shots"] for j in ids if j in ns and "counts_noisy" in ns[j]]
                psi, _ = probs_clbit_order(circuit(ids[0]))
                ideal0 = float(abs(psi[0]) ** 2)
                emp[label] = {"hardware_mean": mean(hwv), "hardware_sd": sd(hwv), "noisy_sim_mean": mean(nv) if nv else None, "ideal": ideal0}
                out("INFO", f"{label:30s} empty-string mass: hardware {mean(hwv):.3f} +- {sd(hwv):.3f}, noisy sim "
                            + (f"{mean(nv):.3f}" if nv else "n/a") + f", ideal {ideal0:.3f}")
            except Exception as e:  # noqa
                out("WARN", f"{label}: {e}")
        REPORT["values_to_patch"]["empty_string_mass"] = emp
        need = ("corrected 2bm2 monolithic", "corrected 1hsg monolithic", "size 2bm2 n=11 P=1.1", "size 1hsg n=11 P=1.1")
        if all(k in emp and emp[k]["noisy_sim_mean"] is not None for k in need):
            E = [emp[k] for k in need]
            claim("8.4 empty-string ideal", f"a probability of {E[0]['ideal']:.3f} and {E[1]['ideal']:.3f} (monolithic 2BM2 and 1HSG) and {E[2]['ideal']:.3f} and {E[3]['ideal']:.3f}")
            claim("8.4 empty-string hardware", ", ".join(f"${e['hardware_mean']:.3f} \\pm {e['hardware_sd']:.3f}$" for e in E[:3]) + f" and ${E[3]['hardware_mean']:.3f} \\pm {E[3]['hardware_sd']:.3f}$")
            claim("8.4 empty-string simulation", f"({E[0]['noisy_sim_mean']:.3f}, {E[1]['noisy_sim_mean']:.3f}, {E[2]['noisy_sim_mean']:.3f})")
            claim("8.4 empty-string simulation 4th", f"({E[3]['noisy_sim_mean']:.3f})")
            below = all(e["hardware_mean"] < e["ideal"] for e in E)
            out("PASS" if below else "FAIL", "8.4: hardware empty-string mass below the noiseless value in all four corrected cases")

        # ---- 1HSG corrected subs and 1HSG size chain: execution window and calibration (tab:circuits TODOs)
        for wkey, wlabel in (("corrected_1hsg_subs", "1HSG corrected subs"), ("size_1hsg", "1HSG size chain")):
          if not ledger.get(wkey):
            continue
          try:
                runs, lastupd = [], set()
                for jid in ledger[wkey]:
                    ts = (ns.get(jid, {}).get("timestamps") or svc.job(jid).metrics().get("timestamps", {}))
                    if ts.get("running"):
                        runs.append(ts["running"])
                    if ns.get(jid, {}).get("properties_last_update"):
                        lastupd.add(ns[jid]["properties_last_update"])
                runs.sort()
                b = svc.backend("ibm_fez")
                if runs and not lastupd:
                    for r in (runs[0], runs[-1]):
                        lastupd.add(str(b.properties(datetime=datetime.datetime.fromisoformat(r.replace("Z", "+00:00"))).last_update_date))

                def medians_at(when_iso):
                    p = b.properties(datetime=datetime.datetime.fromisoformat(when_iso.replace("Z", "+00:00")))
                    cz = [g.parameters[0].value for g in p.gates if g.gate == "cz"]

                    def qp(name):
                        v = []
                        for i in range(b.num_qubits):
                            try:
                                v.append(p.qubit_property(i, name)[0])
                            except Exception:  # noqa
                                pass
                        return v
                    return {"calibration_utc": utc_label(str(p.last_update_date)), "median_cz_error": float(np.median(cz)),
                            "median_T1_us": float(np.median(qp("T1"))) * 1e6, "median_T2_us": float(np.median(qp("T2"))) * 1e6,
                            "median_readout_error": float(np.median(qp("readout_error")))}
                # one representative running time per distinct calibration
                first_on = {}
                for jid in ledger[wkey]:
                    c = ns.get(jid, {}).get("properties_last_update")
                    r = (ns.get(jid, {}).get("timestamps") or {}).get("running")
                    if c and r and (c not in first_on or r < first_on[c]):
                        first_on[c] = r
                per_cal = {utc_label(c): medians_at(r) for c, r in sorted(first_on.items())} or {"?": medians_at(runs[0])}
                REPORT["values_to_patch"][wkey + "_window"] = {"executed_utc": [runs[0], runs[-1]], "n_with_running": len(runs),
                                                               "calibrations_utc": sorted(utc_label(c) for c in lastupd), "medians_per_calibration": per_cal}
                out("INFO", f"{wlabel} executed {utc_label(runs[0])} to {utc_label(runs[-1])} UTC ({len(runs)} jobs with a running timestamp), "
                            f"calibration(s) {sorted(utc_label(c) for c in lastupd)} UTC")
                for cal, med in per_cal.items():
                    out("INFO", f"    calibration {cal} UTC: cz {med['median_cz_error']:.3e}, T1 {med['median_T1_us']:.1f}, "
                                f"T2 {med['median_T2_us']:.1f}, RO {med['median_readout_error']:.3e}")
                if len(lastupd) > 1:
                    out("WARN", f"{wlabel}: executed against {len(lastupd)} calibrations; tab:circuits must list all of them")
          except Exception as e:  # noqa
                out("WARN", f"{wlabel} timing not available ({e})")

        # ---- unaccounted July IDs: what are they?
        for jid in odd:
            try:
                j = svc.job(jid)
                c = j.inputs["pubs"][0][0]
                out("INFO", f"unaccounted July job {jid}: status {j.status()}, backend {j.backend().name}, created {j.creation_date}, "
                            f"{c.num_qubits} qubits in the ISA circuit, ops {dict(c.count_ops())}")
            except Exception as e:  # noqa
                out("INFO", f"unaccounted July job {jid}: {e}")

# ================================================================== SUMMARY
section("SUMMARY")
print(f"PASS {TALLY['PASS']}   FAIL {TALLY['FAIL']}   WARN {TALLY['WARN']}   INFO {TALLY['INFO']}"
      + (f"   SKIPPED {TALLY['SKIP']} number checks (manuscript not found: {TEX})" if TALLY["SKIP"] else ""))
if REPORT["fail"]:
    print("\nFailures to fix before submission:")
    for f in REPORT["fail"]:
        print("  - " + f)
json.dump(REPORT, open("preflight_report.json", "w"), indent=1, default=str)
print("\nwritten preflight_report.json")
