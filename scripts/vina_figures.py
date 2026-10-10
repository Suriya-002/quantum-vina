"""
vina_figures.py  --  figures for the Quantum-Vina paper (10 Oct 2026)

    python vina_figures.py

Reads (from the current folder):
    size_test.json, analysis_size_test.json                2BM2 size chain (ideal + hardware)
    size_test_1hsg.json, analysis_size_test_1hsg.json      1HSG size chain (ideal + hardware; executed Oct 2026)
    depth_penalty_sweep.json, analysis_depth_penalty_sweep.json
Writes fig_size_study.pdf/.png and fig_depth_sweep.pdf/.png; upload the PDFs to the figures/ folder in Overleaf.
Plots whatever exists: if a hardware analysis file is missing, that panel shows the noiseless
prediction only and says so.

Conventions (match the captions in the paper):
    hardware = filled markers, solid lines, error bars = sample s.d. over the five runs (ddof = 1)
    noiseless = open markers, dashed lines
    P = 0.5 w_max: vermillion circles   P = 1.1 w_max: blue squares   (colour-blind and greyscale safe)
    1HSG off-chain controls: diamonds, not joined to the chain
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CM = 1 / 2.54
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 1.0,
                     "lines.markersize": 4.5, "errorbar.capsize": 2, "pdf.fonttype": 42, "savefig.dpi": 300})
STYLE = {0.5: {"color": "#D55E00", "marker": "o", "label": r"$P=0.5\,w_{\max}$"},
         1.1: {"color": "#0072B2", "marker": "s", "label": r"$P=1.1\,w_{\max}$"}}
CHAIN = {"2bm2": ["P79", "P75+P79", "P38+P75+P79", "P75+P76+P79", "P38+P75+P76+P79", "P75+P76+P79+P85", "P38+P75+P76+P79+P85"],
         "1hsg": ["P52", "P51+P52", "P136+P51+P52", "P139+P51+P52", "P136+P139+P51+P52", "P136+P139+P147+P51+P52"]}
CONTROLS = {"1hsg": ["P136+P147+P51", "P136+P139+P147+P52"]}
FILES = {"2bm2": ("size_test.json", "analysis_size_test.json"), "1hsg": ("size_test_1hsg.json", "analysis_size_test_1hsg.json")}
DX = {0.5: -0.12, 1.1: 0.12}


def load(fn):
    if not os.path.exists(fn):
        return None
    with open(fn, encoding="utf-8") as f:
        return json.load(f)


def sd(v):
    return float(np.std(v, ddof=1)) if len(v) > 1 else 0.0


# ======================================================================= figure 1: size study
def size_data(target):
    cfg, an = (load(f) for f in FILES[target])
    if cfg is None:
        return None, None
    ideal = {(c["union"], c["penalty_factor"]): c for c in cfg.get("cases", [])}
    hw = {}
    for r in an or []:
        hw.setdefault((r["union"], r["penalty_factor"]), []).append(r)
    return ideal, (hw if hw else None)


fig1, ax1 = plt.subplots(2, 2, figsize=(16 * CM, 10.5 * CM), sharex="col")
notes = []
for row, target in enumerate(("2bm2", "1hsg")):
    ideal, hw = size_data(target)
    axv, axm = ax1[row, 0], ax1[row, 1]
    name = "2BM2" if target == "2bm2" else "1HSG"
    if ideal is None:
        for a in (axv, axm):
            a.text(0.5, 0.5, f"{FILES[target][0]} not found", ha="center", va="center", transform=a.transAxes)
        continue
    for pf, st in STYLE.items():
        for group, is_chain in ((CHAIN[target], True), (CONTROLS.get(target, []), False)):
            keys = [(u, pf) for u in group if (u, pf) in ideal]
            if not keys:
                continue
            n = np.array([ideal[k]["n"] for k in keys], float) + DX[pf]
            mk = st["marker"] if is_chain else "D"
            ls = "--" if is_chain else "none"
            # noiseless
            axv.plot(n, [ideal[k]["ideal_valid_frac"] for k in keys], ls=ls, marker=mk, mfc="white", color=st["color"], zorder=2)
            axm.plot(n, [1.0 if ideal[k]["ideal_mode_valid"] else 0.0 for k in keys], ls=ls, marker=mk, mfc="white", ms=7.5,
                     color=st["color"], zorder=2)   # larger open ring: a filled hardware marker inside it means agreement
            # hardware
            if hw:
                hk = [k for k in keys if k in hw]
                if hk:
                    nh = np.array([ideal[k]["n"] for k in hk], float) + DX[pf]
                    vf = [np.mean([r["valid_frac"] for r in hw[k]]) for k in hk]
                    vs = [sd([r["valid_frac"] for r in hw[k]]) for k in hk]
                    mf = [np.mean([bool(r["mode_valid"]) for r in hw[k]]) for k in hk]
                    axv.errorbar(nh, vf, yerr=vs, ls="-" if is_chain else "none", marker=mk, color=st["color"], zorder=3)
                    axm.plot(nh, mf, ls="-" if is_chain else "none", marker=mk, color=st["color"], zorder=3)
    if not hw:
        axv.text(0.97, 0.36, "hardware analysis\nfile not found", ha="right", va="center", fontsize=6.5, color="0.35", transform=axv.transAxes)
        axm.text(0.03, 0.5, "hardware analysis\nfile not found", ha="left", va="center", fontsize=6.5, color="0.35", transform=axm.transAxes)
        notes.append(f"{name}: hardware analysis not found ({FILES[target][1]}); noiseless only")
    axv.set_ylabel("valid fraction")
    axm.set_ylabel("fraction of runs")
    dev = "ibm_marrakesh" if target == "2bm2" else "ibm_fez"
    axv.text(0.03, 0.95, f"{name} ({dev})", ha="left", va="top", fontsize=7.5, fontweight="bold", transform=axv.transAxes)
    axv.set_ylim(0, 1)
    axm.set_ylim(-0.08, 1.08)
    axm.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for a in (axv, axm):
        a.set_xlim(2.3, 11.7)
        a.set_xticks(range(3, 12))
        a.grid(axis="y", lw=0.3, color="0.85")
for a in ax1[1]:
    a.set_xlabel("instance size $n$ (nodes)")
ax1[0, 0].set_title("(a) valid fraction", loc="left")
ax1[0, 1].set_title("(b) most frequent string is a clique", loc="left")
handles = [plt.Line2D([], [], color=STYLE[p]["color"], marker=STYLE[p]["marker"], ls="-", label=STYLE[p]["label"] + " hardware") for p in STYLE]
handles += [plt.Line2D([], [], color=STYLE[p]["color"], marker=STYLE[p]["marker"], mfc="white", ls="--", label=STYLE[p]["label"] + " noiseless") for p in STYLE]
handles += [plt.Line2D([], [], color="0.3", marker="D", mfc="white", ls="none", label="1HSG off-chain controls")]
fig1.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.01))
fig1.tight_layout(rect=(0, 0.1, 1, 1))
fig1.savefig("fig_size_study.pdf")
fig1.savefig("fig_size_study.png")

# ======================================================================= figure 2: depth sweep
sw, asw = load("depth_penalty_sweep.json"), load("analysis_depth_penalty_sweep.json")
fig2, ax2 = plt.subplots(1, 3, figsize=(16 * CM, 5.6 * CM), sharey=True)
if sw is None:
    notes.append("depth_penalty_sweep.json not found: figure 2 empty")
else:
    ideal = {(c["group"], c["penalty_factor"], c["p"]): c["ideal_pair_mass"] for c in sw["cases"]}
    hw = {}
    for r in asw or []:
        hw.setdefault((r["group"], r["penalty_factor"], r["p"]), []).append(r["pair_mass"])
    for a, g in zip(ax2, ("P76", "P79", "P85")):
        for pf, st in STYLE.items():
            ps = [p for p in (1, 2, 3) if (g, pf, p) in ideal]
            x = np.array(ps, float) + DX[pf] / 2
            a.plot(x, [ideal[(g, pf, p)] for p in ps], ls="--", marker=st["marker"], mfc="white", color=st["color"], zorder=2)
            hp = [p for p in ps if (g, pf, p) in hw]
            if hp:
                a.errorbar(np.array(hp, float) + DX[pf] / 2, [np.mean(hw[(g, pf, p)]) for p in hp],
                           yerr=[sd(hw[(g, pf, p)]) for p in hp], ls="-", marker=st["marker"], color=st["color"], zorder=3)
        a.axhline(3 / 8, color="0.5", lw=0.6, ls=":", zorder=1)
        a.set_title(g, loc="left")
        a.set_xticks([1, 2, 3])
        a.set_xlim(0.6, 3.4)
        a.set_xlabel("depth $p$")
        a.set_ylim(0, 1)
        a.grid(axis="y", lw=0.3, color="0.85")
    ax2[0].set_ylabel("two-contact shot mass")
    ax2[2].text(3.35, 3 / 8 + 0.02, "uniform", ha="right", va="bottom", fontsize=6.5, color="0.4")
    if not asw:
        notes.append("analysis_depth_penalty_sweep.json not found: figure 2 shows noiseless only")
handles2 = [plt.Line2D([], [], color=STYLE[p]["color"], marker=STYLE[p]["marker"], ls="-", label=STYLE[p]["label"] + " hardware") for p in STYLE]
handles2 += [plt.Line2D([], [], color=STYLE[p]["color"], marker=STYLE[p]["marker"], mfc="white", ls="--", label=STYLE[p]["label"] + " noiseless") for p in STYLE]
fig2.legend(handles=handles2, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.02))
fig2.tight_layout(rect=(0, 0.12, 1, 1))
fig2.savefig("fig_depth_sweep.pdf")
fig2.savefig("fig_depth_sweep.png")

print("written fig_size_study.pdf/.png and fig_depth_sweep.pdf/.png (upload the two PDFs to figures/ in Overleaf)")
for n in notes:
    print("note:", n)
