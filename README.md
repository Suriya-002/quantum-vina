# Quantum-Vina: penalty strength and instance size in quantum approximate optimization of molecular docking on superconducting processors

Data, submitted circuits' job identifiers, analysis scripts and manuscript source for

> S. N. Rajavel, *Penalty strength and instance size in quantum approximate optimization of molecular docking on superconducting processors*, manuscript submitted to Quantum Science and Technology (October 2026).

Two IBM Heron processors (`ibm_marrakesh`, `ibm_fez`), 340 jobs in two campaigns (60 in July 2026, 280 submitted in September 2026 and executed in September and October), eleven-node maximum-weighted-clique instances built from AutoDock Vina-GPU 2.1 poses of human Î²II tryptase (PDB 2BM2) and HIV-1 protease with indinavir (PDB 1HSG). Every number in the manuscript is recomputed from the files in `data/` by `scripts/preflight.py`.

No claim of quantum advantage is made anywhere in this work: the instances are small enough to enumerate, and enumeration is the reference throughout.

## Layout

```
paper/      quantum_vina_qst.tex (IOP iopjournal class), references.bib, figures/
scripts/    the pipeline, in the order listed below
data/       every record the manuscript depends on (flat on purpose: scripts open inputs by bare file name)
```

### Data files

| File | Content |
|---|---|
| `job_records.json` | July 2026 campaign: 60 job IDs with raw counts and metadata |
| `extracted_hamiltonians.json` | executed instances recovered from the submitted July circuits (edge sets, relative weights, penalty ratio, optima, mixer angle; both uniformity readings) |
| `valid_subspace_test.json` | July per-run statistics recomputed from the raw counts in classical-bit order, with noiseless columns |
| `corrected_penalty_arm_executed.json` | corrected-penalty re-execution ledger, 25 jobs (reconstructed by `preflight.py --repair` after the original build file was overwritten by a later dry run) |
| `corrected_penalty_results.json` | corrected-arm analysis, 2BM2 and 1HSG monolithic circuits and the 2BM2 subproblems |
| `corrected_penalty_arm_1hsg-subs.json`, `analysis_corrected_penalty_arm_1hsg-subs.json` | 1HSG corrected subproblems, 15 jobs, and their analysis |
| `depth_penalty_sweep.json`, `analysis_depth_penalty_sweep.json` | depth sweep, 90 jobs, and its analysis |
| `size_test.json`, `analysis_size_test.json` | 2BM2 size study, 70 jobs, and its analysis |
| `size_test_1hsg.json`, `analysis_size_test_1hsg.json` | 1HSG size study, 80 jobs (executed 3 Oct 2026 in two batches), and its analysis |
| `noise_sim.json` | for all 340 jobs: execution timestamps, calibration in force at execution, calibration-based Aer simulation of the submitted circuit |
| `snapshot_medians.json` | device medians (cz error, T1, T2, readout) of every calibration in force at execution, in UTC |
| `results_backup.json` | raw shot counts of all 340 jobs as retrieved from IBM Quantum |
| `preflight_report.json` | output of the last consistency check |

Job IDs are IBM Quantum job identifiers; counts can also be retrieved from the service with them.

## Reproduction

Python 3 with `qiskit` 2.5.2, `qiskit-ibm-runtime` 0.49.0, `qiskit-aer` 0.17.2, `numpy`, `scipy` and `matplotlib` (the September/October environment; the July campaign used `qiskit` 2.5.0 and `qiskit-ibm-runtime` 0.47.0). Steps marked (online) need an IBM Quantum account with access to the job records; everything else runs from `data/` alone. Run every command from `data/`:

```
cd data
python ..\scripts\extract_hamiltonians.py          (online) recover the executed instances from the July circuits
python ..\scripts\valid_subspace_test.py            (online) recompute the July statistics from the raw counts
python ..\scripts\corrected_penalty_arm.py --dry-run         build and simulate the corrected arm, submit nothing (without --dry-run it submits; --only=1hsg-subs restricts)
python ..\scripts\depth_penalty_sweep.py --dry-run           likewise for the sweep (without --dry-run it submits)
python ..\scripts\size_test.py --dry-run                     all unions for 2BM2, submit nothing; add --1hsg for 1HSG; --submit "P79,P75+P79" submits the named unions
python ..\scripts\analyse_arms.py size_test.json    (online) per-run analysis of a ledger (any ledger file)
python ..\scripts\noise_sim.py                      (online) execution timestamps + calibration-based simulation
python ..\scripts\snapshot_medians.py               (online) device medians at execution time
python ..\scripts\vina_figures.py                            the two figures
python ..\scripts\queue_check.py                    (online) status of all 340 jobs + counts backup
python ..\scripts\preflight.py --offline --tex ..\paper\quantum_vina_qst.tex --bib ..\paper\references.bib
python ..\scripts\preflight.py --tex ..\paper\quantum_vina_qst.tex --bib ..\paper\references.bib   (online: also audits every depth-one circuit from its submitted form)
```

Submitting jobs spends IBM Quantum Open Plan time; every submission script has a `--dry-run` that writes a `*_dryrun.json` and submits nothing. Submission scripts never overwrite a ledger that already holds job IDs.

## Conventions worth knowing

- Measurement keys are in Qiskit classical-bit order (rightmost character = clbit 0 = node 0). The July analysis used node order; all July statistics here are recomputed under the correct convention.
- `qiskit-ibm-runtime` returns `BackendProperties.last_update_date` in the local time of the querying machine. `noise_sim.json` stores those strings with their UTC offsets; `snapshot_medians.json` and the manuscript give calibration times in UTC.
- The mixer angle of a depth-one circuit is recoverable from its statevector only up to a shift of Ï€/2 (the other solution reads the bit-complemented Hamiltonian). `extract_hamiltonians.py` and `preflight.py` keep both candidates and select by grid membership or by the recorded angle; the manuscript explains the test.
- The 1HSG size chain executed in two batches against two calibrations; one five-run set (nine nodes, executed penalty) straddles them. Named in the manuscript.

## Licences

Code (`scripts/`) under the MIT licence (`LICENSE`). Data and manuscript (`data/`, `paper/`) under CC BY 4.0 (`data/LICENSE`).
