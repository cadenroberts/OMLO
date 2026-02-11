# Patchset Summary

## PHASE 0 — BASELINE SNAPSHOT

**Current branch:** main  
**Current HEAD:** bfecf86684862d2ed06be34447c2361de04e7bf3  
**Tracked file count:** 39 files (excluding .git, venv, __pycache__, results)

**Primary entry points:**
- `experiments/run_baseline.py` — Standard PyTorch training baseline
- `experiments/run_oram.py` — ORAM-integrated training
- `experiments/run_sweep.py` — Batch-size and dataset-size sweep experiments
- `experiments/analyze_results.py` — Overhead analysis and visualization
- `scripts/run_experiments.sh` — Master orchestrator for all phases

**How the project runs:**
1. Install dependencies: `pip install -r requirements.txt`
2. Run full experiment pipeline: `./scripts/run_experiments.sh`
3. Run individual phases: `./scripts/run_experiments.sh --phase N`
4. Run individual experiments: `python experiments/run_baseline.py`, `python experiments/run_oram.py`, etc.

**Current state:**
- Python package structure with `src/` modules and `experiments/` scripts
- ORAM storage layer wraps PyORAM Path ORAM for CIFAR-10 samples
- Profiler tracks overhead across six categories (I/O, crypto, shuffle, serialize, compute, memory)
- ResNet-18 model adapted for CIFAR-10 (32×32 images)
- Experiment orchestrator with 5 phases: baseline, ORAM, batch sweep, dataset sweep, analysis

---

## CHANGES

(To be populated as phases execute)
