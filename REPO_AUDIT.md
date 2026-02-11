# Repository Audit

## 1. Purpose

Quantifies computational and I/O overhead when integrating Path ORAM into GPU-backed PyTorch training workflows. Specifically measures where overhead originates (block I/O, AES encryption, tree reshuffling, serialization, memory pressure) when Path ORAM replaces standard data loading in a ResNet-18 CIFAR-10 training pipeline.

## 2. Entry Points

**Primary experiment orchestrator:**
- `scripts/run_experiments.sh` — Master bash script that runs all 5 phases sequentially or individually with locking and logging

**Python experiment entry points:**
- `experiments/run_baseline.py` — Standard PyTorch training (no ORAM)
- `experiments/run_oram.py` — ORAM-integrated training
- `experiments/run_sweep.py` — Batch-size and dataset-size sweeps
- `experiments/analyze_results.py` — Overhead breakdown analysis and visualization

**Library modules (imported by experiments):**
- `src/baseline_trainer.py` — Standard ResNet-18 trainer on CIFAR-10
- `src/oram_trainer.py` — ORAM-integrated ResNet-18 trainer
- `src/oram_storage.py` — PyORAM wrapper for CIFAR-10 block storage
- `src/oram_dataloader.py` — Custom PyTorch Dataset backed by ORAM
- `src/profiler.py` — Singleton profiler tracking overhead categories

## 3. Dependency Surface

**Runtime dependencies:**
- `torch>=2.0.0`, `torchvision>=0.15.0` — ML framework
- `pyoram>=0.3.0` — Path ORAM implementation
- `cryptography>=41.0.0`, `pycryptodome>=3.19.0` — AES backend for PyORAM
- `psutil>=5.9.0` — Memory profiling
- `numpy>=1.24.0` — Array operations

**Development/analysis dependencies:**
- `matplotlib>=3.7.0`, `seaborn>=0.12.0`, `pandas>=2.0.0` — Plotting and analysis
- `memory-profiler>=0.61.0` — Memory profiling
- `tqdm>=4.65.0` — Progress bars
- `pyyaml>=6.0` — Configuration (unused currently, appears to be dead dependency)

**External I/O dependencies:**
- CIFAR-10 dataset (downloaded via torchvision)
- Filesystem storage for PyORAM tree (`oram.bin` files, typically in temp directories or `results/`)

## 4. Configuration Surface

**No configuration files currently.** All configuration is via CLI arguments:

**Baseline experiment:**
- `--epochs` (default: 100)
- `--batch-size` (default: 128)
- `--output-dir` (default: `results/baseline`)
- `--device` (default: auto-detect cuda/cpu)

**ORAM experiment:**
- `--epochs` (default: 10, reduced due to overhead)
- `--batch-size` (default: 128)
- `--num-samples` (default: 50000, full CIFAR-10 train set)
- `--output-dir` (default: `results/oram`)
- `--device` (default: auto-detect cuda/cpu)

**Sweep experiments:**
- `--sweep` (choices: `batch_size`, `dataset_size`)
- `--epochs` (default: 3 for batch_size, 2 for dataset_size)
- `--output-dir` (default: `results`)

**Hardcoded constants:**
- ORAM block size: 4096 bytes (4KB)
- CIFAR-10 image size: 3072 bytes (32×32×3)
- Learning rate schedule: MultiStepLR milestones at epochs [50, 75, 90]
- SGD optimizer: lr=0.1, momentum=0.9, weight_decay=5e-4
- ResNet-18 architecture modifications for CIFAR-10: conv1 3×3 stride 1, maxpool removed, fc 512→10

## 5. Data Flow

**Baseline path:**
```
CIFAR-10 download → torchvision.datasets.CIFAR10 → DataLoader (4 workers, shuffle, augmentation) → GPU batches → ResNet-18 → loss/gradients → optimizer step
```

**ORAM path:**
```
CIFAR-10 download → load into ORAMStorage (write 50k samples as encrypted 4KB blocks) → ORAMDataset.__getitem__() → ORAM.read_block(index) → [AES decrypt + tree eviction] → deserialize → transform → DataLoader (0 workers, oblivious batch sampler) → GPU batches → ResNet-18 → loss/gradients → optimizer step
```

**Key differences:**
- ORAM path serializes each sample into a 4KB encrypted block during setup
- Every `__getitem__` traverses O(log N) blocks in the Path ORAM tree
- ORAM requires single-threaded access (num_workers=0)
- Baseline uses multi-worker prefetching and caching

**Profiler data flow:**
- Singleton `Profiler` instance tracks timing for categories: `io`, `oram_read`, `oram_write`, `serialize`, `deserialize`, `shuffle`, `dataload`, `compute`, `transfer`, `batch`, `epoch`, `setup`
- Each category accumulates `TimingStats` (total, count, min, max, samples)
- Memory snapshots taken periodically via `psutil`
- At experiment end, profiler writes JSON to `results/{experiment}_profile.json`

## 6. Determinism Risks

**Nondeterministic sources:**
- Random weight initialization (no seed set)
- CIFAR-10 data augmentation (RandomCrop, RandomHorizontalFlip) uses PyTorch default RNG
- ObliviousBatchSampler shuffling (seeded shuffle available via `seed` parameter, but not used by default)
- CUDA kernel execution order (no deterministic flags set)

**External calls:**
- CIFAR-10 download via HTTP (torchvision) — cached after first download
- PyORAM tree initialization (deterministic given block count and block size, but random key generation internally)
- Filesystem I/O for ORAM storage files (temp directories by default, path nondeterministic)

**Implications:**
- Training curves and final accuracies will vary across runs
- Profiler timing measurements are noisy due to system load
- For reproducibility, would need: fixed seeds, deterministic CUDA, fixed ORAM storage paths, disabled augmentation for testing

## 7. Observability

**Logging:**
- `tqdm` progress bars for epoch and evaluation loops
- Print statements for setup, epoch metrics, final summary
- `scripts/run_experiments.sh` redirects stdout/stderr to timestamped log files in `results/logs/`

**Metrics:**
- Training loss, training accuracy, test loss, test accuracy (per epoch)
- Learning rate schedule (per epoch)
- Timing breakdown by category (via Profiler)
- Memory usage (peak RSS, peak VMS)
- Per-batch metrics (loss, batch_time)

**Error handling:**
- No explicit exception handling in most entry points (crashes propagate)
- `scripts/run_experiments.sh` uses `set -euo pipefail` (fail fast)
- Lock file guard against concurrent runs

**Missing observability:**
- No structured logging (JSON logs, metrics database)
- No real-time monitoring or dashboard
- No alerting for failures
- No GPU utilization tracking (CUDA profiler not integrated)

## 8. Test State

**No automated tests.**

**Test coverage:**
- Unit tests: 0
- Integration tests: 0
- End-to-end tests: 0

**Verification strategy:**
- Manual inspection of results in `results/` directories
- Visual inspection of training curves and overhead plots (generated by `analyze_results.py`)
- Sanity checks: ORAM training should converge to similar accuracy as baseline, overhead ratio should match theoretical O(log N)

**Reliability:**
- Experiments are long-running (baseline: ~1 hour for 100 epochs, ORAM: several hours)
- No checkpointing or resume capability
- Failed runs must restart from scratch

## 9. Reproducibility

**Pinned dependencies:**
- `requirements.txt` specifies minimum versions with `>=`, not exact pins
- No lockfile (e.g., `requirements.lock`, `Pipfile.lock`)

**Build steps:**
1. Create virtual environment: `python -m venv venv`
2. Activate: `source venv/bin/activate`
3. Install: `pip install -r requirements.txt`
4. Download CIFAR-10 (happens automatically on first run)

**Reproducibility issues:**
- Dependency versions will drift over time with `>=` constraints
- No environment specification (Python version, CUDA version, OS)
- No seed management for random operations
- ORAM storage paths are temp directories (nondeterministic, cleaned up after runs)

**Best practices missing:**
- Exact version lockfile
- Docker or conda environment specification
- Seed configuration
- Deterministic CUDA flags

## 10. Security Surface

**Secrets:**
- No credentials, API keys, or tokens required
- `.env` example file exists but is not used by the code

**External APIs:**
- CIFAR-10 download from `https://www.cs.toronto.edu/~kriz/cifar.html` (via torchvision)

**File access:**
- Writes to `results/` directory (user-controlled)
- ORAM storage writes to temp directories or user-specified paths
- No privileged file operations

**Attack surface:**
- Pickle deserialization via PyTorch model loading (no untrusted models loaded in repo)
- Filesystem writes to `results/` (could fill disk with large experiments)
- No input validation on CLI arguments (e.g., negative batch size, out-of-range epochs)

**ORAM security context:**
- PyORAM provides cryptographic guarantees (AES-encrypted blocks, oblivious access patterns)
- Key management is handled by PyORAM internally (not exposed to user)
- This repo measures performance overhead, not security properties

## 11. Ranked Improvement List

### P0 (Critical for reproducibility and correctness)
1. Add minimal smoke test (`scripts/demo.sh`) that runs 1 epoch baseline + 1 epoch ORAM and verifies non-zero accuracy
2. Pin exact dependency versions (`pip freeze > requirements.lock`)
3. Document Python version, CUDA version, OS requirements
4. Add input validation for CLI arguments (positive integers, valid paths)

### P1 (Important for reliability and usability)
5. Add checkpointing to resume interrupted experiments
6. Add seed management for deterministic experiments
7. Add structured logging (JSON format) for automated analysis
8. Add CI workflow (`.github/workflows/ci.yml`) that runs smoke test
9. Add GPU utilization tracking (via `nvidia-smi` or `torch.cuda` profiling)
10. Remove unused `pyyaml` dependency or document its purpose

### P2 (Nice to have for long-term maintainability)
11. Add unit tests for `ORAMStorage`, `Profiler`, `ORAMDataset`
12. Add Docker or conda environment specification
13. Add resume/restart capability for long-running experiments
14. Add real-time monitoring dashboard (e.g., TensorBoard integration)
15. Add ablation studies (e.g., ORAM with different block sizes, tree heights)
