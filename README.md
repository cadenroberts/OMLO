# ORAM-Integrated PyTorch Training

Quantifies computational and bandwidth overhead when integrating Path ORAM into GPU-backed PyTorch training workflows.

## Problem

Oblivious RAM hides data access patterns from adversaries, but the cryptographic and I/O costs are poorly characterized for ML workloads. This system measures where overhead originates — block I/O, AES encryption, tree reshuffling, serialization, and memory pressure — when Path ORAM replaces standard data loading in a ResNet-18 CIFAR-10 training pipeline.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system diagram.

The experiment pipeline has five stages:

1. **Baseline training** — Standard PyTorch DataLoader with ResNet-18 on CIFAR-10 (100 epochs). Establishes performance floor.
2. **ORAM training** — Same model and hyperparameters with ORAM-backed data loading. Each sample access traverses a Path ORAM tree with AES-encrypted 4KB blocks.
3. **Batch-size sweep** — Measures overhead ratio across batch sizes {32, 64, 128, 256} for both pipelines.
4. **Dataset-size sweep** — Validates theoretical O(log N) per-access scaling across dataset sizes {1000, 5000, 10000, 50000}.
5. **Analysis** — Generates overhead breakdown charts, training curve comparisons, and scaling plots.

## Key components

| Module | Responsibility |
|--------|---------------|
| `src/oram_storage.py` | PyORAM Path ORAM wrapper: serialize CIFAR-10 samples into 4KB encrypted blocks, manage tree initialization, read/write with profiling hooks |
| `src/oram_dataloader.py` | Custom PyTorch `Dataset` and `BatchSampler` backed by ORAM storage; enforces `num_workers=0` (single ORAM connection) |
| `src/oram_trainer.py` | ResNet-18 training loop with per-batch profiling of ORAM I/O, compute, and data transfer |
| `src/baseline_trainer.py` | Standard training loop for comparison |
| `src/profiler.py` | Singleton profiler tracking six overhead categories with per-operation timing distributions and memory snapshots |

## Overhead categories

1. **I/O** — ORAM block reads/writes (dominant cost)
2. **Crypto** — AES encryption/decryption inside PyORAM
3. **Shuffling** — Path ORAM eviction and tree reshuffling
4. **Serialization** — Image ↔ block format conversion
5. **Memory** — ORAM tree structure vs. flat array storage
6. **Training compute** — Forward/backward passes (constant across pipelines)

## Theoretical bounds

Path ORAM with N=50,000 samples:
- Bandwidth: O(log N) ≈ 16 block accesses per sample read
- Client storage: O(log N) for position map
- Stash size: O(log N) with high probability
- Block size: 4KB (fits 32×32×3 CIFAR-10 image + metadata)

## Results

See [RESULTS.md](RESULTS.md) for benchmark summaries and overhead analysis.

## Usage

```bash
# Install
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run all phases
./scripts/run_experiments.sh

# Run individual phases
./scripts/run_experiments.sh --phase 1    # Baseline (100 epochs)
./scripts/run_experiments.sh --phase 2    # ORAM (10 epochs, 50k samples)
./scripts/run_experiments.sh --phase 3    # Batch-size sweep
./scripts/run_experiments.sh --phase 4    # Dataset-size sweep
./scripts/run_experiments.sh --phase 5    # Analysis + report generation

# Individual experiments
python experiments/run_baseline.py --epochs 100 --batch-size 128
python experiments/run_oram.py --epochs 10 --batch-size 128
python experiments/run_sweep.py --sweep batch_size --epochs 3
python experiments/run_sweep.py --sweep dataset_size --epochs 2
python experiments/analyze_results.py --baseline results/baseline --oram results/oram
```

## Repo structure

```
oram/
├── src/
│   ├── oram_storage.py        PyORAM wrapper for CIFAR-10 block storage
│   ├── oram_dataloader.py     ORAM-backed PyTorch Dataset + BatchSampler
│   ├── oram_trainer.py        ORAM-integrated training with profiling
│   ├── baseline_trainer.py    Standard training for comparison
│   └── profiler.py            Overhead measurement infrastructure
├── experiments/
│   ├── run_baseline.py        Baseline training experiments
│   ├── run_oram.py            ORAM training experiments
│   ├── run_sweep.py           Batch-size and dataset-size sweeps
│   └── analyze_results.py     Overhead breakdown analysis and plots
├── scripts/
│   ├── run_experiments.sh     Master orchestrator (all phases)
│   └── install_cron.sh        Cron job management
└── results/                   Experiment outputs, plots, logs
```

## References

- Stefanov et al., "Path ORAM: An Extremely Simple Oblivious RAM Protocol" (CCS 2013)
- PyORAM: https://github.com/ghackebeil/PyORAM
- Talur, Demertzis, "SONIC: Concurrent Oblivious RAM" (USENIX Security 2026)
- Mavrogiannakis et al., "OBLIVIATOR" (USENIX Security 2025)

## License

MIT
