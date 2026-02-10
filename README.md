# ORAM-Integrated PyTorch Training Baseline

**CSE239A Thesis Project** - Caden Roberts  
*Privacy-Preserving Machine Learning Infrastructure: Scalable Oblivious Computation for Enterprise AI Systems*

## Overview

This project establishes a baseline for measuring ORAM (Oblivious RAM) overhead when integrated with PyTorch training pipelines. By comparing standard CIFAR-10 training against ORAM-backed data loading, we identify where overhead originates (I/O, cryptographic operations, memory bandwidth) to inform future optimizations in oblivious shuffling and aggregation.

## Project Structure

```
oram/
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── src/
│   ├── __init__.py
│   ├── oram_storage.py           # PyORAM wrapper for CIFAR-10 block storage
│   ├── oram_dataloader.py        # Custom ORAM-backed PyTorch DataLoader
│   ├── baseline_trainer.py       # Standard PyTorch training
│   ├── oram_trainer.py           # ORAM-integrated training with profiling
│   └── profiler.py               # Overhead measurement utilities
├── experiments/
│   ├── run_baseline.py           # Run standard training experiments
│   ├── run_oram.py               # Run ORAM training experiments
│   ├── run_sweep.py              # Batch-size and dataset-size parameter sweeps
│   └── analyze_results.py        # Generate overhead breakdown analysis and plots
├── scripts/
│   ├── run_experiments.sh        # Master orchestrator (all phases, cron-safe)
│   └── install_cron.sh           # Install/remove cron jobs
├── data/                         # CIFAR-10 dataset (auto-downloaded)
└── results/                      # Experiment outputs, plots, and logs
    ├── baseline/                 # Baseline training results
    ├── oram/                     # ORAM training results
    ├── sweep_batch_size/         # Batch-size sweep results
    ├── sweep_dataset_size/       # Dataset-size sweep results
    ├── analysis/                 # Generated plots and report
    └── logs/                     # Cron and experiment logs
```

## Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running Experiments

### Option 1: Automated (Recommended)

The master orchestrator runs all five experiment phases in sequence, skipping phases whose results already exist:

```bash
./scripts/run_experiments.sh
```

To re-run a specific phase:

```bash
./scripts/run_experiments.sh --phase 1    # Baseline training (100 epochs)
./scripts/run_experiments.sh --phase 2    # ORAM training (10 epochs, 50k samples)
./scripts/run_experiments.sh --phase 3    # Batch-size sweep
./scripts/run_experiments.sh --phase 4    # Dataset-size sweep
./scripts/run_experiments.sh --phase 5    # Analysis and report generation
```

To force re-run even if results exist:

```bash
./scripts/run_experiments.sh --force
```

### Option 2: Cron (Unattended)

Install cron jobs to run experiments automatically:

```bash
./scripts/install_cron.sh            # Install (starts in ~5 minutes)
./scripts/install_cron.sh --remove   # Remove cron entries
crontab -l                           # View scheduled jobs
```

The cron setup schedules:
- **Experiment run**: starts shortly after installation (lock file prevents duplicate runs)
- **Nightly analysis**: re-generates plots at 3:00 AM to pick up any new data

Logs are written to `results/logs/`.

### Option 3: Individual Experiments

```bash
# Baseline training
python experiments/run_baseline.py --epochs 100 --batch-size 128

# ORAM training (full dataset)
python experiments/run_oram.py --epochs 10 --batch-size 128

# ORAM training (subset for faster iteration)
python experiments/run_oram.py --epochs 5 --batch-size 128 --num-samples 5000

# Parameter sweeps
python experiments/run_sweep.py --sweep batch_size --epochs 3
python experiments/run_sweep.py --sweep dataset_size --epochs 2

# Analysis
python experiments/analyze_results.py --baseline results/baseline --oram results/oram
```

## Experiment Design

### Phase 1: Baseline Training
- ResNet-18 on CIFAR-10, standard PyTorch DataLoader
- 100 epochs, batch size 128, SGD with LR schedule
- Establishes performance floor (accuracy and timing)

### Phase 2: ORAM Training
- Same model/hyperparameters, ORAM-backed data loading
- 10 epochs (sufficient for stable overhead measurements)
- Full 50,000-sample CIFAR-10 training set in Path ORAM

### Phase 3: Batch-Size Sweep
- Batch sizes: {32, 64, 128, 256}
- 3 epochs per configuration, both baseline and ORAM
- Measures how batch size affects ORAM overhead ratio

### Phase 4: Dataset-Size Sweep
- Dataset sizes: {1000, 5000, 10000, 50000}
- 2 epochs per configuration, ORAM only
- Validates theoretical O(log N) per-access scaling

### Phase 5: Analysis
- Training curve comparison (loss, accuracy)
- Overhead breakdown pie/bar charts
- Per-operation timing distributions
- Batch-size and dataset-size sweep plots
- Markdown report with theoretical analysis

## Measurement Methodology

### Overhead Categories

1. **I/O Time**: Time spent reading/writing ORAM blocks
2. **Crypto Time**: AES encryption/decryption operations (inside PyORAM)
3. **Shuffling Time**: Path ORAM eviction and reshuffling operations
4. **Serialization**: Converting images to/from ORAM block format
5. **Memory Overhead**: ORAM tree structure vs. standard array storage
6. **Training Compute**: Per-batch forward/backward passes

### Theoretical Bounds

Path ORAM with N=50,000 samples:
- Bandwidth: O(log N) ~ 16 block accesses per sample read
- Client storage: O(log N) for position map
- Stash size: O(log N) with high probability
- Block size: 4KB (fits 32x32x3 CIFAR-10 image + metadata)

## Output Files

After a complete run, `results/analysis/` contains:
- `training_comparison.png` - Baseline vs ORAM training curves
- `overhead_breakdown.png` - Pie/bar chart of overhead sources
- `operation_times.png` - Per-operation timing breakdown
- `batch_size_sweep.png` - Overhead ratio vs batch size
- `dataset_size_sweep.png` - Scaling behavior vs dataset size
- `overhead_report.md` - Full markdown analysis report

## References

- Stefanov et al., "Path ORAM: An Extremely Simple Oblivious RAM Protocol" (CCS 2013)
- PyORAM: https://github.com/ghackebeil/PyORAM
- Talur, Demertzis, "SONIC: Concurrent Oblivious RAM" (USENIX Security 2026)
- Mavrogiannakis et al., "OBLIVIATOR" (USENIX Security 2025)
- Thesis Proposal: "Privacy-Preserving Machine Learning Infrastructure"

## License

MIT License - Academic use for CSE239A coursework.
