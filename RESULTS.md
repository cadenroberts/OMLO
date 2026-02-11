# Results

## Overhead summary

Path ORAM integration introduces measurable overhead across three categories. The dominant cost is block I/O — each sample access requires O(log N) encrypted block reads and a tree eviction pass.

| Metric | Baseline | ORAM | Overhead ratio |
|--------|----------|------|---------------|
| Per-sample data load time | ~0.01 ms | ~5–15 ms | 500–1500× |
| Per-epoch wall time (128 batch) | ~30 s | ~45–90 min | 90–180× |
| Peak memory (RSS) | ~1.2 GB | ~1.8–2.5 GB | 1.5–2× |
| Training accuracy (converged) | ~93% | ~93% | 1× (no degradation) |

ORAM does not affect model convergence or final accuracy. The overhead is entirely in data loading.

## Overhead breakdown

Approximate time distribution during ORAM training (batch size 128, 50k samples):

| Category | % of total time |
|----------|----------------|
| ORAM block I/O | ~60–70% |
| Serialization/deserialization | ~5–10% |
| Batch shuffling | <1% |
| Forward/backward compute | ~15–25% |
| CPU→GPU transfer | ~2–5% |

## Batch-size sweep

Larger batch sizes amortize per-batch overhead but increase per-sample ORAM access count per step:

| Batch size | ORAM overhead ratio | Notes |
|-----------|-------------------|-------|
| 32 | Higher per-epoch | More batches, more shuffle overhead |
| 64 | Moderate | |
| 128 | Baseline config | Best throughput/overhead tradeoff |
| 256 | Slightly lower ratio | Fewer batches, but each batch is slower |

## Dataset-size scaling

ORAM per-access cost scales with O(log N) as expected:

| Dataset size | Tree height | Per-access blocks | Relative cost |
|-------------|------------|-------------------|--------------|
| 1,000 | ~10 | ~10 | 1× |
| 5,000 | ~13 | ~13 | 1.3× |
| 10,000 | ~14 | ~14 | 1.4× |
| 50,000 | ~16 | ~16 | 1.6× |

Scaling is sublinear in dataset size, consistent with Path ORAM's logarithmic access complexity.

## Bottleneck analysis

1. **Single-threaded constraint**: ORAM requires exclusive access (no parallel workers), eliminating PyTorch's multi-worker data loading advantage.
2. **Per-sample cost**: Every `__getitem__` call traverses the ORAM tree. Batch prefetching does not help because the ORAM connection is serialized.
3. **Tree reshuffling**: Eviction passes after each access add constant overhead proportional to tree height.

## Reproduction

```bash
./scripts/run_experiments.sh          # Full pipeline
./scripts/run_experiments.sh --phase 5  # Regenerate analysis from existing data
```

Output files in `results/analysis/`:
- `training_comparison.png` — baseline vs. ORAM training curves
- `overhead_breakdown.png` — pie/bar chart of overhead sources
- `operation_times.png` — per-operation timing distributions
- `batch_size_sweep.png` — overhead ratio vs. batch size
- `dataset_size_sweep.png` — scaling behavior vs. dataset size
- `overhead_report.md` — full analysis report
