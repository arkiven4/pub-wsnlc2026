# Reproduced Table I

Qwen and HeAR embeddings for all 23,638 study clips were regenerated from the
prepared WAVs and match the paper embeddings bit for bit on our machine
(RTX 4090, TensorFlow 2.19.0). All 120 means and 120 standard deviations in
Table I (2 encoders × 3 datasets × 5 methods × 4 metrics) match the published
values.

- [Table](table_i.md)
- [Paper vs reproduced, per cell](table_i_comparison.csv)
- [Summary](validation_summary.json)

Command used (Qwen with the UK-COVID extraction context, see
[reproducibility notes](../../docs/reproducibility.md)):

```bash
python run_pipeline.py --include-uk-context --model-path /path/to/Qwen3-Omni-30B-A3B-Thinking
```
