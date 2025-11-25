# Experiments

This directory contains experiment configurations and results for evaluating different RAG pipeline configurations.

## Structure

```
experiments/
├── run.py                  # Main experiment runner script
├── configs/                # Experiment configuration files (YAML)
│   └── example_experiment.yaml
└── runs/                   # Experiment results (auto-generated)
    └── 2025-11-25_12-34-56_my_experiment/
        ├── config.yaml     # Config used for this run
        ├── metrics.json    # Computed metrics
        └── results.jsonl   # Detailed results per query
```

## Quick Start

### 1. Create an Experiment Config

Copy and modify `configs/example_experiment.yaml`:

```yaml
name: my_new_experiment
description: "Testing new preprocessing method"

preprocess_method: pe_domain
preprocess_version: v2

embedding_method: bge_large
embedding_version: v1

retrieval:
  method: hybrid
  version: v1
  top_k: 50
  # ... more config ...
```

### 2. Prepare Evaluation Dataset

Create a JSONL file with your evaluation data:

```jsonl
{"question": "What causes pump cavitation?", "ground_truth": "...", "doc_ids": ["doc123"]}
{"question": "How to fix valve leakage?", "ground_truth": "...", "doc_ids": ["doc456"]}
```

### 3. Run Experiment

```bash
python -m experiments.run \
    --config experiments/configs/my_new_experiment.yaml \
    --dataset data/eval/pe_agent_eval.jsonl \
    --output experiments/runs/my_new_experiment/
```

### 4. View Results

Results are saved in the output directory:

```bash
# View metrics
cat experiments/runs/my_new_experiment/metrics.json

# View detailed results
less experiments/runs/my_new_experiment/results.jsonl
```

## Experiment Workflow

### Testing a New Paper/Method

When you read a new paper and want to test the approach:

1. **Implement the method** in the appropriate module:
   - Preprocessing: `backend/llm_infrastructure/preprocessing/methods/`
   - Embedding: `backend/llm_infrastructure/embedding/embedders/`
   - Retrieval: `backend/llm_infrastructure/retrieval/methods/`

2. **Register the method** using the decorator:
   ```python
   @register_preprocessor("paper_xyz_method", version="v1")
   class PaperXYZPreprocessor(BasePreprocessor):
       ...
   ```

3. **Create experiment config** referencing the new method:
   ```yaml
   preprocess_method: paper_xyz_method
   preprocess_version: v1
   ```

4. **Run experiment** and compare with baseline:
   ```bash
   python -m experiments.run --config configs/paper_xyz.yaml ...
   ```

### Ablation Studies

To test the impact of individual components:

```bash
# Baseline: dense only
python -m experiments.run --config configs/dense_only.yaml ...

# Add sparse: hybrid
python -m experiments.run --config configs/hybrid.yaml ...

# Add multi-query
python -m experiments.run --config configs/hybrid_multiquery.yaml ...

# Add reranking
python -m experiments.run --config configs/hybrid_rerank.yaml ...

# Full pipeline
python -m experiments.run --config configs/full_pipeline.yaml ...
```

### Grid Search / Parameter Sweep

For systematic parameter exploration, you can:

1. Create multiple config files programmatically
2. Run them in a loop
3. Compare results

Example script:
```python
import itertools
from pathlib import Path

chunk_sizes = [256, 512, 1024]
embeddings = ["bge_base", "bge_large"]
top_ks = [10, 20, 50]

for cs, emb, k in itertools.product(chunk_sizes, embeddings, top_ks):
    config_name = f"sweep_cs{cs}_emb{emb}_k{k}"
    # Generate config file
    # Run experiment
```

## Metrics

The experiment runner computes the following metrics:

- **hit@k**: Percentage of queries where relevant doc is in top-k results
- **MRR**: Mean Reciprocal Rank (average of 1/rank of first relevant doc)
- **avg_latency_ms**: Average query processing time

You can add custom metrics in `run.py:compute_metrics()`.

## Tips

1. **Start simple**: Begin with a baseline config, then incrementally add complexity
2. **Version everything**: Use version strings (v1, v2, ...) for all methods
3. **Document changes**: Add descriptions to config files explaining what you're testing
4. **Compare apples to apples**: Keep all other variables constant when testing one change
5. **Save configs**: Never delete experiment configs - they're your research log

## Advanced: Integration with Experiment Tracking

For more sophisticated experiment tracking, you can integrate with:
- **Weights & Biases (wandb)**: `wandb.init()` in `run.py`
- **MLflow**: Track experiments and models
- **TensorBoard**: Visualize metrics over time

Example WandB integration:
```python
import wandb

wandb.init(project="pe-agent-rag", config=config)
wandb.log(metrics)
wandb.save(str(results_path))
```
