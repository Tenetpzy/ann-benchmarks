# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

ANN-Benchmarks is a framework for benchmarking approximate nearest neighbor (ANN) algorithms. It provides standardized datasets, Docker-based testing for 40+ algorithms, and visualization tools to compare performance across different metrics (recall, queries per second, index size, etc.).

## Core Architecture

### Algorithm Plugin System

Each algorithm implementation lives in `ann_benchmarks/algorithms/{ALGORITHM_NAME}/` with:
- `module.py`: Python wrapper implementing the `BaseANN` interface
- `Dockerfile`: Container specification for isolated testing
- `config.yml`: Parameter configurations organized by point type and distance metric

All algorithms must inherit from `BaseANN` (in `ann_benchmarks/algorithms/base/module.py`) and implement:
- `fit(X)`: Build the index from training data
- `query(q, n)`: Return n nearest neighbors for query vector q
- `done()`: Cleanup resources
- Optional: `batch_query(X, n)`, `set_query_arguments()`, and prepared query methods

### Configuration System

Algorithm configurations in `config.yml` use a nested structure:
```yaml
{point_type}:        # float, bit, uint8, etc.
  {distance_metric}: # euclidean, angular, hamming, jaccard, or "any"
    - name: algorithm-display-name
      constructor: PythonClassName
      module: ann_benchmarks.algorithms.{folder}
      docker_tag: ann-benchmarks-{folder}
      base_args: ['@metric']  # Variables like @metric, @dimension, @count
      run_groups:
        group_name:
          args: [[param1_values], [param2_values]]  # Cartesian product
          query_args: [[query_param_values]]
```

The `definitions.py` module parses these configs and generates all parameter combinations via `_generate_combinations()`.

### Execution Flow

1. **Definition Loading** (`main.py` → `definitions.py`):
   - Load dataset metadata to determine point_type and distance_metric
   - Parse all `config.yml` files matching those criteria
   - Generate `Definition` objects for all parameter combinations
   - Filter by existing results, disabled status, Docker availability

2. **Parallel Execution** (`main.py` → `runner.py`):
   - Queue all definitions for parallel processing
   - Spawn worker processes (controlled by `--parallelism`)
   - Each worker runs either locally or in Docker containers
   - CPU/memory limits enforced per container

3. **Benchmarking** (`runner.py`):
   - Load and transform dataset via `get_dataset()`
   - Build index with `algo.fit(X_train)` (timed, memory tracked)
   - Run queries multiple times (controlled by `--runs`), keep best time
   - Compute actual distances to verify correctness
   - Store results in `results/{dataset}/` as JSON

4. **Analysis** (`plot.py`, `create_website.py`):
   - Load all result files for a dataset
   - Compute metrics (recall, QPS, index size, etc.)
   - Generate Pareto frontier plots and comparison tables

### Dataset System

Datasets are pre-processed HDF5 files with structure:
- `train`: Training vectors for index building
- `test`: Query vectors
- `neighbors`: Ground truth (top-100 nearest neighbors)
- `distances`: Ground truth distances
- Attributes: `distance` metric, `point_type`, dimension

Access via `get_dataset(dataset_name)` which downloads if missing.

### Results Storage

Results stored at `results/{dataset}/10/{algorithm}_{params}[_batch].json`:
```json
{
  "algo": "algorithm-name",
  "build_time": float,
  "index_size": int,
  "best_search_time": float,
  "results": [{"time": float, "candidates": [(idx, dist), ...]}, ...]
}
```

## Common Development Tasks

### Building Docker Images

```bash
# Build base image and all algorithms
python install.py

# Build specific algorithm
python install.py --algorithm pgvector

# Parallel builds (faster)
python install.py --proc 4
```

### Running Benchmarks

```bash
# Run all algorithms on a dataset
python run.py --dataset glove-100-angular

# Run specific algorithm
python run.py --dataset glove-100-angular --algorithm annoy

# Local execution (no Docker, for debugging)
python run.py --dataset glove-100-angular --local

# Batch mode (all queries at once)
python run.py --dataset glove-100-angular --batch

# Control parallelism
python run.py --dataset glove-100-angular --parallelism 4

# Force re-run existing results
python run.py --dataset glove-100-angular --force
```

### Plotting Results

```bash
# Generate single plot
python plot.py --dataset glove-100-angular

# Custom axes and scales
python plot.py --dataset glove-100-angular --x-axis k-nn --y-axis qps --x-scale logit --y-scale log

# Show raw results (not just Pareto frontier)
python plot.py --dataset glove-100-angular --raw

# Generate full website with all plots
python create_website.py --outputdir website/
```

### Running Tests

```bash
# Run all unit tests
pytest

# Run specific test file
pytest test/distance_test.py
```

### Adding a New Algorithm

1. Create directory: `ann_benchmarks/algorithms/{name}/`
2. Implement `module.py` with class inheriting from `BaseANN`
3. Create `Dockerfile` (usually `FROM ann-benchmarks` base image)
4. Create `config.yml` with algorithm configurations
5. Add to `.github/workflows/benchmarks.yml` matrix
6. Build and test: `python install.py --algorithm {name}`
7. Run benchmark: `python run.py --algorithm {name} --dataset random-xs-20-angular`

## Key Implementation Details

### Memory Tracking

Memory usage is measured using `psutil.Process().memory_info().rss`. Index size = memory after fit() - memory before fit().

### Query Correctness Verification

After each query, the framework recomputes actual distances between the query and returned candidates to verify correctness. This catches implementation bugs where indices don't match vectors.

### Docker Isolation

Each algorithm runs in its own container with:
- Read-only access to datasets and algorithm code
- Write access to results directory
- Mounted Docker socket (for algorithms that spawn containers, like Elasticsearch)
- CPU pinning (single core unless `--batch` mode)
- Memory limits based on available RAM / parallelism

### Batch vs Individual Mode

- **Individual mode** (default): Queries run one at a time, single-threaded, measures per-query latency
- **Batch mode** (`--batch`): All queries provided at once, algorithm can parallelize, uses all CPUs

Batch mode results stored separately with `_batch` suffix.

### Constants and Directories

- `INDEX_DIR = "indices"`: Temporary index storage (cleaned on each run)
- `SIM_SSD_DIR_HOST = "sim_ssd"`: Simulated SSD for disk-based algorithms
- `SIM_SSD_DIR_CONTAINER = "/home/app/sim_ssd"`: Mount point in containers
- `data/`: Dataset storage (HDF5 files)
- `results/`: Benchmark results (JSON files)

## Code Style

This project uses Black and Ruff with 120-character line length (configured in `pyproject.toml`).
