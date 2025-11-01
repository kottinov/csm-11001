# LATS-RCA Reference Implementation

This repository accompanies the *Multi-Agent Systems for Root Cause Analysis in Microservices* thesis. It contains the LaTeX sources for the thesis as well as the Python implementation of the LATS-RCA agents.

## Project Layout

- `thesis/` – LaTeX sources for the written thesis.
- `src/rca_agents/` – Python package implementing the LATS agents, tools, and orchestrators.
- `lo2-sample/` – Normalised Light OAuth2 dataset used throughout the evaluation.
- `scripts/` – Utility scripts, including the benchmark runner that reproduces the evaluation scenarios.
- `outputs/` – Sample reports produced by the supervisor flow.

## Prerequisites

1. Install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Create a `.env` based on `.env.example` and provide your `ANTHROPIC_API_KEY`. Optional environment variables let you tweak search depth, temperature, and dataset locations.

## Running the Agents

The simplest way to execute both agents and generate a synthesis report is via the supervisor:

```bash
python -m rca_agents.orchestrator.supervisor
```

For individual agents:

```bash
python -m rca_agents.cli.log_agent --no-stream
python -m rca_agents.cli.metrics_agent --no-stream
```

Each CLI emits structured JSON, enabling downstream automation.

## Reproducing the OAuth2 Benchmark

The script below iterates through the Light OAuth2 scenarios and records both agent outputs along with ground-truth metadata.

```bash
python scripts/run_oauth2_benchmark.py \
    --dataset-root lo2-sample \
    --timestamp 1719592986 \
    --output outputs/oauth2_benchmark_results.jsonl
```

Use `--dry-run` to verify configuration without invoking the Anthropic API, or `--limit` to restrict the number of scenarios.

## Testing

A lightweight test suite validates the enhanced LATS mechanics:

```bash
pytest
```

## Building the Thesis

```bash
cd thesis/src
pdflatex gradu.tex
bibtex gradu
pdflatex gradu.tex
pdflatex gradu.tex
```

Alternatively, run `thesis/build.sh` from the repository root.
