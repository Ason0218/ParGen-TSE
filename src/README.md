# ParGen Core Code

This directory contains the core implementation used by the ParGen artifact.
The Python package path is `pargen/`.

## Components

- `pargen/spec_sections.py`: document splitting and section selection.
- `pargen/specification_processing/`: PDF content extraction.
- `pargen/protocol_knowledge/`: protocol-knowledge construction and refinement.
- `pargen/parser_realization/`: parser realization and conformance refinement.
- `pargen/config/`: provider configuration template.
- `pargen/run_pipeline.py`: end-to-end orchestration entry point.

## Installation

```bash
python3 -m pip install -e .
```

## Configuration

Copy the configuration template and set the API key, endpoint, and model:

```bash
cp pargen/config/api_config.example.json pargen/config/api_config.json
```

`api_config.json` is excluded by `.gitignore`.

## Data Layout

The artifact stores data outside the installable package under `../dataset`.
Intermediate outputs are stored under each protocol's `outputs/` directory.
Use `--stage-c-output-dir` to select a different directory for parser outputs.

## Main Entrypoint

```bash
python -m pargen.run_pipeline --help
```

See [Usage](../README_ARTIFACT.md) for a complete example run from the artifact root.
