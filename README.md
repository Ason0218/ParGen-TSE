# ParGen

Implementation of **Protocol Parser Generation from Natural-Language Specifications via Conflict-Aware Refinement**.

ParGen constructs an executable Protocol Knowledge Graph (PKG) from protocol specifications and generates
parsers for Wireshark, Zeek, and Scapy. It refines the PKG and generated parsers using localized conflict reports.

## Setup

Requirements: Python 3.10+, Poppler (`pdftotext`), an LLM API configuration, and the selected parser platform.

```bash
python3 -m pip install -e ./src
cp src/pargen/config/api_config.example.json src/pargen/config/api_config.json
```

Set the API key, endpoint, and model in `api_config.json`.

## Run

```bash
PYTHONPATH=src python3 -m pargen.run_pipeline --help
```

Use the Python entry point to select the protocol, input files, and target platform.
See [Usage](README_ARTIFACT.md) for a Modbus example and evaluation commands.

## Contents

- `src/pargen/`: implementation.
- `dataset/pargen/`: protocol specifications, traffic, and annotations.
- `src/pargen/run_pipeline.py`: end-to-end pipeline entry point.

See [Usage](README_ARTIFACT.md) for configuration and evaluation commands, and
[Dataset](DATASET_MANIFEST.md) for the input inventory.
