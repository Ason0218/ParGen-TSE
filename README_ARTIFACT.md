# Usage

## Setup

```bash
python3 -m pip install -e ./src
cp src/pargen/config/api_config.example.json src/pargen/config/api_config.json
```

Configure the API key, endpoint, and model in `api_config.json`, or set `DEEPSEEK_API_KEY`,
`DEEPSEEK_BASE_URL`, and `DEEPSEEK_MODEL` in the environment. The endpoint uses the OpenAI-compatible
chat-completions API.

Install Poppler for PDF processing (`brew install poppler` on macOS or `apt-get install poppler-utils`
on Debian/Ubuntu). Parser execution requires Wireshark/tshark, Zeek with Spicy, or Scapy for the selected target.

## Pipeline

Run from the repository root. For Modbus and the Wireshark C target:

```bash
STEP2_STATIC_REPAIR_ROUNDS=5 STEP2_TRAFFIC_MCTS_ROUNDS=5 \
PYTHONPATH=src python3 -m pargen.run_pipeline \
  --protocol modbus --spec-sections --spec-select toc_llm \
  --traffic dataset/pargen/data/modbus/inputs/traffic/refinement_packets.txt \
  --semantic-max-packets 4 \
  --stage-c-host wireshark_c \
  --stage-c-pcap dataset/pargen/data/modbus/inputs/pcap/modbus_100.pcap \
  --stage-c-output-dir results/modbus/stage_c \
  --no-stage-c-semantic-fix \
  --stage-c-exp-constraint-fix --stage-c-exp-constraint-fix-max-rounds 5 \
  --stage-c-metric-l2-judge --stage-c-metric-primary-l2 human \
  --no-stage-c-metric-l2-bert
```

This example uses four unlabeled messages for PKG refinement, five static PKG refinement
rounds, five message-guided PKG refinement rounds, and five code refinement rounds.
Code refinement uses messages constructed from the PKG. Dataset scoring uses the evaluation
PCAP and its frame index.

For another protocol, change `--protocol`, `--traffic`, and `--stage-c-pcap` together using
the files in that protocol's dataset directory. Select the target with `--stage-c-host`
(`wireshark_c`, `wireshark_lua`, `zeek_spicy`, or `scapy`). Set `--stage-c-output-dir` to
choose where parser outputs are saved. Use `--help` to see all options.

## Specification Sections

With `--spec-sections`, the pipeline reads `dataset/pargen/data/<protocol>/inputs/pdfs/`. PDF and RFC text are split at
section headings; Word documents are split at heading paragraphs. `toc_llm` sends each document's extracted
heading hierarchy to the model and selects sections covering message layouts, field interpretation, and
single-message validity rules. Selected sections retain their extracted text.

The selection record stores source hashes, target protocol, prompt version, model, prompts, selected and
excluded identifiers, and model replies. Matching records are reused. The default output is
`dataset/pargen/data/<protocol>/outputs/stage_a/spec_sections_toc.json`.

To run section selection alone:

```bash
PYTHONPATH=src python3 -m pargen.run_pipeline --protocol modbus --spec-sections-only --spec-select toc_llm
```

This command writes to `dataset/pargen/data/modbus/outputs/stage_a/`. Use `--spec-select all` to retain
all extracted sections. To replay an existing sections file, replace `--spec-sections --spec-select toc_llm`
with `--sections-file <file> --raw-sections-file <file> --sections-subset <file>`.

## Semantic Evaluation

Human evaluators determine semantic equivalence. The example above also generates LLM suggestions for
review. A semantic match requires an exact field span and equivalent field meaning and decoded value.
Missing fields contribute FN, extra fields contribute FP, and an incorrect aligned pair contributes both.

Create a review file from a saved parser export:

```bash
PYTHONPATH=src python3 -m pargen.parser_realization.target_platforms.tooljson_metrics \
  --pred results/modbus/stage_c/tool_wireshark_c.json \
  --gt-level1 dataset/pargen/data/modbus/gt/modbus_100/modbus_100_level1_gt.json \
  --gt-level2 dataset/pargen/data/modbus/gt/modbus_100/modbus_100_level2_gt.json \
  --frame-index dataset/pargen/data/modbus/inputs/pcap_frame_index.json \
  --out-dir results/semantic_review \
  --no-l2-bert --no-l2-judge --primary-l2-metric human
```

1. Copy `human_semantic_review.template.json` to a separate review file.
2. Enter the reviewer identifier and each pair's `equivalent` decision.
3. Repeat the command with `--human-review <review-file.json>`.

`metrics.json` reports semantic F1 at `paper_metrics.primary_level2`. Applied decisions are saved in
`human_semantic_review.applied.json`.
