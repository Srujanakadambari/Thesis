# Final Thesis Submission Artifacts

This folder contains the final experiment evidence for the thesis project on LLM-based data-to-visualization pipelines. It includes runnable notebooks/scripts, result CSVs, generated chart configuration JSONs, rendered chart images, and Qwen-as-judge evaluation outputs.

## Folder Structure

- `Baseline/`  
  Baseline pipeline code, prompt templates, helper utilities, and baseline notebooks.

- `Architecture_changes/`  
  Main architecture scenario experiment. This includes:
  - `master_run_pipeline.ipynb`: runs the architecture scenarios.
  - `master_run_results/`: generated configs, rendered images, latency/quality CSVs, and Qwen judge outputs.
  - `post_hoc_eval/`: post-hoc structural quality checks and Qwen strict/lenient/configurable judge results.
  - `architecture_rendered_images_quality validator/`: rendered scenario images and quality benchmark tables.

- `Bottlenecks/`  
  Bottleneck and token/latency analysis artifacts, including result CSV/JSON files and bottleneck figures.

- `Qwen3.6-27B_ Baseline S0/`  
  Qwen baseline experiment where Qwen replaces the OpenAI baseline for S0/RQ7-style comparison.

- `cache/`  
  Cache experiment notebooks, cache result CSVs, cached config JSONs, and rendered cache comparison images.

- `combined_strategies/`  
  Combined optimization strategy results, including SA1 cache/parallel strategy CSVs and figures.

- `parallel execution/`  
  RQ5 parallel execution experiment files, consolidated result CSV/JSON files, Gantt/latency/throughput figures, rendered images, and benchmark scripts.

- `prompt style/`  
  Prompt-size and prompt-style experiments, including benchmark notebooks, CSVs, rendered outputs, and summary figures.

- `prompts/`  
  Prompt template definitions used by the baseline pipeline.

- `app.py`  
  Optional Streamlit dashboard for inspecting benchmark scenarios.

- `retrieve_data.py`  
  Local test data loader used by several experiments.

## Key Result Files

- Architecture master run:
  - `Architecture_changes/master_run_results/master_pipeline_runs.csv`
  - `Architecture_changes/master_run_results/configs/*.json`
  - `Architecture_changes/master_run_results/renders/*.png`

- Qwen-as-judge:
  - `Architecture_changes/master_run_results/qwen_freeform_judge/qwen_freeform_verdicts.csv`
  - `Architecture_changes/master_run_results/qwen_freeform_judge/qwen_freeform_summary.csv`
  - `Architecture_changes/master_run_results/qwen_stateless_judge/qwen_stateless_verdicts.csv`
  - `Architecture_changes/master_run_results/qwen_stateless_judge/qwen_stateless_summary.csv`

- Post-hoc evaluation:
  - `Architecture_changes/post_hoc_eval/quality.csv`
  - `Architecture_changes/post_hoc_eval/judge.csv`
  - `Architecture_changes/post_hoc_eval/combined.csv`
  - `Architecture_changes/post_hoc_eval/summary.csv`

- Bottleneck analysis:
  - `Bottlenecks/bottleneck_thesis_results.csv`
  - `Bottlenecks/bottleneck_thesis_summary.json`

- Prompt experiments:
  - `prompt style/rq3_prompt_size.csv`
  - `prompt style/prompt_style_results/style_benchmark.csv`

- Parallel execution:
  - `parallel execution/parallel_exec/rq5_consolidated_outputs/rq5_summary.csv`
  - `parallel execution/parallel_exec/rq5_consolidated_outputs/rq5_raw_results.csv`
  - `parallel execution/parallel_exec/rq5_throughput_results.csv`

## Environment Setup

Create an environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Some notebooks call external LLM endpoints. To rerun those cells, provide the relevant environment variables in a `.env` file, for example:

```bash
OPENAI_API_KEY=...
OPENAI_BASE_URL=...
MODEL_NAME=o4-mini
PHOENIX_COLLECTOR_ENDPOINT=...
```

The Qwen experiments used a vLLM-compatible server endpoint in the notebooks. The saved result files can be inspected without access to that server.

## Notes

- The folder is primarily an evidence package. Most thesis claims can be verified from the saved CSV/JSON/PNG files without rerunning LLM calls.
- `master_run_results` contains outputs from the original architecture generation experiment.
- `post_hoc_eval` contains later quality/judge evaluation of the already-generated architecture configs.
- Qwen judge outputs record the judge model as `Qwen3.6-27B`.
