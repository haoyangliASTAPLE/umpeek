# UMPeek Artifact

This anonymous repository accompanies the paper *Inferring User Models from
Personalized AI Behavior*. It contains the first-party code and a compact set
of records needed to inspect UMPeek, run a small evaluation, and reproduce the
reported aggregation and plotting pipeline.

## What is included

- `src/umpeek/attack_baselines/`: the UMPeek implementation and its shared
  attack input, output, and scoring interfaces.
- `src/umpeek/eval2/`: benchmark splits, metrics, held-out checks, attack
  execution, and aggregation.
- `src/umpeek/real_agent/`: the Qwen3-14B/vLLM victim interface and the Mem0,
  Graphiti, and LangMem+LangGraph memory adapters.
- `src/umpeek/defenses/`: PrivacyChecker, Theory-of-Mind Defense, and Stateful
  Counterfactual Exposure Control as evaluated in the paper.
- `scripts/`: preprocessing, minimal evaluation, full experiment launchers,
  and table/figure exporters.
- `data/release_samples/`: hashed sample identifiers, public requests,
  interface-visible outputs, split manifests, and compact derived records.
- `results/`: paper-facing TeX tables, PDF figures, and their numeric inputs.

The release intentionally does **not** contain third-party datasets, cloned
baseline repositories, model weights, access credentials, provider resources,
private backend state, raw profile/history rows, held-out behavior, or full
internal trajectories. It also does not contain comparison-method code,
ports, adapters, or local reproductions. See [docs/DATA.md](docs/DATA.md) and
[docs/BASELINES.md](docs/BASELINES.md) for dataset and official method sources.

## Quick check without a model

Python 3.10 or later is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[test,plot]"
python scripts/verify_release.py
pytest -q
python scripts/rebuild_release_artifacts.py
```

The verification command parses every released JSON/JSONL/CSV record, checks
the expected anonymous sample counts, and confirms that the paper-facing
tables and figures are present.

## Minimal live evaluation

The paper used Qwen3-14B served by vLLM in non-thinking mode. Start an
OpenAI-compatible endpoint, for example:

```bash
vllm serve Qwen/Qwen3-14B \
  --host 127.0.0.1 --port 8010 \
  --served-model-name Qwen/Qwen3-14B \
  --dtype bfloat16 --max-model-len 32768
```

Acquire the public benchmarks as described in [docs/DATA.md](docs/DATA.md),
then run `python scripts/preprocess_benchmarks.py`. The preprocessing step creates the private
evaluator rows, the public attack-probe rows, and the independent held-out
rows. The latter two are kept separate by construction.

Build the compact release manifest after preprocessing:

```bash
python scripts/build_minimal_manifest.py
```

Run one UMPeek sample from one setting:

```bash
python scripts/run_minimal_evaluation.py \
  --manifest runs/manifest/full_matrix_manifest.json \
  --backend Mem0 \
  --benchmark PersonaMem-v2 \
  --method UMPeek_final \
  --limit 1
```

Add `--dry-run` to validate the setting without calling the model. Environment
variables and the exact frozen evaluation choices are listed in
[docs/REPRODUCTION.md](docs/REPRODUCTION.md) and
[`configs/evaluation.json`](configs/evaluation.json).

## Results and exporters

The compact numeric inputs and reference outputs are organized as follows:

- `results/exp1/`: personalization interventions and token efficiency.
- `results/exp3/`: attack comparison table.
- `results/exp4/`: UMPeek evidence-step ablations.
- `results/adaptive_defense/`: defense table, threshold/budget figure, and
  mechanism ablations.

`scripts/rebuild_release_artifacts.py` regenerates all released TeX tables and
PDF/JPG figures directly from these compact numeric inputs. The full-run
exporters are `scripts/export_exp1_causal_interventions.py`,
`scripts/export_exp3_table2.py`, `scripts/export_exp4_paper_artifacts.py`, and
`scripts/export_adaptive_defense_artifacts.py`; they aggregate newly generated
per-sample runs. Full evaluator records are not redistributed.

The Exp3 aggregates retain the submitted comparison numbers so that Table 2
can be inspected and rebuilt. The live runner in this artifact executes only
UMPeek. Readers who wish to rerun a comparison method should obtain its
official implementation from [docs/BASELINES.md](docs/BASELINES.md).

## Privacy boundary

Files under `data/release_samples/` contain only public task text,
interface-visible model behavior, hashed identifiers, execution metadata, and
derived scores. They do not contain gold user state, private memory contents,
retrieval logs, memory identifiers, hidden benchmark rows, or held-out tasks.
PersonaMem-v2 personas used by this artifact are synthetic; no records were
collected from real users.

## Repository status

This is the anonymous review artifact. If the paper is accepted, the same
materials will be released through a permanent non-anonymous archive.
