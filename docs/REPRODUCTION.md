# Reproduction notes

## Frozen setting

- Victim model: `Qwen/Qwen3-14B`
- Serving layer: vLLM with an OpenAI-compatible local endpoint
- Thinking mode: disabled
- Temperature: `0.0`
- Main query budget grid: `1, 2, 4, 8, 16`
- Backends: Mem0, Graphiti, LangMem+LangGraph
- Benchmarks: PersonaMem-v2, PersonaLens, ETAPP, LoCoMo
- UMPeek method version: `r007_active_bayesian_profile_denoising_v004`
- User-model metric scope: `latent_user_model_v2`

The compact machine-readable setting is in `configs/evaluation.json`.

The artifact was checked with Python 3.12.4, transformers 5.9.0, vLLM 0.22.1,
NumPy 1.26.4, pandas 2.2.2, OpenAI Python 2.32.0, huggingface-hub 1.17.0,
PyYAML 6.0.3, and Matplotlib 3.8.4. The package metadata keeps compatible
lower bounds for the lightweight inspection path and pins vLLM in the optional
`serve` dependency group.

The attacker receives the public task, public tool/backend format, initial
visible behavior, and responses to its bounded ordinary follow-up requests. It
does not receive gold user state, raw private history, backend retrieval logs,
private benchmark rows, or held-out behavior.

The victim prompts are defined in `src/umpeek/real_agent/materializer.py`.
UMPeek's ordinary follow-up prompts and parsing rules are in
`src/umpeek/attack_baselines/adapters/schema_induced_slot_probe.py`. Comparison
methods are not implemented in this artifact; their official sources are
listed in `docs/BASELINES.md`. Defense prompts and response policies are under
`src/umpeek/defenses/`.

## Environment variables

The launchers use the following defaults:

```bash
export UMPEEK_EVAL2_REAL_AGENT_MODE=1
export UMPEEK_REAL_AGENT_MODEL=Qwen/Qwen3-14B
export UMPEEK_REAL_AGENT_VLLM_BASE_URL=http://127.0.0.1:8010/v1
export UMPEEK_REAL_AGENT_REQUIRE_LIVE_ENDPOINT=1
export UMPEEK_REAL_AGENT_ENABLE_THINKING=0
export UMPEEK_REAL_AGENT_STRICT_MODEL_CHECK=1
export UMPEEK_EVAL2_GENERATE_MISSING_VISIBLE=0
export UMPEEK_EVAL2_DISABLE_GENERATION_CACHE=1
export UMPEEK_EVAL2_LATENT_GOLD_MODE=profile
export UMPEEK_EVAL2_PAPER_FACING_ONLY=1
```

## Suggested order

1. Acquire the four public datasets using `docs/DATA.md`.
2. Run `python scripts/preprocess_benchmarks.py` to create canonical rows,
   role-separated splits, and the minimal manifest.
3. Start Qwen3-14B with vLLM.
4. Run `scripts/run_minimal_evaluation.py` with `--limit 1`.
5. Increase the limit only after the one-sample run succeeds.
6. Use the experiment launchers for the UMPeek ablation and defense matrices.
7. Use the export scripts to regenerate the paper-facing artifacts.

To rebuild the released figures and tables without model calls or third-party
datasets, run `python scripts/rebuild_release_artifacts.py`. Outputs are written
under `build/released_artifacts/` by default.

The included `results/` directory contains the submitted aggregate inputs and
reference outputs. It is intended for checking tables and figures without
downloading model weights or third-party datasets.

The released Exp3 aggregate includes the submitted comparison scores, but the
comparison methods cannot be rerun from this repository. Their implementations
must be obtained from the official sources in `docs/BASELINES.md`.
