# Data acquisition and preprocessing

Third-party benchmark data is not redistributed in this repository. Obtain
each dataset from its official source and follow its license and access terms.

| Benchmark | Official source |
|---|---|
| PersonaMem-v2 | [GitHub](https://github.com/bowen-upenn/PersonaMem-v2), [Hugging Face](https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2) |
| PersonaLens | [GitHub](https://github.com/amazon-science/personalens), [Hugging Face](https://huggingface.co/datasets/AmazonScience/PersonaLens) |
| ETAPP | [GitHub](https://github.com/hypasd-art/ETAPP) |
| LoCoMo | [GitHub](https://github.com/snap-research/locomo) |

Expected local locations after acquisition are:

```text
data/external/PersonaMemv2/
data/benchmarks/PersonaLens/
data/benchmarks/ETAPP/
data/benchmarks/LoCoMo/
```

These local dataset directories are excluded by `.gitignore`; only the small,
redistributable examples under `data/release_samples/` belong in the artifact.

After placing the official downloads at those locations, run:

```bash
python scripts/preprocess_benchmarks.py
```

This command invokes the benchmark loaders under `src/umpeek/exp1/` and
`src/umpeek/exp1_whitebox/`, writes canonical task rows, creates the compact
evaluation manifest, and calls `scripts/a200_materialize_strong_query_splits.py`
to create three disjoint roles for each benchmark:

1. private evaluator rows, which contain the scoring target;
2. public attack-probe rows, which contain only the task and public interface;
3. independent held-out behavior rows used by HBPS.

Use `python scripts/preprocess_benchmarks.py --help` to override the
PersonaMem-v2 CSV, PersonaLens directory, or LoCoMo file. Use `--prepare-only`
to inspect canonical rows before creating the role-separated splits.

The exact split metadata and counts from the submitted experiments are in
`data/release_samples/split_manifests/`. The release does not include the
private evaluator rows or held-out task contents. Eight public requests and
twelve interface-visible outputs are included as format examples. The compact
release also includes 24 Experiment 1 record examples, 96 setting-level main
evaluation records, and 1,152 matched adaptive-defense scalar records. Full
per-sample evaluator outputs are intentionally omitted.
