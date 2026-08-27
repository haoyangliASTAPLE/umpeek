# External methods and systems

This artifact does not distribute comparison-method implementations, ports,
adapters, prompts, or method-specific configurations. Obtain and implement
each comparison method from its official source below.

## Attack comparisons

| Method | Official paper or code |
|---|---|
| ADAM | [Paper](https://openreview.net/forum?id=9H1nu8Z6Uy); no stable official code release was available when the artifact was prepared |
| LLM-PBE | [Code](https://github.com/QinbinLi/LLM-PBE) |
| PLeak | [Code](https://github.com/BHui97/PLeak) |
| IPI / InjecAgent | [Code](https://github.com/uiuc-kang-lab/InjecAgent) |
| Imprompter | [Code](https://github.com/Reapor-Yurnero/imprompter) |
| AttrInf | [Code](https://github.com/eth-sri/llmprivacy) |
| PIE | [Code](https://github.com/liu00222/LLM-Based-Personal-Profile-Extraction) |

The submitted aggregate comparison scores remain in
`data/release_samples/main_evaluation_metrics.jsonl` and `results/exp3/` so
that the reported table can be inspected and rebuilt. These records are not a
runnable reproduction of the comparison methods. The local live evaluation
entry point supports UMPeek only.

## Personalization backends

| Backend | Official code |
|---|---|
| Mem0 | [mem0ai/mem0](https://github.com/mem0ai/mem0) |
| Graphiti | [getzep/graphiti](https://github.com/getzep/graphiti) |
| LangMem | [langchain-ai/langmem](https://github.com/langchain-ai/langmem) |
| LangGraph | [langchain-ai/langgraph](https://github.com/langchain-ai/langgraph) |

The evaluated interface adapters are in `src/umpeek/real_agent/backends.py`.

## Adaptive defenses

PrivacyChecker follows the PrivacyInAction component of
[microsoft/ACV](https://github.com/microsoft/ACV). The Theory-of-Mind defense
follows [Vaidehi99/MultiAgentPrivacy](https://github.com/Vaidehi99/MultiAgentPrivacy).
Stateful Counterfactual Exposure Control is implemented in this repository.
All three adapters are under `src/umpeek/defenses/`.
