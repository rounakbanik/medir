# Six-week continuation plan

1. **Data and normalization:** choose a final disease population, expand PubMed sampling with
   coverage targets, review ambiguous ingredients/aliases, add verified chemical identifier joins,
   and audit dictionary linking on sampled papers.
2. **Judgments:** review repoDB outcomes, collect dated verified repurposing events and earlier
   indications, separate development/test queries, and curate the abacavir case without assuming
   its hypothesis is an approved indication.
3. **Retrieval:** compare biomedical encoders, tune candidate depth and fusion on development data,
   introduce document chunking, and replace exact vector search if corpus scale warrants it.
4. **Evidence and ranking:** score treatment versus adverse/background relations, tune aggregation,
   improve the MeSH bridge baseline, and compare biomedical cross-encoders.
5. **Feasibility and learning:** audit BBB coverage, add other supported feasibility signals,
   and train a reranker only after a defensible split and sufficient labels exist. Add DrugBank
   as a new adapter if access becomes available.
6. **Experiments and presentation:** run frozen comparisons, ablations, per-disease/per-direction
   analyses and paired uncertainty estimates; inspect case studies and expose the API through
   a Next.js frontend if useful. Document retrospective/model-date limitations in the report.
