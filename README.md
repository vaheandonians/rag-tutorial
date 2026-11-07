# RAG with LangChain: eight teaching examples

Eight Jupyter notebooks, one technique each. Examples 1–4 and 1a run on the same small corpus so students can compare the techniques one at a time; Example 5 builds a tree over a longer document, and Examples 6 and 7 build knowledge graphs over all four documents. In every notebook the code is split into cells and each cell is explained in markdown.

| # | Notebook | Technique (from the "Choosing a chunker" table) | What changes vs. Example 1 |
|---|----------|------------------------------------------------|----------------------------|
| 1 | `01_simple_rag.ipynb` | Recursive chunking | Baseline: load, chunk, embed, store in Chroma, similarity search, LLM answer |
| 1a | `01a_hybrid_rag.ipynb` | Recursive + hybrid search | Adds a second index over the same chunks, BM25 (a short function using only the standard library), searches both indexes and fuses the two rankings with reciprocal rank fusion. Prints the lexical, dense and fused lists side by side |
| 2 | `02_propositions_rag.ipynb` | Propositions | An LLM turns each chunk into atomic, self-contained facts; those get indexed and linked back to their parent chunk |
| 3 | `03_reranker_rag.ipynb` | Recursive + reranker | Two-stage retrieval: the vector DB fetches 10 candidates, then a cross-encoder re-scores them and keeps the top 4 |
| 4 | `04_late_chunking_rag.ipynb` | Late chunking | Embeds each whole document once, then mean-pools the token vectors per chunk. Prints naive and late retrieval side by side |
| 5 | `05_raptor_rag.ipynb` | RAPTOR (summaries and trees) | Clusters the chunks by embedding, summarises each cluster with the LLM and repeats recursively into a tree. Compares leaves-only, collapsed-tree and tree-traversal retrieval |
| 6 | `06_graphrag.ipynb` | Microsoft GraphRAG (knowledge graph) | An LLM extracts entities and relationships into a knowledge graph, Leiden clusters it into a hierarchy of communities, and every community gets an LLM report. Draws the graph and its hierarchy (static and interactive) and compares global, local and basic search |
| 7 | `07_hirag.ipynb` | HiRAG (hierarchical knowledge graph) | Builds the knowledge graph by hand, clusters its entities as RAPTOR clusters chunks, and lets an LLM add layers of summary entities on top; Leiden communities on the multi-layer graph get LLM reports. Answers from three levels of context – local entities, community reports, and the relations along the shortest paths that bridge them – and compares the communities with and without the summary layers |

Three companion notebooks explain the building blocks of RAPTOR's clustering step, with visualisations and measurements on our own documents. They need no LLM and run in under a minute each (only the Ollama embedding model):

| # | Notebook | Topic | What it shows |
|---|----------|-------|---------------|
| 5a | `05a_dimensionality_reduction.ipynb` | PCA and UMAP | Why 768-dimensional embeddings must be reduced before clustering; how PCA and UMAP work; which one keeps neighbourhoods better in 2-D and at the 10 dimensions RAPTOR clusters in |
| 5b | `05b_gaussian_mixture_models.ipynb` | Gaussian Mixture Models | Gaussians and mixtures, the EM algorithm implemented by hand, soft assignments and RAPTOR's 0.1 threshold, k-means, covariance types, the 30 leaves of Example 5 |
| 5c | `05c_bic_model_selection.ipynb` | Bayesian Information Criterion | Why the likelihood alone cannot choose the number of clusters, how much data BIC needs, why it breaks when clusters have fewer points than dimensions – as on Example 5's leaves – and the paper's complete clustering recipe |

## Corpus (`data/`)

Examples 1–4 and 1a use three short **fictional** documents: a utility's annual report, a bank credit policy and a transit authority review. They were written to show specific failure modes:

- Many chunks say "the company", "its" or "the authority" and never name the entity. Examples 2 and 4 fix this in different ways.
- Several chunks look alike on the surface (limits, percentages, EUR amounts), but only one of them answers the question. Example 3 fixes this.
- Some questions hinge on an exact token, such as the policy version `7.2`; others paraphrase the answer ("how long may an individual borrowing last" for "the maximum term is 84 months"). Example 1a searches with a lexical and a dense index and fuses the two.

Example 5 uses `document_a.md`, the 2019 financial statements of a small company as exported Markdown: four statements as HTML tables and ten notes, about 20,000 characters. Questions about it range from one interest rate to "what kind of company is this", which is what a tree of summaries is for.

Examples 6 and 7 index all four documents together, so that the knowledge graph spans several organisations and the global question "what are these documents about?" has a corpus-sized answer.

## Setup

The project needs **Python 3.13**: Microsoft GraphRAG (Example 6) supports Python 3.11–3.13, and one environment holds everything. Examples 1–5 run locally: the LLM and the embedding model are served by [Ollama](https://ollama.com), and no API key is needed. Examples 6 and 7 use OpenAI's API; see [Example 6: GraphRAG](#example-6-graphrag) and [Example 7: HiRAG](#example-7-hirag) below.

```bash
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

ollama pull mistral-small3.2:24b   # the LLM
ollama pull embeddinggemma:300m    # the embedding model (examples 1-3, 1a, 5 and 5a-5c)
```

Start Jupyter from this folder so that `data/` is next to the notebooks:

```bash
jupyter lab 01_simple_rag.ipynb
```

To use another chat model, set `RAG_LLM` to any `init_chat_model` string before starting Jupyter, e.g. `RAG_LLM="ollama:qwen3:8b"`, or `RAG_LLM="openai:gpt-4o-mini"` with `OPENAI_API_KEY` set in the environment.

| Model | Served by | Used in | Size |
|-------|-----------|---------|------|
| `mistral-small3.2:24b` (LLM) | Ollama | 1, 1a, 2, 3, 4, 5 | ~15 GB |
| `embeddinggemma:300m` (embeddings) | Ollama | 1, 1a, 2, 3, 5, 5a–5c | ~620 MB |
| `BAAI/bge-reranker-base` (cross-encoder) | Hugging Face Hub, via `sentence-transformers` | 3 | ~1.1 GB (lighter option: `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| `nomic-ai/modernbert-embed-base` (8k-context embedder) | Hugging Face Hub, via `transformers` | 4 | ~600 MB |
| `gpt-4.1-mini` (LLM) | OpenAI API | 6, 7 | hosted; key in `.env` |
| `text-embedding-3-small` (embeddings) | OpenAI API | 6, 7 | hosted; key in `.env` |

Two models cannot come from Ollama: it has no reranking API or reranker models (open request since 2024, [ollama/ollama#3368](https://github.com/ollama/ollama/issues/3368)), and its embedding API returns only pooled vectors, while late chunking needs the token-level outputs. Both are downloaded from the Hugging Face Hub on first use and run locally on the CPU.

Examples 2, 5 and 7 make LLM calls while indexing and cache the result in `propositions_cache.json`, `raptor_tree.json` and `hirag_index.json`. Delete the file to index again, for example after you edit a prompt, change the chunking or switch to another LLM. Building the RAPTOR tree takes around ten minutes with a 24B local model.

## Example 6: GraphRAG

[Microsoft GraphRAG](https://github.com/microsoft/graphrag) 3.3 is installed with the other requirements. Example 6 needs an OpenAI key in `.env` in this folder:

```bash
OPENAI_API_KEY=sk-...
```

The notebook uses `gpt-4.1-mini` and `text-embedding-3-small` through GraphRAG's LiteLLM layer. Indexing makes about 80 model calls; with the API it takes under two minutes and costs a few cents, whereas a local 24B model would need roughly 45 minutes. The notebook's last section shows the settings for running it on Ollama anyway.

GraphRAG keeps everything in `graphrag_workspace/`: the input files, `settings.yaml` (written by the notebook), the index in `output/` (Parquet tables, a LanceDB vector store, `graph.graphml`), every model response in `cache/`, and the interactive graph `knowledge_graph.html`. The notebook skips indexing when `output/` exists; delete it to rebuild – the cache makes a rebuild with unchanged settings free.

## Example 7: HiRAG

[HiRAG](https://github.com/hhy-huang/HiRAG) (Huang et al., 2025, [arXiv:2503.10150](https://arxiv.org/abs/2503.10150)) is built by hand in the notebook, step by step, with LangChain, `networkx` and `graspologic-native` – the Leiden implementation that GraphRAG uses internally. Like Example 6 it uses `gpt-4.1-mini` and `text-embedding-3-small` with the key from `.env`, here through LangChain's `init_chat_model` and `init_embeddings`, so `RAG_LLM` still switches the chat model.

Building the index makes 85 LLM calls – two per chunk for entities and relations, one per cluster for the summary entities, one report per community – and takes about a minute. The finished index (extraction, summary layers, communities and reports) is cached in `hirag_index.json`; delete it to rebuild. A local 24B model would need well over an hour.

## Suggested exercises

1. In Example 1, change `CHUNK_SIZE` to 200 and then to 1500. Which answers get worse, and why?
2. Example 1 still has a few headings stuck to the end of the previous chunk. How would a structure-aware splitter (`MarkdownHeaderTextSplitter`) handle that? (Example 5 uses one.)
3. In Example 1a, compare `show_retrieval("97.2")` with `show_retrieval("97.2 percent")`. Why does one common word cost the identifier its lead in the fused list?
4. In Example 2, switch `USE_PARENT_CONTEXT` between `True` and `False` and compare the answers and the token usage.
5. In Example 3, find a question where the reranker moves a chunk from position 6 or lower into the top 4.
6. In Example 4, ask about a Meridian Bank fact. Does late chunking help as much when every chunk already names its entity?
7. In Example 5, replace a retrieved summary node by the full text of its leaves before answering ("index the summary, return the full text"). Does the answer to the debt question get better or just longer?
8. After notebook 5c, change Example 5's clustering so that BIC works in its regular regime (for example PCA to 2 dimensions for the leaves), rebuild the tree, and compare it with the original.
9. Ask Example 5's broad question ("How did the company's results and financial position change from 2018 to 2019?") with Example 6's global search. How do a tree of embedding clusters and a graph of entity communities differ in what they summarise?
10. In Example 7, compare the communities of the flat and the hierarchical graph. Why can Leiden split The Employer Group into topics only once the summary entities are there – and why could GraphRAG in Example 6 not?
11. In Example 7, set `META_TYPES` to HiRAG's original list (organization, person, location, …) and rebuild. What happens to the summary layer, and why?
