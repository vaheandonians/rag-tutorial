# RAG with LangChain: six teaching examples

Six Jupyter notebooks, one technique each. Examples 1–4 run on the same small corpus so students can compare the techniques one at a time; Example 5 builds a tree over a longer document, and Example 6 a knowledge graph over all four documents. In every notebook the code is split into cells and each cell is explained in markdown.

| # | Notebook | Technique (from the "Choosing a chunker" table) | What changes vs. Example 1 |
|---|----------|------------------------------------------------|----------------------------|
| 1 | `01_simple_rag.ipynb` | Recursive chunking | Baseline: load, chunk, embed, store in Chroma, similarity search, LLM answer |
| 2 | `02_propositions_rag.ipynb` | Propositions | An LLM turns each chunk into atomic, self-contained facts; those get indexed and linked back to their parent chunk |
| 3 | `03_reranker_rag.ipynb` | Recursive + reranker | Two-stage retrieval: the vector DB fetches 10 candidates, then a cross-encoder re-scores them and keeps the top 4 |
| 4 | `04_late_chunking_rag.ipynb` | Late chunking | Embeds each whole document once, then mean-pools the token vectors per chunk. Prints naive and late retrieval side by side |
| 5 | `05_raptor_rag.ipynb` | RAPTOR (summaries and trees) | Clusters the chunks by embedding, summarises each cluster with the LLM and repeats recursively into a tree. Compares leaves-only, collapsed-tree and tree-traversal retrieval |
| 6 | `06_graphrag.ipynb` | Microsoft GraphRAG (knowledge graph) | An LLM extracts entities and relationships into a knowledge graph, Leiden clusters it into a hierarchy of communities, and every community gets an LLM report. Draws the graph and its hierarchy (static and interactive) and compares global, local and basic search |

Three companion notebooks explain the building blocks of RAPTOR's clustering step, with visualisations and measurements on our own documents. They need no LLM and run in under a minute each (only the Ollama embedding model):

| # | Notebook | Topic | What it shows |
|---|----------|-------|---------------|
| 5a | `05a_dimensionality_reduction.ipynb` | PCA and UMAP | Why 768-dimensional embeddings must be reduced before clustering; how PCA and UMAP work; which one keeps neighbourhoods better in 2-D and at the 10 dimensions RAPTOR clusters in |
| 5b | `05b_gaussian_mixture_models.ipynb` | Gaussian Mixture Models | Gaussians and mixtures, the EM algorithm implemented by hand, soft assignments and RAPTOR's 0.1 threshold, k-means, covariance types, the 30 leaves of Example 5 |
| 5c | `05c_bic_model_selection.ipynb` | Bayesian Information Criterion | Why the likelihood alone cannot choose the number of clusters, how much data BIC needs, why it breaks when clusters have fewer points than dimensions – as on Example 5's leaves – and the paper's complete clustering recipe |

## Corpus (`data/`)

Examples 1–4 use three short **fictional** documents: a utility's annual report, a bank credit policy and a transit authority review. They were written to show specific failure modes:

- Many chunks say "the company", "its" or "the authority" and never name the entity. Examples 2 and 4 fix this in different ways.
- Several chunks look alike on the surface (limits, percentages, EUR amounts), but only one of them answers the question. Example 3 fixes this.

Example 5 uses `document_a.md`, the 2019 financial statements of a small company as exported Markdown: four statements as HTML tables and ten notes, about 20,000 characters. Questions about it range from one interest rate to "what kind of company is this", which is what a tree of summaries is for.

Example 6 indexes all four documents together, so that the knowledge graph spans several organisations and the global question "what are these documents about?" has a corpus-sized answer.

## Setup

The project needs **Python 3.13**: Microsoft GraphRAG (Example 6) supports Python 3.11–3.13, and one environment holds everything. Examples 1–5 run locally: the LLM and the embedding model are served by [Ollama](https://ollama.com), and no API key is needed. Example 6 uses OpenAI's API; see [Example 6: GraphRAG](#example-6-graphrag) below.

```bash
python3.13 -m venv venv && source venv/bin/activate
pip install -r requirements.txt

ollama pull mistral-small3.2:24b   # the LLM
ollama pull embeddinggemma:300m    # the embedding model (examples 1-3, 5 and 5a-5c)
```

Start Jupyter from this folder so that `data/` is next to the notebooks:

```bash
jupyter lab 01_simple_rag.ipynb
```

To use another chat model, set `RAG_LLM` to any `init_chat_model` string before starting Jupyter, e.g. `RAG_LLM="ollama:qwen3:8b"`, or `RAG_LLM="openai:gpt-4o-mini"` with `OPENAI_API_KEY` and `langchain-openai` installed.

| Model | Served by | Used in | Size |
|-------|-----------|---------|------|
| `mistral-small3.2:24b` (LLM) | Ollama | 1, 2, 3, 4, 5 | ~15 GB |
| `embeddinggemma:300m` (embeddings) | Ollama | 1, 2, 3, 5, 5a–5c | ~620 MB |
| `BAAI/bge-reranker-base` (cross-encoder) | Hugging Face Hub, via `sentence-transformers` | 3 | ~1.1 GB (lighter option: `cross-encoder/ms-marco-MiniLM-L-6-v2`) |
| `nomic-ai/modernbert-embed-base` (8k-context embedder) | Hugging Face Hub, via `transformers` | 4 | ~600 MB |

Two models cannot come from Ollama: it has no reranking API or reranker models (open request since 2024, [ollama/ollama#3368](https://github.com/ollama/ollama/issues/3368)), and its embedding API returns only pooled vectors, while late chunking needs the token-level outputs. Both are downloaded from the Hugging Face Hub on first use and run locally on the CPU.

Examples 2 and 5 make LLM calls while indexing (one per chunk, respectively one per cluster) and cache the result in `propositions_cache.json` and `raptor_tree.json`. Delete the file to index again, for example after you edit a prompt, change the chunking or switch to another LLM. Building the RAPTOR tree takes around ten minutes with a 24B local model.

## Example 6: GraphRAG

[Microsoft GraphRAG](https://github.com/microsoft/graphrag) 3.3 is installed with the other requirements. Example 6 needs an OpenAI key in `.env` in this folder:

```bash
OPENAI_API_KEY=sk-...
```

The notebook uses `gpt-4.1-mini` and `text-embedding-3-small` through GraphRAG's LiteLLM layer. Indexing makes about 80 model calls; with the API it takes under two minutes and costs a few cents, whereas a local 24B model would need roughly 45 minutes. The notebook's last section shows the settings for running it on Ollama anyway.

GraphRAG keeps everything in `graphrag_workspace/`: the input files, `settings.yaml` (written by the notebook), the index in `output/` (Parquet tables, a LanceDB vector store, `graph.graphml`), every model response in `cache/`, and the interactive graph `knowledge_graph.html`. The notebook skips indexing when `output/` exists; delete it to rebuild – the cache makes a rebuild with unchanged settings free.

## Suggested exercises

1. In Example 1, change `CHUNK_SIZE` to 200 and then to 1500. Which answers get worse, and why?
2. Example 1 still has a few headings stuck to the end of the previous chunk. How would a structure-aware splitter (`MarkdownHeaderTextSplitter`) handle that? (Example 5 uses one.)
3. In Example 2, switch `USE_PARENT_CONTEXT` between `True` and `False` and compare the answers and the token usage.
4. In Example 3, find a question where the reranker moves a chunk from position 6 or lower into the top 4.
5. In Example 4, ask about a Meridian Bank fact. Does late chunking help as much when every chunk already names its entity?
6. In Example 5, replace a retrieved summary node by the full text of its leaves before answering ("index the summary, return the full text"). Does the answer to the debt question get better or just longer?
7. After notebook 5c, change Example 5's clustering so that BIC works in its regular regime (for example PCA to 2 dimensions for the leaves), rebuild the tree, and compare it with the original.
8. Ask Example 5's broad question ("How did the company's results and financial position change from 2018 to 2019?") with Example 6's global search. How do a tree of embedding clusters and a graph of entity communities differ in what they summarise?
