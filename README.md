# CS443 Final Project: Financial Sentiment Classification

**Research question:** For specialized-domain text classification (financial sentiment), when is fine-tuning a small model worth it over zero-shot prompting a large model — and does domain pretraining of the small model change that answer?

## Models (5 total)

| Model | Type | Params |
|---|---|---|
| TF-IDF + Logistic Regression | Classical baseline | — |
| RoBERTa-base | General-purpose fine-tuned | ~125M |
| FinBERT (`ProsusAI/finbert`) | Domain-pretrained fine-tuned | ~110M |
| Claude Haiku 4.5 (`claude-haiku-4-5-20251001`) | Zero-shot LLM | — |
| Claude Sonnet 4.6 (`claude-sonnet-4-6`) | Zero-shot LLM | — |

## Datasets

| Dataset | Source | Use |
|---|---|---|
| Financial PhraseBank | `gtfintechlab/financial_phrasebank_sentences_allagree` | Primary train/val/test (70/15/15, seed 42) |
| Twitter Financial News Sentiment | `zeroshot/twitter-financial-news-sentiment` | OOD evaluation only (1 000 stratified samples) |

Label scheme everywhere: **0 = negative, 1 = neutral, 2 = positive**.

## Notebooks

All notebooks are self-contained — they install their own dependencies in the
first cell and inline all data-loading code. Run them in order.

| Notebook | What it does | Where to run |
|---|---|---|
| `01_data_and_baseline.ipynb` | Load data, create splits, save metadata, TF-IDF baseline | Local or Colab |
| `02_finetune_transformers.ipynb` | Fine-tune RoBERTa-base & FinBERT, learning curve sweep | **Colab (GPU)** |
| `03_llm_evaluation.ipynb` | Zero-shot eval with Haiku & Sonnet | Local or Colab |
| `04_generalization.ipynb` | All 5 models on Twitter OOD set | Local or Colab (GPU for transformers) |
| `05_analysis_and_plots.ipynb` | Figures and error analysis from results JSON | Local or Colab |

## Results Files

Each notebook appends to shared JSON files in `results/`:

```
results/
├── splits_metadata.json     # dataset sizes and class distributions
├── main_comparison.json     # all 5 models on PhraseBank test
├── generalization.json      # all 5 models on Twitter OOD
└── learning_curve.json      # RoBERTa & FinBERT at sizes 100/250/500/1000/full
```

Figures are saved to `figures/`. Both directories are committed.

## Setup

### Local (notebooks 01, 03, 05)

No shared virtual environment needed — each notebook installs its own
dependencies. Just run the notebook.

If using the Anthropic API (notebook 03), add your key to `.env`:
```
ANTHROPIC_API_KEY=sk-ant-...
```

### Colab (notebooks 02, 04)

1. Push this repo to GitHub (or zip and upload to Colab)
2. Open the notebook in [Google Colab](https://colab.research.google.com/)
3. Set runtime to **GPU**: Runtime → Change runtime type → T4 GPU
4. *(Optional)* Mount Google Drive to persist checkpoints across sessions:
   - Set `USE_DRIVE = True` at the top of the notebook
   - Update `DRIVE_PATH` to a folder in your Drive
5. For LLM evaluation: add `ANTHROPIC_API_KEY` to Colab Secrets
   (left sidebar → key icon) so notebook 03 can load it automatically
6. Run all cells (Runtime → Run all)
7. Download `results/` at the end — notebook 02 has a download cell

### Running order

```
01_data_and_baseline      →  results/main_comparison.json (tfidf_logreg entry)
                              results/splits_metadata.json
02_finetune_transformers  →  results/main_comparison.json (roberta, finbert entries)
                              results/learning_curve.json
                              models/roberta-base_phrasebank/
                              models/finbert_phrasebank/
03_llm_evaluation         →  results/main_comparison.json (haiku, sonnet entries)
                              cache/llm_*.jsonl
04_generalization         →  results/generalization.json
05_analysis_and_plots     →  figures/*.png
```

## Project Structure

```
├── 01_data_and_baseline.ipynb
├── 02_finetune_transformers.ipynb
├── 03_llm_evaluation.ipynb
├── 04_generalization.ipynb
├── 05_analysis_and_plots.ipynb
├── results/                 # JSON results (committed)
├── figures/                 # generated plots (committed)
├── data/                    # downloaded datasets (gitignored)
├── models/                  # fine-tuned checkpoints (gitignored)
├── cache/                   # LLM response cache (gitignored)
├── paper/                   # LaTeX source
├── notebooks/               # scratch / error analysis notebooks
└── .env                     # ANTHROPIC_API_KEY (gitignored)
```

## Experiment Descriptions

**Experiment 1 — Main comparison** (`01` + `02` + `03`)  
All 5 models evaluated on the PhraseBank test set. Reports accuracy, macro F1,
per-class F1, inference time, and cost per 1K predictions.

**Experiment 2 — OOD generalization** (`04`)  
Same 5 models evaluated on Twitter Financial News Sentiment without any
retraining. Tests domain transfer from formal sentences to social media text.

**Experiment 3 — Learning curve** (`02`)  
RoBERTa-base and FinBERT fine-tuned at training sizes [100, 250, 500, 1000,
full]. The resulting macro F1 curve is plotted against LLM zero-shot scores
as horizontal reference lines, showing the crossover point where fine-tuning
beats zero-shot.

## Key Implementation Notes

- **Determinism:** seed 42 everywhere — data splits, model init, training
- **LLM caching:** responses cached to `cache/llm_<model>_<dataset>.jsonl`
  keyed by SHA-256 hash of the input text; reruns cost nothing
- **Concurrency:** LLM evaluation uses asyncio with a 10-request semaphore
- **Retry:** transient API errors retried with exponential backoff (tenacity)
- **Hyperparameters:** fixed, not tuned (lr 2e-5, batch 16, max 5 epochs,
  early stopping patience 2) — listed as a limitation in the paper
