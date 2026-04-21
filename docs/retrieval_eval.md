RAG Retrieval Eval and Regression (Recall@k / MRR)

1) Prepare cases JSON
- Use docs/retrieval_eval_cases.example.json as template.
- Required fields per case:
  - name
  - question
  - one of: file_path / pdf_path / csv_path / urls
  - at least one relevance target:
    - expected_pages
    - expected_source_contains
    - expected_terms

2) Run raw eval

python rag_retrieval_eval.py --cases docs/retrieval_eval_cases.example.json --k 8

Optional report output:

python rag_retrieval_eval.py --cases docs/retrieval_eval_cases.example.json --k 8 --report reports/rag_eval.json

3) One-command regression with baseline

First-time baseline init:

python rag_retrieval_regression.py --cases docs/retrieval_eval_cases.example.json --k 8 --update-baseline

Daily/CI regression check:

python rag_retrieval_regression.py --cases docs/retrieval_eval_cases.example.json --k 8

Shortcut wrapper (same behavior):

python run_rag_regression.py

Custom tolerance example:

python rag_retrieval_regression.py --cases docs/retrieval_eval_cases.example.json --k 8 --max-recall-drop 0.02 --max-mrr-drop 0.02

Error budget example:

python rag_retrieval_regression.py --cases docs/retrieval_eval_cases.example.json --k 8 --max-error-cases 0

4) CI regression
- Workflow file: .github/workflows/rag-retrieval-regression.yml
- CI uses lightweight case file: docs/retrieval_eval_cases.ci.json
- CI baseline file: docs/retrieval_baseline.ci.json

Run equivalent command locally:

python rag_retrieval_regression.py --cases docs/retrieval_eval_cases.ci.json --baseline docs/retrieval_baseline.ci.json --report reports/rag_retrieval_latest_ci.json --k 6 --max-recall-drop 0.00 --max-mrr-drop 0.00 --max-error-cases 0

5) Metrics
- Recall@k: ratio of cases where at least one relevant chunk appears in top-k.
- MRR@k: average reciprocal rank of first relevant chunk in top-k.

6) Relevance matching rule
A chunk is treated as relevant only when all provided constraints in the case are satisfied:
- expected_pages (if provided)
- expected_source_contains (if provided)
- expected_terms (if provided)

7) Exit codes for rag_retrieval_regression.py
- 0: pass or baseline updated
- 2: baseline missing (unless --allow-missing-baseline)
- 3: regression detected (metric drop exceeds threshold)
- 4: regression detected (error cases exceed --max-error-cases)
- 5: baseline mismatch (baseline k differs from current --k)

Tip
- Start with strict constraints for stable regression checks.
- If you only care page hit, provide expected_pages only.
