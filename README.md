# Where should a RAG system enforce access control?

Measurements of recall loss and leakage in multi-tenant retrieval-augmented generation, on Reuters-21578 with
random tenant, department and executive permissions. Paper: [`paper/main.pdf`](paper/main.pdf).

| Experiment | Finding |
|---|---|
| E1: post-filtering a global top-k | Dense known-item recall fell from 0.61 (pre-filter) to 0.37; 16× over-fetch recovered only 0.59 |
| E2: result count as an existence oracle | Weak: AUC 0.50–0.53 (a negative result) |
| E3: shared BM25 statistics, correct pre-filtering | A tenant recovered the exact count of another tenant's secret documents in 100/100 trials, 6 queries each, from the order of its own results; per-tenant statistics removed the leak |
| E4: semantic answer cache | Keyed on query only: ~90% of hits leaked. Plus tenant: 27–30% of hits still leaked across roles. Plus full permission set: no leaks |

## Reproduce

```bash
pip install -r requirements.txt
mkdir -p data && curl -L -o data/reuters.zip https://raw.githubusercontent.com/nltk/nltk_data/gh-pages/packages/corpora/reuters.zip
unzip data/reuters.zip -d data
python src/rag.py e1 e3 e4      # about two hours on two cores
python src/analyse.py
cd paper && latexmk -pdf main.tex
```

## AI assistance

An AI assistant (Claude, Anthropic) helped write experiment code and draft text. The author designed the
study questions, reviewed the code and results, and is responsible for the content.

## Licence

MIT
