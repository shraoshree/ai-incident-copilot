# Independent Holdout Protocol (Mechanize portfolio improvement)

## Historical result is not independent
The November 4 benchmark used co-authored synthetic queries. After one no-answer test failure, the threshold-selection rule changed. Previously reported 10/10 retrieval and 6/6 abstention are descriptive, NOT unbiased held-out or production accuracy.

## New evaluation protocol
The new evaluation_holdout.py adds a reproducible evaluation protocol, NOT new performance results. This change deliberately contains no new holdout queries: writing variants of already inspected questions would not establish independence.

1. Independently author at least 30 answerable and 30 no-answer cases (50+ each preferred); require another author unfamiliar with old benchmark queries and scores. Include paraphrases, adversarial overlaps, boundary cases, and varied incident categories. Adjudicate ambiguous labels before looking at predictions. Maintain label provenance.
2. Save evaluation/fresh_holdout.json as a JSON list of objects with id, query and expected_title (actual corpus title or null). Review for near-duplicate, semantically overlapping or answer-leaking cases. Automated validation only catches exact normalized duplicates and reused original strings.
3. Fix code, corpus, embedding model, and thresholds before testing. Previously published cutoffs 0.65821 semantic and 3.93992 BM25 are defaults, not production-calibrated.
4. Run: python evaluation_holdout.py freeze --cases evaluation/fresh_holdout.json
5. With pgvector database ingested and Gemini embeddings configured, run once: python evaluation_holdout.py run --cases evaluation/fresh_holdout.json
6. Save the resulting manifest and report. Do not tune thresholds, prompts, labels, or corpus using these results and then call a rerun independently held out. If you make improvements, procure a new test set for a fresh evaluation.

The script freezes SHA-256 hashes for retrieval code, evaluator, incident corpus, and new cases; validates minimum class sizes and label titles; reports both BM25 and semantic results on the same cases; provides Wilson 95% intervals and itemized failures; and refuses overwriting freeze/results files.

## Limitations
The corpus has only 21 synthetic incidents. New paraphrases against the same incident categories do NOT establish performance on unseen incident types or production cases. Even an independently authored, never-inspected query set is not representative production evidence. Record data source, test creation process, model version, case counts, per-category performance, and uncertainty. Use sensitive-data-safe de-identified incident data for any later domain-generalization study.

## Defensible resume bullet (until new run)
Built a Python RAG evaluation harness comparing BM25 and semantic retrieval using Hit@1, Hit@3, MRR@3 and abstention tests; added a frozen holdout protocol with dataset leakage checks, reproducible manifests, and failure-level reporting.
