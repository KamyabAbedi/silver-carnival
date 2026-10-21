# Assignment 1 -- Train a minimal LM from scratch

This kit should help you get started. It contains the tests your code would ideally pass. The adapter file connects the tests to your code. We also provide skeletons for the
data preparation, training and reporting steps.

```
lm/                     
tests/
  adapters.py                  The **only** test file you edit: point each function at your code
  test_tokenizer.py            Part 1
  test_model.py                Part 2 modules and accounting
  test_nn_utils.py             Loss, Clipping, Generation 
  test_optimizer.py            AdamW, cosine schedule 
  test_data.py                 Batches, Checkpoints
  _reference.py                Slow reference implementations the tests compare against; do not edit
  fixtures/corpus.txt          A tiny corpus with special tokens
scripts/prepare_data.py        Tokenise a text file into a uint16 memmap with your tokenizer
scripts/train.py               Training-loop skeleton with the results.json 
notebooks/setup.ipynb          Colab: mount Drive, install, run tests, train
report/report.tex              The report template (At most three pages!)
```

## Getting started

```
uv sync                                         # or: pip install -e .
uv run pytest -q                                # all tests will fail until you fill in tests/adapters.py
uv run pytest -q tests/test_tokenizer.py -x     # one part at a time
```

The tests run on CPU in a few seconds. Signatures, shapes and conventions are documented
in `tests/adapters.py`. If a test and the assingment text seem to disagree, the tests win!

## Submitting

One archive per person containing this directory (**without** data and without checkpoints),
`results.json` (run `python grading/validate_results.py results.json` first), the report
PDF, the notebook you trained with.
# silver-carnival
