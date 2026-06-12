## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Indexer

```bash
python indexer.py --sentences_per_chunk 5  <https://example.com>
```

## Inference

```bash
python src/agent.py
```
