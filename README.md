# Automated LLM Evaluation & Testing Suite

## Summary

An enterprise-grade, modular automated testing framework designed to validate local LLM (Gemma via LM Studio) performance across core quality dimensions using `pytest` and `DeepEval`.

## Requirements

To run and test this project locally, ensure you have the following installed on your machine:

- Python 3.10+
- `pytest` >= 9.0.0
- `deepeval` >= 4.0.0
- Local LLM runner (e.g., LM Studio with Gemma loaded)

## Steps to Install

1. Clone the repository to your local machine.
2. Create and activate a virtual environment:
   - **Windows:**
     ```bash
     python -m venv .venv
     .venv\Scripts\activate
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Steps to Launch

Launch your local model provider (e.g., LM Studio), load the Gemma model, and start the local server on http://localhost:1234/v1.

## Steps to Generate Reports

1. Run the full test suite:

```bash
python -m pytest
```

2. Run a specific quality dimension:

```bash
python -m pytest tests/test_accuracy.py
```
