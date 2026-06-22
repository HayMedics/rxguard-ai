# RxGuard AI — Drug Interaction Intelligence

An AI-powered drug–drug interaction (DDI) checker built by **HayMedics Academy**.
Select two or more medications and the app screens every pair, predicts the most
likely interaction type, explains *why*, and produces a downloadable report.

> ⚠️ **Educational / research project — not medical advice and not a medical device.**
> Predictions come from a machine-learning model and may be wrong. Always consult a
> qualified pharmacist or clinician.

**🔗 Live demo:** _add your Streamlit Cloud link here once deployed_

---

## Features

- **Multi-drug screening** — check any number of medications; every pair is scored and ranked.
- **Search any drug** — pick from a curated list or look up any compound by name via PubChem.
- **Explainability** — global feature importances plus, for the top pair, structural
  (Tanimoto) similarity and the largest physicochemical differences.
- **PDF report** — one-click branded interaction report.
- **Drug profile** — 2D structure, key descriptors, and Lipinski rule-of-5.
- **History** — every check from the current session.

## How it works

- **Data:** DrugBank DDI set (~191,000 drug-pair interactions across 86 interaction types).
- **Features:** each drug → 10 RDKit physicochemical descriptors + a 1024-bit Morgan/ECFP
  fingerprint. Each pair is encoded order-invariantly (sum, absolute difference, product),
  so the prediction for (A, B) equals (B, A).
- **Model:** RandomForest classifier.
- **Held-out performance:** ≈ **0.66 accuracy, 0.64 macro-F1** across 86 classes
  (a deployment-sized model; an unconstrained variant scored higher but was too large to ship).

## Limitations

- Predicts the interaction *type*, not its *severity* or *direction*.
- Trained on known interactions, so the absence of a flag is not proof of safety.
- Educational use only.

## Tech stack

Python · Streamlit · scikit-learn · RDKit · pandas / NumPy · ReportLab

## Run locally

```bash
python -m venv .venv
source .venv/Scripts/activate     # Windows (Git Bash);  use .venv/bin/activate on macOS/Linux
pip install -r requirements.txt
streamlit run app.py
```

The app expects `meddose_model_v2.joblib`, `interaction_type_labels.csv`, and the
`assets/` logo in the same folder. The model and feature pipeline are produced by the
`build_*.py` and `train_*.py` scripts.

---

_Built by **HayMedics Academy** · Data | Research | Innovation_
