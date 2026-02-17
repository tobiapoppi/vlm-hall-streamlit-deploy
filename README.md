# Streamlit Evaluation App Deployment

This folder is a standalone repository-ready package for Streamlit Cloud.

## Run locally

```bash
streamlit run app.py
```

## Streamlit Cloud

- Repository root: this folder (`streamlit_cloud_eval_repo`)
- App file: `app.py`

Note: Streamlit Cloud uses ephemeral filesystem per deployment instance. Updates to:
- `data/val_mixed_dpo_human_labelled_v2.jsonl`
- `data/evaluation_progress_v2.json`
may not persist permanently unless you add external storage.

## Data layout

- `data/val_mixed_dpo.jsonl` (rewritten with repo-relative media paths)
- `data/media/...` (all referenced videos/images)
- `data/val_mixed_dpo_human_labelled_v2.jsonl`
- `data/evaluation_progress_v2.json`
