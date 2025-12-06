# Customer Personality - Streamlit App

## Struktur
- `train_pipeline.py` — training pipeline; run this to produce `model/` artifacts
- `app/` — streamlit app and preprocessing/model wrapper
- `model/` — saved artifacts (created after training)
- `data/marketing_campaign.csv` — raw data

## Structure Detail
customer_personality/
├─ data/
│  └─ data_fix.csv
├─ model/                
│  └─ (auto-generated oleh train_pipeline.py)
├─ app/
│  ├─ model_pipeline.py
│  ├─ api.py
│  └─ app.py
├─ train_pipeline.py
├─ requirements.txt
├─ Dockerfile.fastapi
├─ Dockerfile.streamlit
├─ docker-compose.yml
└─ README.md

