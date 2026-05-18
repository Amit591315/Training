Repository checkpoint: 2026-05-18 12:53:53
Purpose: Saved API-serving part of the project for later review.

Saved files in this checkpoint:
- api_snapshot.py (copy of current API app)
- models_list.txt (model artifact names)

Source paths at time of checkpoint:
- api.py
- models/best_cost_model.pkl
- models/best_sev_model.pkl
- models/label_encoders.pkl

How to run current API:
1) From Training directory, run:
   uvicorn api:app --host 0.0.0.0 --port 8000 --reload
2) Open docs:
   http://localhost:8000/docs
