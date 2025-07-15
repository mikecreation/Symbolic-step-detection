
# Symbolic Discovery App

## Overview
This full stack project unifies three body simulation analysis, symbolic residual probing, contradiction driven question generation, and optional PySR symbolic regression into a single intuitive web interface.

### Backend
* Python FastAPI service (`backend/main.py`)
* Core logic in `symbolic_probe.py`
* Optional PySR regression toggled by `use_pysr` flag
* Run with:
  ```bash
  pip install -r backend/requirements.txt
  uvicorn backend.main:app --reload
  ```
* For PySR enablement, install Julia separately and:
  ```bash
  pip install pysr
  python -c "from pysr import install; install()"
  ```
### Frontend
* Lightweight React + Tailwind UI in `frontend`
* Provide data arrays, click Analyze or Analyze + PySR
* Displays JSON output including symbolic forms and discovered equations

### Usage Flow
1. Start backend API
2. Serve the React app (e.g. `npm install && npm run dev` in `frontend`)
3. Enter simulated time, radius, and momentum arrays
4. Click Analyze to see derivative forms, residual spikes, and transition times
5. Optionally click the PySR button to evolve new formulas from residuals

### Notes
* Minus characters appear in code where math requires. User text responses avoid dashes as requested.
