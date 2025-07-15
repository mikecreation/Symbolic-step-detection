from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import numpy as np

from symbolic_probe import run_symbolic_probe, run_symbolic_regression

app = FastAPI(title="Symbolic Discovery API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimulationRequest(BaseModel):
    time: List[float]
    orbital_radius: List[float]
    angular_momentum: List[float]
    dr_dt: Optional[str] = None
    dL_dt: Optional[str] = None
    transition_times: Optional[List[float]] = None
    residuals: Optional[List[float]] = None
    time_scaled: Optional[List[List[float]]] = None
    use_pysr: Optional[bool] = False

@app.post("/analyze")
async def analyze(req: SimulationRequest):
    try:
        time_arr = np.array(req.time)
        r_arr = np.array(req.orbital_radius)
        L_arr = np.array(req.angular_momentum)
        result = run_symbolic_probe(time_arr, r_arr, L_arr)
        result["time"] = req.time
        result["transition_times"] = req.transition_times if req.transition_times else []

        if req.use_pysr:
            try:
                sym_reg = run_symbolic_regression(result["time_scaled"], result["residuals"])
                result["symbolic_regression"] = str(sym_reg)
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Symbolic regression failed: {e}")

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
