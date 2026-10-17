"""PLACEHOLDER so the platform can be deployed before the real API exists.
The API owner replaces this file. Keep the variable name `app` (uvicorn loads src.api.main:app)."""
import os
import platform

from fastapi import FastAPI

app = FastAPI(title="V12 Engines scoring API (placeholder)")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "machine": platform.machine(),  # expect "s390x"
        "models_mounted": sorted(os.listdir(os.environ.get("MODEL_DIR", "/models"))),
        "test_csv_present": os.path.exists(os.path.join(os.environ.get("DATA_DIR", "/data"), "test.csv")),
    }
