from fastapi import APIRouter, HTTPException
import os

router = APIRouter()

@router.get("/config")
def get_config():
    return {
        "HF_HOME": os.getenv("HF_HOME", ""),
        "TORCH_HOME": os.getenv("TORCH_HOME", ""),
        "PYTHONPATH": os.getenv("PYTHONPATH", ""),
        "CUDA_VISIBLE_DEVICES": os.getenv("CUDA_VISIBLE_DEVICES", ""),
    }
