from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class ExperimentCreate(BaseModel):
    name: str
    model_type: str
    dataset: str
    hyperparameters: dict
    description: Optional[str] = None


class Experiment(ExperimentCreate):
    id: int
    status: str
    created_at: str


@router.post("/")
async def create_experiment(experiment: ExperimentCreate):
    # TODO: Implement experiment creation
    # - Validate experiment parameters
    # - Store in database
    # - Queue for training
    return {"message": "Experiment created successfully"}


@router.get("/")
async def list_experiments(
    model_type: Optional[str] = None, status: Optional[str] = None
):
    # TODO: Implement experiment listing
    # - Filter by model type, status
    # - Retrieve from database
    return {
        "experiments": [
            {
                "id": 1,
                "name": "Sample GAN Experiment",
                "model_type": "GAN",
                "status": "completed",
            }
        ]
    }


@router.get("/{experiment_id}")
async def get_experiment(experiment_id: int):
    # TODO: Retrieve specific experiment details
    return {"id": experiment_id, "name": "Sample Experiment", "status": "running"}
