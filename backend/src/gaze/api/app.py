from fastapi import FastAPI
from .routes import auth, experiments, models

app = FastAPI()

app.include_router(auth.router, prefix="/auth", tags=["Authentication"])
app.include_router(experiments.router, prefix="/experiments", tags=["Experiments"])
app.include_router(models.router, prefix="/models", tags=["Models"])
