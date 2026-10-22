from typing import Literal

from pydantic import BaseModel, Field


class InputData(BaseModel):
    """Physical measurements of one abalone, with the cleaned column names used in training."""

    sex: Literal["M", "F", "I"]
    length: float = Field(gt=0)
    diameter: float = Field(gt=0)
    height: float = Field(gt=0)
    whole_weight: float = Field(gt=0)
    shucked_weight: float = Field(gt=0)
    viscera_weight: float = Field(gt=0)
    shell_weight: float = Field(gt=0)


class PredictionOut(BaseModel):
    """Predicted number of rings of the abalone."""

    rings: float  # Pydantic models for the web service
