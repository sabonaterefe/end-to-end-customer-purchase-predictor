from typing import List
from pydantic import BaseModel

class FeaturesInput(BaseModel):
    features: List[float]

    class Config:
        schema_extra = {
            "example": {
                "features": [35.0, 1.0, 75000.0, 5.0, 15.0, 1.0, 5000.0, 0.0, 0.0, 1.0, 0.0, 1.0]
            }
        }
