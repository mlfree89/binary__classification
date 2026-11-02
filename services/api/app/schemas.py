from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    age: int = Field(ge=17, le=100)
    workclass: str
    fnlwgt: float
    education: str
    education_num: int
    marital_status: str
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(ge=0)
    capital_loss: float = Field(ge=0)
    hours_per_week: int = Field(ge=1, le=100)
    native_country: str


class PredictionResponse(BaseModel):
    prediction: str
