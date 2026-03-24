from pydantic import BaseModel


class CrossDimensionCell(BaseModel):
    dim1_value: str
    dim2_value: str
    repo_count: int


class CrossDimensionResponse(BaseModel):
    dim1: str
    dim2: str
    limit: int
    pairs: list[CrossDimensionCell]
