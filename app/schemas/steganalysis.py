from pydantic import BaseModel


class AnalyzeIn(BaseModel):
    analyzer_key: str
    image_b64: str
    client_ref: str | None = None


class JobOut(BaseModel):
    id: str
    status: str


class AnalyzerOut(BaseModel):
    key: str
    arch: str
    label: str
    training: str
    available: bool
