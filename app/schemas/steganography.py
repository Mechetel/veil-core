from pydantic import BaseModel, ConfigDict


class _Base(BaseModel):
    # `model_key` would otherwise clash with pydantic's protected "model_" namespace.
    model_config = ConfigDict(protected_namespaces=())


class EncodeIn(_Base):
    model_key: str
    message: str
    image_b64: str
    client_ref: str | None = None


class DecodeIn(_Base):
    model_key: str
    image_b64: str
    client_ref: str | None = None


class JobOut(BaseModel):
    id: str
    status: str


class StegModelOut(_Base):
    key: str
    family: str
    label: str
    dataset: str
    data_depth: int
    available: bool
