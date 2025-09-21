from enum import StrEnum
from pydantic import BaseModel, UUID4


class Document(BaseModel):
    doc_id: UUID4
    name: str
    uri: str


class DocumentChunk(BaseModel):
    doc_id: UUID4
    text: str


class EmbeddedDocumentChunk(DocumentChunk):
    dense_vector: list[float]
    bm25_vector: dict[int, float]


class Role(StrEnum):
    agent = "agent"
    user = "user"


class Message(BaseModel):
    role: Role
    text: str


class EmbeddedRequest(BaseModel):
    query: str
    dense_vector: list[float]
    bm25_vector: dict[int, float]
